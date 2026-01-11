from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set

ZIP_NAME = "code_only.zip"

# Preferred includes (zip what exists)
INCLUDE_DIRS = ["src", "tests", "configs", "sql", "tools"]
INCLUDE_FILES = ["pyproject.toml", "README.md", ".gitignore", "LICENSE"]
RUNNER_GLOBS = ["run_*.py"]

# Exclude dirs anywhere
EXCLUDE_DIR_NAMES = {
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    ".venv_test",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".eggs",
    ".ipynb_checkpoints",
    "node_modules",
    "dist",
    "build",
    "artifacts",
    "outputs",
    "runs",
    "logs",
    "data",
}

# Exclude file extensions (binary/large)
EXCLUDE_EXTS = {
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".pdf",
    ".onnx",
    ".pt",
    ".pth",
    ".ckpt",
    ".npz",
    ".npy",
    ".parquet",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pkl",
    ".joblib",
    ".exe",
    ".dll",
    ".bin",
}

MAX_FILE_BYTES = 30 * 1024 * 1024  # 30MB


@dataclass(frozen=True)
class CollectedFile:
    abs_path: Path
    rel_path: Path
    size: int


def repo_root() -> Path:
    # tools/make_code_zip.py -> repo root is parent of tools/
    return Path(__file__).resolve().parents[1]


def _is_egg_info(rel_path: Path) -> bool:
    return any(part.endswith(".egg-info") for part in rel_path.parts)


def _has_excluded_dir(rel_path: Path) -> bool:
    if _is_egg_info(rel_path):
        return True
    return any(part in EXCLUDE_DIR_NAMES for part in rel_path.parts)


def _is_excluded_file(rel_path: Path, abs_path: Path) -> bool:
    if rel_path.name == ZIP_NAME:
        return True
    if rel_path.suffix.lower() in EXCLUDE_EXTS:
        return True
    try:
        if abs_path.stat().st_size > MAX_FILE_BYTES:
            return True
    except OSError:
        return True
    return False


def _iter_dir(root: Path, rel_dir: str) -> Iterable[CollectedFile]:
    base = root / rel_dir
    if not base.exists() or not base.is_dir():
        return
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if _has_excluded_dir(rel):
            continue
        if _is_excluded_file(rel, p):
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        yield CollectedFile(abs_path=p, rel_path=rel, size=sz)


def _iter_root_files(root: Path) -> Iterable[CollectedFile]:
    # explicit root files
    for rel in INCLUDE_FILES:
        p = root / rel
        if p.exists() and p.is_file():
            relp = p.relative_to(root)
            if not _has_excluded_dir(relp) and not _is_excluded_file(relp, p):
                yield CollectedFile(abs_path=p, rel_path=relp, size=p.stat().st_size)

    # runner scripts
    for g in RUNNER_GLOBS:
        for p in root.glob(g):
            if not p.is_file():
                continue
            relp = p.relative_to(root)
            if _has_excluded_dir(relp) or _is_excluded_file(relp, p):
                continue
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            yield CollectedFile(abs_path=p, rel_path=relp, size=sz)

    # FORCE include this script itself (guarantees tools/ appears if script is in tools/)
    self_path = Path(__file__).resolve()
    try:
        self_rel = self_path.relative_to(root)
    except ValueError:
        self_rel = None
    if self_rel is not None and not _has_excluded_dir(self_rel) and not _is_excluded_file(self_rel, self_path):
        yield CollectedFile(abs_path=self_path, rel_path=self_rel, size=self_path.stat().st_size)


def _human(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    u = 0
    while x >= 1024.0 and u < len(units) - 1:
        x /= 1024.0
        u += 1
    return f"{x:.1f} {units[u]}"


def collect_all(root: Path) -> List[CollectedFile]:
    out: List[CollectedFile] = []
    seen: Set[str] = set()

    for d in INCLUDE_DIRS:
        for fi in _iter_dir(root, d):
            k = fi.rel_path.as_posix()
            if k not in seen:
                seen.add(k)
                out.append(fi)

    for fi in _iter_root_files(root):
        k = fi.rel_path.as_posix()
        if k not in seen:
            seen.add(k)
            out.append(fi)

    out.sort(key=lambda f: f.rel_path.as_posix())
    return out


def main() -> int:
    root = repo_root()
    out_zip = root / ZIP_NAME

    print(f"[make_code_zip] Repo root : {root}")
    print(f"[make_code_zip] Output    : {out_zip}")

    files = collect_all(root)
    if not files:
        print("[make_code_zip] ERROR: No files collected.")
        return 2

    total_raw = sum(f.size for f in files)
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f.abs_path, f.rel_path.as_posix())

    zip_size = out_zip.stat().st_size
    print(f"[make_code_zip] Files     : {len(files)}")
    print(f"[make_code_zip] Raw size  : {_human(total_raw)}")
    print(f"[make_code_zip] Zip size  : {_human(zip_size)}")

    # Sanity checks (do NOT hard-fail if optional dirs missing)
    existing_includes = [d for d in INCLUDE_DIRS if (root / d).exists() and (root / d).is_dir()]
    missing_in_zip = []
    for d in existing_includes:
        prefix = f"{d}/"
        if not any(fi.rel_path.as_posix().startswith(prefix) for fi in files):
            missing_in_zip.append(prefix)

    if missing_in_zip:
        print("[make_code_zip] WARNING: Some existing include folders had no files collected:")
        for m in missing_in_zip:
            print(f"  - {m}")

    if any(".egg-info/" in fi.rel_path.as_posix() for fi in files):
        print("[make_code_zip] ERROR: .egg-info artifacts were included (should be excluded).")
        return 4

    largest = sorted(files, key=lambda f: f.size, reverse=True)[:15]
    print("[make_code_zip] Top 15 largest:")
    for i, fi in enumerate(largest, 1):
        print(f"  {i:2d}. {_human(fi.size):>10}  {fi.rel_path.as_posix()}")

    print("[make_code_zip] OK: code_only.zip generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
