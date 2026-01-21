#!/usr/bin/env python3
"""Project bootstrapper (cross-platform).

What it does (in order):
1) Creates a local virtualenv in .venv (unless it already exists).
2) Installs the project in editable mode with dev extras: -e .[dev]
3) Copies .env.example -> .env (if .env missing).
4) Runs pytest.

Usage:
    python scripts/bootstrap_env.py
    python scripts/bootstrap_env.py --venv .venv --skip-tests
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import venv
from pathlib import Path


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def ensure_venv(repo_root: Path, venv_dir: Path) -> Path:
    py = _venv_python(venv_dir)
    if py.exists():
        print(f"Using existing venv: {venv_dir}")
        return py

    print(f"Creating venv: {venv_dir}")
    venv_dir.parent.mkdir(parents=True, exist_ok=True)

    # On Windows, symlinks often require admin/dev mode; avoid breaking bootstrap.
    builder = venv.EnvBuilder(
        with_pip=True,
        clear=False,
        symlinks=(os.name != "nt"),
        upgrade_deps=False,
    )
    builder.create(str(venv_dir))

    py = _venv_python(venv_dir)
    if not py.exists():
        raise RuntimeError(f"Venv python not found at: {py}")

    # Upgrade pip inside venv (avoid old resolver edge cases)
    _run([str(py), "-m", "pip", "install", "-U", "pip"], cwd=repo_root)
    return py


def ensure_env_file(repo_root: Path) -> None:
    env_file = repo_root / ".env"
    example = repo_root / ".env.example"

    if env_file.exists():
        print(".env already exists — leaving it unchanged.")
        return

    if example.exists():
        shutil.copyfile(example, env_file)
        print("Created .env from .env.example")
    else:
        # Not fatal; project can still run without it.
        print("No .env.example found; skipping .env creation.")


def install_project(repo_root: Path, py: Path, *, no_deps: bool) -> None:
    """Install project in editable mode.

    By default we install deps as defined in pyproject.toml.
    Use --no-deps if you only want the editable package for quick editing (tests will fail).
    """
    cmd = [str(py), "-m", "pip", "install", "-e", ".[dev]"]
    if no_deps:
        cmd.append("--no-deps")
    _run(cmd, cwd=repo_root)


def run_tests(repo_root: Path, py: Path) -> None:
    _run([str(py), "-m", "pytest"], cwd=repo_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--venv",
        default=".venv",
        help="Path to the virtual environment directory (default: .venv)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Install only; do not run pytest.",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Do not install dependencies (only install the editable package).",
    )
    args = parser.parse_args()

    if args.no_deps and not args.skip_tests:
        print("ERROR: --no-deps skips runtime dependencies, so the test suite will fail.")
        print("Run one of:")
        print("  python scripts/bootstrap_env.py           # full install + tests")
        print("  python scripts/bootstrap_env.py --no-deps --skip-tests")
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    venv_dir = (repo_root / args.venv).resolve()

    py = ensure_venv(repo_root, venv_dir)
    ensure_env_file(repo_root)
    install_project(repo_root, py, no_deps=args.no_deps)

    if not args.skip_tests:
        run_tests(repo_root, py)

    print("\nBootstrap complete.")
    print(f"Venv python: {py}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

