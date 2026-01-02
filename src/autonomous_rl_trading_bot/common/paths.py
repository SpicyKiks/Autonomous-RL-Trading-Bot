from __future__ import annotations

from pathlib import Path

from .exceptions import PathError


def repo_root() -> Path:
    """
    Resolve repository root by walking up from this file until we find pyproject.toml.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise PathError("Could not locate repo root (pyproject.toml not found in parents).")


def configs_dir() -> Path:
    return repo_root() / "configs"


def artifacts_dir() -> Path:
    return repo_root() / "artifacts"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_artifact_tree() -> None:
    """
    Ensure artifact subfolders exist (datasets/runs/models/reports/logs/db).
    """
    base = artifacts_dir()
    ensure_dir(base / "datasets")
    ensure_dir(base / "runs")
    ensure_dir(base / "models")
    ensure_dir(base / "reports")
    ensure_dir(base / "logs")
    ensure_dir(base / "db")
