from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.write("\n")
    tmp.replace(path)


def copytree_merge(src: Path, dst: Path) -> None:
    """
    Merge-copy src directory into dst (creates dst if missing).
    Existing files are overwritten.
    """
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst)

    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
