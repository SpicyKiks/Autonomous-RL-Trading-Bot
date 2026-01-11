from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def safe_json_dump(path: Path, obj: Any, *, indent: int = 2) -> None:
    """
    Atomically write JSON (write temp then replace) to avoid partial/corrupt artifacts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.write("\n")
    tmp.replace(path)
