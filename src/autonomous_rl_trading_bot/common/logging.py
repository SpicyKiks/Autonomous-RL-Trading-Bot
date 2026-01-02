from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional


def configure_logging(
    *,
    level: str = "INFO",
    console: bool = True,
    file_paths: Optional[Iterable[Path]] = None,
    run_id: Optional[str] = None,
) -> logging.Logger:
    """
    Configure root logger handlers. Safe to call multiple times (clears previous handlers).
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = "%(asctime)s | %(levelname)s"
    if run_id:
        fmt += f" | run={run_id}"
    fmt += " | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt)

    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        root.addHandler(ch)

    if file_paths:
        for fp in file_paths:
            fp.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(fp, encoding="utf-8")
            fh.setFormatter(formatter)
            root.addHandler(fh)

    return logging.getLogger("autonomous_rl_trading_bot")
