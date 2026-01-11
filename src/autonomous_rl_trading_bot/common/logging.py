from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union, Any


def _to_level(level: Union[int, str, None]) -> int:
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    s = str(level).strip().upper()
    return getattr(logging, s, logging.INFO)


def configure_logging(
    name: str = "arbt",
    log_dir: Optional[Union[str, Path]] = None,
    log_file: Optional[str] = None,
    level: Union[int, str, None] = "INFO",
    console: bool = True,
    file: bool = True,
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    propagate: bool = False,
    **_: Any,  # swallow unexpected kwargs safely (future-proof)
) -> logging.Logger:
    """
    Configure and return a logger.

    - Accepts `level=` because run_train.py passes it.
    - Keeps backward compatibility for older callers via get_logger().
    """
    lvl = _to_level(level)

    logger = logging.getLogger(name)
    logger.setLevel(lvl)
    logger.propagate = propagate

    # Clear existing handlers to prevent duplicate logs during re-runs
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(lvl)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if file:
        # Resolve log directory
        if log_dir is None:
            log_dir_path = Path.cwd() / "artifacts" / "logs"
        else:
            log_dir_path = Path(log_dir)

        log_dir_path.mkdir(parents=True, exist_ok=True)

        # Resolve log filename
        filename = log_file or f"{name}.log"
        fh = logging.FileHandler(log_dir_path / filename, encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "arbt", level: Union[int, str, None] = None) -> logging.Logger:
    """
    Backwards-compatible logger accessor used by older modules (e.g. trainer.py).

    If the logger isn't configured yet, configure it with sane defaults.
    If it is configured, just return it (and optionally update its level).
    """
    logger = logging.getLogger(name)

    # If nothing configured yet, configure it now.
    if not logger.handlers:
        configure_logging(name=name, level=level or "INFO")
        logger = logging.getLogger(name)

    # Allow level override without reconfiguring handlers
    if level is not None:
        lvl = _to_level(level)
        logger.setLevel(lvl)
        for h in logger.handlers:
            h.setLevel(lvl)

    return logger


__all__ = ["configure_logging", "get_logger"]
