from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger("arbt")


def safe_git_rev_parse_head() -> str:
    """
    Safely get git HEAD revision hash.
    
    Returns "unknown" if git is not available or if the repo is not a git checkout.
    Never prints to stdout/stderr on failure.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"git metadata unavailable: {type(e).__name__}")
    return "unknown"


def safe_git_describe() -> str:
    """
    Safely get git describe output.
    
    Returns "unknown" if git is not available or if the repo is not a git checkout.
    Never prints to stdout/stderr on failure.
    """
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"git metadata unavailable: {type(e).__name__}")
    return "unknown"
