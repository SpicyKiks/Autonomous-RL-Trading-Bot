from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from autonomous_rl_trading_bot.common.paths import artifacts_dir, ensure_artifact_tree


@dataclass(frozen=True)
class ArtifactIndex:
    datasets: List[str]
    runs: List[str]


def _list_subdirs(p: Path) -> list[str]:
    if not p.exists():
        return []
    return sorted([x.name for x in p.iterdir() if x.is_dir()])


def load_artifact_index() -> ArtifactIndex:
    """Return the current datasets/runs present under artifacts/.

    This is intentionally filesystem-only (no network).
    """
    ensure_artifact_tree()
    base = artifacts_dir()
    return ArtifactIndex(
        datasets=_list_subdirs(base / "datasets"),
        runs=_list_subdirs(base / "runs"),
    )

