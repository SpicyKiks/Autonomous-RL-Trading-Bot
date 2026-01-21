from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autonomous_rl_trading_bot.common.paths import ensure_artifact_tree


def _utc_compact() -> str:
    # 20260104T163012Z
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_segment(s: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in (s or ""))
    out = out.strip("_")
    return out or "x"


@dataclass(frozen=True)
class ArtifactStore:
    """Lightweight helper for managing the project artifact tree.

    This is intentionally tiny: deterministic layout + convenience helpers.

    Default layout (under artifacts/):
      - datasets/<dataset_id>/...
      - runs/<run_id>/run.json
      - backtests/<backtest_id>/{equity.csv,trades.csv,metrics.json,...}
      - models/
      - reports/
      - logs/
      - db/
    """

    base_dir: Path

    @staticmethod
    def default() -> ArtifactStore:
        # Import lazily to avoid circulars
        from autonomous_rl_trading_bot.common.paths import artifacts_dir

        ensure_artifact_tree()
        return ArtifactStore(base_dir=artifacts_dir())

    def datasets_dir(self) -> Path:
        p = self.base_dir / "datasets"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def runs_dir(self) -> Path:
        p = self.base_dir / "runs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def backtests_dir(self) -> Path:
        p = self.base_dir / "backtests"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def logs_dir(self) -> Path:
        p = self.base_dir / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def db_dir(self) -> Path:
        p = self.base_dir / "db"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def new_run_id(self, *, kind: str, mode: str) -> str:
        # Not deterministic by design; deterministic results are validated separately.
        return f"run_{_safe_segment(kind)}_{_safe_segment(mode)}_{_utc_compact()}_{uuid.uuid4().hex[:8]}"

    def run_dir(self, run_id: str) -> Path:
        p = self.runs_dir() / _safe_segment(run_id)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def backtest_dir(self, backtest_id: str) -> Path:
        p = self.backtests_dir() / _safe_segment(backtest_id)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.datasets_dir() / _safe_segment(dataset_id)

    def latest_dataset_dir(
        self,
        *,
        market_type: str | None = None,
        symbol: str | None = None,
        interval: str | None = None,
    ) -> Path | None:
        """Best-effort latest dataset resolver.

        Filters by (market_type/symbol/interval) if provided. Uses meta.json when present.
        Falls back to directory mtime.
        """
        base = self.datasets_dir()
        if not base.exists():
            return None

        best: Path | None = None
        best_key: float = -1.0

        for d in base.iterdir():
            if not d.is_dir():
                continue

            meta_path = d / "meta.json"
            meta: dict[str, Any] = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}

            if market_type and str(meta.get("market_type") or "").lower() != market_type.lower():
                continue
            if symbol and str(meta.get("symbol") or "").upper() != symbol.upper():
                continue
            if interval and str(meta.get("interval") or "") != interval:
                continue

            key = d.stat().st_mtime
            if key > best_key:
                best_key = key
                best = d

        return best

