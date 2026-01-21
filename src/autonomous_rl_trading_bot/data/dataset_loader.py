"""Unified dataset loader supporting both parquet (Day-2) and NPZ (legacy) formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autonomous_rl_trading_bot.common.paths import artifacts_dir


class Dataset:
    """Unified dataset interface."""
    
    def __init__(
        self,
        data: pd.DataFrame | dict[str, np.ndarray],
        meta: dict[str, Any],
        format: str,  # "parquet" or "npz"
        path: Path,
    ):
        self.data = data
        self.meta = meta
        self.format = format
        self.path = path
    
    @property
    def is_parquet(self) -> bool:
        """Check if dataset is parquet format."""
        return self.format == "parquet"
    
    @property
    def is_npz(self) -> bool:
        """Check if dataset is NPZ format."""
        return self.format == "npz"


def resolve_dataset(
    symbol: str | None = None,
    interval: str | None = None,
    dataset_id: str | None = None,
    dataset_path: str | Path | None = None,
    mode: str | None = None,
) -> Dataset:
    """
    Unified dataset resolver with format detection.
    
    Priority order:
    1. Explicit dataset_path (if provided)
    2. dataset_id (if provided) - looks in artifacts/datasets/
    3. symbol + interval (if provided) - looks for parquet in data/processed/
    4. Latest dataset for mode (if mode provided)
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        interval: Timeframe interval (e.g., "1m")
        dataset_id: Dataset ID (for NPZ format)
        dataset_path: Explicit path to dataset file or directory
        mode: Market mode (spot/futures) for fallback lookup
    
    Returns:
        Dataset object with unified interface
    
    Raises:
        FileNotFoundError: If no dataset found
    """
    # 1. Try explicit path
    if dataset_path:
        path = Path(dataset_path)
        if path.is_file() and path.suffix == ".parquet":
            return _load_parquet(path)
        elif path.is_file() and path.name == "dataset.npz":
            return _load_npz(path.parent)
        elif path.is_dir():
            # Check for parquet or NPZ in directory
            parquet_files = list(path.glob("*.parquet"))
            if parquet_files:
                return _load_parquet(parquet_files[0])
            if (path / "dataset.npz").exists():
                return _load_npz(path)
            raise FileNotFoundError(f"No dataset found in directory: {path}")
        else:
            raise FileNotFoundError(f"Invalid dataset path: {path}")
    
    # 2. Try dataset_id (NPZ format)
    if dataset_id:
        dataset_dir = artifacts_dir() / "datasets" / dataset_id
        if dataset_dir.exists():
            return _load_npz(dataset_dir)
        # Fallback to runs directory (legacy)
        runs_dir = artifacts_dir() / "runs" / dataset_id
        if runs_dir.exists() and (runs_dir / "dataset.npz").exists():
            return _load_npz(runs_dir)
        raise FileNotFoundError(f"Dataset not found for dataset_id: {dataset_id}")
    
    # 3. Try symbol + interval (parquet format)
    if symbol and interval:
        parquet_path = Path("data/processed") / f"{symbol.upper()}_{interval}_dataset.parquet"
        if parquet_path.exists():
            return _load_parquet(parquet_path)
    
    # 4. Try latest dataset for mode (NPZ format)
    if mode:
        datasets_dir = artifacts_dir() / "datasets"
        if datasets_dir.exists():
            candidates = []
            for p in datasets_dir.iterdir():
                if not p.is_dir():
                    continue
                meta_path = p / "meta.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        dataset_mode = str(meta.get("market_type", "")).strip().lower()
                        if dataset_mode == mode.lower():
                            candidates.append((p, p.stat().st_mtime))
                    except Exception:
                        continue
            
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                latest_dir = candidates[0][0]
                return _load_npz(latest_dir)
    
    # No dataset found
    error_msg = "Could not resolve dataset. Tried:\n"
    if dataset_path:
        error_msg += f"  - Explicit path: {dataset_path}\n"
    if dataset_id:
        error_msg += f"  - Dataset ID: {dataset_id}\n"
    if symbol and interval:
        error_msg += f"  - Parquet: data/processed/{symbol.upper()}_{interval}_dataset.parquet\n"
    if mode:
        error_msg += f"  - Latest NPZ for mode: {mode}\n"
    raise FileNotFoundError(error_msg)


def _load_parquet(path: Path) -> Dataset:
    """Load parquet dataset."""
    df = pd.read_parquet(path)
    
    # Extract metadata from DataFrame if available
    meta = {
        "format": "parquet",
        "path": str(path),
        "rows": len(df),
    }
    
    # Try to infer symbol/interval from filename
    stem = path.stem
    if "_dataset" in stem:
        parts = stem.replace("_dataset", "").split("_")
        if len(parts) >= 2:
            meta["symbol"] = parts[0]
            meta["interval"] = parts[1]
    
    return Dataset(data=df, meta=meta, format="parquet", path=path)


def _load_npz(dataset_dir: Path) -> Dataset:
    """Load NPZ dataset."""
    meta_path = dataset_dir / "meta.json"
    npz_path = dataset_dir / "dataset.npz"
    
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in: {dataset_dir}")
    if not npz_path.exists():
        raise FileNotFoundError(f"dataset.npz not found in: {dataset_dir}")
    
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays = dict(np.load(npz_path, allow_pickle=False))
    
    return Dataset(data=arrays, meta=meta, format="npz", path=dataset_dir)
