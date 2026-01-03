from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class LoadedDataset:
    dataset_id: str
    dataset_dir: Path
    npz_path: Path
    meta: Dict[str, Any]
    data: Dict[str, np.ndarray]
    scaler: Optional[RobustScaler] = None
    feature_list: Optional[list[str]] = None


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def load_dataset_npz(dataset_dir: Path, split: Optional[str] = None) -> LoadedDataset:
    """
    Load dataset from directory. Optionally filter by split.
    
    Args:
        dataset_dir: Directory containing meta.json and dataset.npz
        split: Optional split name ('train', 'val', 'test') to filter data
    
    Returns:
        LoadedDataset with potentially filtered data based on split
    """
    meta_path = dataset_dir / "meta.json"
    npz_path = dataset_dir / "dataset.npz"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {dataset_dir}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing dataset.npz in {dataset_dir}")
    meta = _read_json(meta_path)
    dataset_id = meta.get("dataset_id") or dataset_dir.name

    npz = np.load(npz_path)
    data = {k: npz[k] for k in npz.files}
    
    # Load scaler if available
    scaler = None
    scaler_path_str = meta.get("scaler_path")
    if scaler_path_str:
        scaler_path = dataset_dir / scaler_path_str
        if scaler_path.exists():
            with scaler_path.open("rb") as f:
                scaler = pickle.load(f)
    
    # Get feature_list from meta
    feature_list = meta.get("feature_list")
    
    # Filter by split if requested
    if split is not None:
        splits = meta.get("splits")
        if not splits:
            raise ValueError(f"Dataset does not have splits metadata, cannot filter by split={split}")
        if split not in splits:
            raise ValueError(f"Unknown split: {split}. Available: {list(splits.keys())}")
        
        split_info = splits[split]
        start_idx = split_info["start_idx"]
        end_idx = split_info["end_idx"]
        
        # Filter all arrays
        data = {k: v[start_idx:end_idx] for k, v in data.items()}
    
    return LoadedDataset(
        dataset_id=str(dataset_id),
        dataset_dir=dataset_dir,
        npz_path=npz_path,
        meta=meta,
        data=data,
        scaler=scaler,
        feature_list=feature_list,
    )


def select_latest_dataset(artifacts_datasets_dir: Path, market_type: str) -> LoadedDataset:
    """
    Select latest dataset folder by mtime where meta.json.market_type matches.
    """
    if market_type not in ("spot", "futures"):
        raise ValueError(f"market_type must be spot|futures, got {market_type}")

    candidates: list[Tuple[float, Path]] = []
    for p in artifacts_datasets_dir.iterdir():
        if not p.is_dir():
            continue
        meta_path = p / "meta.json"
        npz_path = p / "dataset.npz"
        if not meta_path.exists() or not npz_path.exists():
            continue
        try:
            meta = _read_json(meta_path)
        except Exception:
            continue
        if meta.get("market_type") != market_type:
            continue
        candidates.append((p.stat().st_mtime, p))

    if not candidates:
        raise FileNotFoundError(
            f"No datasets found in {artifacts_datasets_dir} for market_type={market_type}"
        )

    candidates.sort(key=lambda x: x[0])
    latest_dir = candidates[-1][1]
    return load_dataset_npz(latest_dir)

