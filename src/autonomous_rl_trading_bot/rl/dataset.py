from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# =========================
# Existing NPZ dataset loader (kept)
# =========================

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

    Expected files in dataset_dir:
      - meta.json
      - dataset.npz
      - optional scaler file (referenced by meta["scaler_path"])
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

    feature_list = meta.get("feature_list")

    # Filter by split if requested
    if split is not None:
        splits = meta.get("splits")
        if not splits:
            raise ValueError(
                f"Dataset does not have splits metadata, cannot filter by split={split}"
            )
        if split not in splits:
            raise ValueError(f"Unknown split: {split}. Available: {list(splits.keys())}")

        split_info = splits[split]
        start_idx = int(split_info["start_idx"])
        end_idx = int(split_info["end_idx"])

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

    if not artifacts_datasets_dir.exists():
        raise FileNotFoundError(f"datasets dir not found: {artifacts_datasets_dir}")

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

    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_dir = candidates[0][1]
    return load_dataset_npz(latest_dir)


# =========================
# Compatibility: training expects `Dataset`
# =========================

@dataclass(frozen=True)
class Dataset:
    """
    Wrapper expected by training code.

    Built on the NPZ pipeline (meta.json + dataset.npz).
    """
    dataset_id: str
    market_type: str
    df: pd.DataFrame
    meta: Dict[str, Any]
    data: Dict[str, np.ndarray]
    scaler: Optional[RobustScaler] = None
    feature_list: Optional[list[str]] = None
    dataset_dir: Optional[Path] = None

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[4]

    @classmethod
    def _find_dataset_dir(cls, dataset_id: str) -> Optional[Path]:
        root = cls._project_root()
        bases = [
            root / "artifacts" / "datasets",
            root / "datasets",
            root / "data",
            root / "artifacts",
        ]

        candidates: list[Path] = []
        for base in bases:
            if not base.exists():
                continue

            exact = base / dataset_id
            if exact.exists() and exact.is_dir():
                candidates.append(exact)

            for p in base.glob(f"*{dataset_id}*"):
                if p.is_dir():
                    candidates.append(p)

        if not candidates:
            return None

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    @staticmethod
    def _to_df(ld: LoadedDataset) -> pd.DataFrame:
        if ld.feature_list:
            cols = {}
            for name in ld.feature_list:
                if name in ld.data:
                    cols[name] = ld.data[name]
            if cols:
                return pd.DataFrame(cols)

        lengths = []
        cols = {}
        for k, v in ld.data.items():
            if isinstance(v, np.ndarray) and v.ndim == 1:
                cols[k] = v
                lengths.append(len(v))
        if cols and len(set(lengths)) == 1:
            return pd.DataFrame(cols)

        return pd.DataFrame()

    @classmethod
    def load(cls, dataset_id: str, market_type: str = "spot", split: Optional[str] = None) -> "Dataset":
        ds_dir = cls._find_dataset_dir(dataset_id)
        if ds_dir is None:
            root = cls._project_root()
            raise FileNotFoundError(
                f"Could not locate dataset_id='{dataset_id}'. Looked under:\n"
                f"  {root / 'artifacts' / 'datasets'}\n"
                f"  {root / 'datasets'}\n"
                f"  {root / 'data'}\n"
                f"  {root / 'artifacts'}"
            )

        ld = load_dataset_npz(ds_dir, split=split)
        df = cls._to_df(ld)

        return cls(
            dataset_id=ld.dataset_id,
            market_type=market_type,
            df=df,
            meta=ld.meta,
            data=ld.data,
            scaler=ld.scaler,
            feature_list=ld.feature_list,
            dataset_dir=ld.dataset_dir,
        )

    @classmethod
    def latest(cls, market_type: str = "spot") -> "Dataset":
        root = cls._project_root()
        ld = select_latest_dataset(root / "artifacts" / "datasets", market_type=market_type)
        df = cls._to_df(ld)
        return cls(
            dataset_id=ld.dataset_id,
            market_type=market_type,
            df=df,
            meta=ld.meta,
            data=ld.data,
            scaler=ld.scaler,
            feature_list=ld.feature_list,
            dataset_dir=ld.dataset_dir,
        )

    def split_time(self, train_frac: float = 0.8) -> Tuple["Dataset", "Dataset"]:
        n = len(self.df)
        if n == 0:
            raise ValueError("Dataset.df is empty; cannot split_time().")

        cut = int(n * train_frac)

        left_df = self.df.iloc[:cut].copy()
        right_df = self.df.iloc[cut:].copy()

        def split_arr(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if isinstance(a, np.ndarray) and a.ndim >= 1 and len(a) == n:
                return a[:cut], a[cut:]
            return a, a

        left_data: Dict[str, np.ndarray] = {}
        right_data: Dict[str, np.ndarray] = {}
        for k, v in self.data.items():
            l, r = split_arr(v)
            left_data[k] = l
            right_data[k] = r

        return (
            Dataset(
                dataset_id=self.dataset_id,
                market_type=self.market_type,
                df=left_df,
                meta=dict(self.meta),
                data=left_data,
                scaler=self.scaler,
                feature_list=self.feature_list,
                dataset_dir=self.dataset_dir,
            ),
            Dataset(
                dataset_id=self.dataset_id,
                market_type=self.market_type,
                df=right_df,
                meta=dict(self.meta),
                data=right_data,
                scaler=self.scaler,
                feature_list=self.feature_list,
                dataset_dir=self.dataset_dir,
            ),
        )


__all__ = [
    "LoadedDataset",
    "load_dataset_npz",
    "select_latest_dataset",
    "Dataset",
]
