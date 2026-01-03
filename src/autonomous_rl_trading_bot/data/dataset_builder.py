from __future__ import annotations

import json
import math
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import RobustScaler

from autonomous_rl_trading_bot.common.timeframes import interval_to_ms


@dataclass(frozen=True)
class QualityReport:
    interval_ms: int
    total_rows: int
    unique_open_times: int
    duplicates: int
    gaps: int
    first_open_time_ms: Optional[int]
    last_open_time_ms: Optional[int]
    first_gap_after_ms: Optional[int]
    max_gap_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interval_ms": self.interval_ms,
            "total_rows": self.total_rows,
            "unique_open_times": self.unique_open_times,
            "duplicates": self.duplicates,
            "gaps": self.gaps,
            "first_open_time_ms": self.first_open_time_ms,
            "last_open_time_ms": self.last_open_time_ms,
            "first_gap_after_ms": self.first_gap_after_ms,
            "max_gap_ms": self.max_gap_ms,
        }


@dataclass(frozen=True)
class DatasetResult:
    dataset_id: str
    out_dir: Path
    report: QualityReport
    window_points: int
    candles_used: int
    features: List[str]
    npz_path: Path
    csv_path: Path
    meta_path: Path


def _query_candles(
    conn: sqlite3.Connection,
    *,
    market_type: str,
    symbol: str,
    interval: str,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Returns rows as: (open_time_ms, open, high, low, close, volume) ordered ASC by open_time_ms.
    """
    where = ["market_type = ?", "symbol = ?", "interval = ?"]
    params: List[Any] = [market_type, symbol.upper(), interval]

    if start_ms is not None:
        where.append("open_time_ms >= ?")
        params.append(int(start_ms))
    if end_ms is not None:
        where.append("open_time_ms <= ?")
        params.append(int(end_ms))

    sql = (
        "SELECT open_time_ms, open, high, low, close, volume "
        "FROM candles "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY open_time_ms ASC"
    )
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))

    rows = conn.execute(sql, params).fetchall()
    out: List[Tuple[int, float, float, float, float, float]] = []
    for r in rows:
        # sqlite Row or tuple
        ot = int(r[0])
        out.append((ot, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
    return out


def validate_candles(open_times: Sequence[int], interval_ms: int) -> QualityReport:
    if not open_times:
        return QualityReport(
            interval_ms=interval_ms,
            total_rows=0,
            unique_open_times=0,
            duplicates=0,
            gaps=0,
            first_open_time_ms=None,
            last_open_time_ms=None,
            first_gap_after_ms=None,
            max_gap_ms=0,
        )

    total = len(open_times)
    sorted_times = np.array(open_times, dtype=np.int64)
    # Candle query is ASC, but enforce monotonic
    sorted_times = np.sort(sorted_times)

    unique = np.unique(sorted_times)
    dup = int(total - unique.size)

    diffs = np.diff(unique)
    gaps_idx = np.where(diffs != interval_ms)[0]
    gaps = int(gaps_idx.size)

    first_gap_after = int(unique[gaps_idx[0]]) if gaps > 0 else None
    max_gap = int(diffs[gaps_idx].max()) if gaps > 0 else 0

    return QualityReport(
        interval_ms=interval_ms,
        total_rows=total,
        unique_open_times=int(unique.size),
        duplicates=dup,
        gaps=gaps,
        first_open_time_ms=int(unique[0]),
        last_open_time_ms=int(unique[-1]),
        first_gap_after_ms=first_gap_after,
        max_gap_ms=max_gap,
    )


def _latest_contiguous_segment(
    rows: List[Tuple[int, float, float, float, float, float]],
    interval_ms: int,
    required_points: int,
) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Finds the latest contiguous segment with >= required_points candles and returns
    the last required_points from that segment.

    Contiguous means consecutive open_time_ms differ by exactly interval_ms.
    """
    if required_points <= 0:
        raise ValueError("required_points must be > 0")
    if not rows:
        return []

    # Dedup by open_time_ms (keep last occurrence)
    dedup: Dict[int, Tuple[int, float, float, float, float, float]] = {}
    for row in rows:
        dedup[int(row[0])] = row
    times_sorted = sorted(dedup.keys())

    # Walk from end backwards building segments
    segment: List[Tuple[int, float, float, float, float, float]] = []
    prev_t: Optional[int] = None

    for t in reversed(times_sorted):
        if prev_t is None:
            segment = [dedup[t]]
        else:
            if prev_t - t == interval_ms:
                segment.append(dedup[t])
            else:
                # segment break; check length
                if len(segment) >= required_points:
                    break
                segment = [dedup[t]]
        prev_t = t

    if len(segment) < required_points:
        return []

    # segment is reversed time (latest->older). Reverse to ASC and take last required_points.
    segment_asc = list(reversed(segment))
    return segment_asc[-required_points:]


def compute_features(
    segment: List[Tuple[int, float, float, float, float, float]],
    features: List[str],
) -> Dict[str, np.ndarray]:
    """
    Produces arrays aligned to time index.
    Base arrays are length T.
    Returns/log_return are length T (first value is 0).
    """
    if not segment:
        raise ValueError("segment is empty")

    times = np.array([r[0] for r in segment], dtype=np.int64)
    o = np.array([r[1] for r in segment], dtype=np.float64)
    h = np.array([r[2] for r in segment], dtype=np.float64)
    low = np.array([r[3] for r in segment], dtype=np.float64)
    c = np.array([r[4] for r in segment], dtype=np.float64)
    v = np.array([r[5] for r in segment], dtype=np.float64)

    out: Dict[str, np.ndarray] = {
        "open_time_ms": times,
        "open": o,
        "high": h,
        "low": low,
        "close": c,
        "volume": v,
    }

    # Returns (aligned length T; r[0]=0)
    if "return" in features or "log_return" in features:
        ret = np.zeros_like(c)
        ret[1:] = (c[1:] / c[:-1]) - 1.0
        out["return"] = ret

    if "log_return" in features:
        eps = 1e-12
        lr = np.zeros_like(c)
        lr[1:] = np.log((c[1:] + eps) / (c[:-1] + eps))
        out["log_return"] = lr

    return out


def _compute_splits(n: int, train_frac: float, val_frac: float, test_frac: float) -> Dict[str, Dict[str, int]]:
    """
    Compute chronological split boundaries (inclusive start, exclusive end).
    Returns dict with 'train', 'val', 'test' each containing 'start_idx' and 'end_idx'.
    """
    total = float(train_frac + val_frac + test_frac)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")
    
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    test_end = n
    
    return {
        "train": {"start_idx": 0, "end_idx": train_end},
        "val": {"start_idx": train_end, "end_idx": val_end},
        "test": {"start_idx": val_end, "end_idx": test_end},
    }


def _build_feature_list(features: List[str], arrays: Dict[str, np.ndarray]) -> List[str]:
    """
    Build ordered feature list for scaling/observation.
    Includes: log_return, return, close_norm, vol_norm (if available).
    Order: log_return, return, close_norm, vol_norm
    """
    feature_list = []
    
    # Always include these if available (in fixed order)
    if "log_return" in arrays:
        feature_list.append("log_return")
    if "return" in arrays:
        feature_list.append("return")
    if "close" in arrays:
        feature_list.append("close_norm")
    if "volume" in arrays:
        feature_list.append("vol_norm")
    
    if not feature_list:
        raise ValueError("No features available for feature_list")
    
    return feature_list


def _prepare_scaled_features(
    arrays: Dict[str, np.ndarray],
    feature_list: List[str],
    splits: Dict[str, Dict[str, int]],
    scaler_type: str = "robust",
) -> Tuple[Dict[str, np.ndarray], RobustScaler]:
    """
    Prepare scaled features. Fit scaler on train split only, apply to all splits.
    Returns (arrays_with_scaled_features, fitted_scaler).
    """
    n = len(arrays["close"])
    
    # Build feature matrix (n, feature_dim)
    feature_matrix = []
    for feat_name in feature_list:
        if feat_name == "close_norm":
            first_close = float(arrays["close"][0])
            feat = arrays["close"] / max(first_close, 1e-12)
        elif feat_name == "vol_norm":
            feat = np.log1p(np.maximum(arrays["volume"], 0.0))
        else:
            if feat_name not in arrays:
                raise KeyError(f"Feature {feat_name} not found in arrays")
            feat = arrays[feat_name]
        feature_matrix.append(feat)
    
    feature_matrix = np.stack(feature_matrix, axis=1).astype(np.float64)  # (n, feature_dim)
    
    # Fit scaler on train split only
    train_start = splits["train"]["start_idx"]
    train_end = splits["train"]["end_idx"]
    train_data = feature_matrix[train_start:train_end]
    
    if scaler_type.lower() == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported scaler_type: {scaler_type}")
    
    scaler.fit(train_data)
    
    # Transform all data
    scaled_features = scaler.transform(feature_matrix).astype(np.float32)
    
    # Add scaled features to arrays dict
    arrays_scaled = dict(arrays)
    for i, feat_name in enumerate(feature_list):
        arrays_scaled[f"{feat_name}_scaled"] = scaled_features[:, i]
    
    return arrays_scaled, scaler


def write_dataset_files(
    out_dir: Path,
    dataset_id: str,
    arrays: Dict[str, np.ndarray],
    report: QualityReport,
    meta_extra: Optional[Dict[str, Any]] = None,
    scaler: Optional[RobustScaler] = None,
    scaler_path: Optional[Path] = None,
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save scaler if provided
    if scaler is not None and scaler_path is not None:
        with scaler_path.open("wb") as f:
            pickle.dump(scaler, f)

    npz_path = out_dir / "dataset.npz"
    np.savez_compressed(npz_path, **arrays)

    # CSV for human inspection
    csv_path = out_dir / "dataset.csv"
    cols = list(arrays.keys())
    # Order columns nicely if present
    preferred = [
        "open_time_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return",
        "log_return",
    ]
    cols = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        n = len(next(iter(arrays.values())))
        for i in range(n):
            row = []
            for c in cols:
                val = arrays[c][i]
                if isinstance(val, (np.integer, int)):
                    row.append(str(int(val)))
                else:
                    row.append(str(float(val)))
            f.write(",".join(row) + "\n")

    meta = {
        "dataset_id": dataset_id,
        "rows": int(len(next(iter(arrays.values())))),
        "columns": cols,
        "quality_report": report.to_dict(),
    }
    if meta_extra:
        meta.update(meta_extra)
    
    if scaler_path is not None:
        meta["scaler_path"] = str(scaler_path.relative_to(out_dir))

    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return npz_path, csv_path, meta_path


def build_dataset_from_db(
    conn: sqlite3.Connection,
    *,
    market_type: str,
    symbol: str,
    interval: str,
    window_minutes: int,
    strict_gaps: bool,
    features: List[str],
    out_base_dir: Path,
    dataset_id: str,
    end_ms: Optional[int] = None,
    train_frac: float = 0.75,
    val_frac: float = 0.10,
    test_frac: float = 0.15,
    scaler_type: str = "robust",
) -> DatasetResult:
    interval_ms = interval_to_ms(interval)
    required_points = int(math.ceil((window_minutes * 60_000) / interval_ms))
    if required_points < 2:
        required_points = 2

    # Pull a larger set to be able to find a contiguous segment.
    # We'll query ~3x the required range.
    end_time = int(end_ms) if end_ms is not None else None
    limit = required_points * 3

    rows = _query_candles(
        conn,
        market_type=market_type,
        symbol=symbol,
        interval=interval,
        start_ms=None,
        end_ms=end_time,
        limit=limit,
    )

    open_times = [r[0] for r in rows]
    report = validate_candles(open_times, interval_ms)

    if strict_gaps and (report.gaps > 0 or report.duplicates > 0):
        raise ValueError(
            f"Dataset strict_gaps=true and quality failed: gaps={report.gaps}, duplicates={report.duplicates}"
        )

    segment = _latest_contiguous_segment(rows, interval_ms, required_points)
    if not segment:
        raise ValueError(
            f"Not enough contiguous candles. Need {required_points} points for {window_minutes} minutes at {interval}."
        )

    arrays = compute_features(segment, features)
    n = len(arrays["close"])
    
    # Compute splits
    splits = _compute_splits(n, train_frac, val_frac, test_frac)
    
    # Build feature list
    feature_list = _build_feature_list(features, arrays)
    
    # Prepare scaled features
    arrays_scaled, scaler = _prepare_scaled_features(arrays, feature_list, splits, scaler_type)
    
    # Save scaler
    out_dir = out_base_dir / dataset_id
    scaler_path = out_dir / "scaler.pkl"
    
    # Get timestamps for splits
    open_time_ms = arrays["open_time_ms"]
    splits_with_timestamps = {
        "train": {
            **splits["train"],
            "start_time_ms": int(open_time_ms[splits["train"]["start_idx"]]) if splits["train"]["start_idx"] < n else None,
            "end_time_ms": int(open_time_ms[splits["train"]["end_idx"] - 1]) if splits["train"]["end_idx"] > splits["train"]["start_idx"] and splits["train"]["end_idx"] <= n else None,
        },
        "val": {
            **splits["val"],
            "start_time_ms": int(open_time_ms[splits["val"]["start_idx"]]) if splits["val"]["start_idx"] < n and splits["val"]["end_idx"] > splits["val"]["start_idx"] else None,
            "end_time_ms": int(open_time_ms[splits["val"]["end_idx"] - 1]) if splits["val"]["end_idx"] > splits["val"]["start_idx"] and splits["val"]["end_idx"] <= n else None,
        },
        "test": {
            **splits["test"],
            "start_time_ms": int(open_time_ms[splits["test"]["start_idx"]]) if splits["test"]["start_idx"] < n and splits["test"]["end_idx"] > splits["test"]["start_idx"] else None,
            "end_time_ms": int(open_time_ms[splits["test"]["end_idx"] - 1]) if splits["test"]["end_idx"] > splits["test"]["start_idx"] and splits["test"]["end_idx"] <= n else None,
        },
    }
    
    npz_path, csv_path, meta_path = write_dataset_files(
        out_dir,
        dataset_id,
        arrays_scaled,
        report,
        meta_extra={
            "market_type": market_type,
            "symbol": symbol.upper(),
            "interval": interval,
            "window_minutes": window_minutes,
            "required_points": required_points,
            "features": features,
            "feature_list": feature_list,
            "splits": splits_with_timestamps,
            "scaler_type": scaler_type,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": test_frac,
        },
        scaler=scaler,
        scaler_path=scaler_path,
    )

    return DatasetResult(
        dataset_id=dataset_id,
        out_dir=out_dir,
        report=report,
        window_points=required_points,
        candles_used=int(len(segment)),
        features=features,
        npz_path=npz_path,
        csv_path=csv_path,
        meta_path=meta_path,
    )
