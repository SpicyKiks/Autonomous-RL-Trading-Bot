from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_features(
    run_dir: Path,
    name: str,
    X: np.ndarray,
    columns: list[str],
    index_ts: np.ndarray | None = None,
) -> None:
    """
    Save feature matrix to disk.

    Args:
        run_dir: Directory to save to
        name: Feature set name
        X: Feature matrix (n_samples, n_features)
        columns: Feature column names
        index_ts: Optional timestamp index
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV for simplicity
    df = pd.DataFrame(X, columns=columns)
    if index_ts is not None:
        df.index = pd.to_datetime(index_ts, unit="ms")
    df.to_csv(run_dir / f"{name}_features.csv")


def load_features(run_dir: Path, name: str) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    """
    Load feature matrix from disk.

    Args:
        run_dir: Directory to load from
        name: Feature set name

    Returns:
        Tuple of (X, columns, index_ts)
    """
    run_dir = Path(run_dir)
    csv_path = run_dir / f"{name}_features.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Feature file not found: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)
    columns = list(df.columns)
    X = df.values.astype(np.float32)

    index_ts = None
    if isinstance(df.index, pd.DatetimeIndex):
        index_ts = df.index.astype(np.int64) // 1_000_000  # Convert to ms

    return X, columns, index_ts

