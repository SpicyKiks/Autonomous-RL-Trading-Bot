from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_parquet(path: Path, df: pd.DataFrame, **kwargs) -> None:
    """Write DataFrame to Parquet file."""
    try:
        import pyarrow.parquet as pq

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, **kwargs)
    except ImportError:
        raise RuntimeError("pyarrow is required for Parquet support. Install: pip install pyarrow")


def read_parquet(path: Path, **kwargs) -> pd.DataFrame:
    """Read DataFrame from Parquet file."""
    try:
        import pyarrow.parquet as pq

        return pd.read_parquet(path, **kwargs)
    except ImportError:
        raise RuntimeError("pyarrow is required for Parquet support. Install: pip install pyarrow")

