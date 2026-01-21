from __future__ import annotations

import pandas as pd


def asof_align(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: str = "timestamp",
    by: str | None = None,
    tolerance: pd.Timedelta | None = None,
    direction: str = "backward",
) -> pd.DataFrame:
    """
    Align two DataFrames using pandas merge_asof to prevent lookahead bias.

    Args:
        left_df: Base DataFrame (will keep all rows)
        right_df: DataFrame to align to left_df
        on: Column name to merge on (must be sorted)
        by: Optional grouping column
        tolerance: Maximum time distance for matching
        direction: "backward" (default) or "forward"

    Returns:
        Merged DataFrame with right_df columns aligned to left_df timestamps
    """
    left = left_df.copy()
    right = right_df.copy()

    if on not in left.columns or on not in right.columns:
        raise ValueError(f"Column '{on}' must exist in both DataFrames")

    # Ensure sorted
    left = left.sort_values(by=[on] + ([by] if by else []))
    right = right.sort_values(by=[on] + ([by] if by else []))

    result = pd.merge_asof(
        left,
        right,
        on=on,
        by=by,
        tolerance=tolerance,
        direction=direction,
    )

    return result


def asof_align_many(
    base_df: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    on: str = "timestamp",
    by: str | None = None,
    tolerance: pd.Timedelta | None = None,
    direction: str = "backward",
) -> pd.DataFrame:
    """
    Align multiple DataFrames to a base DataFrame using asof alignment.

    Args:
        base_df: Base DataFrame to align others to
        frames: Dictionary mapping name -> DataFrame to align
        on: Column name to merge on
        by: Optional grouping column
        tolerance: Maximum time distance for matching
        direction: "backward" (default) or "forward"

    Returns:
        Merged DataFrame with all frames aligned to base_df
    """
    result = base_df.copy()

    for name, df in frames.items():
        aligned = asof_align(result, df, on=on, by=by, tolerance=tolerance, direction=direction)
        # Add columns with prefix to avoid collisions
        for col in df.columns:
            if col != on and (by is None or col != by):
                result[f"{name}_{col}"] = aligned[col]

    return result

