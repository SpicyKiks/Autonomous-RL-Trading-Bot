from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.preprocessing import RobustScaler


def fit_scaler(train_X: np.ndarray, kind: str = "robust") -> Dict[str, Any]:
    """
    Fit a scaler on training data (leakage-safe).

    Args:
        train_X: Training feature matrix (n_samples, n_features)
        kind: "robust" (default) or "standard"

    Returns:
        Dictionary with scaler parameters (JSON-serializable)
    """
    train_X = np.asarray(train_X, dtype=np.float64)
    if train_X.ndim != 2:
        raise ValueError("train_X must be 2D (n_samples, n_features)")

    if kind == "robust":
        scaler = RobustScaler()
    elif kind == "standard":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler kind: {kind}")

    scaler.fit(train_X)

    return {
        "kind": kind,
        "center": scaler.center_.tolist() if hasattr(scaler, "center_") else None,
        "scale": scaler.scale_.tolist(),
        "n_features": int(train_X.shape[1]),
    }


def transform(X: np.ndarray, scaler_params: Dict[str, Any]) -> np.ndarray:
    """
    Transform features using fitted scaler parameters.

    Args:
        X: Feature matrix to transform
        scaler_params: Parameters from fit_scaler()

    Returns:
        Transformed feature matrix
    """
    X = np.asarray(X, dtype=np.float64)
    kind = scaler_params.get("kind", "robust")
    scale = np.asarray(scaler_params["scale"], dtype=np.float64)

    if kind == "robust":
        center = scaler_params.get("center")
        if center is not None:
            center = np.asarray(center, dtype=np.float64)
            X = X - center
        X = X / scale
    elif kind == "standard":
        center = scaler_params.get("center")
        if center is not None:
            center = np.asarray(center, dtype=np.float64)
            X = X - center
        X = X / scale
    else:
        raise ValueError(f"Unknown scaler kind: {kind}")

    return X.astype(np.float32)


def save_scaler_json(path: Path, scaler_params: Dict[str, Any]) -> None:
    """Save scaler parameters to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scaler_params, indent=2), encoding="utf-8")


def load_scaler_json(path: Path) -> Dict[str, Any]:
    """Load scaler parameters from JSON file."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))

