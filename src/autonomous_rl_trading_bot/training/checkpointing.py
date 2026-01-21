from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_path(run_dir: Path) -> Path:
    return Path(run_dir) / "model.zip"


def best_model_path(run_dir: Path) -> Path:
    return Path(run_dir) / "best_model.zip"


def tensorboard_dir(run_dir: Path) -> Path:
    return ensure_dir(Path(run_dir) / "tb")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_training_manifest(
    *,
    run_dir: Path,
    train_params: dict[str, Any],
    split: dict[str, Any],
    dataset_meta: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "train_params": train_params,
        "split": split,
        "dataset_meta": dataset_meta or {},
    }
    if extra:
        payload.update(extra)
    write_json(Path(run_dir) / "train_manifest.json", payload)

