from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> dict[str, Any]:
    """
    Set seeds for reproducibility. Torch is optional.
    Returns a report dict for run.json.
    """
    report: dict[str, Any] = {"seed": seed}

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    report["pythonhashseed"] = os.environ.get("PYTHONHASHSEED")

    # Torch is optional
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        report["torch"] = {
            "available": True,
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": torch.cuda.is_available(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
    except Exception as e:
        report["torch"] = {"available": False, "error": str(e)}

    return report
