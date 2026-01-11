from __future__ import annotations

import random

import numpy as np
import pytest

from autonomous_rl_trading_bot.common.reproducibility import set_global_seed


def test_set_global_seed_numpy():
    """Test that setting seed produces deterministic numpy outputs."""
    seed_report1 = set_global_seed(42)
    arr1 = np.random.rand(10)
    arr2 = np.random.rand(10)

    seed_report2 = set_global_seed(42)
    arr3 = np.random.rand(10)
    arr4 = np.random.rand(10)

    assert np.allclose(arr1, arr3)
    assert np.allclose(arr2, arr4)
    assert seed_report1["seed"] == 42
    assert seed_report2["seed"] == 42


def test_set_global_seed_python_random():
    """Test that setting seed produces deterministic Python random outputs."""
    set_global_seed(123)
    py_vals1 = [random.random() for _ in range(5)]
    py_vals2 = [random.randint(1, 100) for _ in range(5)]

    set_global_seed(123)
    py_vals3 = [random.random() for _ in range(5)]
    py_vals4 = [random.randint(1, 100) for _ in range(5)]

    assert py_vals1 == py_vals3
    assert py_vals2 == py_vals4


def test_set_global_seed_torch():
    """Test that setting seed produces deterministic torch outputs (if available)."""
    try:
        import torch

        set_global_seed(999)
        torch_vals1 = torch.rand(5).numpy()

        set_global_seed(999)
        torch_vals2 = torch.rand(5).numpy()

        assert np.allclose(torch_vals1, torch_vals2)
    except ImportError:
        pytest.skip("torch not available")


def test_set_global_seed_different_seeds():
    """Test that different seeds produce different outputs."""
    set_global_seed(100)
    vals1 = np.random.rand(10)

    set_global_seed(200)
    vals2 = np.random.rand(10)

    # Should be different (very unlikely to be identical)
    assert not np.allclose(vals1, vals2, rtol=1e-10)


def test_set_global_seed_numpy_generator():
    """Test that numpy Generator can be seeded explicitly (set_global_seed uses legacy API)."""
    # Note: set_global_seed uses np.random.seed() which affects legacy API, not Generator API
    # For Generator API, we need to pass seed explicitly
    set_global_seed(555)
    rng1 = np.random.default_rng(555)  # Explicit seed for Generator
    gen_vals1 = rng1.random(5)

    set_global_seed(555)
    rng2 = np.random.default_rng(555)  # Same explicit seed
    gen_vals2 = rng2.random(5)

    assert np.allclose(gen_vals1, gen_vals2)
