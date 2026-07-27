"""Checks for the Torch, JAX, and Warp primer result table."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.framework_comparison import segmented_sum_comparison_table  # noqa: E402


def _table(**overrides):
    values = {
        "expected_totals": np.array([3.0, 7.0], dtype=np.float32),
        "torch_result": np.array([3.0, 7.0], dtype=np.float32),
        "jax_result": np.array([3.0, 7.0], dtype=np.float32),
        "warp_result": np.array([3.0, 7.0], dtype=np.float32),
        "torch_gradient": np.ones(4, dtype=np.float32),
        "jax_gradient": np.ones(4, dtype=np.float32),
        "torch_dtype": "torch.float32",
        "torch_device": "cuda:0",
        "jax_dtype": "float32",
        "jax_device": "cuda:0",
        "warp_dtype": "float32",
        "warp_device": "cuda:0",
    }
    values.update(overrides)
    return segmented_sum_comparison_table(**values)


def test_framework_table_preserves_three_interface_differences() -> None:
    table = _table()

    assert table["Path"].tolist() == ["PyTorch binding", "JAX binding", "raw Warp"]
    assert table["Totals"].tolist() == [[3.0, 7.0]] * 3
    assert table.loc[2, "Output"] == "fills supplied output"


def test_framework_table_rejects_a_wrong_result() -> None:
    with pytest.raises(AssertionError):
        _table(warp_result=np.array([3.0, 8.0], dtype=np.float32))
