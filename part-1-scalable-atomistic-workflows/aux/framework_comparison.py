"""Validation and display rows for the short Torch, JAX, and Warp primer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def segmented_sum_comparison_table(
    *,
    expected_totals: Any,
    torch_result: Any,
    jax_result: Any,
    warp_result: Any,
    torch_gradient: Any,
    jax_gradient: Any,
    torch_dtype: str,
    torch_device: str,
    jax_dtype: str,
    jax_device: str,
    warp_dtype: str,
    warp_device: str,
) -> pd.DataFrame:
    """Check the three reductions and return their interface differences."""

    expected = np.asarray(expected_totals, dtype=np.float32)
    results = {
        "PyTorch binding": np.asarray(torch_result),
        "JAX binding": np.asarray(jax_result),
        "raw Warp": np.asarray(warp_result),
    }
    for result in results.values():
        np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(np.asarray(torch_gradient), 1.0)
    np.testing.assert_allclose(np.asarray(jax_gradient), 1.0)

    return pd.DataFrame(
        [
            (
                "PyTorch binding",
                "torch.Tensor",
                torch_dtype,
                torch_device,
                results["PyTorch binding"].tolist(),
                "returns a tensor",
            ),
            (
                "JAX binding",
                "jax.Array",
                jax_dtype,
                jax_device,
                results["JAX binding"].tolist(),
                "returns an array",
            ),
            (
                "raw Warp",
                "wp.array",
                warp_dtype,
                warp_device,
                results["raw Warp"].tolist(),
                "fills supplied output",
            ),
        ],
        columns=["Path", "Array", "Dtype", "Device", "Totals", "Output"],
    )


__all__ = ["segmented_sum_comparison_table"]
