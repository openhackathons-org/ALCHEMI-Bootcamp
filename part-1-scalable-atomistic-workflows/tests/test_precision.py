"""Focused checks for the compact precision-lesson helper."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.precision import precision_display_table, summarize_model_precision  # noqa: E402


class TinyFloatModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
        self.register_buffer("scale", torch.tensor([3.0], dtype=torch.float32))


def test_precision_summary_counts_storage_and_spacing() -> None:
    summary = summarize_model_precision(TinyFloatModel(), reference_energy_eV=10.0)

    assert summary.parameter_count == 2
    assert summary.parameter_dtypes == ("torch.float32",)
    assert summary.buffer_dtypes == ("torch.float32",)
    assert summary.parameter_storage_mib == pytest.approx(8 / 2**20)
    assert summary.float64_parameter_storage_mib == pytest.approx(16 / 2**20)
    assert summary.float32_spacing_eV > summary.float64_spacing_eV > 0.0
    assert summary.widening_preserves_stored_values


def test_precision_table_requires_the_four_visible_dtype_observations() -> None:
    summary = summarize_model_precision(TinyFloatModel(), reference_energy_eV=1.0)
    observed = {
        "hello-world coordinates": "torch.float32",
        "float64 probe coordinates": "torch.float64",
        "coordinates passed to AIMNet": "torch.float32",
        "probe energy / forces / charges": "torch.float32 / torch.float64 / torch.float32",
    }

    table = precision_display_table(
        summary, observed_dtypes=observed, matmul_precision="highest"
    )
    assert table.shape == (12, 2)
    assert set(observed).issubset(set(table["Quantity"]))

    with pytest.raises(ValueError, match="observed_dtypes"):
        precision_display_table(
            summary,
            observed_dtypes={"hello-world coordinates": "torch.float32"},
            matmul_precision="highest",
        )
