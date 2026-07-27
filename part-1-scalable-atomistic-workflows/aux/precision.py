"""Small floating-point summaries for the notebook precision lesson.

The notebook keeps the Toolkit input conversion and model call visible.  This
module owns the repetitive parameter counting, storage arithmetic, and table
assembly used to explain the observed dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd
import torch


@dataclass(frozen=True)
class ModelPrecisionSummary:
    """Storage and spacing facts for one model checkpoint."""

    parameter_count: int
    parameter_storage_mib: float
    float64_parameter_storage_mib: float
    parameter_dtypes: tuple[str, ...]
    buffer_dtypes: tuple[str, ...]
    float32_spacing_eV: float
    float64_spacing_eV: float
    widening_preserves_stored_values: bool


def summarize_model_precision(
    model: torch.nn.Module,
    *,
    reference_energy_eV: float,
) -> ModelPrecisionSummary:
    """Inspect stored floating tensors and resolution near one energy value."""

    parameters = [
        parameter for parameter in model.parameters() if parameter.is_floating_point()
    ]
    buffers = [buffer for buffer in model.buffers() if buffer.is_floating_point()]
    if not parameters:
        raise ValueError("model has no floating-point parameters")

    parameter_count = sum(parameter.numel() for parameter in parameters)
    parameter_storage_mib = sum(
        parameter.numel() * parameter.element_size() for parameter in parameters
    ) / 2**20
    float64_storage_mib = (
        parameter_count * torch.empty((), dtype=torch.float64).element_size() / 2**20
    )

    energy32 = torch.tensor(abs(float(reference_energy_eV)), dtype=torch.float32)
    energy64 = torch.tensor(abs(float(reference_energy_eV)), dtype=torch.float64)
    spacing32 = float(
        torch.nextafter(energy32, torch.full_like(energy32, torch.inf)) - energy32
    )
    spacing64 = float(
        torch.nextafter(energy64, torch.full_like(energy64, torch.inf)) - energy64
    )
    stored_sample = parameters[0].detach().reshape(-1)[:16].cpu()
    widening_preserves = torch.equal(
        stored_sample.to(torch.float64).to(stored_sample.dtype), stored_sample
    )

    return ModelPrecisionSummary(
        parameter_count=parameter_count,
        parameter_storage_mib=parameter_storage_mib,
        float64_parameter_storage_mib=float64_storage_mib,
        parameter_dtypes=tuple(sorted({str(value.dtype) for value in parameters})),
        buffer_dtypes=tuple(sorted({str(value.dtype) for value in buffers})),
        float32_spacing_eV=spacing32,
        float64_spacing_eV=spacing64,
        widening_preserves_stored_values=bool(widening_preserves),
    )


def precision_display_table(
    summary: ModelPrecisionSummary,
    *,
    observed_dtypes: Mapping[str, str],
    matmul_precision: str,
) -> pd.DataFrame:
    """Combine model storage facts with dtypes observed in the visible cell."""

    required = {
        "hello-world coordinates",
        "float64 probe coordinates",
        "coordinates passed to AIMNet",
        "probe energy / forces / charges",
    }
    if set(observed_dtypes) != required:
        raise ValueError(
            "observed_dtypes must contain exactly: " + ", ".join(sorted(required))
        )

    rows = [
        {"Quantity": name, "Observed": str(value)}
        for name, value in observed_dtypes.items()
    ]
    rows.extend(
        [
            {
                "Quantity": "checkpoint floating parameters",
                "Observed": ", ".join(summary.parameter_dtypes),
            },
            {
                "Quantity": "checkpoint floating buffers",
                "Observed": ", ".join(summary.buffer_dtypes),
            },
            {
                "Quantity": "floating parameter count",
                "Observed": f"{summary.parameter_count:,}",
            },
            {
                "Quantity": "float32 parameter storage only",
                "Observed": f"{summary.parameter_storage_mib:.1f} MiB",
            },
            {
                "Quantity": "same parameter count in float64",
                "Observed": f"{summary.float64_parameter_storage_mib:.1f} MiB",
            },
            {
                "Quantity": "float32 spacing at |E(H2O)|",
                "Observed": f"{summary.float32_spacing_eV:.3e} eV",
            },
            {
                "Quantity": "float64 spacing at |E(H2O)|",
                "Observed": f"{summary.float64_spacing_eV:.3e} eV",
            },
            {"Quantity": "float32 matmul setting", "Observed": matmul_precision},
        ]
    )
    return pd.DataFrame(rows)


__all__ = [
    "ModelPrecisionSummary",
    "precision_display_table",
    "summarize_model_precision",
]
