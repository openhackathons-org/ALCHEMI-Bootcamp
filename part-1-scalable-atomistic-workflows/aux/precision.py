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


def validate_precision_observation(
    *,
    hello_coordinates_dtype: torch.dtype,
    probe_coordinates_before_dtype: torch.dtype,
    probe_coordinates_after_adapt_dtype: torch.dtype,
    model_input_coordinates_dtype: torch.dtype,
    probe_coordinates_after_forward_dtype: torch.dtype,
    probe_output_dtypes: Mapping[str, torch.dtype],
) -> dict[str, str]:
    """Check the visible AIMNet dtype observations and format their display rows.

    The caller performs the Toolkit conversion, input adaptation, and model
    call.  Passing the observed dtypes here keeps those public operations
    visible while centralizing the exact assertions used by the lesson.
    """

    required_outputs = {"energy", "forces", "charges"}
    if set(probe_output_dtypes) != required_outputs:
        raise ValueError(
            "probe_output_dtypes must contain exactly: "
            + ", ".join(sorted(required_outputs))
        )

    expected = (
        ("hello-world coordinates", hello_coordinates_dtype, torch.float32),
        (
            "probe coordinates before wrapper call",
            probe_coordinates_before_dtype,
            torch.float64,
        ),
        (
            "probe coordinates after input adaptation",
            probe_coordinates_after_adapt_dtype,
            torch.float64,
        ),
        (
            "coordinates passed to AIMNet",
            model_input_coordinates_dtype,
            torch.float32,
        ),
        (
            "probe coordinates after wrapper call",
            probe_coordinates_after_forward_dtype,
            torch.float32,
        ),
        ("probe energy", probe_output_dtypes["energy"], torch.float64),
        ("probe forces", probe_output_dtypes["forces"], torch.float32),
        ("probe charges", probe_output_dtypes["charges"], torch.float32),
    )
    for label, observed, required in expected:
        if observed != required:
            raise AssertionError(
                f"{label} used {observed}; expected {required}"
            )

    return {
        "hello-world coordinates": str(hello_coordinates_dtype),
        "probe coordinates before wrapper call": str(
            probe_coordinates_before_dtype
        ),
        "coordinates passed to AIMNet": str(model_input_coordinates_dtype),
        "probe coordinates after wrapper call": str(
            probe_coordinates_after_forward_dtype
        ),
        "probe energy / forces / charges": " / ".join(
            str(probe_output_dtypes[name])
            for name in ("energy", "forces", "charges")
        ),
    }


def precision_display_table(
    summary: ModelPrecisionSummary,
    *,
    observed_dtypes: Mapping[str, str],
    matmul_precision: str,
) -> pd.DataFrame:
    """Combine model storage facts with dtypes observed in the visible cell."""

    required = {
        "hello-world coordinates",
        "probe coordinates before wrapper call",
        "coordinates passed to AIMNet",
        "probe coordinates after wrapper call",
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
    "validate_precision_observation",
]
