"""Focused tests for strict post-run acceptance gates."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = ROOT / "scripts" / "validate_part1_ir_run.py"
SPEC = importlib.util.spec_from_file_location("validate_part1_ir_run", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def valid_manifest() -> dict[str, object]:
    return {
        "gates": {
            "residual_serial_batch_max_abs_eV": 1e-7,
            "full_serial_batch_max_abs_eV": 2e-7,
            "component_closure_max_abs_eV": 3e-7,
            "official_calculator_parity": {
                "energy_eV": 1e-7,
                "forces_eV_A": 2e-7,
                "charges_e": 3e-9,
            },
            "analytic_coulomb": {
                "energy_eV": 1e-7,
                "forces_eV_A": 2e-7,
            },
            "compiled_ir_eager_parity": {
                "energy": 1e-7,
                "forces": 2e-7,
                "charges": 3e-9,
            },
            "compiled_ir_repeat_parity": {
                "energy": 0.0,
                "forces": 0.0,
                "charges": 0.0,
            },
            "finite_difference_force_reference_eV_A": 0.1,
            "finite_difference_force_pipeline_eV_A": 0.101,
            "finite_difference_force_abs_error_eV_A": 0.001,
        }
    }


def test_composition_validator_accepts_complete_gate_record() -> None:
    result = VALIDATOR.validate_composition_gates(valid_manifest())

    assert result["official_calculator_parity"]["charges_e"] == 3e-9
    assert result["finite_difference_force"]["abs_tolerance_eV_A"] == 0.004


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("full_serial_batch_max_abs_eV",), 2e-5, "composition gate failed"),
        (
            ("official_calculator_parity", "forces_eV_A"),
            2e-6,
            "composition gate failed",
        ),
        (
            ("analytic_coulomb", "energy_eV"),
            2e-6,
            "composition gate failed",
        ),
        (
            ("finite_difference_force_abs_error_eV_A",),
            0.002,
            "was not reproduced",
        ),
    ],
)
def test_composition_validator_rejects_failed_or_inconsistent_gate(
    path: tuple[str, ...], value: float, message: str
) -> None:
    manifest = deepcopy(valid_manifest())
    target = manifest["gates"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(RuntimeError, match=message):
        VALIDATOR.validate_composition_gates(manifest)


def test_execution_runner_is_hashed_and_required_in_bundle() -> None:
    assert "scripts/run_notebook_no_timeout.py" in VALIDATOR.SOURCE_PATHS
    assert "run_notebook_no_timeout.py" in VALIDATOR.BUNDLE_SOURCE_FILES
