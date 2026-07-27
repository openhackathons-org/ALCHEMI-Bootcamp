"""Checks for the named NCI numerical settings."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


PART_DIR = Path(__file__).resolve().parents[1]
if str(PART_DIR) not in sys.path:
    sys.path.insert(0, str(PART_DIR))

from aux.composition_config import (  # noqa: E402
    COMPOSITION_FD_FORCE_TOLERANCE_EV_A,
    COMPOSITION_FD_STEP_A,
    COMPOSITION_FORCE_AGREEMENT_TOLERANCE_EV_A,
)
from aux.nci_config import (  # noqa: E402
    NCI_COMPLETE_MAE_LIMIT_KCAL_MOL,
    NCI_VALIDATION,
    NCIValidationSettings,
    nci_validation_settings_table,
)


def test_default_nci_settings_preserve_the_current_numerical_method() -> None:
    assert NCI_COMPLETE_MAE_LIMIT_KCAL_MOL == 0.5
    assert NCI_VALIDATION.method_id == "nci-composition-check-v2"
    assert NCI_VALIDATION.charge_atol_e == 2.0e-4
    assert NCI_VALIDATION.interaction_energy_atol_eV == 3.0e-3
    assert NCI_VALIDATION.net_force_atol_eV_A == 5.0e-3
    assert NCI_VALIDATION.finite_difference_step_A == COMPOSITION_FD_STEP_A
    assert (
        NCI_VALIDATION.finite_difference_atol_eV_A
        == COMPOSITION_FD_FORCE_TOLERANCE_EV_A
    )
    assert NCI_VALIDATION.finite_difference_rtol == 2.0e-2
    assert (
        NCI_VALIDATION.toolkit_official_force_atol_eV_A
        == COMPOSITION_FORCE_AGREEMENT_TOLERANCE_EV_A
    )


def test_settings_table_displays_all_seven_values_with_units_and_reasons() -> None:
    table = nci_validation_settings_table(NCI_VALIDATION)

    assert len(table) == 7
    assert table.columns.tolist() == [
        "check",
        "value",
        "unit",
        "why it is checked",
    ]
    assert table["check"].is_unique
    assert table["unit"].str.strip().ne("").all()
    assert table["why it is checked"].str.endswith(".").all()
    assert table["value"].tolist() == list(NCI_VALIDATION.as_record().values())[1:]


def test_settings_reject_missing_names_and_nonpositive_values() -> None:
    values = {
        "method_id": "unit-test",
        "charge_atol_e": 1.0,
        "interaction_energy_atol_eV": 1.0,
        "net_force_atol_eV_A": 1.0,
        "finite_difference_step_A": 1.0,
        "finite_difference_atol_eV_A": 1.0,
        "finite_difference_rtol": 1.0,
        "toolkit_official_force_atol_eV_A": 1.0,
    }

    with pytest.raises(ValueError, match="method_id"):
        NCIValidationSettings(**{**values, "method_id": ""})
    with pytest.raises(ValueError, match="finite_difference_step_A"):
        NCIValidationSettings(**{**values, "finite_difference_step_A": 0.0})
