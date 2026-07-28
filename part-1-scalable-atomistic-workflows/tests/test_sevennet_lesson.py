"""Focused tests for the SevenNet adapter parity check."""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


torch = pytest.importorskip("torch")
pytest.importorskip("nvalchemi")

PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.models.sevennet_lesson import (  # noqa: E402
    _SEVENNET_NO_ACCELERATOR_WARNING,
    _official_calculator_comparison,
)


class _ChattyCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.received = kwargs
        warnings.warn_explicit(
            _SEVENNET_NO_ACCELERATOR_WARNING,
            UserWarning,
            filename="sevenn/calculator.py",
            lineno=93,
            module="sevenn.calculator",
        )
        warnings.warn_explicit(
            "unexpected calculator warning",
            UserWarning,
            filename="sevenn/calculator.py",
            lineno=94,
            module="sevenn.calculator",
        )
        print("Converting model backend...")
        print("unexpected calculator output")

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": -1.25,
            "forces": np.zeros((len(atoms), 3), dtype=np.float64),
            "num_edges": 2,
        }


def test_official_comparison_hides_only_expected_sevennet_chatter(
    capsys,
) -> None:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]])

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        display, agreement = _official_calculator_comparison(
            atoms=atoms,
            structure_key="h2",
            checkpoint_path=Path("model.pth"),
            modality="mpa",
            device=torch.device("cpu"),
            adapter_energy_eV=-1.25,
            adapter_forces_eV_A=np.zeros((2, 3), dtype=np.float64),
            calculator_factory=_ChattyCalculator,
        )

    assert [str(item.message) for item in captured] == ["unexpected calculator warning"]
    assert capsys.readouterr().out == "unexpected calculator output\n"
    assert display.loc[0, "energy_difference_eV"] == 0.0
    assert agreement.loc[0, "max_force_component_difference_eV_A"] == 0.0
