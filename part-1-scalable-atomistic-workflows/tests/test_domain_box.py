"""Tests for the periodic NCI Atlas molecular-box helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
import numpy as np
import pandas as pd
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux.domain.packing import (  # noqa: E402
    ATOMIC_MASS_VOLUME_FACTOR,
    COMPONENT_NAMES,
    PackmolProcessResult,
    balanced_repeat_factors,
    box_summary_table,
    build_molecular_supercell,
    build_nci_molecular_box,
    molecule_charge_tables,
    plan_nci_molecular_box,
    render_packmol_input,
    validate_molecular_box,
)
from aux.nci_atlas import load_nci_atlas_subset  # noqa: E402


DATA_FILE = PART_DIR / "data" / "nci_atlas" / "nci-atlas-curves.csv.gz"


@pytest.fixture(scope="module")
def nci_table() -> pd.DataFrame:
    return load_nci_atlas_subset(DATA_FILE)


@pytest.mark.parametrize(
    ("target_pair_count", "expected"),
    [
        (128, (1, 1, 1)),
        (256, (1, 1, 2)),
        (512, (1, 2, 2)),
        (1_024, (2, 2, 2)),
        (2_048, (2, 2, 4)),
        (4_096, (2, 4, 4)),
        (8_192, (4, 4, 4)),
        (16_384, (4, 4, 8)),
        (32_768, (4, 8, 8)),
    ],
)
def test_balanced_repeat_factors_cover_the_capacity_ladder(
    target_pair_count: int,
    expected: tuple[int, int, int],
) -> None:
    factors = balanced_repeat_factors(
        base_pair_count=128,
        target_pair_count=target_pair_count,
    )

    assert factors == expected
    assert np.prod(factors) == target_pair_count // 128
    assert max(factors) <= 2 * min(factors)


@pytest.mark.parametrize("target_pair_count", [64, 192, 384])
def test_balanced_repeat_factors_reject_non_doubling_sizes(
    target_pair_count: int,
) -> None:
    with pytest.raises(ValueError):
        balanced_repeat_factors(
            base_pair_count=128,
            target_pair_count=target_pair_count,
        )


def test_supercell_preserves_pbc_density_labels_and_unique_ids() -> None:
    base = Atoms(
        symbols=["H", "He", "Li", "Be"],
        positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.5, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        cell=[4.0, 5.0, 6.0],
        pbc=True,
    )
    base.set_array("source_atom_id", np.arange(4, dtype=np.int64))
    base.set_array("molecule_id", np.array([0, 0, 1, 1], dtype=np.int64))
    base.set_array(
        "molecule_component",
        np.array([0, 0, 1, 1], dtype=np.int8),
    )
    base.set_array("molecule_kind", np.array([0, 0, 1, 1], dtype=np.int32))
    base.set_array(
        "template_atom_index",
        np.array([0, 1, 0, 1], dtype=np.int32),
    )
    base.info["charge"] = 0
    base_density = (
        np.sum(base.get_masses())
        * ATOMIC_MASS_VOLUME_FACTOR
        / base.get_volume()
    )

    expanded, factors = build_molecular_supercell(
        base,
        base_pair_count=1,
        target_pair_count=4,
    )

    assert factors == (1, 2, 2)
    assert len(expanded) == 16
    assert expanded.pbc.all()
    assert np.allclose(expanded.cell, np.diag([4.0, 10.0, 12.0]))
    assert np.array_equal(
        expanded.arrays["source_atom_id"],
        np.arange(16, dtype=np.int64),
    )
    assert np.array_equal(
        expanded.arrays["molecule_id"],
        np.repeat(np.arange(8, dtype=np.int64), 2),
    )
    for name in (
        "molecule_component",
        "molecule_kind",
        "template_atom_index",
    ):
        assert np.array_equal(expanded.arrays[name], np.tile(base.arrays[name], 4))
    expanded_density = (
        np.sum(expanded.get_masses())
        * ATOMIC_MASS_VOLUME_FACTOR
        / expanded.get_volume()
    )
    assert expanded_density == pytest.approx(base_density, rel=1.0e-12)
    assert expanded.info["base_pair_count"] == 1
    assert expanded.info["pair_count"] == 4
    assert [
        expanded.info[f"supercell_repeat_{axis}"] for axis in ("x", "y", "z")
    ] == [1, 2, 2]


def test_plan_selects_scale_one_phenol_nma_and_derives_box(
    nci_table: pd.DataFrame,
) -> None:
    plan = plan_nci_molecular_box(
        nci_table,
        nci_system_id="1.041",
        nci_scale=1.0,
        molecules_per_species=128,
        construction_density_g_cm3=1.0,
        packmol_tolerance_a=2.0,
        packmol_precision_a=1.0e-3,
        packmol_seed=20260723,
    )

    assert tuple(template.name for template in plan.templates) == COMPONENT_NAMES
    assert tuple(template.atom_count for template in plan.templates) == (13, 12)
    assert tuple(template.charge_e for template in plan.templates) == (0, 0)
    assert plan.molecule_counts == (128, 128)
    assert plan.molecules_per_species == 128
    assert plan.molecule_count == 256
    assert plan.atom_count == 3200
    assert plan.net_charge_e == 0
    assert plan.system_id == "1.041"
    assert plan.scale == 1.0
    assert plan.packmol_precision_a == pytest.approx(1.0e-3)
    expected_volume = (
        plan.total_mass_u * ATOMIC_MASS_VOLUME_FACTOR / plan.construction_density_g_cm3
    )
    assert plan.box_length_a == pytest.approx(expected_volume ** (1.0 / 3.0))
    assert plan.box_length_a == pytest.approx(32.88, abs=0.02)

    summary = box_summary_table(plan)
    assert summary["component"].tolist() == [
        "phenol",
        "N-methylacetamide",
        "periodic box",
    ]
    box_row = summary.iloc[-1]
    assert box_row["construction_density_g_cm3"] == 1.0
    assert box_row["packmol_tolerance_a"] == pytest.approx(2.0)
    assert box_row["packmol_precision_a"] == pytest.approx(1.0e-3)
    assert box_row["min_distance_required_a"] == pytest.approx(1.999)


def test_plan_rejects_an_unavailable_visible_nci_selection(
    nci_table: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="NCI Atlas system missing at scale 1 must contain AB, A, and B",
    ):
        plan_nci_molecular_box(
            nci_table,
            nci_system_id="missing",
            nci_scale=1.0,
            molecules_per_species=1,
            construction_density_g_cm3=0.02,
            packmol_tolerance_a=2.0,
            packmol_precision_a=1.0e-3,
            packmol_seed=17,
        )


def test_packmol_input_is_deterministic_and_uses_global_pbc(
    nci_table: pd.DataFrame,
) -> None:
    plan = plan_nci_molecular_box(
        nci_table,
        molecules_per_species=2,
        construction_density_g_cm3=0.5,
        packmol_tolerance_a=2.0,
        packmol_precision_a=1.0e-3,
        packmol_seed=17,
    )
    filenames = {
        "phenol": "phenol.xyz",
        "N-methylacetamide": "n_methylacetamide.xyz",
    }

    first = render_packmol_input(
        plan,
        template_filenames=filenames,
        output_filename="box.xyz",
    )
    second = render_packmol_input(
        plan,
        template_filenames=filenames,
        output_filename="box.xyz",
    )

    assert first == second
    assert "tolerance 2.00000000" in first
    assert "precision 0.00100000" in first
    assert "seed 17" in first
    assert (
        f"pbc {plan.box_length_a:.8f} {plan.box_length_a:.8f} {plan.box_length_a:.8f}"
    ) in first
    assert first.count("  number 2") == 2
    assert "inside box" not in first


def test_packmol_precision_must_be_smaller_than_tolerance(
    nci_table: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="smaller than packmol_tolerance_a"):
        plan_nci_molecular_box(
            nci_table,
            molecules_per_species=1,
            construction_density_g_cm3=0.02,
            packmol_tolerance_a=2.0,
            packmol_precision_a=2.0,
            packmol_seed=17,
        )


class PackedBoxRunner:
    """Deterministic Packmol stand-in passed through the normal runner API."""

    def __init__(self, plan) -> None:
        self.plan = plan
        self.calls: list[tuple[str | Path | None, Path, str]] = []

    def __call__(
        self,
        binary: str | Path | None,
        cwd: Path,
        input_text: str,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((binary, cwd, input_text))
        components: list[Atoms] = []
        centers = (
            np.array([0.25, 0.5, 0.5]) * self.plan.box_length_a,
            np.array([0.75, 0.5, 0.5]) * self.plan.box_length_a,
        )
        for template, count, center in zip(
            self.plan.templates,
            self.plan.molecule_counts,
            centers,
            strict=True,
        ):
            assert count == 1
            molecule = template.atoms.copy()
            molecule.positions -= molecule.positions.mean(axis=0)
            molecule.positions += center
            components.append(molecule)
        packed = components[0] + components[1]
        output_line = next(
            line for line in input_text.splitlines() if line.startswith("output ")
        )
        ase_write(cwd / output_line.split(maxsplit=1)[1], packed, format="xyz")
        return subprocess.CompletedProcess(
            args=["injected-packmol-runner"],
            returncode=0,
            stdout="packed",
            stderr="",
        )


def test_build_injects_runner_validates_box_and_saves_extxyz(
    nci_table: pd.DataFrame,
    tmp_path: Path,
) -> None:
    plan = plan_nci_molecular_box(
        nci_table,
        molecules_per_species=1,
        construction_density_g_cm3=0.02,
        packmol_tolerance_a=2.0,
        packmol_precision_a=1.0e-3,
        packmol_seed=23,
    )
    runner = PackedBoxRunner(plan)
    extxyz_path = tmp_path / "validated.extxyz"

    atoms = build_nci_molecular_box(
        plan,
        tmp_path / "packmol",
        runner=runner,
        extxyz_path=extxyz_path,
    )

    assert len(runner.calls) == 1
    assert runner.calls[0][0] is None
    assert "pbc " in runner.calls[0][2]
    assert len(atoms) == 25
    assert atoms.pbc.all()
    assert np.allclose(atoms.cell, np.eye(3) * plan.box_length_a)
    assert np.unique(atoms.arrays["molecule_id"]).tolist() == [0, 1]
    assert np.unique(atoms.arrays["molecule_component"]).tolist() == [0, 1]
    assert atoms.info["charge"] == 0
    assert atoms.info["construction_density_g_cm3"] == pytest.approx(0.02)
    assert atoms.info["packmol_precision_a"] == pytest.approx(1.0e-3)
    assert atoms.info["periodic_min_distance_lower_bound_a"] >= 1.999
    assert (tmp_path / "packmol" / "packmol.inp").is_file()
    assert (tmp_path / "packmol" / "packmol.stdout.log").read_text() == "packed"
    assert (tmp_path / "packmol" / "packmol.stderr.log").read_text() == ""
    assert (tmp_path / "packmol" / "phenol.xyz").is_file()
    assert (tmp_path / "packmol" / "n_methylacetamide.xyz").is_file()

    restored = ase_read(extxyz_path, format="extxyz")
    assert restored.pbc.all()
    assert np.array_equal(restored.arrays["molecule_id"], atoms.arrays["molecule_id"])
    assert np.array_equal(
        restored.arrays["molecule_component"],
        atoms.arrays["molecule_component"],
    )

    summary = box_summary_table(plan, atoms)
    box_row = summary.iloc[-1]
    assert box_row["density_from_mass_and_cell_g_cm3"] == pytest.approx(0.02)
    assert box_row["packmol_tolerance_a"] == pytest.approx(2.0)
    assert box_row["min_distance_required_a"] == pytest.approx(1.999)
    validation = validate_molecular_box(plan, atoms)
    assert validation.density_from_mass_and_cell_g_cm3 == pytest.approx(0.02)
    assert validation.min_distance_required_a == pytest.approx(1.999)

    charges, charge_summary = molecule_charge_tables(
        plan,
        atoms,
        np.array([0.08, -0.08]),
    )
    assert charges.to_dict(orient="records") == [
        {
            "molecule_id": 0,
            "component": "phenol",
            "predicted_charge_e": 0.08,
        },
        {
            "molecule_id": 1,
            "component": "N-methylacetamide",
            "predicted_charge_e": -0.08,
        },
    ]
    assert charge_summary["component"].tolist() == [
        "phenol",
        "N-methylacetamide",
        "all molecules",
    ]
    assert charge_summary.iloc[-1]["molecules"] == 2
    assert charge_summary.iloc[-1]["total_charge_e"] == pytest.approx(0.0)
    assert charge_summary.iloc[-1]["mean_abs_charge_e"] == pytest.approx(0.08)


def test_validation_rejects_short_periodic_intermolecular_contact(
    nci_table: pd.DataFrame,
) -> None:
    plan = plan_nci_molecular_box(
        nci_table,
        molecules_per_species=1,
        construction_density_g_cm3=0.02,
        packmol_tolerance_a=2.0,
        packmol_precision_a=1.0e-3,
        packmol_seed=23,
    )
    # The scale=1.0 NCI dimer contains the chemically meaningful close contact,
    # so it is deliberately too close for a Packmol tolerance of 2 Å.
    packed = plan.templates[0].atoms + plan.templates[1].atoms
    packed.set_cell([plan.box_length_a] * 3)
    packed.set_pbc(True)

    with pytest.raises(ValueError, match="intermolecular contact"):
        validate_molecular_box(plan, packed)


def test_validation_rejects_wrong_charge_metadata(
    nci_table: pd.DataFrame,
    tmp_path: Path,
) -> None:
    plan = plan_nci_molecular_box(
        nci_table,
        molecules_per_species=1,
        construction_density_g_cm3=0.02,
        packmol_tolerance_a=1.0,
        packmol_precision_a=1.0e-3,
        packmol_seed=23,
    )
    runner = PackedBoxRunner(plan)
    atoms = build_nci_molecular_box(plan, tmp_path, runner=runner)
    atoms.info["charge"] = 1

    with pytest.raises(ValueError, match="net charge"):
        validate_molecular_box(plan, atoms)


def test_executor_requires_explicit_binary_or_runner(
    nci_table: pd.DataFrame,
    tmp_path: Path,
) -> None:
    plan = plan_nci_molecular_box(
        nci_table,
        molecules_per_species=1,
        construction_density_g_cm3=0.02,
        packmol_tolerance_a=2.0,
        packmol_precision_a=1.0e-3,
        packmol_seed=23,
    )

    with pytest.raises(ValueError, match="packmol_binary or"):
        build_nci_molecular_box(plan, tmp_path)


def test_nonzero_packmol_exit_is_not_accepted(
    nci_table: pd.DataFrame,
    tmp_path: Path,
) -> None:
    plan = plan_nci_molecular_box(
        nci_table,
        molecules_per_species=1,
        construction_density_g_cm3=0.02,
        packmol_tolerance_a=2.0,
        packmol_precision_a=1.0e-3,
        packmol_seed=23,
    )

    def failed_runner(
        binary: str | Path | None,
        cwd: Path,
        input_text: str,
    ) -> PackmolProcessResult:
        del binary, cwd, input_text
        return PackmolProcessResult(
            returncode=173,
            stderr="ENDED WITHOUT PERFECT PACKING",
        )

    with pytest.raises(RuntimeError, match="status 173"):
        build_nci_molecular_box(plan, tmp_path, runner=failed_runner)
