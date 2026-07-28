"""Focused tests for production-trajectory diagnostic post-processing."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
import inspect
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux import analysis as analysis_module  # noqa: E402
from aux.diagnostics import (  # noqa: E402
    ProductionDiagnostics,
    analyze_production_trajectory,
    build_production_diagnostics_display_tables,
    cluster_integrity,
)
from aux.structures import make_ir_structures  # noqa: E402


LABELS = ("H2O", "D2O", "(H2O)6", "(D2O)6")
KB_EV_PER_K = 8.617_333_262_145e-5
ENERGY_SLOPES_MEV_ATOM_PS = np.array([1.0, -2.0, 0.5, -0.25])


@dataclass(frozen=True)
class _Trajectory:
    dipoles_e_angstrom: np.ndarray
    charge_sums_e: np.ndarray
    kinetic_energies_eV: np.ndarray
    total_energies_eV: np.ndarray
    positions_angstrom: np.ndarray
    atomic_numbers: np.ndarray
    atomic_masses_u: np.ndarray
    batch_idx: np.ndarray
    batch_ptr: np.ndarray
    dt_fs: float


def _trajectory() -> tuple[_Trajectory, np.ndarray]:
    structures, labels = make_ir_structures()
    assert tuple(labels) == LABELS
    atom_counts = np.array([len(atoms) for atoms in structures], dtype=np.int64)
    batch_ptr = np.cumsum(np.concatenate(([0], atom_counts)))
    positions = np.concatenate([atoms.positions for atoms in structures])
    atomic_numbers = np.concatenate([atoms.numbers for atoms in structures])
    atomic_masses = np.concatenate([atoms.get_masses() for atoms in structures])
    batch_idx = np.repeat(np.arange(len(structures)), atom_counts)

    frame_count = 4
    dt_fs = 1_000.0
    time_ps = np.arange(frame_count, dtype=np.float64)
    target_temperature_K = np.array(
        [
            [60.0, 65.0, 70.0, 75.0],
            [70.0, 75.0, 80.0, 85.0],
            [80.0, 85.0, 90.0, 95.0],
            [90.0, 95.0, 100.0, 105.0],
        ]
    )
    kinetic = (
        0.5
        * 3.0
        * atom_counts[None, :]
        * KB_EV_PER_K
        * target_temperature_K
    )
    energy_delta_eV = (
        time_ps[:, None]
        * ENERGY_SLOPES_MEV_ATOM_PS[None, :]
        * atom_counts[None, :]
        / 1_000.0
    )
    energy_baseline_eV = np.array([-10.0, -11.0, -60.0, -61.0])
    charge_sums_e = np.array(
        [
            [1.0e-7, -2.0e-7, 3.0e-7, -4.0e-7],
            [-2.0e-7, 3.0e-7, -4.0e-7, 5.0e-7],
            [3.0e-7, -4.0e-7, 5.0e-7, -6.0e-7],
            [-4.0e-7, 5.0e-7, -6.0e-7, 7.0e-7],
        ]
    )
    trajectory = _Trajectory(
        dipoles_e_angstrom=np.zeros((frame_count, len(structures), 3)),
        charge_sums_e=charge_sums_e,
        kinetic_energies_eV=kinetic,
        total_energies_eV=energy_baseline_eV[None, :] + energy_delta_eV,
        positions_angstrom=np.repeat(positions[None, :, :], frame_count, axis=0),
        atomic_numbers=atomic_numbers,
        atomic_masses_u=atomic_masses,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        dt_fs=dt_fs,
    )
    return trajectory, target_temperature_K


def _settings(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "labels": LABELS,
        "cluster_graph_indices": (2, 3),
        "boltzmann_constant_eV_per_K": KB_EV_PER_K,
        "energy_spacing_dtype": np.float32,
        "oxygen_connectivity_cutoff_angstrom": 4.0,
        "covalent_oh_cutoff_angstrom": 1.25,
        "h_acceptor_cutoff_angstrom": 2.5,
        "oo_cutoff_angstrom": 3.5,
        "hbond_angle_cutoff_deg": 140.0,
        "energy_excursion_advisory_meV_per_atom": 10.0,
    }
    settings.update(overrides)
    return settings


def _analyze(
    trajectory: _Trajectory | None = None,
    **overrides: Any,
) -> ProductionDiagnostics:
    if trajectory is None:
        trajectory = _trajectory()[0]
    return analyze_production_trajectory(
        trajectory,
        **_settings(**overrides),
    )


def test_production_result_matches_original_notebook_formulas_and_tables() -> None:
    trajectory, expected_temperature = _trajectory()
    result = _analyze(trajectory)
    atom_counts = np.diff(trajectory.batch_ptr)
    time_ps = np.arange(len(trajectory.positions_angstrom)) * trajectory.dt_fs / 1000.0
    expected_drift = []
    expected_excursion = []
    for graph, atom_count in enumerate(atom_counts):
        delta_eV = (
            trajectory.total_energies_eV[:, graph]
            - trajectory.total_energies_eV[0, graph]
        )
        expected_drift.append(
            1_000.0 * np.polyfit(time_ps, delta_eV, 1)[0] / atom_count
        )
        expected_excursion.append(
            1_000.0 * np.max(np.abs(delta_eV)) / atom_count
        )
    expected_drift_array = np.asarray(expected_drift)
    expected_excursion_array = np.asarray(expected_excursion)
    expected_charge_error = np.max(np.abs(trajectory.charge_sums_e), axis=0)
    expected_spacing = (
        1_000.0
        * np.spacing(
            np.abs(trajectory.total_energies_eV[0]).astype(np.float32)
        )
        / atom_counts
    )
    expected_table = pd.DataFrame(
        {
            "system": LABELS,
            "NVE_start_T_3N_K": expected_temperature[0],
            "NVE_mean_T_3N_K": expected_temperature.mean(axis=0),
            "max_charge_error_e": expected_charge_error,
            "energy_drift_meV_atom_ps": expected_drift_array,
            "max_energy_excursion_meV_atom": expected_excursion_array,
        }
    ).set_index("system")
    expected_integrity = pd.DataFrame.from_records(
        [
            cluster_integrity(
                trajectory,
                graph,
                oxygen_cutoff_angstrom=4.0,
                h_acceptor_cutoff_angstrom=2.5,
                oo_cutoff_angstrom=3.5,
                hbond_angle_cutoff_deg=140.0,
            )
            for graph in (2, 3)
        ],
        index=LABELS[2:],
    )

    pdt.assert_frame_equal(result.diagnostic_table, expected_table)
    pdt.assert_frame_equal(result.integrity_table, expected_integrity)
    np.testing.assert_allclose(result.nve_temperature_3n_K, expected_temperature)
    np.testing.assert_allclose(
        result.energy_drift_meV_atom_ps,
        expected_drift_array,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.max_energy_excursion_meV_atom,
        expected_excursion_array,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.energy_spacing_meV_atom,
        expected_spacing,
        rtol=0.0,
        atol=0.0,
    )
    assert result.cluster_intact is True
    assert result.cluster_dft_comparison_valid is True
    assert result.energy_within_advisory is True


def test_learner_display_tables_are_compact_and_keep_system_labels() -> None:
    result = _analyze()

    display_tables = build_production_diagnostics_display_tables(result)

    assert list(display_tables.diagnostics.columns) == [
        "System",
        "Start T / K",
        "Mean T / K",
        "Max |Δq| / e",
        "Drift / meV atom⁻¹ ps⁻¹",
        "Max excursion / meV atom⁻¹",
    ]
    assert display_tables.diagnostics["System"].tolist() == list(LABELS)
    np.testing.assert_allclose(
        display_tables.diagnostics["Start T / K"],
        result.diagnostic_table["NVE_start_T_3N_K"],
    )
    np.testing.assert_allclose(
        display_tables.diagnostics["Mean T / K"],
        result.diagnostic_table["NVE_mean_T_3N_K"],
    )

    assert list(display_tables.integrity.columns) == [
        "System",
        "Max O-H / Å",
        "Connected",
        "H-bonds min-max",
        "Initial-ring fraction",
        "First ring change / ps",
    ]
    assert display_tables.integrity["System"].tolist() == list(LABELS[2:])
    np.testing.assert_allclose(
        display_tables.integrity["Max O-H / Å"],
        result.integrity_table["max_OH_angstrom"],
    )
    assert display_tables.integrity["Connected"].tolist() == [True, True]
    assert display_tables.integrity["H-bonds min-max"].tolist() == ["6-6", "6-6"]
    np.testing.assert_allclose(
        display_tables.integrity["Initial-ring fraction"],
        result.integrity_table["initial_ring_fraction"],
    )
    np.testing.assert_allclose(
        display_tables.integrity["First ring change / ps"],
        result.integrity_table["first_initial_ring_loss_ps"],
        equal_nan=True,
    )


def test_learner_display_tables_do_not_change_saved_raw_tables() -> None:
    result = _analyze()
    raw_diagnostics = result.diagnostic_table.copy(deep=True)
    raw_integrity = result.integrity_table.copy(deep=True)

    display_tables = build_production_diagnostics_display_tables(result)
    display_tables.diagnostics.iloc[0, 1] = -1.0
    display_tables.integrity.iloc[0, 1] = -1.0

    pdt.assert_frame_equal(result.diagnostic_table, raw_diagnostics)
    pdt.assert_frame_equal(result.integrity_table, raw_integrity)


def test_result_arrays_are_read_only_copies_and_result_is_frozen() -> None:
    trajectory, _ = _trajectory()
    result = _analyze(trajectory)

    for array in (
        result.nve_temperature_3n_K,
        result.energy_drift_meV_atom_ps,
        result.max_energy_excursion_meV_atom,
        result.energy_spacing_meV_atom,
    ):
        assert array.flags.owndata
        assert not array.flags.writeable
    original = result.nve_temperature_3n_K.copy()
    trajectory.kinetic_energies_eV[:] = 0.0
    np.testing.assert_array_equal(result.nve_temperature_3n_K, original)
    with pytest.raises(FrozenInstanceError):
        result.cluster_intact = False  # type: ignore[misc]


def test_cluster_topology_results_are_typed_read_only_and_keep_settings() -> None:
    result = _analyze()

    assert [entry.graph_index for entry in result.cluster_topologies] == [2, 3]
    assert [entry.system_label for entry in result.cluster_topologies] == [
        "(H2O)6",
        "(D2O)6",
    ]
    for entry in result.cluster_topologies:
        assert entry.oxygen_connectivity_cutoff_angstrom == 4.0
        assert entry.h_acceptor_cutoff_angstrom == 2.5
        assert entry.oo_cutoff_angstrom == 3.5
        assert entry.hbond_angle_cutoff_deg == 140.0
        topology = entry.observables
        arrays = (
            topology.hydrogen_bond_count,
            topology.ring_masks.any_cycle,
            topology.ring_masks.exact_single_ring,
            topology.ring_masks.initial_cycle,
            topology.ring_masks.cycle_multiplicity,
            topology.oxygen_component_count,
            topology.oxygen_radius_gyration_angstrom,
            topology.oxygen_rmsd_angstrom,
            topology.max_assigned_oh_angstrom,
        )
        assert all(
            array is not None and array.flags.owndata and not array.flags.writeable
            for array in arrays
        )

    with pytest.raises(FrozenInstanceError):
        result.cluster_topologies[0].graph_index = 3  # type: ignore[misc]


def test_topology_timeline_reuses_cached_observables_without_recalculation() -> None:
    trajectory, _ = _trajectory()
    result = _analyze(trajectory)
    expected = {
        entry.graph_index: analysis_module.topology_time_series(
            trajectory,
            entry.graph_index,
            h_acceptor_cutoff_angstrom=2.5,
            oo_cutoff_angstrom=3.5,
            hbond_angle_cutoff_deg=140.0,
        )
        for entry in result.cluster_topologies
    }

    class ShapeOnlyPositions:
        def __init__(self, positions: np.ndarray) -> None:
            self._positions = positions

        def __array__(
            self,
            dtype: np.dtype[Any] | None = None,
            copy: bool | None = None,
        ) -> np.ndarray:
            values = np.asarray(self._positions, dtype=dtype)
            return values.copy() if copy else values

        def __getitem__(self, key: Any) -> np.ndarray:
            raise AssertionError("cached topology should not slice trajectory data")

    cached_trajectory = replace(
        trajectory,
        positions_angstrom=ShapeOnlyPositions(trajectory.positions_angstrom),
    )
    for entry in result.cluster_topologies:
        actual = analysis_module.topology_time_series(
            cached_trajectory,
            entry.graph_index,
            h_acceptor_cutoff_angstrom=2.5,
            oo_cutoff_angstrom=3.5,
            hbond_angle_cutoff_deg=140.0,
            precomputed_topology=entry,
        )
        pdt.assert_frame_equal(actual, expected[entry.graph_index])


def test_cached_topology_rejects_graph_and_cutoff_mismatches() -> None:
    trajectory, _ = _trajectory()
    cached = _analyze(trajectory).cluster_topologies[0]

    with pytest.raises(ValueError, match="graph does not match"):
        analysis_module.topology_time_series(
            trajectory,
            3,
            h_acceptor_cutoff_angstrom=2.5,
            oo_cutoff_angstrom=3.5,
            hbond_angle_cutoff_deg=140.0,
            precomputed_topology=cached,
        )
    with pytest.raises(ValueError, match="h_acceptor_cutoff_angstrom"):
        analysis_module.topology_time_series(
            trajectory,
            2,
            h_acceptor_cutoff_angstrom=2.4,
            oo_cutoff_angstrom=3.5,
            hbond_angle_cutoff_deg=140.0,
            precomputed_topology=cached,
        )


def test_every_scientific_cutoff_and_advisory_is_a_required_argument() -> None:
    signature = inspect.signature(analyze_production_trajectory)
    explicit_parameters = (
        "boltzmann_constant_eV_per_K",
        "energy_spacing_dtype",
        "oxygen_connectivity_cutoff_angstrom",
        "covalent_oh_cutoff_angstrom",
        "h_acceptor_cutoff_angstrom",
        "oo_cutoff_angstrom",
        "hbond_angle_cutoff_deg",
        "energy_excursion_advisory_meV_per_atom",
    )

    for name in explicit_parameters:
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty


def test_boltzmann_constant_controls_reported_temperature() -> None:
    baseline = _analyze()
    changed = _analyze(boltzmann_constant_eV_per_K=2.0 * KB_EV_PER_K)

    np.testing.assert_allclose(
        changed.nve_temperature_3n_K,
        0.5 * baseline.nve_temperature_3n_K,
    )


def test_energy_spacing_dtype_controls_only_the_roundoff_estimate() -> None:
    float32_result = _analyze(energy_spacing_dtype=np.float32)
    float64_result = _analyze(energy_spacing_dtype=np.float64)

    assert np.all(float32_result.energy_spacing_meV_atom > 0.0)
    assert np.all(float64_result.energy_spacing_meV_atom > 0.0)
    assert np.all(
        float32_result.energy_spacing_meV_atom
        > float64_result.energy_spacing_meV_atom
    )
    pdt.assert_frame_equal(
        float32_result.diagnostic_table,
        float64_result.diagnostic_table,
    )


def test_energy_advisory_is_inclusive_and_does_not_raise_on_failure() -> None:
    baseline = _analyze()
    maximum = float(np.max(baseline.max_energy_excursion_meV_atom))

    at_limit = _analyze(energy_excursion_advisory_meV_per_atom=maximum)
    below_limit = _analyze(
        energy_excursion_advisory_meV_per_atom=np.nextafter(maximum, 0.0)
    )

    assert at_limit.energy_within_advisory is True
    assert below_limit.energy_within_advisory is False


def test_covalent_cutoff_controls_cluster_acceptance_without_hiding_result() -> None:
    baseline = _analyze()
    max_oh = float(baseline.integrity_table["max_OH_angstrom"].max())

    at_limit = _analyze(covalent_oh_cutoff_angstrom=max_oh)
    above_limit = _analyze(
        covalent_oh_cutoff_angstrom=np.nextafter(max_oh, np.inf)
    )

    assert at_limit.cluster_intact is False
    assert above_limit.cluster_intact is True
    assert at_limit.cluster_dft_comparison_valid is True


def test_oxygen_connectivity_cutoff_controls_cluster_acceptance() -> None:
    result = _analyze(oxygen_connectivity_cutoff_angstrom=0.1)

    assert (result.integrity_table["max_oxygen_components"] > 1).all()
    assert result.cluster_intact is False


@pytest.mark.parametrize(
    ("parameter", "value"),
    (
        ("h_acceptor_cutoff_angstrom", 0.1),
        ("oo_cutoff_angstrom", 0.1),
        ("hbond_angle_cutoff_deg", 180.0),
    ),
)
def test_each_hbond_cutoff_controls_the_ring_comparison_flag(
    parameter: str,
    value: float,
) -> None:
    result = _analyze(**{parameter: value})

    assert result.cluster_intact is True
    assert result.cluster_dft_comparison_valid is False


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    (
        (
            {"boltzmann_constant_eV_per_K": 0.0},
            ValueError,
            "boltzmann_constant_eV_per_K",
        ),
        (
            {"energy_spacing_dtype": np.int64},
            TypeError,
            "floating-point dtype",
        ),
        (
            {"oxygen_connectivity_cutoff_angstrom": -1.0},
            ValueError,
            "oxygen_connectivity_cutoff_angstrom",
        ),
        (
            {"covalent_oh_cutoff_angstrom": np.inf},
            ValueError,
            "covalent_oh_cutoff_angstrom",
        ),
        (
            {"h_acceptor_cutoff_angstrom": 0.0},
            ValueError,
            "h_acceptor_cutoff_angstrom",
        ),
        (
            {"oo_cutoff_angstrom": np.nan},
            ValueError,
            "oo_cutoff_angstrom",
        ),
        (
            {"hbond_angle_cutoff_deg": 181.0},
            ValueError,
            "hbond_angle_cutoff_deg",
        ),
        (
            {"energy_excursion_advisory_meV_per_atom": -1.0},
            ValueError,
            "energy_excursion_advisory_meV_per_atom",
        ),
    ),
)
def test_invalid_explicit_settings_fail_clearly(
    overrides: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        _analyze(**overrides)


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    (
        ({"labels": LABELS[:-1]}, ValueError, "exactly 4"),
        ({"labels": ("H2O", "D2O", "cluster", "cluster")}, ValueError, "unique"),
        ({"cluster_graph_indices": ()}, ValueError, "at least one"),
        ({"cluster_graph_indices": (2, 2)}, ValueError, "duplicates"),
        ({"cluster_graph_indices": (4,)}, IndexError, "outside"),
        ({"cluster_graph_indices": (True,)}, TypeError, "integers"),
    ),
)
def test_system_and_cluster_selection_is_validated(
    overrides: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        _analyze(**overrides)


@pytest.mark.parametrize(
    ("field", "replacement", "error", "match"),
    (
        (
            "positions_angstrom",
            np.zeros((4, 3)),
            ValueError,
            "positions_angstrom must have shape",
        ),
        (
            "kinetic_energies_eV",
            np.zeros((4, 3)),
            ValueError,
            "kinetic_energies_eV must have shape",
        ),
        (
            "total_energies_eV",
            np.full((4, 4), np.nan),
            ValueError,
            "total_energies_eV contains non-finite",
        ),
        (
            "charge_sums_e",
            np.zeros((4, 3)),
            ValueError,
            "charge_sums_e must have shape",
        ),
        (
            "batch_ptr",
            np.array([0.0, 3.0, 6.0, 24.0, 42.0]),
            TypeError,
            "batch_ptr must contain integers",
        ),
        (
            "dt_fs",
            0.0,
            ValueError,
            "trajectory.dt_fs",
        ),
    ),
)
def test_trajectory_layout_errors_name_the_invalid_field(
    field: str,
    replacement: Any,
    error: type[Exception],
    match: str,
) -> None:
    trajectory, _ = _trajectory()
    broken = replace(trajectory, **{field: replacement})

    with pytest.raises(error, match=match):
        _analyze(broken)


def test_negative_kinetic_energy_and_single_frame_are_rejected() -> None:
    trajectory, _ = _trajectory()
    negative_kinetic = trajectory.kinetic_energies_eV.copy()
    negative_kinetic[0, 0] = -1.0
    with pytest.raises(ValueError, match="must be non-negative"):
        _analyze(replace(trajectory, kinetic_energies_eV=negative_kinetic))

    one_frame = replace(
        trajectory,
        positions_angstrom=trajectory.positions_angstrom[:1],
        kinetic_energies_eV=trajectory.kinetic_energies_eV[:1],
        total_energies_eV=trajectory.total_energies_eV[:1],
        charge_sums_e=trajectory.charge_sums_e[:1],
    )
    with pytest.raises(ValueError, match="at least two frames"):
        _analyze(one_frame)
