"""Dependency-light scientific diagnostics for the water IR workflow.

The production-trajectory interface in this module owns array validation,
unit conversions, table assembly, and reproducible pass/fail calculations.
Notebook cells still supply every scientific cutoff and advisory value, then
decide how to display or act on the returned results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

from .topology import (
    WaterTopologyObservables,
    longest_false_run,
    water_topology_observables,
)

if TYPE_CHECKING:
    import pandas as pd

    from .capture import IRTrajectory


@dataclass(frozen=True)
class ClusterTopologyDiagnostics:
    """One cached cluster calculation and the choices that produced it.

    ``graph_index`` and ``system_label`` keep the cached arrays tied to the
    trajectory slice they describe.  The distance and angle values are stored
    with the result so a later table builder can reject an accidental mismatch
    instead of silently reusing data calculated with different choices.
    """

    graph_index: int
    system_label: str
    oxygen_connectivity_cutoff_angstrom: float
    h_acceptor_cutoff_angstrom: float
    oo_cutoff_angstrom: float
    hbond_angle_cutoff_deg: float
    observables: WaterTopologyObservables


@dataclass(frozen=True)
class ProductionDiagnostics:
    """Complete post-processing result for one production trajectory.

    Arrays are read-only copies so later notebook code cannot accidentally
    alter a result that has already been displayed or saved.  The two tables
    retain the notebook's established column and index names. Per-cluster
    topology records keep the frame observables available for later tables and
    plots without recalculating them. Boolean fields expose the three decisions
    used by downstream cells without deciding what the notebook should print,
    report, or raise.
    """

    diagnostic_table: pd.DataFrame
    integrity_table: pd.DataFrame
    nve_temperature_3n_K: NDArray[np.float64]
    energy_drift_meV_atom_ps: NDArray[np.float64]
    max_energy_excursion_meV_atom: NDArray[np.float64]
    energy_spacing_meV_atom: NDArray[np.float64]
    cluster_topologies: tuple[ClusterTopologyDiagnostics, ...]
    cluster_intact: bool
    cluster_dft_comparison_valid: bool
    energy_within_advisory: bool


def _to_numpy(value: Any) -> np.ndarray:
    """Detach a Torch-like value if needed, then return a NumPy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def mass_only_invariance(
    batch: Any,
    outputs: Mapping[str, Any],
    *,
    position_rtol: float,
    position_atol_angstrom: float,
    energy_tolerance_eV: float,
    force_tolerance_eV_A: float,
    charge_tolerance_e: float,
) -> dict[str, float]:
    """Assert that H/D changes only masses, not PES/model predictions.

    Every acceptance value is required at the call site.  This helper performs
    the repeated reductions and comparisons but does not choose the numerical
    definition of agreement.
    """

    for name, value in (
        ("position_rtol", position_rtol),
        ("position_atol_angstrom", position_atol_angstrom),
        ("energy_tolerance_eV", energy_tolerance_eV),
        ("force_tolerance_eV_A", force_tolerance_eV_A),
        ("charge_tolerance_e", charge_tolerance_e),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be non-negative and finite")

    energies = _to_numpy(outputs["energy"]).reshape(-1)
    forces = _to_numpy(outputs["forces"])
    charges = _to_numpy(outputs["charges"]).reshape(-1)
    positions = _to_numpy(batch.positions)
    atomic_numbers = _to_numpy(batch.atomic_numbers)
    atomic_masses = _to_numpy(batch.atomic_masses)
    boundaries = _to_numpy(batch.batch_ptr).astype(int).tolist()

    deltas: dict[str, float] = {}
    for left, right, label in ((0, 1, "monomer"), (2, 3, "hexamer")):
        l0, l1 = boundaries[left], boundaries[left + 1]
        r0, r1 = boundaries[right], boundaries[right + 1]
        if (l1 - l0) != (r1 - r0):
            raise AssertionError(f"{label}: isotope pair topology differs")
        if not np.array_equal(atomic_numbers[l0:l1], atomic_numbers[r0:r1]):
            raise AssertionError(f"{label}: deuterium must keep atomic number 1")
        np.testing.assert_allclose(
            positions[l0:l1],
            positions[r0:r1],
            rtol=position_rtol,
            atol=position_atol_angstrom,
        )

        energy_delta = float(abs(energies[left] - energies[right]))
        force_delta = float(np.max(abs(forces[l0:l1] - forces[r0:r1])))
        charge_delta = float(np.max(abs(charges[l0:l1] - charges[r0:r1])))
        if energy_delta > energy_tolerance_eV:
            raise AssertionError(f"{label}: isotope energy changed by {energy_delta}")
        if force_delta > force_tolerance_eV_A:
            raise AssertionError(f"{label}: isotope forces changed by {force_delta}")
        if charge_delta > charge_tolerance_e:
            raise AssertionError(f"{label}: isotope charges changed by {charge_delta}")
        deltas[f"{label}_energy_eV"] = energy_delta
        deltas[f"{label}_force_eV_A"] = force_delta
        deltas[f"{label}_charge_e"] = charge_delta

    h_mass = atomic_masses[atomic_numbers == 1][0]
    d_start = boundaries[1]
    d_numbers = atomic_numbers[d_start : boundaries[2]]
    d_mass = atomic_masses[d_start : boundaries[2]][d_numbers == 1][0]
    deltas["D_over_H_mass"] = float(d_mass / h_mass)
    return deltas


def _cluster_topology_observables(
    trajectory: "IRTrajectory",
    graph_index: int,
    *,
    oxygen_cutoff_angstrom: float,
    h_acceptor_cutoff_angstrom: float,
    oo_cutoff_angstrom: float,
    hbond_angle_cutoff_deg: float,
) -> WaterTopologyObservables:
    """Calculate topology arrays for one explicitly selected graph."""

    start, stop = map(int, trajectory.batch_ptr[graph_index : graph_index + 2])
    return water_topology_observables(
        trajectory.positions_angstrom[:, start:stop],
        trajectory.atomic_numbers[start:stop],
        oxygen_connectivity_cutoff_angstrom=oxygen_cutoff_angstrom,
        h_acceptor_cutoff_angstrom=h_acceptor_cutoff_angstrom,
        oo_cutoff_angstrom=oo_cutoff_angstrom,
        hbond_angle_cutoff_deg=hbond_angle_cutoff_deg,
    )


def _cluster_integrity_from_topology(
    topology: WaterTopologyObservables,
    *,
    dt_fs: float,
) -> dict[str, float | int | bool]:
    """Reduce cached frame observables to the established integrity columns."""

    ring_masks = topology.ring_masks
    first_initial_loss = np.flatnonzero(~ring_masks.initial_cycle)
    components = topology.oxygen_component_count
    if components is None:
        raise RuntimeError("cluster integrity requires oxygen component counts")
    radius_gyration = topology.oxygen_radius_gyration_angstrom
    return {
        "oxygen_count": topology.oxygen_count,
        "max_OH_angstrom": float(topology.max_assigned_oh_angstrom.max()),
        "max_oxygen_components": int(components.max()),
        "H_bonds_initial": int(topology.hydrogen_bond_count[0]),
        "H_bonds_min": int(topology.hydrogen_bond_count.min()),
        "H_bonds_max": int(topology.hydrogen_bond_count.max()),
        "single_ring_fraction": float(ring_masks.exact_single_ring.mean()),
        "all_frames_single_ring": bool(ring_masks.exact_single_ring.all()),
        "directed_six_cycle_fraction": float(ring_masks.any_cycle.mean()),
        "all_frames_have_directed_six_cycle": bool(ring_masks.any_cycle.all()),
        "initial_ring_fraction": float(ring_masks.initial_cycle.mean()),
        "all_frames_initial_ring": bool(ring_masks.initial_cycle.all()),
        "first_initial_ring_loss_ps": (
            float(first_initial_loss[0] * dt_fs / 1000.0)
            if len(first_initial_loss)
            else float("nan")
        ),
        "longest_initial_ring_absence_ps": float(
            longest_false_run(ring_masks.initial_cycle) * dt_fs / 1000.0
        ),
        "maximum_directed_cycle_multiplicity": int(ring_masks.cycle_multiplicity.max()),
        "Rg_initial_angstrom": float(radius_gyration[0]),
        "Rg_min_over_initial": float(
            radius_gyration.min() / max(radius_gyration[0], 1e-12)
        ),
        "Rg_max_over_initial": float(
            radius_gyration.max() / max(radius_gyration[0], 1e-12)
        ),
        "oxygen_RMSD_max_angstrom": float(
            topology.oxygen_rmsd_angstrom.max()
        ),
    }


def cluster_integrity(
    trajectory: "IRTrajectory",
    graph_index: int,
    *,
    oxygen_cutoff_angstrom: float,
    h_acceptor_cutoff_angstrom: float,
    oo_cutoff_angstrom: float,
    hbond_angle_cutoff_deg: float,
) -> dict[str, float | int | bool]:
    """Summarize covalent, H-bond-ring, and oxygen-skeleton integrity."""

    topology = _cluster_topology_observables(
        trajectory,
        graph_index,
        oxygen_cutoff_angstrom=oxygen_cutoff_angstrom,
        h_acceptor_cutoff_angstrom=h_acceptor_cutoff_angstrom,
        oo_cutoff_angstrom=oo_cutoff_angstrom,
        hbond_angle_cutoff_deg=hbond_angle_cutoff_deg,
    )
    return _cluster_integrity_from_topology(
        topology,
        dt_fs=float(trajectory.dt_fs),
    )


def _finite_numeric_array(value: Any, *, field: str) -> np.ndarray:
    """Return a finite numeric array or raise a field-specific error."""

    array = _to_numpy(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"trajectory.{field} must be a numeric array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"trajectory.{field} contains non-finite values")
    return array


def _positive_finite(value: float, *, field: str) -> float:
    """Validate one positive finite scalar without selecting a default."""

    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be positive and finite")
    return result


def _nonnegative_finite(value: float, *, field: str) -> float:
    """Validate one non-negative finite scalar without selecting a default."""

    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be non-negative and finite")
    return result


def _readonly_copy(
    value: Any,
    *,
    dtype: DTypeLike | None = np.float64,
) -> np.ndarray:
    """Copy an array and prevent accidental mutation by later notebook cells."""

    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _trajectory_layout(
    trajectory: "IRTrajectory",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    """Validate the arrays used by production diagnostics.

    Returns positions, batch pointers, kinetic energies, total energies,
    charge sums, frame count, and graph count in that order.
    """

    positions = _finite_numeric_array(
        trajectory.positions_angstrom,
        field="positions_angstrom",
    )
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(
            "trajectory.positions_angstrom must have shape (frames, atoms, 3)"
        )
    n_frames, n_atoms, _ = positions.shape
    if n_frames < 2:
        raise ValueError("production diagnostics require at least two frames")

    batch_ptr = _to_numpy(trajectory.batch_ptr)
    if batch_ptr.ndim != 1 or len(batch_ptr) < 2:
        raise ValueError("trajectory.batch_ptr must contain graph boundaries")
    if not np.issubdtype(batch_ptr.dtype, np.integer):
        raise TypeError("trajectory.batch_ptr must contain integers")
    batch_ptr = batch_ptr.astype(np.int64, copy=False)
    if batch_ptr[0] != 0 or batch_ptr[-1] != n_atoms:
        raise ValueError(
            "trajectory.batch_ptr must start at zero and end at the atom count"
        )
    if np.any(np.diff(batch_ptr) <= 0):
        raise ValueError("trajectory.batch_ptr must describe non-empty graphs")
    n_graphs = len(batch_ptr) - 1

    atomic_numbers = _finite_numeric_array(
        trajectory.atomic_numbers,
        field="atomic_numbers",
    )
    if atomic_numbers.ndim != 1 or len(atomic_numbers) != n_atoms:
        raise ValueError(
            "trajectory.atomic_numbers must contain one value per atom"
        )

    expected_graph_series_shape = (n_frames, n_graphs)
    kinetic = _finite_numeric_array(
        trajectory.kinetic_energies_eV,
        field="kinetic_energies_eV",
    )
    if kinetic.shape != expected_graph_series_shape:
        raise ValueError(
            "trajectory.kinetic_energies_eV must have shape "
            f"{expected_graph_series_shape}; got {kinetic.shape}"
        )
    if np.any(kinetic < 0.0):
        raise ValueError("trajectory.kinetic_energies_eV must be non-negative")

    total = _finite_numeric_array(
        trajectory.total_energies_eV,
        field="total_energies_eV",
    )
    if total.shape != expected_graph_series_shape:
        raise ValueError(
            "trajectory.total_energies_eV must have shape "
            f"{expected_graph_series_shape}; got {total.shape}"
        )

    charge_sums = _finite_numeric_array(
        trajectory.charge_sums_e,
        field="charge_sums_e",
    )
    if charge_sums.shape != expected_graph_series_shape:
        raise ValueError(
            "trajectory.charge_sums_e must have shape "
            f"{expected_graph_series_shape}; got {charge_sums.shape}"
        )

    _positive_finite(trajectory.dt_fs, field="trajectory.dt_fs")
    return (
        positions,
        batch_ptr,
        kinetic,
        total,
        charge_sums,
        n_frames,
        n_graphs,
    )


def _system_labels(labels: Sequence[str], *, n_graphs: int) -> tuple[str, ...]:
    """Validate stable, unique table labels for every graph."""

    if isinstance(labels, (str, bytes)):
        raise TypeError("labels must be a sequence of system names")
    result = tuple(labels)
    if len(result) != n_graphs:
        raise ValueError(f"labels must contain exactly {n_graphs} system names")
    if any(not isinstance(label, str) or not label.strip() for label in result):
        raise ValueError("labels must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("labels must be unique")
    return result


def _cluster_graphs(
    graph_indices: Sequence[int],
    *,
    n_graphs: int,
) -> tuple[int, ...]:
    """Validate the explicitly selected cluster graphs."""

    if isinstance(graph_indices, (str, bytes)):
        raise TypeError("cluster_graph_indices must be a sequence of integers")
    result: list[int] = []
    for value in graph_indices:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError("cluster_graph_indices must contain integers")
        graph = int(value)
        if not 0 <= graph < n_graphs:
            raise IndexError(f"cluster graph index {graph} is outside the trajectory")
        result.append(graph)
    if not result:
        raise ValueError("cluster_graph_indices must select at least one graph")
    if len(set(result)) != len(result):
        raise ValueError("cluster_graph_indices must not contain duplicates")
    return tuple(result)


def analyze_production_trajectory(
    trajectory: "IRTrajectory",
    *,
    labels: Sequence[str],
    cluster_graph_indices: Sequence[int],
    boltzmann_constant_eV_per_K: float,
    energy_spacing_dtype: DTypeLike,
    oxygen_connectivity_cutoff_angstrom: float,
    covalent_oh_cutoff_angstrom: float,
    h_acceptor_cutoff_angstrom: float,
    oo_cutoff_angstrom: float,
    hbond_angle_cutoff_deg: float,
    energy_excursion_advisory_meV_per_atom: float,
) -> ProductionDiagnostics:
    """Calculate the notebook's complete production-trajectory diagnostics.

    Temperature uses the unconstrained ``3N`` kinetic convention already named
    in the notebook.  Energy drift is the least-squares slope of ``E(t)-E(0)``
    over every saved frame.  Drift and maximum absolute excursion are both
    normalized per atom.  ``energy_spacing_dtype`` makes the precision used by
    the round-off callout explicit rather than inferring it from the archive.

    All topology cutoffs, the covalent O-H limit, and the energy-excursion
    advisory are required keyword arguments.  This function deliberately has
    no scientific threshold defaults.  It returns decisions but never raises
    merely because a cluster or advisory check failed; the notebook owns that
    learner-facing control flow. Each selected cluster topology is calculated
    once and retained in the result for subsequent timeline construction.
    """

    import pandas as pd

    (
        _positions,
        batch_ptr,
        kinetic_energies_eV,
        total_energies_eV,
        charge_sums_e,
        n_frames,
        n_graphs,
    ) = _trajectory_layout(trajectory)
    system_labels = _system_labels(labels, n_graphs=n_graphs)
    cluster_graphs = _cluster_graphs(
        cluster_graph_indices,
        n_graphs=n_graphs,
    )

    boltzmann_constant = _positive_finite(
        boltzmann_constant_eV_per_K,
        field="boltzmann_constant_eV_per_K",
    )
    oxygen_cutoff = _positive_finite(
        oxygen_connectivity_cutoff_angstrom,
        field="oxygen_connectivity_cutoff_angstrom",
    )
    covalent_oh_cutoff = _positive_finite(
        covalent_oh_cutoff_angstrom,
        field="covalent_oh_cutoff_angstrom",
    )
    h_acceptor_cutoff = _positive_finite(
        h_acceptor_cutoff_angstrom,
        field="h_acceptor_cutoff_angstrom",
    )
    oo_cutoff = _positive_finite(
        oo_cutoff_angstrom,
        field="oo_cutoff_angstrom",
    )
    hbond_angle_cutoff = float(hbond_angle_cutoff_deg)
    if not np.isfinite(hbond_angle_cutoff) or not (
        0.0 <= hbond_angle_cutoff <= 180.0
    ):
        raise ValueError("hbond_angle_cutoff_deg must lie in [0, 180]")
    energy_advisory = _nonnegative_finite(
        energy_excursion_advisory_meV_per_atom,
        field="energy_excursion_advisory_meV_per_atom",
    )

    spacing_dtype = np.dtype(energy_spacing_dtype)
    if not np.issubdtype(spacing_dtype, np.floating):
        raise TypeError("energy_spacing_dtype must be a floating-point dtype")

    atoms_per_graph = np.diff(batch_ptr).astype(np.float64, copy=False)
    nve_temperature = (
        2.0
        * kinetic_energies_eV.astype(np.float64, copy=False)
        / (3.0 * atoms_per_graph[None, :] * boltzmann_constant)
    )

    time_ps = (
        np.arange(n_frames, dtype=np.float64)
        * float(trajectory.dt_fs)
        / 1000.0
    )
    energy_delta_eV = (
        total_energies_eV.astype(np.float64, copy=False)
        - total_energies_eV[0].astype(np.float64, copy=False)
    )
    energy_drift = np.asarray(
        [
            1000.0
            * float(np.polyfit(time_ps, energy_delta_eV[:, graph], 1)[0])
            / atoms_per_graph[graph]
            for graph in range(n_graphs)
        ],
        dtype=np.float64,
    )
    energy_excursion = (
        1000.0
        * np.max(np.abs(energy_delta_eV), axis=0)
        / atoms_per_graph
    )
    charge_error = np.max(np.abs(charge_sums_e), axis=0)
    energy_spacing = (
        1000.0
        * np.spacing(
            np.abs(total_energies_eV[0]).astype(spacing_dtype, copy=False)
        )
        / atoms_per_graph
    )

    cluster_topologies = tuple(
        ClusterTopologyDiagnostics(
            graph_index=graph,
            system_label=system_labels[graph],
            oxygen_connectivity_cutoff_angstrom=oxygen_cutoff,
            h_acceptor_cutoff_angstrom=h_acceptor_cutoff,
            oo_cutoff_angstrom=oo_cutoff,
            hbond_angle_cutoff_deg=hbond_angle_cutoff,
            observables=_cluster_topology_observables(
                trajectory,
                graph,
                oxygen_cutoff_angstrom=oxygen_cutoff,
                h_acceptor_cutoff_angstrom=h_acceptor_cutoff,
                oo_cutoff_angstrom=oo_cutoff,
                hbond_angle_cutoff_deg=hbond_angle_cutoff,
            ),
        )
        for graph in cluster_graphs
    )
    integrity_records = [
        _cluster_integrity_from_topology(
            result.observables,
            dt_fs=float(trajectory.dt_fs),
        )
        for result in cluster_topologies
    ]
    integrity_table = pd.DataFrame.from_records(
        integrity_records,
        index=[system_labels[graph] for graph in cluster_graphs],
    )
    integrity_table.index.name = None

    diagnostic_table = pd.DataFrame(
        {
            "system": system_labels,
            "NVE_start_T_3N_K": nve_temperature[0],
            "NVE_mean_T_3N_K": nve_temperature.mean(axis=0),
            "max_charge_error_e": charge_error,
            "energy_drift_meV_atom_ps": energy_drift,
            "max_energy_excursion_meV_atom": energy_excursion,
        }
    ).set_index("system")

    cluster_intact = bool(
        (integrity_table["max_oxygen_components"] == 1).all()
        and (integrity_table["max_OH_angstrom"] < covalent_oh_cutoff).all()
    )
    cluster_dft_comparison_valid = bool(
        integrity_table["all_frames_initial_ring"].all()
    )
    energy_within_advisory = bool(
        np.max(energy_excursion) <= energy_advisory
    )

    return ProductionDiagnostics(
        diagnostic_table=diagnostic_table,
        integrity_table=integrity_table,
        nve_temperature_3n_K=_readonly_copy(nve_temperature),
        energy_drift_meV_atom_ps=_readonly_copy(energy_drift),
        max_energy_excursion_meV_atom=_readonly_copy(energy_excursion),
        energy_spacing_meV_atom=_readonly_copy(energy_spacing),
        cluster_topologies=cluster_topologies,
        cluster_intact=cluster_intact,
        cluster_dft_comparison_valid=cluster_dft_comparison_valid,
        energy_within_advisory=energy_within_advisory,
    )
