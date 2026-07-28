"""Scientific tables for the water-IR tutorial.

The notebook keeps the scientific choices visible and passes every comparison
condition and frequency window into these functions. This module owns the
repetitive array handling, table assembly, and mode bookkeeping; it does not
run Toolkit models or dynamics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .reference import (
    IRSpectrumComparison,
    IsotopologueModeMatch,
    compare_md_to_harmonic_reference,
    match_isotopologue_modes,
    reference_water_monomer_mode_labels,
    reference_water_ring_mode_characters,
)
from .spectra import band_centroid, welch_dipole_current_spectrum
from .topology import water_topology_observables

if TYPE_CHECKING:
    from .diagnostics import ClusterTopologyDiagnostics


EV_PER_MOLECULE_TO_KJ_MOL = 96.485_332_123_310_02

_IR_LABELS = ("H2O", "D2O", "(H2O)6", "(D2O)6")
_TEMPERATURE_PAIR_NAMES = (
    "H2O_D2O",
    "H6_D6",
    "H2O_H6",
    "D2O_D6",
)


@dataclass(frozen=True)
class IRSpectrumAnalysis:
    """Per-system classical spectra and their fixed-window metrics."""

    spectra: dict[str, tuple[np.ndarray, np.ndarray]]
    metrics: pd.DataFrame


@dataclass(frozen=True)
class IRComparisonAnalysis:
    """The four tutorial comparisons and the checks controlling each value."""

    temperature_relative_differences: dict[str, float]
    temperature_matches: dict[str, bool]
    table: pd.DataFrame


@dataclass(frozen=True)
class ReferenceComparisonAnalysis:
    """MD-versus-harmonic comparisons allowed by the topology check."""

    comparisons: dict[str, IRSpectrumComparison]
    metrics: pd.DataFrame


@dataclass(frozen=True)
class ModeMappingAnalysis:
    """H-to-D mode table plus the mappings used to build it."""

    table: pd.DataFrame
    monomer: IsotopologueModeMatch
    hexamer: IsotopologueModeMatch
    hexamer_fine: IsotopologueModeMatch


def _labels_once(labels: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(label) for label in labels)
    if not result:
        raise ValueError("labels must not be empty")
    if len(set(result)) != len(result):
        raise ValueError("labels must be unique")
    return result


def _window_for_label(
    label: str,
    region_windows_cm1: Mapping[str, tuple[float, float]],
) -> tuple[float, float]:
    isotope = "D" if "D" in label else "H"
    try:
        low, high = map(float, region_windows_cm1[isotope])
    except KeyError as exc:
        raise ValueError(f"missing {isotope!r} spectral window") from exc
    if not np.isfinite((low, high)).all() or not low < high:
        raise ValueError(f"{isotope!r} spectral window must have increasing bounds")
    return low, high


def first_atomic_data_table(
    *,
    num_atoms: int,
    atomic_numbers: Sequence[int] | np.ndarray,
    positions_shape: Sequence[int],
    cell: Any | None,
    positions_dtype: str,
    device: str,
) -> pd.DataFrame:
    """Format fields observed from the first notebook-built ``AtomicData``.

    The notebook still constructs and inspects the Toolkit object.  This helper
    only validates the values passed from that visible code and assembles the
    learner-facing table.
    """

    if isinstance(num_atoms, bool) or not isinstance(num_atoms, (int, np.integer)):
        raise TypeError("num_atoms must be an integer")
    atom_count = int(num_atoms)
    if atom_count <= 0:
        raise ValueError("num_atoms must be positive")
    numbers = np.asarray(atomic_numbers)
    if numbers.shape != (atom_count,):
        raise ValueError("atomic_numbers must contain one value per atom")
    if not np.issubdtype(numbers.dtype, np.integer):
        raise TypeError("atomic_numbers must contain integers")
    shape = tuple(int(value) for value in positions_shape)
    if shape != (atom_count, 3):
        raise ValueError("positions_shape must be (num_atoms, 3)")

    rows = [
        ("atoms", atom_count),
        ("atomic numbers", numbers.tolist()),
        ("positions shape", shape),
        ("cell / PBC", "none / nonperiodic" if cell is None else "periodic"),
        ("dtype", str(positions_dtype)),
        ("device", str(device)),
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def first_model_result_tables(
    *,
    num_graphs: int,
    energy_eV: float,
    symbols: Sequence[str],
    charges_e: Sequence[float] | np.ndarray,
    forces_eV_A: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the system- and atom-level tables from the first Toolkit result.

    The caller performs the model call and passes all returned scientific
    values.  The first table preserves the graph count, energy, and total
    predicted charge.  The second preserves each atomic charge, Cartesian
    force component, and force norm.
    """

    if isinstance(num_graphs, bool) or not isinstance(
        num_graphs, (int, np.integer)
    ):
        raise TypeError("num_graphs must be an integer")
    graph_count = int(num_graphs)
    if graph_count <= 0:
        raise ValueError("num_graphs must be positive")
    energy = float(energy_eV)
    if not np.isfinite(energy):
        raise ValueError("energy_eV must be finite")

    atom_symbols = tuple(str(symbol) for symbol in symbols)
    if not atom_symbols or any(not symbol.strip() for symbol in atom_symbols):
        raise ValueError("symbols must contain non-empty atom labels")
    charges = np.asarray(charges_e, dtype=np.float64)
    forces = np.asarray(forces_eV_A, dtype=np.float64)
    if charges.shape != (len(atom_symbols),):
        raise ValueError("charges_e must contain one value per atom")
    if forces.shape != (len(atom_symbols), 3):
        raise ValueError("forces_eV_A must have shape (atoms, 3)")
    if not np.isfinite(charges).all() or not np.isfinite(forces).all():
        raise ValueError("charges_e and forces_eV_A must contain finite values")

    system_table = pd.DataFrame(
        [
            ("Graphs", graph_count),
            ("Energy / eV", energy),
            ("Total predicted charge / e", float(charges.sum())),
        ],
        columns=["System result", "Value"],
    )
    atom_table = pd.DataFrame(
        {
            "atom": [
                f"{symbol}{index + 1}"
                for index, symbol in enumerate(atom_symbols)
            ],
            "charge (e)": charges,
            "Fx (eV/Å)": forces[:, 0],
            "Fy (eV/Å)": forces[:, 1],
            "Fz (eV/Å)": forces[:, 2],
            "|F| (eV/Å)": np.linalg.norm(forces, axis=1),
        }
    ).round(6)
    return system_table, atom_table


def ir_spectrum_metrics(
    dipoles_e_angstrom: np.ndarray,
    labels: Sequence[str],
    *,
    dt_fs: float,
    segment_time_fs: float,
    overlap: float,
    region_windows_cm1: Mapping[str, tuple[float, float]],
) -> IRSpectrumAnalysis:
    """Compute all batched-system spectra and preserve the notebook columns."""

    names = _labels_once(labels)
    dipoles = np.asarray(dipoles_e_angstrom, dtype=np.float64)
    if dipoles.ndim != 3 or dipoles.shape[1:] != (len(names), 3):
        raise ValueError("dipoles must have shape (frames, systems, 3)")

    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, float | int | str]] = []
    for graph, label in enumerate(names):
        wavenumber, intensity, per_segment = welch_dipole_current_spectrum(
            dipoles[:, graph],
            dt_fs,
            segment_time_fs=segment_time_fs,
            overlap=overlap,
        )
        window = _window_for_label(label, region_windows_cm1)
        centroid = band_centroid(wavenumber, intensity, window)
        segment_centroids = np.asarray(
            [band_centroid(wavenumber, segment, window) for segment in per_segment]
        )
        spectra[label] = (wavenumber, intensity)
        rows.append(
            {
                "system": label,
                "OH_OD_region_centroid_cm-1": centroid,
                "Welch_segment_std_cm-1": segment_centroids.std(ddof=1),
                "Welch_segments": len(segment_centroids),
            }
        )
    return IRSpectrumAnalysis(
        spectra=spectra,
        metrics=pd.DataFrame(rows).set_index("system"),
    )


def _relative_difference(left: float, right: float) -> float:
    values = np.asarray((left, right), dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("mean NVE temperatures must be finite and positive")
    return float(abs(left - right) / (0.5 * (left + right)))


def ir_comparison_table(
    metrics: pd.DataFrame,
    nve_temperature_K: np.ndarray,
    labels: Sequence[str],
    *,
    pair_temperature_relative_tolerance: float,
    cluster_reference_allowed: bool,
) -> IRComparisonAnalysis:
    """Build the four comparisons used by the tutorial.

    A comparison that does not meet its temperature or topology requirement
    leaves ``value`` as NaN instead of printing a number with a warning.
    """

    names = _labels_once(labels)
    missing = set(_IR_LABELS) - set(names)
    if missing:
        raise ValueError(f"missing required systems: {sorted(missing)!r}")
    temperatures = np.asarray(nve_temperature_K, dtype=np.float64)
    if temperatures.ndim != 2 or temperatures.shape[1] != len(names):
        raise ValueError("nve_temperature_K must have shape (frames, systems)")
    tolerance = float(pair_temperature_relative_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("pair temperature tolerance must be finite and non-negative")
    missing_metrics = set(_IR_LABELS) - set(map(str, metrics.index))
    if missing_metrics:
        raise ValueError(f"metrics are missing systems: {sorted(missing_metrics)!r}")

    column = {label: names.index(label) for label in _IR_LABELS}
    mean_temperature = {
        label: float(temperatures[:, column[label]].mean()) for label in _IR_LABELS
    }
    relative = {
        "H2O_D2O_temperature_relative_difference": _relative_difference(
            mean_temperature["H2O"], mean_temperature["D2O"]
        ),
        "H6_D6_temperature_relative_difference": _relative_difference(
            mean_temperature["(H2O)6"], mean_temperature["(D2O)6"]
        ),
        "H2O_H6_temperature_relative_difference": _relative_difference(
            mean_temperature["H2O"], mean_temperature["(H2O)6"]
        ),
        "D2O_D6_temperature_relative_difference": _relative_difference(
            mean_temperature["D2O"], mean_temperature["(D2O)6"]
        ),
    }
    temperature_matches = {
        pair: bool(value <= tolerance)
        for pair, value in zip(_TEMPERATURE_PAIR_NAMES, relative.values(), strict=True)
    }

    rows: list[dict[str, Any]] = []

    def add(
        name: str,
        candidate_value: float,
        *,
        temperatures_match: bool,
        topology_unchanged: bool = True,
    ) -> None:
        shown = bool(temperatures_match and topology_unchanged)
        reasons: list[str] = []
        if not temperatures_match:
            reasons.append("mean temperatures differ by more than the allowed tolerance")
        if not topology_unchanged:
            reasons.append("initial ring changed")
        rows.append(
            {
                "comparison": name,
                "value": float(candidate_value) if shown else np.nan,
                # These names are part of the saved v1 result format. The
                # notebook presents friendlier labels through
                # ``comparison_display_table`` below.
                "reported": shown,
                "thermal_gate_passed": bool(temperatures_match),
                "topology_gate_passed": bool(topology_unchanged),
                "status": "reported" if shown else "; ".join(reasons),
            }
        )

    centroid = metrics["OH_OD_region_centroid_cm-1"]
    add(
        "H2O_over_D2O_centroid",
        centroid.loc["H2O"] / centroid.loc["D2O"],
        temperatures_match=temperature_matches["H2O_D2O"],
    )
    add(
        "H6_over_D6_centroid",
        centroid.loc["(H2O)6"] / centroid.loc["(D2O)6"],
        temperatures_match=temperature_matches["H6_D6"],
        topology_unchanged=cluster_reference_allowed,
    )
    add(
        "H_cluster_minus_monomer_OH_region_centroid_cm-1",
        centroid.loc["(H2O)6"] - centroid.loc["H2O"],
        temperatures_match=temperature_matches["H2O_H6"],
        topology_unchanged=cluster_reference_allowed,
    )
    add(
        "D_cluster_minus_monomer_OD_region_centroid_cm-1",
        centroid.loc["(D2O)6"] - centroid.loc["D2O"],
        temperatures_match=temperature_matches["D2O_D6"],
        topology_unchanged=cluster_reference_allowed,
    )
    return IRComparisonAnalysis(
        temperature_relative_differences=relative,
        temperature_matches=temperature_matches,
        table=pd.DataFrame(rows).set_index("comparison"),
    )


def comparison_display_table(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Return the comparison table with plain learner-facing column labels."""

    required = {
        "value",
        "reported",
        "thermal_gate_passed",
        "topology_gate_passed",
        "status",
    }
    missing = required - set(comparisons.columns)
    if missing:
        raise ValueError(f"comparison table is missing columns: {sorted(missing)!r}")
    return comparisons.rename(
        columns={
            "reported": "shown",
            "thermal_gate_passed": "temperatures_match",
            "topology_gate_passed": "topology_unchanged",
        }
    )


def reference_comparison_metrics(
    spectra: Mapping[str, tuple[np.ndarray, np.ndarray]],
    references: Mapping[str, Any],
    labels: Sequence[str],
    *,
    dt_fs: float,
    segment_time_fs: float,
    region_windows_cm1: Mapping[str, tuple[float, float]],
    cluster_reference_allowed: bool,
) -> ReferenceComparisonAnalysis:
    """Compare MD and harmonic references only where topology permits it."""

    names = _labels_once(labels)
    allowed = set(names if cluster_reference_allowed else ("H2O", "D2O"))
    comparisons: dict[str, IRSpectrumComparison] = {}
    rows: list[dict[str, float | str]] = []
    for label in names:
        if label not in allowed:
            continue
        try:
            wavenumber, intensity = spectra[label]
            reference = references[label]
        except KeyError as exc:
            raise ValueError(f"missing spectrum or reference for {label!r}") from exc
        comparison = compare_md_to_harmonic_reference(
            wavenumber,
            intensity,
            reference,
            dt_fs=dt_fs,
            segment_time_fs=segment_time_fs,
            summary_window_cm1=_window_for_label(label, region_windows_cm1),
        )
        comparisons[label] = comparison
        rows.append(
            {
                "system": label,
                "MD_OH_OD_region_centroid_cm-1": comparison.md_summary.centroid_cm1,
                "DFT_stick_OH_OD_region_centroid_cm-1": (
                    comparison.reference_stick_summary.centroid_cm1
                ),
                "MD_width_10_90_cm-1": comparison.md_summary.width_10_90_cm1,
                "DFT_Hann_width_10_90_cm-1": (
                    comparison.reference_envelope_summary.width_10_90_cm1
                ),
                "comparison_scope": "qualitative region inspection",
            }
        )
    return ReferenceComparisonAnalysis(
        comparisons=comparisons,
        metrics=pd.DataFrame(rows).set_index("system"),
    )


def assignments_are_subspace_equivalent(
    first: IsotopologueModeMatch,
    second: IsotopologueModeMatch,
) -> bool:
    """Accept reordered members only inside a declared degenerate subspace."""

    if len(first.source_to_target) != len(second.source_to_target):
        return False
    for source, (first_target, second_target) in enumerate(
        zip(first.source_to_target, second.source_to_target, strict=True)
    ):
        if first_target == second_target:
            continue
        target_subspace: set[int] = set()
        for result in (first, second):
            for block in result.ambiguous_subspaces:
                if source in block.source_indices:
                    target_subspace.update(map(int, block.target_indices))
        if not {int(first_target), int(second_target)} <= target_subspace:
            return False
    return True


def grouped_mapping_rows(
    mode_map: IsotopologueModeMatch,
    selected_modes: Sequence[int] | np.ndarray,
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...], float]]:
    """Yield modes individually, but near-degenerate modes as whole subspaces."""

    selected = set(map(int, selected_modes))
    by_source: dict[int, tuple[tuple[int, ...], tuple[int, ...], float]] = {}
    for block in mode_map.ambiguous_subspaces:
        source = tuple(mode for mode in block.source_indices if mode in selected)
        if not source:
            continue
        if len(source) != len(block.source_indices):
            raise RuntimeError("Displayed band cuts through a degenerate subspace")
        item = (
            tuple(map(int, source)),
            tuple(map(int, block.target_indices)),
            float(block.minimum_principal_overlap),
        )
        for mode in source:
            by_source[int(mode)] = item

    emitted: set[int] = set()
    for source in sorted(selected):
        if source in emitted:
            continue
        if source in by_source:
            source_modes, target_modes, overlap = by_source[source]
        else:
            source_modes = (source,)
            target_modes = (int(mode_map.source_to_target[source]),)
            overlap = float(mode_map.minimum_path_squared_overlaps[source])
        emitted.update(source_modes)
        yield source_modes, target_modes, overlap


def h_to_d_mode_mapping_table(
    references: Mapping[str, Any],
    *,
    coarse_mass_path_steps: int,
    fine_mass_path_steps: int,
    degeneracy_tolerance_cm1: float,
    covalent_oh_cutoff_angstrom: float,
    h_acceptor_cutoff_angstrom: float,
    oo_cutoff_angstrom: float,
    hbond_angle_cutoff_deg: float,
) -> ModeMappingAnalysis:
    """Build the H/D mode table and verify path-grid equivalence."""

    return _h_to_d_mode_mapping_table_with_dependencies(
        references,
        coarse_mass_path_steps=coarse_mass_path_steps,
        fine_mass_path_steps=fine_mass_path_steps,
        degeneracy_tolerance_cm1=degeneracy_tolerance_cm1,
        covalent_oh_cutoff_angstrom=covalent_oh_cutoff_angstrom,
        h_acceptor_cutoff_angstrom=h_acceptor_cutoff_angstrom,
        oo_cutoff_angstrom=oo_cutoff_angstrom,
        hbond_angle_cutoff_deg=hbond_angle_cutoff_deg,
        mode_matcher=match_isotopologue_modes,
        monomer_mode_labeler=reference_water_monomer_mode_labels,
        ring_mode_characterizer=reference_water_ring_mode_characters,
    )


def _h_to_d_mode_mapping_table_with_dependencies(
    references: Mapping[str, Any],
    *,
    coarse_mass_path_steps: int,
    fine_mass_path_steps: int,
    degeneracy_tolerance_cm1: float,
    covalent_oh_cutoff_angstrom: float,
    h_acceptor_cutoff_angstrom: float,
    oo_cutoff_angstrom: float,
    hbond_angle_cutoff_deg: float,
    mode_matcher: Callable[..., IsotopologueModeMatch],
    monomer_mode_labeler: Callable[[Any], Sequence[str]],
    ring_mode_characterizer: Callable[..., Any],
) -> ModeMappingAnalysis:
    """Implementation seam for deterministic tests of mode-table assembly."""

    missing = set(_IR_LABELS) - set(references)
    if missing:
        raise ValueError(f"missing harmonic references: {sorted(missing)!r}")
    match_kwargs = {"degeneracy_tolerance_cm1": degeneracy_tolerance_cm1}
    monomer = mode_matcher(
        references["H2O"],
        references["D2O"],
        mass_path_steps=coarse_mass_path_steps,
        **match_kwargs,
    )
    hexamer = mode_matcher(
        references["(H2O)6"],
        references["(D2O)6"],
        mass_path_steps=coarse_mass_path_steps,
        **match_kwargs,
    )
    hexamer_fine = mode_matcher(
        references["(H2O)6"],
        references["(D2O)6"],
        mass_path_steps=fine_mass_path_steps,
        **match_kwargs,
    )
    if not assignments_are_subspace_equivalent(hexamer, hexamer_fine):
        raise RuntimeError(
            f"H/D mode mapping changed between {coarse_mass_path_steps} and "
            f"{fine_mass_path_steps} mass steps"
        )

    monomer_characters = monomer_mode_labeler(references["H2O"])
    hexamer_character_data = ring_mode_characterizer(
        references["(H2O)6"],
        covalent_oh_cutoff_angstrom=covalent_oh_cutoff_angstrom,
        h_acceptor_cutoff_angstrom=h_acceptor_cutoff_angstrom,
        oo_cutoff_angstrom=oo_cutoff_angstrom,
        hbond_angle_cutoff_deg=hbond_angle_cutoff_deg,
        require_single_ring=True,
    )
    expected_categories = ("bend", "hbonded_oh", "free_oh", "intermolecular")
    if tuple(hexamer_character_data.categories) != expected_categories:
        raise RuntimeError("unexpected water-ring mode-character categories")

    rows: list[dict[str, Any]] = []
    for family, h_label, d_label, mode_map, characters, character_fractions in (
        (
            "monomer",
            "H2O",
            "D2O",
            monomer,
            monomer_characters,
            None,
        ),
        (
            "cyclic hexamer",
            "(H2O)6",
            "(D2O)6",
            hexamer,
            hexamer_character_data.dominant_labels,
            hexamer_character_data.fractions,
        ),
    ):
        h_reference = references[h_label]
        d_reference = references[d_label]
        selected = np.arange(h_reference.n_modes)
        if family == "cyclic hexamer":
            selected = selected[np.asarray(characters) != "intermolecular"]
        for h_modes, d_modes, overlap in grouped_mapping_rows(mode_map, selected):
            h_modes_array = np.asarray(h_modes)
            d_modes_array = np.asarray(d_modes)
            h_frequency = h_reference.frequencies_cm1[h_modes_array]
            d_frequency = d_reference.frequencies_cm1[d_modes_array]
            unique_characters = sorted({characters[mode] for mode in h_modes})
            fractions = (
                np.full(4, np.nan)
                if character_fractions is None
                else character_fractions[h_modes_array].mean(axis=0)
            )
            rows.append(
                {
                    "system": family,
                    "mapping_unit": "subspace" if len(h_modes) > 1 else "mode",
                    "H_mode_1based": ",".join(str(mode + 1) for mode in h_modes),
                    "D_mode_1based": ",".join(str(mode + 1) for mode in d_modes),
                    "character": "+".join(unique_characters),
                    "H_center_cm-1": h_frequency.mean(),
                    "H_span_cm-1": np.ptp(h_frequency),
                    "D_center_cm-1": d_frequency.mean(),
                    "D_span_cm-1": np.ptp(d_frequency),
                    "H_over_D_center": h_frequency.mean() / d_frequency.mean(),
                    "H_IR_sum_km_mol": h_reference.ir_intensities_km_mol[
                        h_modes_array
                    ].sum(),
                    "D_IR_sum_km_mol": d_reference.ir_intensities_km_mol[
                        d_modes_array
                    ].sum(),
                    "mapping_overlap": overlap,
                    "bend_fraction": fractions[0],
                    "hbonded_OH_fraction": fractions[1],
                    "free_OH_fraction": fractions[2],
                    "intermolecular_fraction": fractions[3],
                }
            )
    return ModeMappingAnalysis(
        table=pd.DataFrame(rows),
        monomer=monomer,
        hexamer=hexamer,
        hexamer_fine=hexamer_fine,
    )


def topology_time_series(
    trajectory: Any,
    graph_index: int,
    *,
    h_acceptor_cutoff_angstrom: float,
    oo_cutoff_angstrom: float,
    hbond_angle_cutoff_deg: float,
    precomputed_topology: ClusterTopologyDiagnostics | None = None,
) -> pd.DataFrame:
    """Return frame-resolved H-bond and oxygen-skeleton observables.

    Covalent O-H ownership is assigned once from the first frame, matching the
    integrity check. The returned table is diagnostic data; it does not decide
    whether a comparison is scientifically valid. ``precomputed_topology``
    reuses the immutable result from ``analyze_production_trajectory`` after
    checking its graph and H-bond cutoffs against this explicit call.
    """

    if precomputed_topology is None:
        start, stop = map(int, trajectory.batch_ptr[graph_index : graph_index + 2])
        topology = water_topology_observables(
            trajectory.positions_angstrom[:, start:stop],
            trajectory.atomic_numbers[start:stop],
            oxygen_connectivity_cutoff_angstrom=None,
            h_acceptor_cutoff_angstrom=h_acceptor_cutoff_angstrom,
            oo_cutoff_angstrom=oo_cutoff_angstrom,
            hbond_angle_cutoff_deg=hbond_angle_cutoff_deg,
        )
    else:
        if precomputed_topology.graph_index != int(graph_index):
            raise ValueError(
                "precomputed topology graph does not match graph_index"
            )
        requested_cutoffs = {
            "h_acceptor_cutoff_angstrom": float(h_acceptor_cutoff_angstrom),
            "oo_cutoff_angstrom": float(oo_cutoff_angstrom),
            "hbond_angle_cutoff_deg": float(hbond_angle_cutoff_deg),
        }
        cached_cutoffs = {
            "h_acceptor_cutoff_angstrom": (
                precomputed_topology.h_acceptor_cutoff_angstrom
            ),
            "oo_cutoff_angstrom": precomputed_topology.oo_cutoff_angstrom,
            "hbond_angle_cutoff_deg": precomputed_topology.hbond_angle_cutoff_deg,
        }
        mismatched = [
            name
            for name, requested in requested_cutoffs.items()
            if requested != cached_cutoffs[name]
        ]
        if mismatched:
            raise ValueError(
                "precomputed topology was calculated with different values for: "
                + ", ".join(mismatched)
            )
        topology = precomputed_topology.observables
        frame_count = np.asarray(trajectory.positions_angstrom).shape[0]
        if len(topology.hydrogen_bond_count) != frame_count:
            raise ValueError(
                "precomputed topology frame count does not match the trajectory"
            )

    ring = topology.ring_masks
    frame_count = len(topology.hydrogen_bond_count)
    return pd.DataFrame(
        {
            "time_ps": np.arange(frame_count) * float(trajectory.dt_fs) / 1000.0,
            "H_bonds": topology.hydrogen_bond_count,
            "exact_single_ring": ring.exact_single_ring,
            "has_directed_cycle": ring.any_cycle,
            "initial_ring_present": ring.initial_cycle,
            "directed_cycle_multiplicity": ring.cycle_multiplicity,
            "oxygen_Rg_angstrom": topology.oxygen_radius_gyration_angstrom,
            "oxygen_RMSD_angstrom": topology.oxygen_rmsd_angstrom,
            "max_assigned_OH_angstrom": topology.max_assigned_oh_angstrom,
        }
    )


def dimer_interaction_energy_table(
    distances_angstrom: Sequence[float] | np.ndarray,
    components_eV: Mapping[
        str,
        tuple[
            Sequence[float] | np.ndarray,
            float | Sequence[float] | np.ndarray,
            float | Sequence[float] | np.ndarray,
        ],
    ],
) -> pd.DataFrame:
    """Unpack ``E(AB) - E(A) - E(B)`` for one or more energy components."""

    distances = np.asarray(distances_angstrom, dtype=np.float64)
    if distances.ndim != 1 or distances.size == 0:
        raise ValueError("distances_angstrom must be a non-empty vector")
    if not np.isfinite(distances).all():
        raise ValueError("distances_angstrom contains non-finite values")
    if not components_eV:
        raise ValueError("components_eV must not be empty")
    table: dict[str, np.ndarray] = {"distance_angstrom": distances}
    for raw_name, (dimer, monomer_a, monomer_b) in components_eV.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("component names must not be empty")
        dimer_values = np.asarray(dimer, dtype=np.float64)
        try:
            dimer_values = np.broadcast_to(dimer_values, distances.shape)
            a_values = np.broadcast_to(
                np.asarray(monomer_a, dtype=np.float64), distances.shape
            )
            b_values = np.broadcast_to(
                np.asarray(monomer_b, dtype=np.float64), distances.shape
            )
        except ValueError as exc:
            raise ValueError(
                f"energy component {name!r} cannot broadcast to the distance grid"
            ) from exc
        interaction = dimer_values - a_values - b_values
        if not np.isfinite(interaction).all():
            raise ValueError(f"energy component {name!r} contains non-finite values")
        table[f"{name}_interaction_eV"] = interaction
        table[f"{name}_interaction_kJ_mol"] = (
            interaction * EV_PER_MOLECULE_TO_KJ_MOL
        )
    return pd.DataFrame(table)


__all__ = [
    "EV_PER_MOLECULE_TO_KJ_MOL",
    "IRComparisonAnalysis",
    "IRSpectrumAnalysis",
    "ModeMappingAnalysis",
    "ReferenceComparisonAnalysis",
    "assignments_are_subspace_equivalent",
    "comparison_display_table",
    "dimer_interaction_energy_table",
    "first_atomic_data_table",
    "first_model_result_tables",
    "grouped_mapping_rows",
    "h_to_d_mode_mapping_table",
    "ir_comparison_table",
    "ir_spectrum_metrics",
    "reference_comparison_metrics",
    "topology_time_series",
]
