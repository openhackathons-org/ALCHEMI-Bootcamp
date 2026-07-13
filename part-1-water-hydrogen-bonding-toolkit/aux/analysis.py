"""Scientific tables for the water-IR tutorial.

The notebook keeps the scientific choices visible and passes every gate and
window into these functions.  This module owns the repetitive array handling,
table assembly, and mode-bookkeeping; it does not run Toolkit models or
dynamics.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
from .topology import directed_ring_masks, kabsch_rmsd_frames


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
class IRComparisonGateAnalysis:
    """The four tutorial comparisons and the gates controlling each value."""

    temperature_relative_differences: dict[str, float]
    thermal_gates: dict[str, bool]
    table: pd.DataFrame


@dataclass(frozen=True)
class ReferenceComparisonAnalysis:
    """MD-versus-harmonic comparisons that survived the topology gate."""

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


def ir_comparison_gate_table(
    metrics: pd.DataFrame,
    nve_temperature_K: np.ndarray,
    labels: Sequence[str],
    *,
    pair_temperature_relative_tolerance: float,
    cluster_topology_gate: bool,
) -> IRComparisonGateAnalysis:
    """Build the four gated comparisons used by the tutorial.

    A failed gate leaves ``value`` as NaN.  The candidate value is never
    silently reported with a warning, and the legacy output columns and status
    strings remain unchanged for downstream validators.
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
    thermal_gates = {
        pair: bool(value <= tolerance)
        for pair, value in zip(_TEMPERATURE_PAIR_NAMES, relative.values(), strict=True)
    }

    rows: list[dict[str, Any]] = []

    def add(
        name: str,
        candidate_value: float,
        *,
        thermal_gate: bool,
        topology_gate: bool = True,
    ) -> None:
        reported = bool(thermal_gate and topology_gate)
        reasons: list[str] = []
        if not thermal_gate:
            reasons.append("thermal-state gate failed")
        if not topology_gate:
            reasons.append("initial-ring persistence gate failed")
        rows.append(
            {
                "comparison": name,
                "value": float(candidate_value) if reported else np.nan,
                "reported": reported,
                "thermal_gate_passed": bool(thermal_gate),
                "topology_gate_passed": bool(topology_gate),
                "status": "reported" if reported else "; ".join(reasons),
            }
        )

    centroid = metrics["OH_OD_region_centroid_cm-1"]
    add(
        "H2O_over_D2O_centroid",
        centroid.loc["H2O"] / centroid.loc["D2O"],
        thermal_gate=thermal_gates["H2O_D2O"],
    )
    add(
        "H6_over_D6_centroid",
        centroid.loc["(H2O)6"] / centroid.loc["(D2O)6"],
        thermal_gate=thermal_gates["H6_D6"],
        topology_gate=cluster_topology_gate,
    )
    add(
        "H_cluster_minus_monomer_OH_region_centroid_cm-1",
        centroid.loc["(H2O)6"] - centroid.loc["H2O"],
        thermal_gate=thermal_gates["H2O_H6"],
        topology_gate=cluster_topology_gate,
    )
    add(
        "D_cluster_minus_monomer_OD_region_centroid_cm-1",
        centroid.loc["(D2O)6"] - centroid.loc["D2O"],
        thermal_gate=thermal_gates["D2O_D6"],
        topology_gate=cluster_topology_gate,
    )
    return IRComparisonGateAnalysis(
        temperature_relative_differences=relative,
        thermal_gates=thermal_gates,
        table=pd.DataFrame(rows).set_index("comparison"),
    )


def reference_comparison_metrics(
    spectra: Mapping[str, tuple[np.ndarray, np.ndarray]],
    references: Mapping[str, Any],
    labels: Sequence[str],
    *,
    dt_fs: float,
    segment_time_fs: float,
    region_windows_cm1: Mapping[str, tuple[float, float]],
    cluster_topology_gate: bool,
) -> ReferenceComparisonAnalysis:
    """Compare MD and harmonic references only where topology permits it."""

    names = _labels_once(labels)
    allowed = set(names if cluster_topology_gate else ("H2O", "D2O"))
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
                "MD_minus_DFT_cm-1": (
                    comparison.md_summary.centroid_cm1
                    - comparison.reference_stick_summary.centroid_cm1
                ),
                "MD_width_10_90_cm-1": comparison.md_summary.width_10_90_cm1,
                "DFT_Hann_width_10_90_cm-1": (
                    comparison.reference_envelope_summary.width_10_90_cm1
                ),
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
    """Build the existing H/D mode table and verify path-grid equivalence."""

    missing = set(_IR_LABELS) - set(references)
    if missing:
        raise ValueError(f"missing harmonic references: {sorted(missing)!r}")
    match_kwargs = {"degeneracy_tolerance_cm1": degeneracy_tolerance_cm1}
    monomer = match_isotopologue_modes(
        references["H2O"],
        references["D2O"],
        mass_path_steps=coarse_mass_path_steps,
        **match_kwargs,
    )
    hexamer = match_isotopologue_modes(
        references["(H2O)6"],
        references["(D2O)6"],
        mass_path_steps=coarse_mass_path_steps,
        **match_kwargs,
    )
    hexamer_fine = match_isotopologue_modes(
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

    monomer_characters = reference_water_monomer_mode_labels(references["H2O"])
    hexamer_character_data = reference_water_ring_mode_characters(
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
) -> pd.DataFrame:
    """Return frame-resolved H-bond and oxygen-skeleton observables.

    Covalent O-H ownership is assigned once from the first frame, matching the
    integrity gate.  The returned table is diagnostic data; it does not decide
    whether a comparison is scientifically valid.
    """

    start, stop = map(int, trajectory.batch_ptr[graph_index : graph_index + 2])
    numbers = np.asarray(trajectory.atomic_numbers[start:stop])
    frames = np.asarray(trajectory.positions_angstrom[:, start:stop], dtype=float)
    oxygen_local = np.flatnonzero(numbers == 8)
    hydrogen_local = np.flatnonzero(numbers == 1)
    if len(oxygen_local) < 2 or len(hydrogen_local) == 0:
        raise ValueError("topology_time_series requires a multi-water graph")
    if frames.ndim != 3 or frames.shape[0] == 0:
        raise ValueError("trajectory must contain position frames")

    reference = frames[0]
    assignment = np.argmin(
        np.linalg.norm(
            reference[hydrogen_local, None] - reference[oxygen_local][None, :],
            axis=-1,
        ),
        axis=1,
    )
    oxygen_frames = frames[:, oxygen_local]
    hydrogen_frames = frames[:, hydrogen_local]
    donor_oxygen = oxygen_frames[:, assignment]
    h_to_donor = donor_oxygen - hydrogen_frames
    h_to_acceptor = oxygen_frames[:, None] - hydrogen_frames[:, :, None]
    h_acceptor_distance = np.linalg.norm(h_to_acceptor, axis=-1)
    cosine = np.sum(h_to_donor[:, :, None] * h_to_acceptor, axis=-1)
    cosine /= np.linalg.norm(h_to_donor, axis=-1)[:, :, None]
    cosine /= np.linalg.norm(h_to_acceptor, axis=-1)
    hbond_angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    oo_distance = np.linalg.norm(
        donor_oxygen[:, :, None] - oxygen_frames[:, None], axis=-1
    )
    is_donor = (
        np.arange(len(oxygen_local))[None, None, :] == assignment[None, :, None]
    )
    hbond_mask = (
        (h_acceptor_distance <= float(h_acceptor_cutoff_angstrom))
        & (oo_distance <= float(oo_cutoff_angstrom))
        & (hbond_angle >= float(hbond_angle_cutoff_deg))
        & ~is_donor
    )
    adjacency = np.zeros(
        (frames.shape[0], len(oxygen_local), len(oxygen_local)), dtype=bool
    )
    for hydrogen, donor in enumerate(assignment):
        adjacency[:, donor] |= hbond_mask[:, hydrogen]
    ring = directed_ring_masks(adjacency)
    centered = oxygen_frames - oxygen_frames.mean(axis=1, keepdims=True)
    radius_gyration = np.sqrt(np.mean(np.sum(centered**2, axis=-1), axis=1))
    rmsd = kabsch_rmsd_frames(oxygen_frames[0], oxygen_frames)
    assigned_oh = np.linalg.norm(
        hydrogen_frames - donor_oxygen,
        axis=-1,
    )
    return pd.DataFrame(
        {
            "time_ps": np.arange(frames.shape[0]) * float(trajectory.dt_fs) / 1000.0,
            "H_bonds": hbond_mask.sum(axis=(1, 2)),
            "exact_single_ring": ring.exact_single_ring,
            "has_directed_cycle": ring.any_cycle,
            "initial_ring_present": ring.initial_cycle,
            "directed_cycle_multiplicity": ring.cycle_multiplicity,
            "oxygen_Rg_angstrom": radius_gyration,
            "oxygen_RMSD_angstrom": rmsd,
            "max_assigned_OH_angstrom": assigned_oh.max(axis=1),
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
    "IRComparisonGateAnalysis",
    "IRSpectrumAnalysis",
    "ModeMappingAnalysis",
    "ReferenceComparisonAnalysis",
    "assignments_are_subspace_equivalent",
    "dimer_interaction_energy_table",
    "grouped_mapping_rows",
    "h_to_d_mode_mapping_table",
    "ir_comparison_gate_table",
    "ir_spectrum_metrics",
    "reference_comparison_metrics",
    "topology_time_series",
]
