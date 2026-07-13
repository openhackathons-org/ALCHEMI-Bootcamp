"""Focused CPU tests for Part 1 analysis tables and side-effect-free plots."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux import analysis  # noqa: E402
from aux.analysis import (  # noqa: E402
    dimer_interaction_energy_table,
    h_to_d_mode_mapping_table,
    ir_comparison_gate_table,
    ir_spectrum_metrics,
    topology_time_series,
)
from aux.plotting import (  # noqa: E402
    COMPONENT_COLORS,
    COMPONENT_STYLES,
    FIGURE_SIZE,
    SYSTEM_COLORS,
    SYSTEM_LINESTYLES,
    plot_dimer_interaction_energies,
    plot_md_dft_comparison,
    plot_topology_timeline,
)
from aux.reference import (  # noqa: E402
    IsotopologueModeMatch,
    ModeSubspaceMatch,
)
from aux.structures import make_ir_structures  # noqa: E402


LABELS = ("H2O", "D2O", "(H2O)6", "(D2O)6")


def test_ir_metrics_and_four_gate_rows_preserve_output_contract() -> None:
    sample = np.arange(65, dtype=float)
    dipoles = np.zeros((sample.size, 4, 3), dtype=float)
    for graph, cycles_per_sample in enumerate((0.10, 0.12, 0.15, 0.18)):
        dipoles[:, graph, 0] = np.sin(2.0 * np.pi * cycles_per_sample * sample)
        dipoles[:, graph, 1] = np.cos(2.0 * np.pi * cycles_per_sample * sample)

    spectrum_result = ir_spectrum_metrics(
        dipoles,
        LABELS,
        dt_fs=1.0,
        segment_time_fs=16.0,
        overlap=0.5,
        region_windows_cm1={"H": (1000.0, 12000.0), "D": (1000.0, 12000.0)},
    )

    assert list(spectrum_result.metrics.columns) == [
        "OH_OD_region_centroid_cm-1",
        "Welch_segment_std_cm-1",
        "Welch_segments",
    ]
    assert list(spectrum_result.metrics.index) == list(LABELS)
    assert set(spectrum_result.spectra) == set(LABELS)
    assert (spectrum_result.metrics["Welch_segments"] == 7).all()

    temperatures = np.tile(np.array((300.0, 315.0, 600.0, 600.0)), (20, 1))
    gated = ir_comparison_gate_table(
        spectrum_result.metrics,
        temperatures,
        LABELS,
        pair_temperature_relative_tolerance=0.20,
        cluster_topology_gate=False,
    )

    assert list(gated.table.index) == [
        "H2O_over_D2O_centroid",
        "H6_over_D6_centroid",
        "H_cluster_minus_monomer_OH_region_centroid_cm-1",
        "D_cluster_minus_monomer_OD_region_centroid_cm-1",
    ]
    assert list(gated.table.columns) == [
        "value",
        "reported",
        "thermal_gate_passed",
        "topology_gate_passed",
        "status",
    ]
    assert bool(gated.table.loc["H2O_over_D2O_centroid", "reported"])
    assert not bool(gated.table.loc["H6_over_D6_centroid", "reported"])
    assert np.isnan(gated.table.loc["H6_over_D6_centroid", "value"])
    assert (
        gated.table.loc["H6_over_D6_centroid", "status"]
        == "initial-ring persistence gate failed"
    )
    assert (
        gated.table.loc[
            "H_cluster_minus_monomer_OH_region_centroid_cm-1", "status"
        ]
        == "thermal-state gate failed; initial-ring persistence gate failed"
    )


def _mode_match(
    assignment: list[int],
    *,
    ambiguous: tuple[ModeSubspaceMatch, ...] = (),
) -> IsotopologueModeMatch:
    count = len(assignment)
    return IsotopologueModeMatch(
        source_to_target=np.asarray(assignment),
        endpoint_squared_overlaps=np.full(count, 0.99),
        minimum_path_squared_overlaps=np.full(count, 0.98),
        ambiguous_subspaces=ambiguous,
        mass_path_fractions=np.linspace(0.0, 1.0, 5),
    )


def test_mode_table_groups_degenerate_modes_and_checks_fine_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = ModeSubspaceMatch(
        source_indices=(1, 2),
        target_indices=(1, 2),
        minimum_principal_overlap=0.94,
    )
    monomer_map = _mode_match([0, 1, 2])
    hexamer_map = _mode_match([0, 1, 2, 3], ambiguous=(block,))
    hexamer_fine = _mode_match([0, 2, 1, 3], ambiguous=(block,))

    references = {
        "H2O": SimpleNamespace(
            label="H2O",
            n_modes=3,
            frequencies_cm1=np.array((1600.0, 3650.0, 3750.0)),
            ir_intensities_km_mol=np.array((10.0, 20.0, 30.0)),
        ),
        "D2O": SimpleNamespace(
            label="D2O",
            n_modes=3,
            frequencies_cm1=np.array((1200.0, 2700.0, 2800.0)),
            ir_intensities_km_mol=np.array((8.0, 18.0, 28.0)),
        ),
        "(H2O)6": SimpleNamespace(
            label="(H2O)6",
            n_modes=4,
            frequencies_cm1=np.array((1500.0, 3300.0, 3301.0, 200.0)),
            ir_intensities_km_mol=np.array((10.0, 20.0, 30.0, 5.0)),
        ),
        "(D2O)6": SimpleNamespace(
            label="(D2O)6",
            n_modes=4,
            frequencies_cm1=np.array((1100.0, 2450.0, 2451.0, 180.0)),
            ir_intensities_km_mol=np.array((8.0, 18.0, 28.0, 4.0)),
        ),
    }
    seen_ring_kwargs: dict[str, object] = {}

    def fake_match(source, target, *, mass_path_steps, **kwargs):
        assert kwargs["degeneracy_tolerance_cm1"] == 2.0
        if source.label == "H2O":
            return monomer_map
        return hexamer_map if mass_path_steps == 65 else hexamer_fine

    def fake_ring_characters(reference, **kwargs):
        seen_ring_kwargs.update(kwargs)
        return SimpleNamespace(
            categories=("bend", "hbonded_oh", "free_oh", "intermolecular"),
            dominant_labels=("bend", "hbonded_oh", "free_oh", "intermolecular"),
            fractions=np.eye(4),
        )

    monkeypatch.setattr(analysis, "match_isotopologue_modes", fake_match)
    monkeypatch.setattr(
        analysis,
        "reference_water_monomer_mode_labels",
        lambda reference: ("bend", "symmetric_stretch", "antisymmetric_stretch"),
    )
    monkeypatch.setattr(
        analysis,
        "reference_water_ring_mode_characters",
        fake_ring_characters,
    )

    result = h_to_d_mode_mapping_table(
        references,
        coarse_mass_path_steps=65,
        fine_mass_path_steps=129,
        degeneracy_tolerance_cm1=2.0,
        covalent_oh_cutoff_angstrom=1.25,
        h_acceptor_cutoff_angstrom=2.5,
        oo_cutoff_angstrom=3.5,
        hbond_angle_cutoff_deg=140.0,
    )

    assert len(result.table) == 5
    grouped = result.table[
        (result.table["system"] == "cyclic hexamer")
        & (result.table["mapping_unit"] == "subspace")
    ].iloc[0]
    assert grouped["H_mode_1based"] == "2,3"
    assert grouped["D_mode_1based"] == "2,3"
    assert grouped["mapping_overlap"] == pytest.approx(0.94)
    assert seen_ring_kwargs == {
        "covalent_oh_cutoff_angstrom": 1.25,
        "h_acceptor_cutoff_angstrom": 2.5,
        "oo_cutoff_angstrom": 3.5,
        "hbond_angle_cutoff_deg": 140.0,
        "require_single_ring": True,
    }


def test_topology_timeline_and_dimer_energy_table() -> None:
    hexamer = make_ir_structures()[0][2]
    trajectory = SimpleNamespace(
        positions_angstrom=np.stack((hexamer.positions, hexamer.positions)),
        atomic_numbers=hexamer.numbers,
        batch_ptr=np.array((0, len(hexamer))),
        dt_fs=0.5,
    )

    timeline = topology_time_series(
        trajectory,
        0,
        h_acceptor_cutoff_angstrom=2.5,
        oo_cutoff_angstrom=3.5,
        hbond_angle_cutoff_deg=140.0,
    )

    assert list(timeline["H_bonds"]) == [6, 6]
    assert timeline["initial_ring_present"].all()
    assert timeline["oxygen_RMSD_angstrom"].max() == pytest.approx(0.0, abs=1e-12)

    energy = dimer_interaction_energy_table(
        [2.7, 3.0],
        {
            "full": ([-20.2, -20.1], -10.0, -10.0),
            "D3": ([-0.12, -0.08], 0.0, 0.0),
        },
    )
    np.testing.assert_allclose(energy["full_interaction_eV"], [-0.2, -0.1])
    np.testing.assert_allclose(
        energy["D3_interaction_kJ_mol"],
        np.array([-0.12, -0.08]) * analysis.EV_PER_MOLECULE_TO_KJ_MOL,
    )


def test_plot_builders_render_without_saving_or_showing(tmp_path: Path) -> None:
    wavenumber = np.linspace(0.0, 5000.0, 101)
    intensity = np.sin(np.linspace(0.0, np.pi, 101)) ** 2
    spectra = {label: (wavenumber, intensity) for label in LABELS}
    comparison = SimpleNamespace(
        wavenumber_cm1=wavenumber,
        md_intensity_normalized=intensity,
        reference_envelope_normalized=np.roll(intensity, 3),
        stick_wavenumber_cm1=np.array((1600.0, 3650.0, 3750.0)),
        stick_intensity_normalized=np.array((0.2, 0.8, 1.0)),
    )
    fig, axes = plot_md_dft_comparison(
        LABELS,
        spectra,
        {"H2O": comparison, "D2O": comparison},
        wavenumber_limits_cm1=(500.0, 4200.0),
    )
    assert axes.shape == (2, 2)
    assert fig._suptitle.get_text() == (
        "AIMNet MD with harmonic B97-3c shown for inspection"
    )
    assert tuple(fig.get_size_inches()) == FIGURE_SIZE
    assert axes[0, 0].get_title() == r"H$_2$O"
    assert axes[1, 0].get_title() == (
        r"cyc-(H$_2$O)$_6$ — cyclic overlay withheld"
    )
    assert "ring-persistence gate failed" in axes[1, 0].texts[0].get_text()
    assert not axes[0, 0].spines["top"].get_visible()
    assert not axes[0, 0].spines["right"].get_visible()
    comparison_path = tmp_path / "comparison.png"
    fig.savefig(comparison_path)
    assert comparison_path.stat().st_size > 0
    plt.close(fig)

    timeline = pd.DataFrame(
        {
            "time_ps": [0.0, 0.5, 1.0],
            "H_bonds": [6, 5, 6],
            "initial_ring_present": [True, False, True],
        }
    )
    fig, axes = plot_topology_timeline(
        {"(H2O)6": timeline, "(D2O)6": timeline}
    )
    assert axes.shape == (2,)
    assert tuple(fig.get_size_inches()) == FIGURE_SIZE
    assert axes[1].get_yticklabels()[0].get_text() == "absent"
    assert axes[0].get_legend().get_texts()[0].get_text() == r"cyc-(H$_2$O)$_6$"
    assert axes[0].lines[0].get_color() == SYSTEM_COLORS["(H2O)6"]
    assert axes[0].lines[0].get_linestyle() == SYSTEM_LINESTYLES["(H2O)6"]
    assert axes[0].lines[1].get_linestyle() == SYSTEM_LINESTYLES["(D2O)6"]
    plt.close(fig)

    energy = pd.DataFrame(
        {
            "distance_angstrom": [2.7, 3.0],
            "residual_interaction_kJ_mol": [-10.0, -5.0],
            "residual_plus_D3_interaction_kJ_mol": [-12.0, -7.0],
            "residual_plus_Coulomb_interaction_kJ_mol": [-18.0, -10.0],
            "full_interaction_kJ_mol": [-20.0, -11.0],
            "B97_3c_interaction_kJ_mol": [-21.0, -12.0],
        }
    )
    fig, axis = plot_dimer_interaction_energies(energy)
    assert tuple(fig.get_size_inches()) == FIGURE_SIZE
    assert axis.get_xlabel() == "O–O distance / Å"
    assert axis.get_title() == (
        "Water-dimer interaction ablation against full B97-3c"
    )
    assert axis.lines[3].get_color() == COMPONENT_COLORS["full_interaction_kJ_mol"]
    component_lines = axis.lines[:-1]
    style_tokens = {
        (line.get_linestyle(), line.get_marker()) for line in component_lines
    }
    assert len(style_tokens) == len(COMPONENT_STYLES)
    assert len(axis.lines) == 6  # five interaction curves plus zero-energy guide
    plt.close(fig)
