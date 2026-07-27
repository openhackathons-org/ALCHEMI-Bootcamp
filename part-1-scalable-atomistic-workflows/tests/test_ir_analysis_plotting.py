"""Focused CPU tests for Part 1 analysis tables and side-effect-free plots."""

from __future__ import annotations

from inspect import signature
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
    comparison_display_table,
    dimer_interaction_energy_table,
    ir_comparison_table,
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
    plot_distributed_pipeline_scaling,
    plot_harmonic_monomer_comparison,
    plot_md_dft_comparison,
    plot_monomer_ir_comparison,
    plot_pipeline_campaign,
    plot_topology_timeline,
)
from aux.reference import (  # noqa: E402
    IsotopologueModeMatch,
    ModeSubspaceMatch,
)
from aux.structures import make_ir_structures  # noqa: E402


LABELS = ("H2O", "D2O", "(H2O)6", "(D2O)6")


def test_mode_table_public_signature_contains_only_analysis_choices() -> None:
    parameters = signature(analysis.h_to_d_mode_mapping_table).parameters

    assert "mode_matcher" not in parameters
    assert "monomer_mode_labeler" not in parameters
    assert "ring_mode_characterizer" not in parameters


def test_ir_metrics_and_four_comparison_rows_have_plain_statuses() -> None:
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
    comparisons = ir_comparison_table(
        spectrum_result.metrics,
        temperatures,
        LABELS,
        pair_temperature_relative_tolerance=0.20,
        cluster_reference_allowed=False,
    )

    assert list(comparisons.table.index) == [
        "H2O_over_D2O_centroid",
        "H6_over_D6_centroid",
        "H_cluster_minus_monomer_OH_region_centroid_cm-1",
        "D_cluster_minus_monomer_OD_region_centroid_cm-1",
    ]
    display_table = comparison_display_table(comparisons.table)
    assert list(display_table.columns) == [
        "value",
        "shown",
        "temperatures_match",
        "topology_unchanged",
        "status",
    ]
    assert bool(display_table.loc["H2O_over_D2O_centroid", "shown"])
    assert not bool(display_table.loc["H6_over_D6_centroid", "shown"])
    assert np.isnan(display_table.loc["H6_over_D6_centroid", "value"])
    assert (
        display_table.loc["H6_over_D6_centroid", "status"]
        == "initial ring changed"
    )
    assert (
        display_table.loc[
            "H_cluster_minus_monomer_OH_region_centroid_cm-1", "status"
        ]
        == "mean temperatures differ by more than the allowed tolerance; "
        "initial ring changed"
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


def test_mode_table_groups_degenerate_modes_and_checks_fine_path() -> None:
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

    result = analysis._h_to_d_mode_mapping_table_with_dependencies(
        references,
        coarse_mass_path_steps=65,
        fine_mass_path_steps=129,
        degeneracy_tolerance_cm1=2.0,
        covalent_oh_cutoff_angstrom=1.25,
        h_acceptor_cutoff_angstrom=2.5,
        oo_cutoff_angstrom=3.5,
        hbond_angle_cutoff_deg=140.0,
        mode_matcher=fake_match,
        monomer_mode_labeler=lambda reference: (
            "bend",
            "symmetric_stretch",
            "antisymmetric_stretch",
        ),
        ring_mode_characterizer=fake_ring_characters,
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


def _harmonic_comparison_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "system": ["H2O"] * 3 + ["D2O"] * 3,
            "mode": ["ν2 bend", "ν1 symmetric", "ν3 antisymmetric"] * 2,
            "AIMNet+Coulomb+D3_harmonic_cm-1": [
                1610,
                3820,
                3910,
                1185,
                2760,
                2870,
            ],
            "AIMNet_point_charge_IR_km_mol": [
                42.0,
                7.0,
                126.0,
                31.0,
                11.0,
                96.0,
            ],
            "B97-3c_harmonic_cm-1": [1620, 3830, 3930, 1190, 2780, 2890],
            "B97-3c_IR_intensity_km_mol": [65.0, 16.0, 102.0, 52.0, 14.0, 78.0],
            "observed_gas_cm-1": [1595, 3657, 3756, 1178, 2671, 2788],
        }
    )


def test_harmonic_plot_requires_absolute_intensity_columns() -> None:
    table = _harmonic_comparison_table().drop(
        columns=["AIMNet_point_charge_IR_km_mol"]
    )

    with pytest.raises(ValueError, match="harmonic comparison table is missing"):
        plot_harmonic_monomer_comparison(table)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    (
        ("AIMNet+Coulomb+D3_harmonic_cm-1", np.nan, "non-finite"),
        ("B97-3c_IR_intensity_km_mol", np.inf, "non-finite"),
        ("AIMNet_point_charge_IR_km_mol", -0.01, "must be non-negative"),
        ("observed_gas_cm-1", -1.0, "wavenumbers must be positive"),
    ),
)
def test_harmonic_plot_rejects_invalid_physical_values(
    column: str,
    value: float,
    message: str,
) -> None:
    table = _harmonic_comparison_table()
    table.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        plot_harmonic_monomer_comparison(table)


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
        r"cyc-(H$_2$O)$_6$: cyclic overlay not shown"
    )
    assert "initial ring changed" in axes[1, 0].texts[0].get_text()
    assert not axes[0, 0].spines["top"].get_visible()
    assert not axes[0, 0].spines["right"].get_visible()
    comparison_path = tmp_path / "comparison.png"
    fig.savefig(comparison_path)
    assert comparison_path.stat().st_size > 0
    plt.close(fig)

    experimental = pd.DataFrame(
        {
            "isotopologue": ["H2O"] * 3 + ["D2O"] * 3,
            "mode_index": [1, 2, 3, 1, 2, 3],
            "wavenumber_cm1": [3657.1, 1594.8, 3755.9, 2671.7, 1178.4, 2787.7],
        }
    )
    fig, axes = plot_monomer_ir_comparison(
        {"H2O": comparison, "D2O": comparison},
        experimental,
        harmonic_mode_indices={"H2O": (2, 1, 3), "D2O": (2, 1, 3)},
    )
    assert axes.shape == (2,)
    assert fig._suptitle.get_text() == (
        "Water monomer IR: three sources, three explicit comparison lanes"
    )
    assert axes[0].get_title() == r"H$_2$O"
    assert [tick.get_text() for tick in axes[0].get_yticklabels()] == [
        "experiment\npositions",
        "B97-3c\nharmonic",
        "AIMNet MD\nfinite T",
    ]
    assert len(axes[0].collections) >= 3  # DFT sticks + observed lines + markers
    assert "observed gas-phase fundamentals" in {
        text.get_text() for text in axes[0].get_legend().get_texts()
    }
    assert any("independently max-normalized" in text.get_text() for text in fig.texts)
    assert len(axes[0].texts) == 6  # three DFT and three experimental mode labels
    plt.close(fig)

    harmonic_rows = _harmonic_comparison_table()
    fig, axes = plot_harmonic_monomer_comparison(harmonic_rows)
    assert axes.shape == (2,)
    assert fig._suptitle.get_text() == (
        "Water monomer harmonic frequencies and model-specific intensities"
    )
    assert axes[0].get_ylabel() == (
        "model-specific integrated intensity / km mol$^{-1}$"
    )
    assert axes[0].get_ylim() == axes[1].get_ylim()
    assert (
        "AIMNet + Coulomb + D3 frequency; "
        "AIMNet point-charge dipole intensity"
    ) in {text.get_text() for text in axes[0].get_legend().get_texts()}
    assert "B97-3c electronic dipole derivative" in {
        text.get_text() for text in axes[0].get_legend().get_texts()
    }
    assert "experiment (positions only)" in {
        text.get_text() for text in axes[0].get_legend().get_texts()
    }
    assert [text.get_text() for text in axes[1].get_legend().get_texts()] == [
        "ν2 bend",
        "ν1 symmetric",
        "ν3 antisymmetric",
    ]
    assert any(
        "absolute intensities are not compared" in text.get_text()
        for text in fig.texts
    )

    harmonic_stick_heights = sorted(
        float(segments[0][1, 1])
        for collection in axes[0].collections
        if hasattr(collection, "get_segments")
        and (segments := collection.get_segments())
    )
    assert harmonic_stick_heights == sorted(
        harmonic_rows.loc[
            harmonic_rows["system"] == "H2O",
            [
                "AIMNet_point_charge_IR_km_mol",
                "B97-3c_IR_intensity_km_mol",
            ],
        ].to_numpy(dtype=float).ravel()
    )
    plt.close(fig)

    distributed_runs = pd.DataFrame(
        {
            "mode": ["strong", "strong", "strong", "weak", "weak", "weak"],
            "nodes": [2, 2, 4, 2, 4, 4],
            "success": [True, True, True, True, True, False],
            "elapsed_s": [10.0, 11.0, 6.0, 8.0, 8.5, np.nan],
        }
    )
    common = {
        "nodes": [2, 4],
        "successful_runs": [2, 1],
        "failed_runs": [0, 1],
        "baseline_nodes": [2, 2],
    }
    distributed = SimpleNamespace(
        runs=distributed_runs,
        strong_summary=pd.DataFrame(
            common
            | {
                "median_elapsed_s": [10.5, 6.0],
                "speedup_vs_baseline": [1.0, 1.75],
            }
        ),
        weak_summary=pd.DataFrame(
            common
            | {
                "median_elapsed_s": [8.0, 8.5],
                "weak_elapsed_efficiency_vs_baseline": [1.0, 8.0 / 8.5],
            }
        ),
    )
    fig, axes = plot_distributed_pipeline_scaling(distributed)
    assert axes.shape == (2,)
    assert axes[0].get_title() == "Fixed total work"
    assert axes[1].get_title() == "Fixed work per two-GPU pipeline"
    assert "1 pass / 1 fail" in {text.get_text() for text in axes[1].texts}
    assert fig._suptitle.get_text() == (
        "Saved Toolkit DistributedPipeline timing check"
    )
    plt.close(fig)

    failed_allocation = SimpleNamespace(
        runs=pd.DataFrame(
            {
                "mode": ["strong", "strong", "weak", "weak"],
                "nodes": [2, 4, 2, 4],
                "success": [False, True, True, False],
                "elapsed_s": [np.nan, 6.0, 8.0, np.nan],
            }
        ),
        strong_summary=pd.DataFrame(
            {
                "nodes": [2, 4],
                "successful_runs": [0, 1],
                "failed_runs": [1, 0],
                "baseline_nodes": [4, 4],
                "median_elapsed_s": [np.nan, 6.0],
                "speedup_vs_baseline": [np.nan, 1.0],
            }
        ),
        weak_summary=pd.DataFrame(
            {
                "nodes": [2, 4],
                "successful_runs": [1, 0],
                "failed_runs": [0, 1],
                "baseline_nodes": [2, 2],
                "median_elapsed_s": [8.0, np.nan],
                "weak_elapsed_efficiency_vs_baseline": [1.0, np.nan],
            }
        ),
    )
    fig, axes = plot_distributed_pipeline_scaling(failed_allocation)
    assert "0 pass / 1 fail" in {
        text.get_text() for axis in axes for text in axis.texts
    }
    assert "all repeats failed" in {
        text.get_text() for text in fig.legends[0].get_texts()
    }
    plt.close(fig)

    campaign_runs = pd.DataFrame(
        {
            "route": [
                "fused_1gpu",
                "fused_1gpu",
                "pipeline_2gpu",
                "pipeline_2gpu",
                "pipeline_4gpu",
                "pipeline_4gpu",
            ],
            "success": [True, True, True, True, True, False],
            "elapsed_s": [120.0, 126.0, 72.0, 75.0, 39.0, np.nan],
            "gpu_seconds_per_structure": [
                0.24,
                0.25,
                0.29,
                0.30,
                0.31,
                np.nan,
            ],
        }
    )
    campaign_summary = pd.DataFrame(
        {
            "route": ["fused_1gpu", "pipeline_2gpu", "pipeline_4gpu"],
            "successful_runs": [2, 2, 1],
            "failed_runs": [0, 0, 1],
            "median_elapsed_s": [123.0, 73.5, 39.0],
            "elapsed_q25_s": [121.5, 72.75, 39.0],
            "elapsed_q75_s": [124.5, 74.25, 39.0],
            "median_gpu_seconds_per_structure": [0.245, 0.295, 0.31],
            "speedup_vs_1gpu": [1.0, 123.0 / 73.5, 123.0 / 39.0],
        }
    )
    campaign = SimpleNamespace(
        runs=campaign_runs,
        summary=campaign_summary,
        manifest={"campaign": {"systems_total": 8_192}},
    )
    fig, axes = plot_pipeline_campaign(campaign)
    assert axes.shape == (2,)
    assert axes[0].get_title() == "Time to finish the campaign"
    assert axes[1].get_title() == "Timed GPU work per structure"
    assert axes[1].get_ylabel() == "timed H100-seconds / structure"
    assert (
        fig._suptitle.get_text()
        == "8,192 water hexamers: same workload, three node layouts"
    )
    assert [tick.get_text() for tick in axes[0].get_xticklabels()] == [
        "1 node · 1 GPU\nfused stages",
        "2 nodes · 2 GPUs\none pipeline",
        "4 nodes · 4 GPUs\ntwo pipelines",
    ]
    assert "1 pass / 1 fail" in {text.get_text() for text in axes[1].texts}
    assert "3.15×" in {text.get_text() for text in axes[0].texts}
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
