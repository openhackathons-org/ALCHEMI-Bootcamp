# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import HostMemory

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

from helpers import viewers


@dataclass(frozen=True)
class _Spec:
    structure: Atoms
    display_name: str
    domain: str
    purpose: str

    def build(self) -> Atoms:
        return self.structure.copy()


class _BatchFrame:
    def __init__(self, second_distance: float) -> None:
        self.batch_ptr = np.array([0, 2, 5])
        self.atomic_numbers = np.array([1, 1, 7, 1, 1])
        self.positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0 + second_distance, 0.0, 0.0],
                [10.0, 1.0, 0.0],
            ]
        )
        self.cell = np.zeros((2, 3, 3))
        self.pbc = np.zeros((2, 3), dtype=bool)


class _SurfaceFrameWithoutCell:
    def __init__(self) -> None:
        self.num_graphs = 1
        self.batch_ptr = np.array([0, 3])
        self.atomic_numbers = np.array([78, 6, 8])
        self.positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [1.0, 1.0, 3.1]])
        self.forces = np.zeros((3, 3))

    def get_data(self, graph_index: int) -> _SurfaceFrameWithoutCell:
        if graph_index != 0:
            raise IndexError(graph_index)
        return self


def _urea_like() -> Atoms:
    return Atoms(
        "CON2H4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [-1.2, 0.0, 0.0],
            [-1.8, 0.8, 0.0],
            [-1.8, -0.8, 0.0],
            [1.5, 0.8, 0.0],
            [1.5, -0.8, 0.0],
            [0.0, 1.1, 0.0],
        ],
    )


def _phenol_like() -> Atoms:
    angles = np.arange(6) * np.pi / 3
    ring = np.column_stack([1.4 * np.cos(angles), 1.4 * np.sin(angles), np.zeros(6)])
    oxygen = np.array([[2.76, 0.0, 0.0]])
    hydroxyl_hydrogen = np.array([[3.72, 0.0, 0.0]])
    ring_hydrogens = np.column_stack(
        [2.45 * np.cos(angles[1:]), 2.45 * np.sin(angles[1:]), np.zeros(5)]
    )
    return Atoms(
        symbols=["C"] * 6 + ["O"] + ["H"] * 6,
        positions=np.vstack([ring, oxygen, hydroxyl_hydrogen, ring_hydrogens]),
    )


def test_structure_explorer_uses_catalog_metadata_and_runtime_fallback() -> None:
    catalog = {
        "urea": _Spec(
            _urea_like(),
            "Urea",
            "Agricultural chemistry",
            "Nitrogen fertilizer",
        ),
        "caffeine": _Spec(
            molecule("C6H6"),
            "Caffeine alternate",
            "Biochemistry and pharma",
            "Familiar bioactive molecule",
        ),
    }

    selection = viewers.show_structure_explorer(catalog, selected="urea")
    bundle = selection._repr_mimebundle_()
    fallback = bundle["text/html"]

    assert selection.selected_keys == ("urea",)
    assert selection.selected_specs() == [catalog["urea"]]
    assert "application/vnd.jupyter.widget-view+json" in bundle
    assert "data-runtime-note" in fallback
    assert 'role="note"' in fallback
    assert "Urea" not in fallback
    assert "Caffeine alternate" not in fallback
    header = selection.widget.children[1]
    grid = selection.widget.children[2]
    first_card = grid.children[0]
    first_viewer = first_card.children[0].value
    first_facts = first_card.children[1].value
    first_control = first_card.children[2]
    assert grid.layout.grid_template_columns == "repeat(4, minmax(0, 1fr))"
    assert "Choose systems for the Batch" in header.children[0].value
    assert first_control.description == "✓ In batch"
    assert "3dmolviewer_" in first_viewer
    assert "createViewer" in first_viewer
    assert "window.$3Dmolpromise=Promise.resolve" in first_viewer
    assert "https://3dmol" not in first_viewer
    assert "Nitrogen fertilizer · Agricultural chemistry" in first_facts
    assert first_viewer.index(
        "window.$3Dmolpromise=Promise.resolve"
    ) < first_viewer.index("loadScriptAsync")


def test_molecule_viewer_uses_shared_runtime_notice() -> None:
    html = viewers.show_molecule(molecule("NH3"), height=240).data

    assert "data-runtime-note" in html
    assert 'role="note"' in html
    assert "window.devicePixelRatio||1" in html
    assert "fontSize:Math.round(fontSize*labelPixelRatio)" in html
    assert "label.sprite.scale.set(1/labelPixelRatio,1/labelPixelRatio,1)" in html
    assert "const highResolutionLabel = function" in html
    assert html.count("highResolutionLabel(") == 3


def test_structure_selection_updates_the_python_batch_choice() -> None:
    catalog = {
        "urea": _Spec(
            _urea_like(),
            "Urea",
            "Agricultural chemistry",
            "Nitrogen fertilizer",
        ),
        "water": _Spec(
            molecule("H2O"),
            "Water",
            "General chemistry",
            "Small polar molecule",
        ),
    }
    selection = viewers.show_structure_explorer(catalog)
    controls = {
        key: card.children[2]
        for key, card in zip(
            catalog, selection.widget.children[2].children, strict=True
        )
    }

    controls["urea"].value = True
    controls["water"].value = True
    controls["urea"].value = False
    assert selection.selected_keys == ("water",)
    assert selection.selected_specs() == [catalog["water"]]
    assert controls["water"].description == "✓ In batch"
    assert controls["urea"].description == "+ Add to batch"

    controls["water"].value = False
    with pytest.raises(ValueError, match="Select at least one molecular card"):
        selection.selected_specs()


def test_structure_selection_enforces_the_five_system_limit() -> None:
    catalog = {
        f"system-{index}": _Spec(
            molecule("H2O"),
            f"Water {index}",
            "General chemistry",
            "Small polar molecule",
        )
        for index in range(6)
    }
    selection = viewers.show_structure_explorer(catalog)
    controls = {
        key: card.children[2]
        for key, card in zip(
            catalog, selection.widget.children[2].children, strict=True
        )
    }

    for key in tuple(catalog)[:5]:
        controls[key].value = True
    controls["system-5"].value = True

    assert selection.selected_keys == tuple(catalog)[:5]
    assert controls["system-5"].value is False

    with pytest.raises(ValueError):
        viewers.show_structure_explorer(catalog, selected=tuple(catalog))


def test_relaxation_player_has_native_grid_force_colors_and_trace() -> None:
    first = molecule("H2O")
    second = molecule("NH3")
    trajectories = {
        "Water": [first, first.copy()],
        "Ammonia": [second, second.copy()],
    }
    html = viewers.show_relaxation_player(
        trajectories,
        force_history={"Water": [0.8, 0.3], "Ammonia": [0.7, 0.2]},
        device="NVIDIA H100",
        elapsed_s=1.25,
        convergence={"Water": "bounded preview", "Ammonia": "converged"},
        batch_multiplier=16,
    ).data

    assert "Synchronized relaxation player" in html
    assert "Shared step" in html
    assert "▶ Play" in html
    assert "MAX FORCE" in html
    assert "eV/Å" in html
    assert "grid-template-columns:repeat(5,minmax(0,1fr))" in html
    assert html.count("data-empty-slot") == 3
    assert "fmax 0.800 → 0.300 eV/Å" in html
    assert "Static initial/final preview" in html
    assert "data-runtime-note" in html
    assert 'role="note"' in html
    assert "position:relative;isolation:isolate" in html
    assert "Synchronized py3Dmol relaxation player" in html
    assert "$3Dmol.createViewer(node" in html
    assert "addModelsAsFrames" in html
    assert "viewer.setFrame(index)" in html
    assert "window.requestAnimationFrame(tick)" in html
    assert "Maximum force (eV/Å)" not in html
    assert "viewBox=&quot;0 0 300 42&quot;" in html
    assert "Color atoms by force" in html
    assert "item.atom_forces[0]" in html
    assert "item.forces[index]" in html
    assert "grid-template-rows:repeat(1,286px)" in html
    assert "card-title" in html
    assert "metric" in html
    assert "class=&quot;of&quot;" in html
    assert "data-progress-fill" in html
    assert 'data-batch-multiplier="16"' in html
    assert 'data-total-graphs="32"' in html
    assert 'data-total-atoms="112"' in html
    assert "Batch · 2 molecules × 16 replicas = 32 graphs · 112 atoms" in html
    assert "data-atom-count=&quot;3&quot;" in html
    assert "data-atom-count=&quot;4&quot;" in html
    assert "transition:transform 110ms linear" in html
    assert "transition:cx 110ms linear,cy 110ms linear" in html
    assert "cursor.style.setProperty" in html
    assert "dot.style.setProperty" in html
    assert "max=&quot;1&quot;" in html
    assert "forceColor" in html
    assert "pointerdown" in html
    assert "Run status" not in html
    update_script = html.split("const update=function(index){", 1)[1].split(
        "const stop=function()", 1
    )[0]
    assert "drawStructure" not in update_script
    assert "removeAllModels" not in update_script
    assert "rotate(" not in update_script
    assert "$3Dmolpromise.then(function()" in html


def test_relaxation_player_keeps_five_equal_slots_for_one_selection() -> None:
    water = molecule("H2O")

    html = viewers.show_relaxation_player(
        {"Water": [water, water.copy()]},
        force_history={"Water": [0.8, 0.0]},
        batch_multiplier=16,
    ).data

    assert "grid-template-columns:repeat(5,minmax(0,1fr))" in html
    assert html.count("data-empty-slot") == 4
    assert "height:372px" in html
    assert "html,body{height:100%;overflow:hidden" in html
    assert "Batch · 1 molecule × 16 replicas = 16 graphs · 48 atoms" in html
    assert 'data-total-atoms="48"' in html
    assert "data-atom-count=&quot;3&quot;" in html
    assert "3 atoms" in html
    assert "x1=&quot;4.00&quot;" in html
    assert "y2=&quot;37&quot;" in html


def _nci_scan(equilibrium_distance: float) -> list[Atoms]:
    frames = []
    for relative_scale in (0.8, 1.0, 1.2):
        frames.append(
            Atoms(
                "HOHe",
                positions=[
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [equilibrium_distance * relative_scale, 0.0, 0.0],
                ],
            )
        )
    return frames


def test_nci_display_interpolation_preserves_calculated_frames() -> None:
    source_frames = _nci_scan(5.0)
    display_frames, display_scales = viewers._interpolate_nci_display_frames(
        {"complex": source_frames},
        np.asarray((0.8, 1.0, 1.2)),
    )

    assert len(display_scales) == 41
    assert np.diff(display_scales).max() <= 0.01 + 1.0e-12
    for source_scale, source_frame in zip((0.8, 1.0, 1.2), source_frames, strict=True):
        display_index = int(np.flatnonzero(np.isclose(display_scales, source_scale))[0])
        np.testing.assert_allclose(
            display_frames["complex"][display_index].positions,
            source_frame.positions,
        )


def test_nci_separation_player_aligns_scans_contacts_and_shared_control() -> None:
    labels = (
        "phenol - N-methylacetamide",
        "propyne - methyl azide",
        "ammonia - benzoate",
    )
    grouped_frames = {
        label: _nci_scan(equilibrium_distance)
        for label, equilibrium_distance in zip(labels, (4.0, 5.0, 6.0), strict=True)
    }

    html = viewers.show_nci_separation_player(
        grouped_frames,
        fragment_sizes={label: 2 for label in labels},
        scale_values=(0.8, 1.0, 1.2),
    ).data

    for label in labels:
        assert label in html
    assert (
        html.count("data-frame aria-label=&quot;Shared NCI Atlas scan coordinate&quot;")
        == 1
    )
    assert html.count("data-play aria-label=&quot;Play NCI separation scan&quot;") == 1
    assert "max=&quot;40&quot;" in html
    assert "Synchronized py3Dmol NCI separation player" in html
    assert "viewer.addModelsAsFrames(item.trajectory_xyz,&#x27;xyz&#x27;)" in html
    assert "viewer.setFrame(index)" in html
    assert "dashed:true" in html
    assert "dashLength:.12" in html
    assert 'stroke-dasharray="3 4"' in html
    assert "line.distance_a.toFixed(3)" in html
    assert "line.delta_distance_a.toFixed(3)" in html
    assert "R/R&lt;sub&gt;e&lt;/sub&gt;" not in html
    assert "data-runtime-note" in html
    assert 'role="note"' in html

    first_equilibrium = grouped_frames[labels[0]][1]
    expected_distance = np.linalg.norm(
        first_equilibrium.positions[2] - first_equilibrium.positions[1]
    )
    assert f"{expected_distance:.3f} Å" in html
    assert f"&quot;distance_a&quot;:{expected_distance}" in html

    for state in ("initial", "equilibrium", "far"):
        assert f">{state}</text>" in html


@pytest.mark.parametrize(
    ("frames", "fragment_sizes", "scales", "message"),
    [
        (
            {
                "first": _nci_scan(4.0),
                "second": _nci_scan(5.0)[:2],
                "third": _nci_scan(6.0),
            },
            (2, 2, 2),
            (0.8, 1.0, 1.2),
            "same frame count",
        ),
        (
            {
                "first": _nci_scan(4.0),
                "second": _nci_scan(5.0),
                "third": _nci_scan(6.0),
            },
            (2, 2, 2),
            (0.8, 1.0),
            "one value per aligned frame",
        ),
        (
            {
                "first": _nci_scan(4.0),
                "second": _nci_scan(5.0),
                "third": _nci_scan(6.0),
            },
            (2, 2),
            (0.8, 1.0, 1.2),
            "one value per NCI system",
        ),
    ],
)
def test_nci_separation_player_rejects_unaligned_inputs(
    frames: dict[str, list[Atoms]],
    fragment_sizes: tuple[int, ...],
    scales: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        viewers.show_nci_separation_player(frames, fragment_sizes, scales)


def _bias_trajectory(reference: float, offsets: tuple[float, ...]) -> list[Atoms]:
    return [
        Atoms("HHe", positions=[[0.0, 0.0, 0.0], [reference + offset, 0.0, 0.0]])
        for offset in offsets
    ]


def test_bias_comparison_player_coordinates_six_cards_and_bias_force() -> None:
    labels = ("first complex", "second complex", "third complex")
    references = (4.0, 5.0, 6.0)
    unrestrained = {
        label: _bias_trajectory(reference, (0.0, 0.4, 0.8))
        for label, reference in zip(labels, references, strict=True)
    }
    restrained = {
        label: _bias_trajectory(reference, (0.0, 0.1, 0.2))
        for label, reference in zip(labels, references, strict=True)
    }

    html = viewers.show_bias_comparison_player(
        unrestrained,
        restrained,
        labels=labels,
        fragment_sizes=(1, 1, 1),
        reference_distances=references,
        force_constant_ev_per_a2=10.0,
    ).data

    assert html.count("&lt;section class=&quot;trajectory-card&quot;") == 6
    assert html.count("data-play aria-label=&quot;Play matched trajectories&quot;") == 1
    assert html.count("data-frame aria-label=&quot;Shared trajectory frame&quot;") == 1
    assert "viewer.addModelsAsFrames(item.trajectory_xyz,&#x27;xyz&#x27;)" in html
    assert "viewer.setFrame(index)" in html
    assert "dashed:true" in html
    assert "maximum_bias_force" in html
    assert "MAX BIAS FORCE" in html
    assert "SEPARATION" in html
    assert "2.000 eV/Å" in html
    assert "data-runtime-note" in html
    assert 'role="note"' in html
    assert "forceColor" not in html


def test_bias_comparison_player_rejects_misaligned_conditions() -> None:
    labels = ("first", "second", "third")
    frames = {
        label: _bias_trajectory(reference, (0.0, 0.1, 0.2))
        for label, reference in zip(labels, (4.0, 5.0, 6.0), strict=True)
    }
    shorter = {**frames, "third": frames["third"][:2]}

    with pytest.raises(ValueError, match="same frame count"):
        viewers.show_bias_comparison_player(
            frames,
            shorter,
            labels=labels,
            fragment_sizes=(1, 1, 1),
            reference_distances=(4.0, 5.0, 6.0),
            force_constant_ev_per_a2=10.0,
        )


def test_surface_viewers_disable_bonds_and_use_larger_spheres() -> None:
    surface = Atoms(
        "PtCO",
        positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [1.0, 1.0, 3.1]],
        cell=[4.0, 4.0, 8.0],
        pbc=True,
    )
    selection = viewers.show_structure_explorer(
        {
            "surface": _Spec(
                surface,
                "CO on Pt(111)",
                "Heterogeneous catalysis",
                "Adsorbate on a metal surface",
            )
        }
    )
    card = selection.widget.children[2].children[0].children[0].value
    player = viewers.show_relaxation_player(
        {"CO on Pt(111)": [surface, surface.copy()]},
        force_history={"CO on Pt(111)": [1.0, 0.5]},
    ).data

    assert "&quot;assignBonds&quot;: false" in card
    assert "&quot;scale&quot;: 0.38" in card
    assert "&quot;stick&quot;" not in card
    assert "assignBonds:!item.sphere_only" in player
    assert "item.sphere_only ? .38 : .23" in player


def test_relaxation_player_keeps_surface_style_when_snapshots_lose_cell() -> None:
    reference = Atoms(
        "PtCO",
        positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [1.0, 1.0, 3.1]],
        cell=[4.0, 4.0, 8.0],
        pbc=True,
    )
    frames = [_SurfaceFrameWithoutCell(), _SurfaceFrameWithoutCell()]

    html = viewers.show_relaxation_player(
        frames,
        labels=["CO on Pt(111)"],
        force_history={"CO on Pt(111)": [1.0, 0.5]},
        reference_structures={"CO on Pt(111)": reference},
    ).data

    assert "sphere_only&quot;:true" in html
    assert "cell_edges&quot;:[{" in html
    assert "viewer.setStyle({serial:atom}" in html


def test_snapshot_batches_rebuilds_each_host_memory_write() -> None:
    water = molecule("H2O")
    ammonia = molecule("NH3")
    later_water = water.copy()
    later_ammonia = ammonia.copy()
    later_water.positions += [0.1, 0.0, 0.0]
    later_ammonia.positions += [0.2, 0.0, 0.0]
    sink = HostMemory(capacity=4)
    sink.write(
        Batch.from_data_list(
            [AtomicData.from_atoms(water), AtomicData.from_atoms(ammonia)]
        )
    )
    sink.write(
        Batch.from_data_list(
            [AtomicData.from_atoms(later_water), AtomicData.from_atoms(later_ammonia)]
        )
    )

    frames = viewers.snapshot_batches(sink, num_graphs=2)

    assert len(frames) == 2
    assert [frame.num_graphs for frame in frames] == [2, 2]
    assert [frame.num_nodes_per_graph.tolist() for frame in frames] == [
        [3, 4],
        [3, 4],
    ]
    np.testing.assert_allclose(
        frames[1].get_data(1).positions.numpy(), later_ammonia.positions
    )


def test_structure_explorer_requires_at_least_one_structure() -> None:
    with pytest.raises(ValueError, match="at least one structure"):
        viewers.show_structure_explorer({})


def test_structure_explorer_requires_buildable_catalog_entries() -> None:
    with pytest.raises(TypeError, match=r"catalog entry 'water' must provide a build"):
        viewers.show_structure_explorer({"water": molecule("H2O")})
