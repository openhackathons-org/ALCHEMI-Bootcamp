# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import HostMemory

_NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NOTEBOOK_DIR))

from helpers import core as helpers_core
from helpers import viewers


@dataclass(frozen=True)
class _Spec:
    structure: Atoms
    display_name: str
    domain: str
    purpose: str
    source: str
    model_scope: str
    is_default: bool = True

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
        self.positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [1.0, 1.0, 3.1]]
        )
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
    ring = np.column_stack(
        [1.4 * np.cos(angles), 1.4 * np.sin(angles), np.zeros(6)]
    )
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
            "RDKit SMILES",
            "Exploratory model application",
        ),
        "caffeine": _Spec(
            molecule("C6H6"),
            "Caffeine alternate",
            "Biochemistry and pharma",
            "Familiar bioactive molecule",
            "RDKit SMILES",
            "Exploratory model application",
            False,
        ),
    }

    selection = viewers.show_structure_explorer(catalog)
    bundle = selection._repr_mimebundle_()
    fallback = bundle["text/html"]
    preview = viewers._single_structure_markup(
        viewers._metadata_from_spec("urea", catalog["urea"], _urea_like())
    )

    assert selection.selected_keys == ()
    selection._checkboxes["urea"].value = True
    assert selection.selected_keys == ("urea",)
    assert selection.selected_specs() == [catalog["urea"]]
    assert selection.selected_atoms()[0].get_chemical_symbols() == _urea_like().get_chemical_symbols()
    assert "application/vnd.jupyter.widget-view+json" in bundle
    assert "Interactive molecular explorer" in fallback
    assert "needs a running notebook" in fallback
    assert "Jupyter or VS Code" in fallback
    assert "choose systems and explore them in 3D" in fallback
    assert "Urea" not in fallback
    assert "Caffeine alternate" not in fallback
    assert "window.$3Dmolpromise=Promise.resolve()" in preview
    assert "https://3dmol" not in preview
    assert 'aria-label="Rotate left"' in preview
    assert "Math.hypot" in preview
    assert "item.domain+' · '+item.purpose" in preview
    header = selection.widget.children[1]
    grid = selection.widget.children[2]
    first_card = selection._cards["urea"]
    first_viewer = first_card.children[0].value
    first_control = first_card.children[2]
    assert grid.layout.grid_template_columns == "repeat(4, minmax(0, 1fr))"
    assert "Choose systems for the Batch" in header.children[0].value
    assert first_control.description == "✓ In batch"
    assert "3dmolviewer_" in first_viewer
    assert "createViewer" in first_viewer


def test_structure_selection_updates_the_python_batch_choice() -> None:
    catalog = {
        "urea": _Spec(
            _urea_like(),
            "Urea",
            "Agricultural chemistry",
            "Nitrogen fertilizer",
            "RDKit SMILES",
            "Exploratory model application",
        ),
        "water": _Spec(
            molecule("H2O"),
            "Water",
            "General chemistry",
            "Small polar molecule",
            "ASE",
            "Exploratory model application",
        ),
    }
    selection = viewers.show_structure_explorer(catalog)

    selection._checkboxes["urea"].value = True
    selection._checkboxes["water"].value = True
    selection._checkboxes["urea"].value = False
    assert selection.selected_keys == ("water",)
    assert selection.selected_specs() == [catalog["water"]]
    assert selection._checkboxes["water"].description == "✓ In batch"
    assert selection._checkboxes["urea"].description == "+ Add to batch"

    selection._checkboxes["water"].value = False
    with pytest.raises(ValueError, match="Select at least one molecular card"):
        selection.selected_specs()


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
    ).data

    assert "Synchronized relaxation player" in html
    assert "Shared step" in html
    assert "▶ Play" in html
    assert "max force" in html
    assert "eV/Å" in html
    assert "grid-template-columns:repeat(3,minmax(0,1fr))" in html
    assert "fmax 0.800 → 0.300 eV/Å" in html
    assert "Static initial/final preview" in html
    assert "position:relative;isolation:isolate" in html
    assert "Synchronized py3Dmol relaxation player" in html
    assert "$3Dmol.createViewer(node" in html
    assert "addModelsAsFrames" in html
    assert "viewer.setFrame(index)" in html
    assert "window.requestAnimationFrame(tick)" in html
    assert html.count("Maximum force (eV/Å)") == 2
    assert "Color atoms by force" in html
    assert "item.atom_forces[0]" in html
    assert "item.forces[index]" in html
    assert "grid-template-columns:repeat(5,minmax(0,1fr))" in html
    assert "grid-template-rows:repeat(1,286px)" in html
    assert "flex:0 0 118px" in html
    assert "data-progress-fill" in html
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


def test_surface_viewers_disable_bonds_and_use_larger_spheres() -> None:
    surface = Atoms(
        "PtCO",
        positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [1.0, 1.0, 3.1]],
        cell=[4.0, 4.0, 8.0],
        pbc=True,
    )
    item = viewers._structure_payload(surface)
    card = viewers._py3dmol_card_markup(item)
    player = viewers.show_relaxation_player(
        {"CO on Pt(111)": [surface, surface.copy()]},
        force_history={"CO on Pt(111)": [1.0, 0.5]},
    ).data

    assert item["sphere_only"] is True
    assert '"assignBonds": false' in card
    assert '"scale": 0.38' in card
    assert '"stick"' not in card
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


def test_bond_slider_uses_one_sided_hookean_response_and_static_endpoints() -> None:
    atoms = _urea_like()
    html = viewers.show_bond_slider(
        atoms,
        (2, 4),
        spring_constant=12.5,
        activation_distance=1.1,
        distance_range=(0.8, 2.0),
    ).data

    assert "one-sided Hookean bond experiment" in html
    assert "N–H bond experiment" in html
    assert "Static start/end preview" in html
    assert "Hookean energy" in html
    assert "Restoring force" in html
    assert "Math.max(0,distance-threshold)" in html
    assert "k=12.5,threshold=1.1" in html
    assert "eV/Å" in html
    assert 'min="0.8" max="2"' in html

    update_script = html.split("const update=function(){", 1)[1].split(
        "slider.addEventListener", 1
    )[0]
    assert "const view=viewer.getView()" in update_script
    assert "viewer.addModel(item.xyz,'xyz')" in update_script
    assert "viewer.setView(view)" in update_script
    assert "drawStructure" not in update_script
    assert "zoomTo" not in update_script
    assert "rotate(" not in update_script
    assert "E = ½kΔr²" in html
    assert "|F| = kΔr" in html


def test_phenol_hook_experiment_commits_once_and_updates_both_branches() -> None:
    phenol = _phenol_like()
    oxygen = phenol.get_chemical_symbols().index("O")
    hydrogen = min(
        (
            index
            for index, symbol in enumerate(phenol.get_chemical_symbols())
            if symbol == "H"
        ),
        key=lambda index: phenol.get_distance(oxygen, index),
    )
    calls: list[Atoms] = []
    callback_states: list[str] = []
    holder: dict[str, Any] = {}

    def run_callback(edited: Atoms) -> dict[str, Any]:
        calls.append(edited.copy())
        callback_states.append(holder["experiment"].status)
        baseline = [edited.copy(), edited.copy()]
        hookean = [edited.copy(), edited.copy()]
        hookean[-1].positions[hydrogen] = (
            hookean[-1].positions[oxygen]
            + (hookean[-1].positions[hydrogen] - hookean[-1].positions[oxygen])
            / hookean[-1].get_distance(oxygen, hydrogen)
        )
        return {
            "trajectories": {"baseline": baseline, "hookean": hookean},
            "force_history": {"baseline": [0.8, 0.4], "hookean": [0.8, 0.2]},
            "device": "CUDA GPU",
            "elapsed_s": 1.5,
            "convergence": {"baseline": "fixed steps", "hookean": "converged"},
        }

    fallback = {
        "trajectories": {
            "baseline": [phenol.copy(), phenol.copy()],
            "hookean": [phenol.copy(), phenol.copy()],
        }
    }
    experiment = viewers.show_phenol_hook_experiment(
        phenol,
        (oxygen, hydrogen),
        run_callback,
        fallback,
        activation_distance=1.10,
        distance_range=(0.95, 1.45),
    )
    holder["experiment"] = experiment

    experiment.preview(1.30)
    experiment.preview(1.35)
    assert calls == []

    experiment.commit(1.35)
    assert len(calls) == 1
    assert callback_states == ["running"]
    assert calls[0].get_distance(oxygen, hydrogen) == pytest.approx(1.35)
    assert experiment.status == "success"
    assert "AIMNet2 + FIRE2" in experiment.result_html
    assert "AIMNet2 + Hookean hook + FIRE2" in experiment.result_html
    assert "O–H 1.350 → 1.350 Å" in experiment.result_html
    assert "O–H 1.350 → 1.000 Å" in experiment.result_html


def test_copy_with_bond_distance_returns_an_independent_structure() -> None:
    phenol = _phenol_like()
    original_positions = phenol.positions.copy()
    carbon_hydrogen = (1, 8)

    moved = viewers.copy_with_bond_distance(phenol, carbon_hydrogen, 1.35)

    assert moved is not phenol
    np.testing.assert_allclose(phenol.positions, original_positions)
    assert moved.get_distance(*carbon_hydrogen) == pytest.approx(1.35)
    np.testing.assert_allclose(moved.positions[:8], original_positions[:8])
    assert helpers_core.copy_with_bond_distance is viewers.copy_with_bond_distance


def test_phenol_hook_experiment_accepts_and_labels_carbon_hydrogen() -> None:
    phenol = _phenol_like()
    carbon_hydrogen = (1, 8)
    fallback = {
        "trajectories": {
            "baseline": [phenol.copy()],
            "hookean": [phenol.copy()],
        }
    }

    experiment = viewers.show_phenol_hook_experiment(
        phenol,
        carbon_hydrogen,
        lambda edited: fallback,
        fallback,
        distance_range=(0.90, 1.50),
    )
    html = experiment._repr_mimebundle_()["text/html"]

    assert "Phenol C–H hook experiment" in html
    assert "C–H distance" in html
    assert "against C–H distance" in html
    assert "C–H" in experiment.status_message


def test_phenol_hook_experiment_rejects_other_or_reversed_bonds() -> None:
    phenol = _phenol_like()
    fallback = {
        "trajectories": {
            "baseline": [phenol.copy()],
            "hookean": [phenol.copy()],
        }
    }

    with pytest.raises(ValueError, match="carbon or oxygen, then hydrogen"):
        viewers.show_phenol_hook_experiment(
            phenol, (7, 6), lambda edited: fallback, fallback
        )


def test_phenol_hook_experiment_has_correct_curves_units_and_static_fallback() -> None:
    phenol = _phenol_like()
    oxygen = phenol.get_chemical_symbols().index("O")
    hydrogen = min(
        (
            index
            for index, symbol in enumerate(phenol.get_chemical_symbols())
            if symbol == "H"
        ),
        key=lambda index: phenol.get_distance(oxygen, index),
    )
    fallback = {
        "trajectories": {
            "baseline": [phenol.copy(), phenol.copy()],
            "hookean": [phenol.copy(), phenol.copy()],
        }
    }
    experiment = viewers.show_phenol_hook_experiment(
        phenol,
        (oxygen, hydrogen),
        lambda edited: fallback,
        fallback,
        spring_constant=10.0,
        activation_distance=1.10,
        distance_range=(0.95, 1.45),
    )
    html = experiment._repr_mimebundle_()["text/html"]

    assert "Phenol O–H hook experiment" in html
    assert "Hookean potential energy, E (eV)" in html
    assert "Restoring-force magnitude, |F| (eV/Å)" in html
    assert "E = ½kΔr²" in html
    assert "|F| = kΔr" in html
    assert "energy=.5*k*extension*extension" in html
    assert "force=k*extension" in html
    assert "data-energy-marker" in html
    assert "data-force-marker" in html
    assert 'data-distance type="range"' in html
    assert "const slider=root.querySelector('[data-distance]')" in html
    assert "slider.addEventListener('input',updatePreview)" in html
    assert "slider.addEventListener('change',commitPreview)" in html
    assert (
        "window.parent.document.querySelector"
        not in html.split("const updatePreview", 1)[0]
    )
    assert html.count("window.$3Dmol.createViewer") >= 2
    assert "Static fallback" in html
    assert "AIMNet2 + FIRE2" in html
    assert "AIMNet2 + Hookean hook + FIRE2" in html
    assert "#76B900" in html


def test_phenol_hook_iframe_tracks_responsive_plot_height() -> None:
    iframe = viewers._viewer_iframe(
        "<div>responsive phenol widget</div>", height=720, auto_height=True
    )

    assert "ResizeObserver" in iframe
    assert "document.documentElement.scrollHeight" in iframe
    assert "frame.style.height" in iframe
    assert "overflow:visible" in iframe


def test_phenol_hook_result_uses_isolated_dark_iframe() -> None:
    phenol = _phenol_like()
    oxygen_hydrogen = (6, 7)
    fallback = {
        "trajectories": {
            "baseline": [phenol.copy()],
            "hookean": [phenol.copy()],
        }
    }
    experiment = viewers.show_phenol_hook_experiment(
        phenol, oxygen_hydrogen, lambda edited: fallback, fallback
    )

    class _ResultWidget:
        value = ""

    result_widget = _ResultWidget()
    experiment._result_widget = result_widget
    experiment._replace_result_output()

    assert "<iframe" in result_widget.value
    assert "document.documentElement.scrollHeight" not in result_widget.value
    assert "height:500px" in result_widget.value
    assert "background:#171D20" in result_widget.value


def test_phenol_hook_experiment_reports_callback_errors() -> None:
    phenol = _phenol_like()
    oxygen = phenol.get_chemical_symbols().index("O")
    hydrogen = min(
        (
            index
            for index, symbol in enumerate(phenol.get_chemical_symbols())
            if symbol == "H"
        ),
        key=lambda index: phenol.get_distance(oxygen, index),
    )
    fallback = {
        "trajectories": {
            "baseline": [phenol.copy()],
            "hookean": [phenol.copy()],
        }
    }

    def fail(_: Atoms) -> dict[str, Any]:
        raise RuntimeError("optimizer failed")

    experiment = viewers.show_phenol_hook_experiment(
        phenol, (oxygen, hydrogen), fail, fallback
    )
    with pytest.raises(RuntimeError, match="optimizer failed"):
        experiment.commit(1.25)

    assert experiment.status == "error"
    assert "optimizer failed" in experiment.status_message


def test_trajectory_comparison_applies_batch_offsets_to_local_bond() -> None:
    frames = [_BatchFrame(2.0), _BatchFrame(3.0)]

    second_graph = viewers._graph_atoms(frames[0], 1)
    html = viewers.show_trajectory_comparison(
        frames,
        labels=("pull only", "pull + Hookean restraint"),
        bond=(0, 1),
        force_history={
            "pull only": [0.9, 0.7],
            "pull + Hookean restraint": [0.8, 0.2],
        },
        device="NVIDIA H100",
        elapsed_s=2.5,
        convergence={
            "pull only": "fixed-step preview",
            "pull + Hookean restraint": "converged",
        },
    ).data

    np.testing.assert_allclose(second_graph.positions[0], [10.0, 0.0, 0.0])
    np.testing.assert_allclose(second_graph.positions[1], [12.0, 0.0, 0.0])
    assert '"name":"pull only"' in html
    assert '"name":"pull + Hookean restraint"' in html
    assert '"distances":[1.0,1.0]' in html
    assert '"distances":[2.0,3.0]' in html
    assert '"bond_label":"H–H"' in html
    assert '"bond_label":"N–H"' in html
    assert "item.bond_label+' distance: '" in html
    assert "max force" in html
    assert "fmax 0.900 → 0.700 eV/Å" in html
    assert "NVIDIA H100" in html
    assert "2.500 s" in html
    assert "fixed-step preview" in html
    assert "Run status" in html
    assert "repeat(auto-fit,minmax(240px,1fr))" in html
    assert "fixed-step frame" in html
    assert "Static initial/final comparison" in html


def test_trajectory_comparison_derives_oh_label_from_supplied_atoms() -> None:
    phenol = _phenol_like()
    oxygen = phenol.get_chemical_symbols().index("O")
    hydrogen = min(
        (
            index
            for index, symbol in enumerate(phenol.get_chemical_symbols())
            if symbol == "H"
        ),
        key=lambda index: phenol.get_distance(oxygen, index),
    )
    html = viewers.show_trajectory_comparison(
        {
            "AIMNet2 + FIRE2": [phenol.copy()],
            "AIMNet2 + Hookean hook + FIRE2": [phenol.copy()],
        },
        bond=(oxygen, hydrogen),
    ).data

    assert "O–H" in html
    assert "N–H" not in html


def test_phenol_hook_helper_contains_no_toolkit_construction() -> None:
    source = Path(viewers.__file__).read_text(encoding="utf-8")
    public_section = source.split("class PhenolHookExperiment", 1)[1]
    public_section = public_section.split("\ndef show_trajectory_comparison", 1)[0]

    for forbidden in (
        "AIMNet2Wrapper",
        "FIRE2(",
        "BiasedPotentialHook",
        "DynamicsStage",
        "nvalchemi",
    ):
        assert forbidden not in public_section


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


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: viewers.show_structure_explorer({}), "at least one structure"),
        (
            lambda: viewers.show_trajectory_comparison(
                {"one": [molecule("H2O")]}, bond=(0, 1)
            ),
            "exactly two systems",
        ),
        (
            lambda: viewers.show_bond_slider(molecule("H2O"), (0, 0)),
            "two different valid atom indices",
        ),
    ],
)
def test_viewer_inputs_fail_with_clear_messages(
    call: Callable[[], Any], message: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        call()
