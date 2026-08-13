"""Unit checks for Part 08 inputs, provenance, and evidence gating."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import helpers
import numpy as np
import pytest
import torch


def test_accessible_figure_exposes_alt_text_to_html_and_png_renderers() -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    bundle, metadata = helpers.accessible_figure(
        figure, "A diagonal line from zero to one."
    )._repr_mimebundle_()
    plt.close(figure)

    assert 'aria-label="A diagonal line from zero to one."' in bundle["text/html"]
    assert "<title>A diagonal line from zero to one.</title>" in bundle["text/html"]
    assert isinstance(bundle["image/png"], bytes)
    assert metadata["image/png"]["alt"] == "A diagonal line from zero to one."


def test_checked_argon_control_has_periodic_fields_and_stable_ids() -> None:
    batch = helpers.build_argon_control(device="cpu")

    assert batch.num_graphs == 1
    assert batch.num_nodes == 32
    assert batch.pbc.tolist() == [[True, True, True]]
    assert torch.linalg.det(batch.cell).item() > 0.0
    assert batch.energy.shape == (1, 1)
    assert batch.forces.shape == (32, 3)
    assert batch.source_atom_id.tolist() == list(range(32))


def test_base_box_identity_matches_checked_part1_source() -> None:
    identity = helpers.base_box_identity()

    assert identity["atom_count"] == 3_200
    assert identity["molecule_count"] == 256
    assert identity["structure_sha256"] == (
        "5fcfc9394ebed3583267f20f322f60fb7b9311650e3b8dec4b8e8edaa4e0c0da"
    )
    assert identity["manifest_sha256"] == (
        "ea30e3f12f042f98f136147e783b56ab2e0da622f3486718b9fec69f3cde74b4"
    )
    assert identity["pbc"] == [True, True, True]


def test_control_summary_refuses_to_call_one_rank_decomposed() -> None:
    batch = helpers.build_argon_control(device="cpu")
    summary = helpers.control_summary(
        world_size=1,
        full_batch=batch,
        owned_batch=batch,
        gathered_batch=batch,
    )

    assert summary["world size"] == 1
    assert summary["input atoms"] == 32
    assert summary["rank-0 owned atoms"] == 32
    assert summary["gathered atoms"] == 32
    assert summary["spatially decomposed"] is False
    assert summary["interpretation"] == "one-process control; no partition"


def test_reorder_by_source_id_restores_input_order() -> None:
    values = np.array([[30.0], [10.0], [20.0]])
    source_ids = np.array([2, 0, 1])

    ordered = helpers.reorder_by_source_id(values, source_ids)

    np.testing.assert_array_equal(ordered, [[10.0], [20.0], [30.0]])
    with pytest.raises(ValueError, match="permutation"):
        helpers.reorder_by_source_id(values, np.array([0, 0, 2]))


def test_placeholder_campaign_is_not_plot_eligible() -> None:
    spec = helpers.read_campaign_spec()
    report = helpers.validate_campaign(spec)

    assert spec["status"] == "NOT REPORTED"
    assert report.ready is False
    assert report.status == "NOT REPORTED"
    assert "current-pin" in report.reason
    assert report.table.empty
    with pytest.raises(RuntimeError, match="not plot-eligible"):
        helpers.plot_campaign(report)


def _write_case(
    root: Path,
    *,
    spec: dict[str, object],
    world_size: int,
    forces: np.ndarray,
    energy_ev: float,
    median_s: float,
) -> str:
    case_dir = root / f"gpus-{world_size:02d}"
    case_dir.mkdir(parents=True)
    source_ids = np.arange(forces.shape[0], dtype=np.int64)
    artifact = case_dir / "result.npz"
    np.savez(artifact, forces=forces, source_atom_id=source_ids)
    pins = deepcopy(spec["current_pins"])
    workload = spec["workload"]
    checkpoint = workload["model"]["checkpoint_sha256"]
    owned = np.array_split(np.arange(forces.shape[0]), world_size)
    record = {
        "schema": "alchemi.part08-domain-case.v1",
        "status": "complete",
        "world_size": world_size,
        "current_pins": pins,
        "input": {
            "atom_count": forces.shape[0],
            "base_structure_sha256": workload["base_structure_sha256"],
            "tensor_sha256": "b" * 64,
        },
        "model": {
            "alias": workload["model"]["alias"],
            "checkpoint_sha256": checkpoint,
        },
        "distributed": {
            "mesh_shape": [world_size],
            "owned_atom_counts": [len(indices) for indices in owned],
            "halo_atom_counts": None,
            "halo_atom_counts_reason": "not exposed by the public API",
        },
        "output": {
            "energy_ev": energy_ev,
            "artifact": artifact.name,
            "artifact_sha256": helpers.sha256_file(artifact),
            "forces_sha256": helpers.array_sha256(forces),
            "source_atom_id_sha256": helpers.array_sha256(source_ids),
        },
        "timing": {
            "warmup_count": 1,
            "pass_times_s": [median_s * 1.05, median_s, median_s * 0.95],
            "median_s": median_s,
            "ranks_synchronized": True,
            "publishable_benchmark": False,
        },
        "runtime": {
            "gpu_names": ["Synthetic test GPU"] * world_size,
            "python_version": "3.12.13",
            "torch_version": "2.12.0+cu130",
            "torch_cuda_version": "13.0",
            "driver_version": "test",
        },
    }
    case_path = case_dir / "case.json"
    case_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return str(case_path.relative_to(root))


def test_complete_campaign_requires_identity_ownership_and_parity(tmp_path: Path) -> None:
    spec = deepcopy(helpers.read_campaign_spec())
    spec["status"] = "complete"
    spec["workload"]["atom_count"] = 4
    spec["workload"]["input_tensor_sha256"] = "b" * 64
    reference = np.array(
        [[1.0, -1.0, 0.0], [0.5, 0.0, -0.5], [0.0, 0.2, 0.0], [-0.1, 0.0, 0.1]],
        dtype=np.float32,
    )
    spec["cases"] = {
        str(world_size): _write_case(
            tmp_path,
            spec=spec,
            world_size=world_size,
            forces=reference + np.float32(0.0004 * (world_size - 1)),
            energy_ev=-20.0 + 0.00002 * (world_size - 1),
            median_s=1.0 / world_size,
        )
        for world_size in (1, 2, 4)
    }

    report = helpers.validate_campaign(spec, root=tmp_path)

    assert report.ready is True
    assert report.status == "complete"
    assert report.table["GPUs"].tolist() == [1, 2, 4]
    assert report.table["owned atoms"].tolist() == [4, 4, 4]
    assert report.table["parity"].tolist() == ["reference", "pass", "pass"]
    assert report.table["speedup"].tolist() == pytest.approx([1.0, 2.0, 4.0])


def test_campaign_rejects_changed_pin_or_failed_force_parity(tmp_path: Path) -> None:
    spec = deepcopy(helpers.read_campaign_spec())
    spec["status"] = "complete"
    spec["workload"]["atom_count"] = 4
    spec["workload"]["input_tensor_sha256"] = "b" * 64
    reference = np.zeros((4, 3), dtype=np.float32)
    spec["cases"] = {
        str(world_size): _write_case(
            tmp_path,
            spec=spec,
            world_size=world_size,
            forces=reference if world_size < 4 else np.ones_like(reference),
            energy_ev=-1.0,
            median_s=1.0,
        )
        for world_size in (1, 2, 4)
    }

    failed_parity = helpers.validate_campaign(spec, root=tmp_path)
    assert failed_parity.ready is False
    assert "force parity" in failed_parity.reason

    case_two = tmp_path / spec["cases"]["2"]
    record = json.loads(case_two.read_text(encoding="utf-8"))
    record["current_pins"]["toolkit"]["commit"] = "0" * 40
    case_two.write_text(json.dumps(record), encoding="utf-8")
    changed_pin = helpers.validate_campaign(spec, root=tmp_path)
    assert changed_pin.ready is False
    assert "pin mismatch" in changed_pin.reason


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_domain_and_ready_campaign_plots_have_questions_worth_answering(
    tmp_path: Path,
) -> None:
    domain_figure = helpers.plot_domain_ownership()
    assert len(domain_figure.axes) == 1
    assert "Owned" in domain_figure.axes[0].get_title()
    assert "ghost" in " ".join(text.get_text() for text in domain_figure.axes[0].texts)

    spec = deepcopy(helpers.read_campaign_spec())
    spec["status"] = "complete"
    spec["workload"]["atom_count"] = 4
    spec["workload"]["input_tensor_sha256"] = "b" * 64
    forces = np.zeros((4, 3), dtype=np.float32)
    spec["cases"] = {
        str(world_size): _write_case(
            tmp_path,
            spec=spec,
            world_size=world_size,
            forces=forces,
            energy_ev=-1.0,
            median_s=1.0 / world_size,
        )
        for world_size in (1, 2, 4)
    }
    report = helpers.validate_campaign(spec, root=tmp_path)
    campaign_figure = helpers.plot_campaign(report)
    labels = {axis.get_ylabel() for axis in campaign_figure.axes}
    assert "Median evaluation time (s)" in labels
    assert "Maximum force difference (eV/Å)" in labels
