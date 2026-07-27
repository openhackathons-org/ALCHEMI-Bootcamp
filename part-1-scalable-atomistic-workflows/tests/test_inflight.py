"""Checks for the tutorial-specific inflight source preparation."""

from __future__ import annotations

from pathlib import Path
import sys

from ase import Atoms
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("nvalchemi")

PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.dynamics import DynamicsStage  # noqa: E402

from aux.inflight import (  # noqa: E402
    InflightTraceCollector,
    InflightTraceEvent,
    inflight_trace_table,
    prepare_inflight_dimer_source,
    register_inflight_trace,
)


def _dimer(offset: float) -> Atoms:
    positions = [
        [offset + 0.00, 0.00, 0.00],
        [offset + 0.96, 0.00, 0.00],
        [offset - 0.24, 0.93, 0.00],
        [offset + 2.90, 0.00, 0.00],
        [offset + 3.86, 0.00, 0.00],
        [offset + 2.66, 0.93, 0.00],
    ]
    return Atoms(numbers=[8, 1, 1, 8, 1, 1], positions=positions, pbc=False)


def _scan(dimers: list[Atoms], *, truncate_first_ab: bool = False) -> Batch:
    graphs: list[AtomicData] = []
    for index, dimer in enumerate(dimers):
        ab = dimer[:-1] if truncate_first_ab and index == 0 else dimer
        for atoms in (ab, dimer[:3], dimer[3:]):
            atoms = atoms.copy()
            atoms.info["charge"] = 0
            graphs.append(
                AtomicData.from_atoms(atoms, device="cpu", dtype=torch.float32)
            )
    return Batch.from_data_list(graphs, device="cpu")


def _outputs(scan: Batch) -> dict[str, torch.Tensor]:
    return {
        "energy": torch.arange(scan.num_graphs, dtype=torch.float64).reshape(-1, 1)
        + 0.25,
        "forces": torch.arange(scan.num_nodes * 3, dtype=torch.float64).reshape(
            scan.num_nodes, 3
        )
        / 100.0,
        "charges": torch.linspace(
            -0.4, 0.4, scan.num_nodes, dtype=torch.float64
        ).reshape(-1, 1),
    }


def _trace_batch(system_ids: list[int], *, failed: list[bool] | None = None) -> Batch:
    if failed is not None and len(failed) != len(system_ids):
        raise ValueError("failed markers must align with system_ids")
    graphs: list[AtomicData] = []
    for index, system_id in enumerate(system_ids):
        graph = AtomicData.from_atoms(
            Atoms("H", positions=[[float(system_id), 0.0, 0.0]]),
            device="cpu",
            dtype=torch.float32,
        )
        graph.add_system_property(
            "system_id", torch.tensor([[system_id]], dtype=torch.long)
        )
        if failed is not None:
            graph.add_system_property(
                "failed", torch.tensor([[failed[index]]], dtype=torch.bool)
            )
        graphs.append(graph)
    return Batch.from_data_list(graphs, device="cpu")


class _FusedStageProbe:
    """Minimal public fused-hook registration interface for trace tests."""

    def __init__(self, *, refill_frequency: int) -> None:
        self.refill_frequency = refill_frequency
        self.step_count = 0
        self.active_batch: Batch | None = None
        self.done = False
        self.fused_hooks: list[object] = []

    def register_fused_hook(self, hook: object) -> None:
        self.fused_hooks.append(hook)

    def observe(self, batch: Batch, *, step: int) -> None:
        self.active_batch = batch
        self.step_count = step
        for hook in self.fused_hooks:
            if step % hook.frequency == 0 and hook.stage is DynamicsStage.BEFORE_STEP:
                hook(
                    type("Context", (), {"batch": batch, "step_count": step})(),
                    DynamicsStage.BEFORE_STEP,
                )


def test_registered_inflight_trace_records_real_membership_changes() -> None:
    stage = _FusedStageProbe(refill_frequency=2)
    trace = register_inflight_trace(stage)
    first = _trace_batch([101, 102])
    replacement = _trace_batch([103, 104])
    first_ids_before = first.system_id.clone()

    stage.observe(first, step=0)
    stage.observe(first, step=2)
    stage.observe(replacement, step=4)

    assert trace.events() == (
        InflightTraceEvent(
            refill=0,
            fused_step=0,
            active=2,
            completed=0,
            entered_system_ids=(101, 102),
            leaving_system_ids=(),
            failures=None,
        ),
        InflightTraceEvent(
            refill=1,
            fused_step=4,
            active=2,
            completed=2,
            entered_system_ids=(103, 104),
            leaving_system_ids=(101, 102),
            failures=None,
        ),
    )
    assert torch.equal(first.system_id, first_ids_before)


def test_inflight_trace_table_closes_the_final_refill_and_counts_failures() -> None:
    stage = _FusedStageProbe(refill_frequency=1)
    trace = register_inflight_trace(stage)
    stage.observe(_trace_batch([201, 202]), step=0)
    stage.observe(_trace_batch([203]), step=3)
    completed = _trace_batch(
        [201, 202, 203],
        failed=[False, True, False],
    )
    stage.active_batch = None
    stage.step_count = 6
    stage.done = True

    with pytest.raises(ValueError, match="do not match the trace"):
        trace.finalize(
            completed_system_ids=torch.tensor([201, 999]),
        )
    trace.finalize(
        completed_system_ids=completed.system_id,
        failure_count=int(completed.failed.sum()),
    )
    table = inflight_trace_table(trace)

    assert table.to_dict(orient="records") == [
        {
            "Refill": 0,
            "Fused step": 0,
            "Active": 2,
            "Completed": 0,
            "Entered": 2,
            "Entered IDs": "201-202",
            "Leaving": 0,
            "Leaving IDs": "none",
            "Failures": "NOT REPORTED",
        },
        {
            "Refill": 1,
            "Fused step": 3,
            "Active": 1,
            "Completed": 2,
            "Entered": 1,
            "Entered IDs": "203",
            "Leaving": 2,
            "Leaving IDs": "201-202",
            "Failures": "NOT REPORTED",
        },
        {
            "Refill": 2,
            "Fused step": 6,
            "Active": 0,
            "Completed": 3,
            "Entered": 0,
            "Entered IDs": "none",
            "Leaving": 1,
            "Leaving IDs": "203",
            "Failures": 1,
        },
    ]


def test_inflight_trace_finalizes_2048_systems_without_expanding_id_cells() -> None:
    trace = InflightTraceCollector(frequency=1)
    all_ids = torch.arange(2048, dtype=torch.long)

    for refill, start in enumerate(range(0, 2048, 256)):
        system_ids = all_ids[start : start + 256].reshape(-1, 1)
        batch = type(
            "RuntimeBatch",
            (),
            {"system_id": system_ids, "num_graphs": len(system_ids)},
        )()
        trace.record(batch, fused_step=refill * 5)

    trace.finalize(
        completed_system_ids=all_ids,
        fused_step=40,
    )
    events = trace.events()
    table = inflight_trace_table(events)

    assert events[-1].active == 0
    assert events[-1].completed == 2048
    assert table.iloc[0]["Entered IDs"] == "0-255"
    assert table.iloc[-1]["Leaving IDs"] == "1792-2047"
    assert table.iloc[-1]["Failures"] == "NOT REPORTED"
    assert table["Entered IDs"].str.len().max() < 32
    assert table["Leaving IDs"].str.len().max() < 32


def test_prepare_inflight_dimer_source_is_cpu_float32_and_reproducible() -> None:
    dimers = [_dimer(0.0), _dimer(0.2)]
    scan = _scan(dimers)
    outputs = _outputs(scan)

    source = prepare_inflight_dimer_source(
        scan_dimers=dimers,
        scan_batch=scan,
        full_outputs=outputs,
        num_systems=4,
        temperature_k=75.0,
        velocity_seed=404,
    )
    repeated = prepare_inflight_dimer_source(
        scan_dimers=dimers,
        scan_batch=scan,
        full_outputs=outputs,
        num_systems=4,
        temperature_k=75.0,
        velocity_seed=404,
    )

    assert source.device.type == "cpu"
    assert source.num_graphs == 4
    assert source.positions.dtype == torch.float32
    assert source.energy.dtype == torch.float32
    assert source.forces.dtype == torch.float32
    assert source.charges.dtype == torch.float32
    assert source.charges.ndim == 1
    assert source.nvt_steps_done.dtype == torch.int64
    assert source.nve_steps_done.dtype == torch.int64
    assert torch.count_nonzero(source.nvt_steps_done) == 0
    assert torch.count_nonzero(source.nve_steps_done) == 0
    assert torch.isfinite(source.velocities).all()
    float32_eps = torch.finfo(torch.float32).eps
    torch.testing.assert_close(
        source.velocities,
        repeated.velocities,
        rtol=2 * float32_eps,
        atol=float32_eps,
    )

    torch.testing.assert_close(
        source.energy,
        outputs["energy"][[0, 3, 0, 3]].to(torch.float32),
    )
    source_ptr = source.batch_ptr.tolist()
    scan_ptr = scan.batch_ptr.tolist()
    for output_graph, scan_graph in enumerate((0, 3, 0, 3)):
        out_start, out_stop = source_ptr[output_graph : output_graph + 2]
        scan_start, scan_stop = scan_ptr[scan_graph : scan_graph + 2]
        torch.testing.assert_close(
            source.forces[out_start:out_stop],
            outputs["forces"][scan_start:scan_stop].to(torch.float32),
        )
        torch.testing.assert_close(
            source.charges[out_start:out_stop],
            outputs["charges"][scan_start:scan_stop].reshape(-1).to(torch.float32),
        )
        torch.testing.assert_close(
            source.positions[out_start:out_stop],
            torch.tensor(
                dimers[output_graph % len(dimers)].positions,
                dtype=torch.float32,
            ),
        )


def test_prepare_inflight_dimer_source_requires_complete_outputs() -> None:
    dimers = [_dimer(0.0), _dimer(0.2)]
    scan = _scan(dimers)
    outputs = _outputs(scan)
    outputs.pop("charges")

    with pytest.raises(ValueError, match="tensor output 'charges'"):
        prepare_inflight_dimer_source(
            scan_dimers=dimers,
            scan_batch=scan,
            full_outputs=outputs,
            num_systems=4,
            temperature_k=75.0,
            velocity_seed=404,
        )


def test_prepare_inflight_dimer_source_requires_one_charge_per_atom() -> None:
    dimers = [_dimer(0.0), _dimer(0.2)]
    scan = _scan(dimers)
    outputs = _outputs(scan)
    outputs["charges"] = outputs["charges"].expand(-1, 2)

    with pytest.raises(ValueError, match=r"\[n_atoms\] or \[n_atoms, 1\]"):
        prepare_inflight_dimer_source(
            scan_dimers=dimers,
            scan_batch=scan,
            full_outputs=outputs,
            num_systems=4,
            temperature_k=75.0,
            velocity_seed=404,
        )


def test_prepare_inflight_dimer_source_checks_ab_a_b_layout() -> None:
    dimers = [_dimer(0.0), _dimer(0.2)]
    scan = _scan(dimers)

    with pytest.raises(ValueError, match="one AB, A, B triplet"):
        prepare_inflight_dimer_source(
            scan_dimers=dimers,
            scan_batch=scan.index_select(list(range(5))),
            full_outputs=_outputs(scan),
            num_systems=4,
            temperature_k=75.0,
            velocity_seed=404,
        )

    bad_scan = _scan(dimers, truncate_first_ab=True)
    with pytest.raises(ValueError, match="AB graph atom count"):
        prepare_inflight_dimer_source(
            scan_dimers=dimers,
            scan_batch=bad_scan,
            full_outputs=_outputs(bad_scan),
            num_systems=4,
            temperature_k=75.0,
            velocity_seed=404,
        )


def test_prepare_inflight_dimer_source_rejects_heterogeneous_dimers() -> None:
    dimers = [_dimer(0.0), _dimer(0.2)]
    scan = _scan(dimers)
    with pytest.raises(ValueError, match="homogeneous nonempty"):
        prepare_inflight_dimer_source(
            scan_dimers=[dimers[0], dimers[1][:-1]],
            scan_batch=scan,
            full_outputs=_outputs(scan),
            num_systems=4,
            temperature_k=75.0,
            velocity_seed=404,
        )


@pytest.mark.parametrize(
    ("num_systems", "temperature_k", "message"),
    ((0, 75.0, "num_systems must be positive"), (4, 0.0, "temperature_k")),
)
def test_prepare_inflight_dimer_source_checks_sizes_and_temperature(
    num_systems: int, temperature_k: float, message: str
) -> None:
    dimers = [_dimer(0.0), _dimer(0.2)]
    scan = _scan(dimers)
    with pytest.raises(ValueError, match=message):
        prepare_inflight_dimer_source(
            scan_dimers=dimers,
            scan_batch=scan,
            full_outputs=_outputs(scan),
            num_systems=num_systems,
            temperature_k=temperature_k,
            velocity_seed=404,
        )
