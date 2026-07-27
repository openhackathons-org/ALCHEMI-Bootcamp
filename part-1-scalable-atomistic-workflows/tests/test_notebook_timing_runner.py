"""Focused checks for the cell and stage timing report."""

from __future__ import annotations

import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType

import nbformat
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPOSITORY_ROOT / "scripts" / "run_notebook_no_timeout.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "part1_notebook_runner",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def test_stage_contexts_follow_the_preceding_stage_card() -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("setup = True", id="setup"),
            nbformat.v4.new_markdown_cell(
                '<h2 id="alchemi-stage-1-heading">First result</h2>',
                id="stage-1",
            ),
            nbformat.v4.new_code_cell("one = 1", id="one"),
            nbformat.v4.new_markdown_cell(
                '<h2 id="alchemi-stage-2-heading">One batch</h2>',
                id="stage-2",
            ),
            nbformat.v4.new_code_cell("two = 2", id="two"),
        ]
    )

    assert RUNNER.stage_contexts(notebook, expected_stages=(1, 2)) == {
        0: (0, "Setup"),
        2: (1, "First result"),
        4: (2, "One batch"),
    }


def test_stage_contexts_reject_missing_or_reordered_stage_ids() -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("Stage 1", id="stage-1"),
            nbformat.v4.new_code_cell("one = 1", id="one"),
            nbformat.v4.new_markdown_cell("Stage 3", id="stage-3"),
            nbformat.v4.new_code_cell("three = 3", id="three"),
        ]
    )

    with pytest.raises(ValueError, match="must appear once in order"):
        RUNNER.stage_contexts(notebook)


def test_refresh_timing_summary_recomputes_counts_and_stage_totals() -> None:
    report = {
        "cell_timings": [
            {
                "stage": 0,
                "stage_title": "Setup",
                "status": "complete",
                "elapsed_s": 1.25,
            },
            {
                "stage": 1,
                "stage_title": "First result",
                "status": "complete",
                "elapsed_s": 2.5,
            },
            {
                "stage": 1,
                "stage_title": "First result",
                "status": "failed",
                "elapsed_s": 0.25,
            },
        ]
    }

    RUNNER.refresh_timing_summary(report)

    assert report["code_cells_started"] == 3
    assert report["code_cells_completed"] == 2
    assert report["code_cells_failed"] == 1
    assert report["total_code_elapsed_s"] == 4.0
    assert report["stage_timings"] == [
        {
            "stage": 0,
            "title": "Setup",
            "code_cells_started": 1,
            "code_cells_completed": 1,
            "code_cells_failed": 0,
            "elapsed_s": 1.25,
        },
        {
            "stage": 1,
            "title": "First result",
            "code_cells_started": 2,
            "code_cells_completed": 1,
            "code_cells_failed": 1,
            "elapsed_s": 2.75,
        },
    ]


def test_atomic_write_json_replaces_the_complete_report(tmp_path: Path) -> None:
    destination = tmp_path / "timings.json"

    RUNNER.atomic_write_json(destination, {"status": "running"})
    RUNNER.atomic_write_json(destination, {"status": "complete"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "complete"
    }
    assert not (tmp_path / ".timings.json.tmp").exists()


def test_temporary_cell_is_not_stored_or_counted_as_a_learner_cell() -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("learner = True", id="learner")]
    )
    learner_cell = notebook.cells[0]

    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int | None, bool]] = []

        def execute_cell(
            self,
            cell,
            cell_index,
            execution_count=None,
            store_history=True,
        ):
            assert notebook.cells[cell_index] is cell
            self.calls.append((cell.source, execution_count, store_history))
            return cell

    client = RecordingClient()
    RUNNER.execute_temporary_cell(
        client,
        notebook,
        RUNNER.CUDA_SYNCHRONIZE_SOURCE,
    )

    assert notebook.cells == [learner_cell]
    assert client.calls == [
        (RUNNER.CUDA_SYNCHRONIZE_SOURCE, None, False),
    ]


def test_failed_cell_saves_a_partial_notebook_and_failed_report(
    tmp_path: Path,
) -> None:
    cells = [nbformat.v4.new_code_cell("setup = True", id="setup")]
    for stage in range(1, 8):
        cells.extend(
            [
                nbformat.v4.new_markdown_cell(
                    f'<h2 id="alchemi-stage-{stage}-heading">Stage {stage}</h2>',
                    id=f"stage-{stage}",
                ),
                nbformat.v4.new_code_cell(
                    f"value_{stage} = {stage}",
                    id=f"code-{stage}",
                ),
            ]
        )
    source = tmp_path / "source.ipynb"
    output = tmp_path / "executed.ipynb"
    timings = tmp_path / "timings.json"
    nbformat.write(nbformat.v4.new_notebook(cells=cells), source)
    original_sources = [cell.source for cell in cells]

    class FailingClient:
        def __init__(self, notebook, **_kwargs):
            self.notebook = notebook
            self.calls: list[tuple[str, int | None, bool]] = []

        def reset_execution_trackers(self) -> None:
            return None

        def setup_kernel(self):
            return nullcontext()

        def set_widgets_metadata(self) -> None:
            return None

        def execute_cell(
            self,
            cell,
            cell_index,
            execution_count=None,
            store_history=True,
        ):
            assert self.notebook.cells[cell_index] is cell
            self.calls.append((cell.source, execution_count, store_history))
            if cell.id == "code-2":
                raise RuntimeError("deliberate test failure")
            if store_history:
                cell.execution_count = execution_count
            return cell

    clients: list[FailingClient] = []

    def client_factory(notebook, **kwargs):
        client = FailingClient(notebook, **kwargs)
        clients.append(client)
        return client

    with pytest.raises(RuntimeError, match="deliberate test failure"):
        RUNNER.main(
            [
                str(source),
                "--output",
                str(output),
                "--timing-output",
                str(timings),
                "--kernel",
                "test-kernel",
            ],
            client_factory=client_factory,
        )

    assert output.is_file()
    executed = nbformat.read(output, as_version=4)
    assert [cell.source for cell in executed.cells] == original_sources
    assert len(executed.cells) == len(cells)

    calls = clients[0].calls
    synchronization_calls = [
        call for call in calls if call[0] == RUNNER.CUDA_SYNCHRONIZE_SOURCE
    ]
    learner_calls = [
        call for call in calls if call[0] != RUNNER.CUDA_SYNCHRONIZE_SOURCE
    ]
    assert [
        "sync" if call[0] == RUNNER.CUDA_SYNCHRONIZE_SOURCE else call[0]
        for call in calls
    ] == [
        "sync",
        "setup = True",
        "sync",
        "sync",
        "value_1 = 1",
        "sync",
        "sync",
        "value_2 = 2",
        "sync",
    ]
    assert synchronization_calls == [
        (RUNNER.CUDA_SYNCHRONIZE_SOURCE, None, False),
    ] * 6
    assert [call[1] for call in learner_calls] == [1, 2, 3]
    assert all(call[2] is True for call in learner_calls)

    report = json.loads(timings.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["code_cells_started"] == 3
    assert report["code_cells_completed"] == 2
    assert report["code_cells_failed"] == 1
    assert report["cell_timings"][-1]["cell_id"] == "code-2"
    assert report["cell_timings"][-1]["status"] == "failed"
    assert report["runner_error_type"] == "RuntimeError"
    assert "deliberate test failure" in report["runner_error_message"]
    assert "conditional temporary torch.cuda.synchronize()" in report[
        "cell_timing_boundary"
    ]


def test_pre_sync_failure_does_not_count_an_unstarted_learner_cell(
    tmp_path: Path,
) -> None:
    cells = [nbformat.v4.new_code_cell("setup = True", id="setup")]
    for stage in range(1, 8):
        cells.append(
            nbformat.v4.new_markdown_cell(
                f'<h2 id="alchemi-stage-{stage}-heading">Stage {stage}</h2>',
                id=f"stage-{stage}",
            )
        )
    source = tmp_path / "source.ipynb"
    output = tmp_path / "executed.ipynb"
    timings = tmp_path / "timings.json"
    nbformat.write(nbformat.v4.new_notebook(cells=cells), source)

    class PreSyncFailingClient:
        def __init__(self, notebook, **_kwargs):
            self.notebook = notebook
            self.learner_calls = 0

        def reset_execution_trackers(self) -> None:
            return None

        def setup_kernel(self):
            return nullcontext()

        def set_widgets_metadata(self) -> None:
            return None

        def execute_cell(
            self,
            cell,
            cell_index,
            execution_count=None,
            store_history=True,
        ):
            assert self.notebook.cells[cell_index] is cell
            if store_history:
                self.learner_calls += 1
                return cell
            raise RuntimeError("pre-sync exploded")

    clients: list[PreSyncFailingClient] = []

    def client_factory(notebook, **kwargs):
        client = PreSyncFailingClient(notebook, **kwargs)
        clients.append(client)
        return client

    with pytest.raises(
        RUNNER.CellSynchronizationError,
        match="failed before learner cell 0 .* started",
    ):
        RUNNER.main(
            [
                str(source),
                "--output",
                str(output),
                "--timing-output",
                str(timings),
            ],
            client_factory=client_factory,
        )

    report = json.loads(timings.read_text(encoding="utf-8"))
    assert clients[0].learner_calls == 0
    assert report["status"] == "failed"
    assert report["code_cells_started"] == 0
    assert report["code_cells_completed"] == 0
    assert report["code_cells_failed"] == 0
    assert report["cell_timings"] == []
    assert report["runner_error_type"] == "CellSynchronizationError"
    assert "pre-sync exploded" in report["runner_error_message"]
    executed = nbformat.read(output, as_version=4)
    assert [cell.source for cell in executed.cells] == [
        cell.source for cell in cells
    ]


def test_learner_exception_is_preserved_when_post_sync_also_fails(
    tmp_path: Path,
) -> None:
    cells = [nbformat.v4.new_code_cell("setup = True", id="setup")]
    for stage in range(1, 8):
        cells.append(
            nbformat.v4.new_markdown_cell(
                f'<h2 id="alchemi-stage-{stage}-heading">Stage {stage}</h2>',
                id=f"stage-{stage}",
            )
        )
    source = tmp_path / "source.ipynb"
    output = tmp_path / "executed.ipynb"
    timings = tmp_path / "timings.json"
    nbformat.write(nbformat.v4.new_notebook(cells=cells), source)
    learner_error = ValueError("learner exploded")

    class DualFailingClient:
        def __init__(self, notebook, **_kwargs):
            self.notebook = notebook
            self.calls = 0

        def reset_execution_trackers(self) -> None:
            return None

        def setup_kernel(self):
            return nullcontext()

        def set_widgets_metadata(self) -> None:
            return None

        def execute_cell(
            self,
            cell,
            cell_index,
            execution_count=None,
            store_history=True,
        ):
            assert self.notebook.cells[cell_index] is cell
            self.calls += 1
            if self.calls == 1:
                assert store_history is False
                return cell
            if self.calls == 2:
                assert store_history is True
                raise learner_error
            assert self.calls == 3
            assert store_history is False
            raise RuntimeError("post-sync exploded")

    def client_factory(notebook, **kwargs):
        return DualFailingClient(notebook, **kwargs)

    with pytest.raises(ValueError, match="learner exploded") as exc_info:
        RUNNER.main(
            [
                str(source),
                "--output",
                str(output),
                "--timing-output",
                str(timings),
            ],
            client_factory=client_factory,
        )

    assert exc_info.value is learner_error
    report = json.loads(timings.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["code_cells_started"] == 1
    assert report["code_cells_completed"] == 0
    assert report["code_cells_failed"] == 1
    record = report["cell_timings"][0]
    assert record["status"] == "failed"
    assert record["error_type"] == "ValueError"
    assert "learner exploded" in record["error_message"]
    assert "post-sync exploded" in record["error_message"]
    assert report["runner_error_type"] == "ValueError"
    assert "learner exploded" in report["runner_error_message"]
    assert "post-sync exploded" in report["runner_error_message"]
