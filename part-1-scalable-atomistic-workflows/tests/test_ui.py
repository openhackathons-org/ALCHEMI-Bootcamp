from __future__ import annotations

from base64 import b64decode
from inspect import signature
from pathlib import Path
import re
import sys

import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux import ui  # noqa: E402


class _RecordingDisplayHandle:
    def __init__(self) -> None:
        self.updates: list[object] = []

    def update(self, value: object) -> None:
        self.updates.append(value)


class _RecordingNotebookProgress(ui.NotebookProgress):
    def __init__(self, *, title: str, total: int) -> None:
        self.recorded_handle = _RecordingDisplayHandle()
        self.displayed: list[tuple[object, bool | None]] = []
        super().__init__(title=title, total=total)

    def _display(
        self,
        value: object,
        *,
        display_id: bool | None = None,
    ) -> _RecordingDisplayHandle:
        self.displayed.append((value, display_id))
        return self.recorded_handle


def test_progress_constructor_hides_display_internals() -> None:
    assert "display_fn" not in signature(ui.NotebookProgress).parameters


def test_figure_with_alt_embeds_png_and_escapes_description() -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [1.0, 0.0])
    rendered = ui.figure_with_alt(
        figure,
        alt_text='Falling line <script>alert("x")</script>',
        dpi=72,
    )
    plt.close(figure)

    html = rendered.data
    assert '<img src="data:image/png;base64,' in html
    assert 'alt="Falling line &lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;"' in html
    assert "<script>" not in html
    encoded = html.split("base64,", 1)[1].split('"', 1)[0]
    assert b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    ("alt_text", "dpi", "message"),
    [
        ("", 180, "alt_text"),
        ("Description", 0, "dpi"),
    ],
)
def test_figure_with_alt_rejects_inaccessible_or_invalid_output(
    alt_text: str, dpi: int, message: str
) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure()
    try:
        with pytest.raises(ValueError, match=message):
            ui.figure_with_alt(figure, alt_text=alt_text, dpi=dpi)
    finally:
        plt.close(figure)


def test_stage_card_is_accessible_and_escapes_caller_text() -> None:
    html = ui.stage_card_html(
        stage=2,
        total=6,
        title='Batch <script>alert("x")</script>',
        outcome="Compare A&B\nwithout hidden loops",
        state="active",
    )

    assert 'role="region"' in html
    assert 'aria-labelledby="alchemi-stage-2-heading"' in html
    assert '<h2 id="alchemi-stage-2-heading"' in html
    assert 'role="progressbar"' in html
    assert 'aria-valuemin="1"' in html
    assert 'aria-valuemax="6"' in html
    assert 'aria-valuenow="2"' in html
    assert 'aria-current="step"' in html
    assert "STAGE 2 OF 6" in html
    assert "IN PROGRESS" in html
    assert "width:33.33%" in html
    assert "<script" not in html
    assert "&lt;script&gt;" in html
    assert "A&amp;B<br>without hidden loops" in html
    assert "Compute time:" not in html


def test_stage_card_can_show_an_escaped_compute_time() -> None:
    html = ui.stage_card_html(
        stage=3,
        total=6,
        title="Complete the model",
        outcome="Compare with reference data",
        compute_time='about 45 seconds on one H100 <fast & measured>',
    )

    assert "Compute time:" in html
    assert "about 45 seconds on one H100 &lt;fast &amp; measured&gt;" in html
    assert "<fast" not in html


def test_stage_card_rejects_an_empty_compute_time() -> None:
    with pytest.raises(ValueError, match="compute_time"):
        ui.stage_card_html(
            stage=1,
            total=6,
            title="First call",
            outcome="Evaluate one structure",
            compute_time="   ",
        )


@pytest.mark.parametrize(
    ("kind", "result_state", "label"),
    [
        ("before", None, "BEFORE YOU RUN"),
        ("check", None, "WHAT TO CHECK"),
        ("note", None, "NOTE"),
        ("result", "observed", "RESULT"),
        ("result", "pass", "CHECK PASSED"),
        ("result", "not_reported", "NOT REPORTED"),
        ("result", "action", "NEEDS ATTENTION"),
    ],
)
def test_callout_has_visible_and_accessible_state_label(
    kind: str, result_state: str | None, label: str
) -> None:
    html = ui.callout_html(
        "Learner-facing conclusion",
        kind=kind,
        result_state=result_state,
    )

    assert 'role="note"' in html
    assert f'aria-label="{label}"' in html
    assert label in html


def test_legacy_callout_states_render_plain_learner_labels() -> None:
    note_html = ui.callout_html("Legacy note", kind="boundary")
    result_html = ui.callout_html(
        "Legacy result", kind="result", result_state="withheld"
    )

    assert 'aria-label="NOTE"' in note_html
    assert 'aria-label="NOT REPORTED"' in result_html
    assert "BOUNDARY" not in note_html
    assert "WITHHELD" not in result_html


def test_callout_escapes_body_and_rejects_ambiguous_state() -> None:
    html = ui.callout_html(
        '<img src="x" onerror="alert(1)"> & done',
        kind="check",
    )

    assert "<img" not in html
    assert "&lt;img" in html
    assert "&amp; done" in html
    with pytest.raises(ValueError, match="only"):
        ui.callout_html("No", kind="note", result_state="pass")
    with pytest.raises(ValueError, match="callout kind"):
        ui.callout_html("No", kind="unknown")


def test_progress_card_exposes_text_count_and_aria_values() -> None:
    html = ui.notebook_progress_html_string(
        title='H<sub>2</sub>O "run"',
        done=5_000,
        total=55_000,
        message="NVT finished; starting <NVE>",
        elapsed_s=62.25,
        unit="steps",
        average_label="step",
    )

    assert 'role="progressbar"' in html
    assert 'aria-valuemin="0"' in html
    assert 'aria-valuemax="55000"' in html
    assert 'aria-valuenow="5000"' in html
    assert "5,000 / 55,000 steps" in html
    assert "1 min 02.2 s" in html
    assert "RUNNING" in html
    assert "width:9.09%" in html
    assert "<sub>" not in html
    assert "&lt;sub&gt;" in html
    assert "<NVE>" not in html
    assert "&lt;NVE&gt;" in html
    assert "&quot;run&quot;" in html
    assert 'aria-live="polite"' in html


def test_notebook_progress_updates_without_display_side_effects() -> None:
    progress = ui.NotebookProgress(
        title="Trajectory",
        total=4,
        unit="passes",
        auto_display=False,
    )

    assert progress.done == 0
    assert "RUNNING" in progress.render_string()
    progress.advance(message="First pass")
    assert progress.done == 1
    assert "RUNNING" in progress.render_string()
    progress.complete("All four passes finished")
    html = progress.render_string()
    assert progress.done == 4
    assert "COMPLETE" in html
    assert "4 / 4 passes" in html
    assert "All four passes finished" in html


def test_notebook_progress_prefers_exportable_display_handle() -> None:
    """Store plain HTML so static exports do not depend on widget JavaScript."""

    progress = _RecordingNotebookProgress(
        title="Trajectory",
        total=2,
    )

    assert progress._display_handle is progress.recorded_handle
    assert progress._widget is None
    assert len(progress.displayed) == 1
    assert progress.displayed[0][1] is True
    assert "RUNNING" in progress.displayed[0][0].data

    progress.complete("Finished")
    assert len(progress.recorded_handle.updates) == 1
    assert "COMPLETE" in progress.recorded_handle.updates[0].data


def test_early_completion_fills_rail_without_hiding_step_limit() -> None:
    progress = ui.NotebookProgress(
        title="FIRE2",
        total=5_000,
        unit="steps",
        auto_display=False,
    )

    progress.update(done=181, message="Converged", state="complete")
    html = progress.render_string()

    assert "181 steps used · limit 5,000" in html
    assert 'aria-valuenow="181"' in html
    assert 'aria-valuetext="181 steps used · limit 5,000; status COMPLETE; Converged"' in html
    assert "width:100.00%" in html
    assert "COMPLETE" in html


def test_action_progress_uses_matching_red_rail_and_label() -> None:
    html = ui.notebook_progress_html_string(
        title="Scientific gate",
        done=2,
        total=4,
        message="Validation failed",
        elapsed_s=3.0,
        state="action",
    )

    assert "ACTION NEEDED" in html
    assert "background:#DC2626" in html
    assert 'aria-valuenow="2"' in html


def test_hero_summary_and_placeholder_share_accessible_visual_system() -> None:
    hero = ui.notebook_hero_html(
        image_path='assets/banner<bad>.png',
        image_alt='Water "dimer"',
        title="Dimer < trajectory",
        subtitle="One & many",
    )
    summary = ui.lesson_summary_html(
        do="Run <one>", learn="Batch & inspect", need="One GPU"
    )
    placeholder = ui.figure_placeholder_html(
        title="Atomistic loop", description="atoms → model → outputs"
    )

    assert 'role="region"' in hero
    assert 'aria-labelledby="alchemi-notebook-title"' in hero
    assert '<h1 id="alchemi-notebook-title"' in hero
    assert 'aspect-ratio:2/1' in hero
    assert ".jp-MarkdownOutput{max-width:880px;" in hero
    assert "overflow-x:auto" in hero
    assert 'assets/banner&lt;bad&gt;.png' in hero
    assert 'alt="Water &quot;dimer&quot;"' in hero
    assert "Dimer &lt; trajectory" in hero
    assert 'aria-label="Lesson summary"' in summary
    assert "Run &lt;one&gt;" in summary
    assert "Batch &amp; inspect" in summary
    assert 'role="img"' in placeholder
    assert 'aria-label="Illustration slot: Atomistic loop.' in placeholder
    assert "ILLUSTRATION SLOT · VISUAL REVIEW" in placeholder

    for html in (hero, summary, placeholder):
        assert "max-width:880px" in html


def test_process_diagram_is_accessible_escaped_and_self_contained() -> None:
    diagram = ui.process_diagram_html(
        title="Model < composition",
        steps=("positions & elements", "predicted charges", "total energy"),
        caption='Read the numbered steps from left to right.',
    )

    assert 'role="img"' in diagram
    assert 'aria-label="Process diagram: Model &lt; composition.' in diagram
    assert "positions &amp; elements then predicted charges then total energy" in diagram
    assert "Model &lt; composition" in diagram
    assert "STEP 1" in diagram
    assert "STEP 3" in diagram
    assert "→" not in diagram
    assert "max-width:880px" in diagram
    assert "<script" not in diagram.lower()


def test_five_step_process_diagram_fits_the_shared_reading_width() -> None:
    diagram = ui.process_diagram_html(
        title="Five steps",
        steps=("one", "two", "three", "four", "five"),
        caption="One desktop row.",
    )

    assert diagram.count("min-width:124px") == 5
    assert diagram.count("flex:1 1 124px") == 5


@pytest.mark.parametrize(
    ("steps", "error"),
    [
        (("only one",), "at least two"),
        (("first", ""), "cannot be empty"),
    ],
)
def test_process_diagram_rejects_incomplete_flows(
    steps: tuple[str, ...], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        ui.process_diagram_html(title="Bad", steps=steps, caption="Bad")


def test_readable_table_preserves_full_reasons_and_plain_missing_values() -> None:
    import numpy as np
    import pandas as pd

    reason = (
        "mean temperatures differ by more than the allowed tolerance; "
        "initial ring changed"
    )
    table = pd.DataFrame(
        {
            "Result": ["H2O/D2O shift", "Cluster/monomer shift"],
            "Measured": [reason, np.nan],
            "Unsafe": ["<script>alert('x')</script>", "safe"],
        }
    )

    rendered = ui.readable_table_html(
        table,
        label="Comparison results",
        show_index=False,
        missing="NOT REPORTED",
    )

    assert reason in rendered
    assert "..." not in rendered
    assert "NaN" not in rendered
    assert "NOT REPORTED" in rendered
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert 'aria-label="Comparison results"' in rendered
    assert "max-width:880px" in rendered
    assert "overflow-wrap:anywhere" in rendered
    assert '<th></th>' not in rendered


def test_readable_table_rejects_invalid_input() -> None:
    import pandas as pd

    with pytest.raises(TypeError, match="DataFrame"):
        ui.readable_table_html([{"value": 1}], label="Bad")
    with pytest.raises(ValueError, match="label"):
        ui.readable_table_html(pd.DataFrame({"value": [1]}), label="")


def test_invalid_live_progress_state_does_not_mutate_last_good_count() -> None:
    progress = ui.NotebookProgress(title="Trajectory", total=4, auto_display=False)

    with pytest.raises(ValueError, match="progress state"):
        progress.update(done=2, state="unknown")

    assert progress.done == 0
    assert "RUNNING" in progress.render_string()


@pytest.mark.parametrize(
    "operation",
    [
        lambda: ui.stage_card_html(
            stage=0, total=6, title="Bad", outcome="Bad", state="ready"
        ),
        lambda: ui.stage_card_html(
            stage=7, total=6, title="Bad", outcome="Bad", state="ready"
        ),
        lambda: ui.notebook_progress_html_string(
            title="Bad", done=-1, total=4, message="Bad", elapsed_s=0
        ),
        lambda: ui.notebook_progress_html_string(
            title="Bad", done=5, total=4, message="Bad", elapsed_s=0
        ),
        lambda: ui.notebook_progress_html_string(
            title="Bad", done=0, total=0, message="Bad", elapsed_s=0
        ),
    ],
)
def test_invalid_progress_ranges_fail_loudly(operation) -> None:
    with pytest.raises(ValueError):
        operation()


def _relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(foreground: str, background: str) -> float:
    first = _relative_luminance(foreground)
    second = _relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def test_semantic_text_colors_meet_wcag_aa_contrast() -> None:
    styles = [
        *ui._STAGE_STYLES.values(),
        *ui._CALLOUT_STYLES.values(),
        *ui._RESULT_STYLES.values(),
        *ui._PROGRESS_STYLES.values(),
    ]
    assert all(
        _contrast_ratio(style.foreground, style.background) >= 4.5
        for style in styles
    )


def test_rendered_html_is_self_contained_and_green_is_not_text() -> None:
    samples = [
        ui.stage_card_html(
            stage=1, total=6, title="First result", outcome="Evaluate one system"
        ),
        ui.callout_html("Predict before running", kind="before"),
        ui.notebook_progress_html_string(
            title="Run", done=1, total=2, message="Working", elapsed_s=1
        ),
        ui.lesson_summary_html(do="Run", learn="Inspect", need="GPU"),
        ui.figure_placeholder_html(title="Diagram", description="Visual slot"),
        ui.process_diagram_html(
            title="Flow", steps=("one", "two"), caption="Offline HTML"
        ),
    ]

    for html in samples:
        lowered = html.lower()
        assert "<script" not in lowered
        assert "<link" not in lowered
        assert "@import" not in lowered
        assert "url(" not in lowered
        assert re.search(r"color\s*:\s*#76b900", lowered) is None
    assert "background:#76B900" in samples[0]
    assert "background:#76B900" in samples[2]
