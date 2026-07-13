from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


PART_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_DIR))

from aux import ui


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


@pytest.mark.parametrize(
    ("kind", "result_state", "label"),
    [
        ("before", None, "BEFORE YOU RUN"),
        ("check", None, "CHECK"),
        ("result", "observed", "RESULT — OBSERVED"),
        ("result", "pass", "RESULT — PASS"),
        ("result", "withheld", "RESULT — WITHHELD"),
        ("result", "action", "RESULT — ACTION NEEDED"),
        ("boundary", None, "BOUNDARY"),
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


def test_callout_escapes_body_and_rejects_ambiguous_state() -> None:
    html = ui.callout_html(
        '<img src="x" onerror="alert(1)"> & done',
        kind="check",
    )

    assert "<img" not in html
    assert "&lt;img" in html
    assert "&amp; done" in html
    with pytest.raises(ValueError, match="only"):
        ui.callout_html("No", kind="boundary", result_state="pass")
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
