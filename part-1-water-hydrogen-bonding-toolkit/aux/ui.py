"""Self-contained, accessible presentation helpers for the Part 1 notebook.

The helpers in this module intentionally render plain HTML with inline styles:
they work in JupyterLab and VS Code without a stylesheet, JavaScript, or a
network connection.  All caller-provided text is HTML-escaped.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from time import perf_counter
from typing import Literal


StageState = Literal["ready", "active", "complete", "withheld", "action"]
CalloutKind = Literal["before", "check", "result", "boundary"]
ResultState = Literal["observed", "pass", "withheld", "action"]
ProgressState = Literal["ready", "running", "complete", "action"]

__all__ = [
    "CalloutKind",
    "NotebookProgress",
    "ProgressState",
    "ResultState",
    "StageState",
    "callout",
    "callout_html",
    "figure_placeholder_html",
    "format_elapsed",
    "lesson_summary_html",
    "notebook_hero_html",
    "notebook_progress_html",
    "notebook_progress_html_string",
    "stage_card",
    "stage_card_html",
]


_NVIDIA_GREEN = "#76B900"
_TEXT = "#111827"
_MUTED = "#4B5563"
_SURFACE = "#FFFFFF"
_BORDER = "#D1D5DB"
_RAIL = "#E5E7EB"
_FONT_STACK = "system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif"
_CONTENT_WIDTH_PX = 880
_CARD_RADIUS_PX = 12


@dataclass(frozen=True)
class _SemanticStyle:
    label: str
    foreground: str
    background: str


# Each foreground/background pair has at least 4.5:1 WCAG contrast.  The
# brighter NVIDIA green is therefore used as a non-text accent only.
_STAGE_STYLES: dict[str, _SemanticStyle] = {
    # Static stage cards are navigation, not execution state. A neutral label
    # stays truthful in both a fresh and a fully executed notebook.
    "ready": _SemanticStyle("STAGE", "#374151", "#F3F4F6"),
    "active": _SemanticStyle("IN PROGRESS", "#075985", "#E0F2FE"),
    "complete": _SemanticStyle("COMPLETE", "#365314", "#ECFCCB"),
    "withheld": _SemanticStyle("WITHHELD", "#9A3412", "#FFEDD5"),
    "action": _SemanticStyle("ACTION NEEDED", "#991B1B", "#FEE2E2"),
}

_CALLOUT_STYLES: dict[str, _SemanticStyle] = {
    "before": _SemanticStyle("BEFORE YOU RUN", "#075985", "#E0F2FE"),
    "check": _SemanticStyle("CHECK", "#854D0E", "#FEF9C3"),
    "boundary": _SemanticStyle("BOUNDARY", "#9A3412", "#FFEDD5"),
}

_RESULT_STYLES: dict[str, _SemanticStyle] = {
    "observed": _SemanticStyle("RESULT — OBSERVED", "#374151", "#F3F4F6"),
    "pass": _SemanticStyle("RESULT — PASS", "#365314", "#ECFCCB"),
    "withheld": _SemanticStyle("RESULT — WITHHELD", "#9A3412", "#FFEDD5"),
    "action": _SemanticStyle("RESULT — ACTION NEEDED", "#991B1B", "#FEE2E2"),
}

_PROGRESS_STYLES: dict[str, _SemanticStyle] = {
    "ready": _SemanticStyle("READY", "#374151", "#F3F4F6"),
    "running": _SemanticStyle("RUNNING", "#075985", "#E0F2FE"),
    "complete": _SemanticStyle("COMPLETE", "#365314", "#ECFCCB"),
    "action": _SemanticStyle("ACTION NEEDED", "#991B1B", "#FEE2E2"),
}


def _escaped_text(value: object) -> str:
    """Escape untrusted text while preserving intentional line breaks."""

    return escape(str(value), quote=True).replace("\n", "<br>")


def _escaped_attribute(value: object) -> str:
    """Escape untrusted text for use in a quoted HTML attribute."""

    return escape(str(value), quote=True)


def _positive_int(value: int, *, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _progress_values(done: int, total: int) -> tuple[int, int]:
    total = _positive_int(total, name="total")
    done = int(done)
    if not 0 <= done <= total:
        raise ValueError(f"done must be between 0 and total ({total})")
    return done, total


def _lookup_style(
    styles: dict[str, _SemanticStyle], value: str, *, name: str
) -> _SemanticStyle:
    try:
        return styles[value]
    except KeyError as exc:
        choices = ", ".join(styles)
        raise ValueError(f"unknown {name} {value!r}; choose one of: {choices}") from exc


def stage_card_html(
    *,
    stage: int,
    total: int,
    title: str,
    outcome: str,
    state: StageState = "ready",
) -> str:
    """Return a styled stage card as a safe, self-contained HTML string.

    ``stage`` identifies the current stage in the notebook's full sequence.
    The rail communicates that position; ``state`` communicates whether this
    particular stage is ready, active, complete, withheld, or needs action.
    """

    stage = _positive_int(stage, name="stage")
    total = _positive_int(total, name="total")
    if stage > total:
        raise ValueError("stage cannot be greater than total")
    style = _lookup_style(_STAGE_STYLES, state, name="stage state")
    percent = 100.0 * stage / total
    count = f"STAGE {stage} OF {total}"
    heading_id = f"alchemi-stage-{stage}-heading"
    aria_current = ' aria-current="step"' if state == "active" else ""

    return (
        f'<section role="region" aria-labelledby="{heading_id}" '
        f'style="max-width:{_CONTENT_WIDTH_PX}px;margin:18px 0;background:{_SURFACE};'
        f'border:1px solid {_BORDER};border-radius:{_CARD_RADIUS_PX}px;overflow:hidden;'
        f'box-shadow:0 1px 2px rgba(0,0,0,0.05);font-family:{_FONT_STACK};">'
        f'<div role="progressbar" aria-label="Notebook stage" aria-valuemin="1" '
        f'aria-valuemax="{total}" aria-valuenow="{stage}" '
        f'aria-valuetext="{count}"{aria_current} '
        f'style="height:6px;background:{_RAIL};overflow:hidden;">'
        f'<div aria-hidden="true" style="height:100%;width:{percent:.2f}%;'
        f'background:{_NVIDIA_GREEN};"></div></div>'
        '<div style="padding:16px 18px 17px;">'
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'gap:12px;flex-wrap:wrap;margin-bottom:9px;">'
        f'<span style="color:{_MUTED};font-size:12px;font-weight:700;'
        f'letter-spacing:0.08em;">{count}</span>'
        f'<span style="display:inline-block;padding:3px 9px;border-radius:999px;'
        f'background:{style.background};color:{style.foreground};font-size:12px;'
        f'font-weight:750;letter-spacing:0.04em;">{style.label}</span></div>'
        f'<h2 id="{heading_id}" style="color:{_TEXT};font-size:21px;'
        f'font-weight:750;line-height:1.25;margin:0 0 7px;">'
        f'{_escaped_text(title)}</h2>'
        f'<div style="color:{_MUTED};font-size:14px;line-height:1.55;">'
        f'<strong style="color:{_TEXT};">Outcome:</strong> '
        f'{_escaped_text(outcome)}</div></div></section>'
    )


def stage_card(**kwargs):
    """Return an ``IPython.display.HTML`` stage card for a notebook cell."""

    from IPython.display import HTML

    return HTML(stage_card_html(**kwargs))


def callout_html(
    body: str,
    *,
    kind: CalloutKind,
    result_state: ResultState | None = None,
) -> str:
    """Return a safe action-oriented callout as a self-contained HTML string.

    ``result_state`` is accepted only for ``kind="result"`` and yields one of
    ``RESULT — OBSERVED``, ``PASS``, ``WITHHELD``, or ``ACTION NEEDED``.
    """

    if kind == "result":
        style = _lookup_style(
            _RESULT_STYLES,
            "observed" if result_state is None else result_state,
            name="result state",
        )
    else:
        if result_state is not None:
            raise ValueError('result_state is valid only when kind="result"')
        style = _lookup_style(_CALLOUT_STYLES, kind, name="callout kind")

    return (
        f'<aside role="note" aria-label="{_escaped_attribute(style.label)}" '
        f'style="max-width:{_CONTENT_WIDTH_PX}px;margin:14px 0;padding:13px 16px;'
        f'border:1px solid {style.foreground};border-left-width:5px;'
        f'border-radius:{_CARD_RADIUS_PX}px;background:{style.background};'
        f'font-family:{_FONT_STACK};">'
        f'<div style="color:{style.foreground};font-size:12px;font-weight:800;'
        f'letter-spacing:0.07em;margin-bottom:5px;">{style.label}</div>'
        f'<div style="color:{_TEXT};font-size:14px;line-height:1.55;">'
        f'{_escaped_text(body)}</div></aside>'
    )


def callout(body: str, **kwargs):
    """Return an ``IPython.display.HTML`` callout for a notebook cell."""

    from IPython.display import HTML

    return HTML(callout_html(body, **kwargs))


def notebook_hero_html(
    *,
    image_path: str,
    image_alt: str,
    title: str,
    subtitle: str,
    eyebrow: str = "NVIDIA ALCHEMI TOOLKIT",
    badge: str = "PART 1 · LIVE GPU",
) -> str:
    """Return the accessible, text-safe notebook hero used above the first cell."""

    return (
        f'<section role="region" aria-labelledby="alchemi-notebook-title" '
        f'style="position:relative;max-width:{_CONTENT_WIDTH_PX}px;aspect-ratio:2/1;'
        f'margin:0 0 18px;overflow:hidden;border-radius:{_CARD_RADIUS_PX}px;'
        f'background:#050607;border:1px solid #252A30;box-shadow:0 2px 8px '
        f'rgba(0,0,0,0.18);font-family:{_FONT_STACK};">'
        f'<img src="{_escaped_attribute(image_path)}" '
        f'alt="{_escaped_attribute(image_alt)}" style="position:absolute;inset:0;'
        f'width:100%;height:100%;object-fit:cover;display:block;">'
        '<div aria-hidden="true" style="position:absolute;inset:0;'
        'background:linear-gradient(90deg,rgba(3,5,6,0.98) 0%,'
        'rgba(3,5,6,0.91) 39%,rgba(3,5,6,0.18) 68%,rgba(3,5,6,0.03) 100%);">'
        '</div><div style="position:relative;z-index:1;width:55%;height:100%;'
        'box-sizing:border-box;padding:34px 30px;display:flex;flex-direction:column;'
        'justify-content:center;align-items:flex-start;">'
        f'<div style="color:#B8C0C7;font-size:11px;font-weight:800;'
        f'letter-spacing:0.12em;margin-bottom:10px;">{_escaped_text(eyebrow)}</div>'
        f'<h1 id="alchemi-notebook-title" style="color:#FFFFFF;font-size:'
        f'clamp(25px,4.1vw,42px);font-weight:500;line-height:1.10;letter-spacing:'
        f'-0.02em;margin:0 0 13px;">{_escaped_text(title)}</h1>'
        f'<p style="color:#D7DCE0;font-size:14px;line-height:1.45;margin:0 0 18px;'
        f'max-width:430px;">{_escaped_text(subtitle)}</p>'
        '<div style="display:inline-flex;align-items:center;gap:9px;background:#FFFFFF;'
        'color:#111827;border-radius:8px;padding:8px 11px;font-size:11px;'
        'font-weight:800;letter-spacing:0.06em;box-shadow:0 1px 2px rgba(0,0,0,0.22);">'
        f'<span aria-hidden="true" style="display:inline-block;width:4px;height:18px;'
        f'border-radius:999px;background:{_NVIDIA_GREEN};"></span>'
        f'{_escaped_text(badge)}</div></div></section>'
    )


def lesson_summary_html(*, do: str, learn: str, need: str) -> str:
    """Return one consistent three-part lesson summary without raw bold labels."""

    cards = []
    for label, body in (("DO", do), ("LEARN", learn), ("NEED", need)):
        cards.append(
            '<div style="min-width:0;padding:12px 13px;background:#FFFFFF;'
            f'border:1px solid {_BORDER};border-top:3px solid {_NVIDIA_GREEN};'
            f'border-radius:10px;"><dt style="color:{_MUTED};font-size:11px;'
            f'font-weight:800;letter-spacing:0.09em;margin:0 0 5px;">{label}</dt>'
            f'<dd style="color:{_TEXT};font-size:13px;line-height:1.48;margin:0;">'
            f'{_escaped_text(body)}</dd></div>'
        )
    return (
        f'<dl aria-label="Lesson summary" style="max-width:{_CONTENT_WIDTH_PX}px;'
        f'margin:14px 0;display:grid;grid-template-columns:repeat(auto-fit,minmax('
        f'220px,1fr));gap:10px;font-family:{_FONT_STACK};">'
        + "".join(cards)
        + "</dl>"
    )


def figure_placeholder_html(*, title: str, description: str) -> str:
    """Return the shared visual-review slot used for intentionally open figures."""

    aria = _escaped_attribute(f"Illustration slot: {title}. {description}")
    return (
        f'<div role="img" aria-label="{aria}" style="max-width:{_CONTENT_WIDTH_PX}px;'
        f'margin:14px 0;padding:22px 24px;border:1px dashed #9CA3AF;'
        f'border-radius:{_CARD_RADIUS_PX}px;text-align:left;color:{_MUTED};'
        f'background:#F9FAFB;font-family:{_FONT_STACK};">'
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;">'
        f'<span aria-hidden="true" style="display:inline-block;width:4px;height:22px;'
        f'border-radius:999px;background:{_NVIDIA_GREEN};"></span>'
        f'<strong style="color:{_TEXT};font-size:14px;letter-spacing:0.02em;">'
        f'{_escaped_text(title)}</strong></div>'
        f'<div style="font-size:13px;line-height:1.5;">{_escaped_text(description)}</div>'
        '<div style="font-size:10px;font-weight:800;letter-spacing:0.10em;'
        'margin-top:9px;color:#6B7280;">ILLUSTRATION SLOT · VISUAL REVIEW</div></div>'
    )


def format_elapsed(seconds: float) -> str:
    """Format non-negative elapsed seconds for a compact progress card."""

    seconds = float(seconds)
    if seconds < 0:
        raise ValueError("seconds cannot be negative")
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        minutes, remainder = divmod(seconds, 60)
        return f"{int(minutes)} min {remainder:04.1f} s"
    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    return f"{int(hours)} h {int(minutes):02d} min {remainder:04.1f} s"


def _inferred_progress_state(done: int, total: int) -> ProgressState:
    if done == 0:
        return "ready"
    if done == total:
        return "complete"
    return "running"


def notebook_progress_html_string(
    *,
    title: str,
    done: int,
    total: int,
    message: str,
    elapsed_s: float,
    unit: str = "steps",
    average_label: str | None = None,
    state: ProgressState | None = None,
    width_px: int = _CONTENT_WIDTH_PX,
) -> str:
    """Return the live progress card's safe, self-contained HTML string."""

    done, total = _progress_values(done, total)
    width_px = _positive_int(width_px, name="width_px")
    state = _inferred_progress_state(done, total) if state is None else state
    status = _lookup_style(_PROGRESS_STYLES, state, name="progress state")
    completed_early = state == "complete" and done < total
    percent = 100.0 if completed_early else 100.0 * done / total
    count_text = (
        f"{done:,} {unit} used · limit {total:,}"
        if completed_early
        else f"{done:,} / {total:,} {unit}"
    )
    aria_text = _escaped_attribute(
        f"{count_text}; status {status.label}; {message}"
    )
    aria_now = done
    bar_color = "#DC2626" if state == "action" else _NVIDIA_GREEN
    average = ""
    if done and average_label:
        average = (
            f'<span style="color:{_MUTED};">avg {elapsed_s / done:.3f} s/'
            f'{_escaped_text(average_label)}</span>'
        )

    return (
        f'<section role="group" aria-label="{_escaped_attribute(title)} progress" '
        f'style="width:{width_px}px;max-width:100%;margin:10px 0;background:{_SURFACE};'
        f'border:1px solid {_BORDER};border-radius:{_CARD_RADIUS_PX}px;'
        f'padding:13px 15px;'
        f'box-sizing:border-box;font-family:{_FONT_STACK};">'
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;'
        'gap:14px;flex-wrap:wrap;margin-bottom:9px;">'
        f'<strong style="color:{_TEXT};font-size:14px;line-height:1.35;">'
        f'{_escaped_text(title)}</strong>'
        f'<span style="display:inline-block;padding:3px 9px;border-radius:999px;'
        f'background:{status.background};color:{status.foreground};font-size:11px;'
        f'font-weight:800;letter-spacing:0.05em;">{status.label}</span></div>'
        '<div style="display:flex;justify-content:space-between;gap:12px;'
        'flex-wrap:wrap;margin-bottom:7px;font-size:12px;">'
        f'<span style="color:{_TEXT};font-weight:700;">'
        f'{_escaped_text(count_text)}</span>'
        f'<span style="color:{_MUTED};">elapsed {format_elapsed(elapsed_s)}</span>'
        f'{average}</div>'
        f'<div role="progressbar" aria-label="{_escaped_attribute(title)}" '
        f'aria-valuemin="0" aria-valuemax="{total}" aria-valuenow="{aria_now}" '
        f'aria-valuetext="{aria_text}" style="height:9px;width:100%;'
        f'background:{_RAIL};border-radius:999px;overflow:hidden;">'
        f'<div aria-hidden="true" style="height:100%;width:{percent:.2f}%;'
        f'background:{bar_color};border-radius:999px;"></div></div>'
        f'<div aria-live="polite" style="color:{_MUTED};font-size:12px;'
        f'line-height:1.45;margin-top:8px;">{_escaped_text(message)}</div></section>'
    )


def notebook_progress_html(**kwargs):
    """Return an ``IPython.display.HTML`` live-progress card snapshot."""

    from IPython.display import HTML

    return HTML(notebook_progress_html_string(**kwargs))


class NotebookProgress:
    """Update a progress card in-place during a long notebook cell.

    The class prefers an ``ipywidgets.HTML`` value when widgets are available,
    then falls back to an IPython display handle.  ``auto_display=False`` keeps
    the object side-effect free for scripts and tests; call :meth:`show` later
    if a notebook display is wanted.
    """

    def __init__(
        self,
        *,
        title: str,
        total: int,
        unit: str = "steps",
        message: str = "Starting",
        average_label: str | None = None,
        width_px: int = _CONTENT_WIDTH_PX,
        auto_display: bool = True,
    ) -> None:
        self.title = str(title)
        self.total = _positive_int(total, name="total")
        self.unit = str(unit)
        self.average_label = None if average_label is None else str(average_label)
        self.width_px = _positive_int(width_px, name="width_px")
        self.started = perf_counter()
        self._done = 0
        self._message = str(message)
        # A live progress object is constructed immediately before its work
        # starts.  Render that truthfully from the first visible frame; static
        # stage cards use the separate timeless ``STAGE`` state.
        self._state: ProgressState = "running"
        self._widget = None
        self._display_handle = None
        if auto_display:
            self.show()

    @property
    def done(self) -> int:
        """Return the latest completed count."""

        return self._done

    def elapsed(self) -> float:
        """Return elapsed wall time since this progress object was created."""

        return perf_counter() - self.started

    def render_string(self) -> str:
        """Render the current state as a widget-updatable HTML string."""

        return notebook_progress_html_string(
            title=self.title,
            done=self._done,
            total=self.total,
            message=self._message,
            elapsed_s=self.elapsed(),
            unit=self.unit,
            average_label=self.average_label,
            state=self._state,
            width_px=self.width_px,
        )

    def render(self):
        """Render the current state as an ``IPython.display.HTML`` object."""

        from IPython.display import HTML

        return HTML(self.render_string())

    def show(self) -> None:
        """Display the progress card once; subsequent updates replace it."""

        if self._widget is not None or self._display_handle is not None:
            return
        try:
            import ipywidgets as widgets
            from IPython.display import display

            self._widget = widgets.HTML(
                value=self.render_string(),
                layout=widgets.Layout(
                    width=f"{self.width_px}px",
                    max_width="100%",
                    padding="0",
                    margin="0",
                ),
            )
            display(self._widget)
            return
        except Exception:
            self._widget = None

        try:
            from IPython.display import display

            self._display_handle = display(self.render(), display_id=True)
        except Exception:
            # The computational workflow remains usable in a plain Python
            # process; callers can still inspect ``render_string()``.
            self._display_handle = None

    def update(
        self,
        *,
        done: int,
        message: str | None = None,
        state: ProgressState | None = None,
    ) -> None:
        """Set the completed count and refresh the visible card."""

        done, _ = _progress_values(done, self.total)
        next_state = (
            _inferred_progress_state(done, self.total) if state is None else state
        )
        _lookup_style(_PROGRESS_STYLES, next_state, name="progress state")
        self._done = done
        if message is not None:
            self._message = str(message)
        self._state = next_state
        self._refresh_display()

    def advance(
        self,
        count: int = 1,
        *,
        message: str | None = None,
    ) -> None:
        """Advance by ``count`` completed units and refresh the card."""

        count = int(count)
        if count <= 0:
            raise ValueError("count must be greater than zero")
        self.update(done=self._done + count, message=message)

    def complete(self, message: str = "Complete") -> None:
        """Mark all units complete and refresh the card."""

        self.update(done=self.total, message=message, state="complete")

    def _refresh_display(self) -> None:
        if self._widget is not None:
            self._widget.value = self.render_string()
            return
        if self._display_handle is None:
            return
        try:
            self._display_handle.update(self.render())
        except Exception:
            try:
                from IPython.display import display

                display(self.render())
            except Exception:
                pass
