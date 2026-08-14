# Shared presentation files

The active design decisions live in
[`TUTORIAL_GUIDE.md`](../TUTORIAL_GUIDE.md). This
directory contains their reusable implementation files:

- [`banner.md`](banner.md) and
  [`alchemi-banner-left.png`](alchemi-banner-left.png): exact top cell and
  course banner used by the Core playbook and future lessons;
- [`callouts.md`](callouts.md): the approved 💡 highlight and muted API
  blockquote templates;
- [`alchemi-dark.mplstyle`](alchemi-dark.mplstyle): quantitative plot style;
- [`3dmol-2.5.5/`](3dmol-2.5.5/): pinned 3Dmol.js renderer and its two
  BSD-3-Clause license files. `helpers.show_molecule` inlines this JavaScript
  into its own HTML output, so structure views need no network fetch during the
  lesson and stay interactive in an exported notebook.

These files implement decisions from the authoring guide, which remains the
single source for tutorial design.

The Core playbook uses [`alchemi-dark.mplstyle`](alchemi-dark.mplstyle) for
plots: black canvas, NVIDIA green (`#76B900`) as the primary series, blue
(`#00A3E0`) for the first comparison or highlight, light text, and restrained
gray grids.

Use it from a notebook launched at the repository root:

```python
import matplotlib.pyplot as plt

plt.style.use("shared/alchemi-dark.mplstyle")
```

Keep the quantity and comparison visible. Put repeated Matplotlib layout,
colors, labels, alternative text, and figure cleanup in a tested presentation
helper.

Plot rules:

- NVIDIA green is the main result or Toolkit path.
- Blue is the first comparison or selected highlight.
- Amber and light gray are later comparison series.
- Use direct titles and axis labels with units.
- Put explanations in the surrounding Markdown and keep each figure focused.
- Save with the black figure face color.
- Inspect every plot at notebook width and in the exported HTML.

## Progress displays

Use Rich 14.1.0 for cells with visible waiting time. Rich supplies Unicode spinners and bars, color styles, elapsed time, counts, and Jupyter rendering. In Jupyter, pass `refresh=True` when updating because automatic refresh is disabled there.

Use this column set consistently:

```python
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

progress = Progress(
    SpinnerColumn("dots", style="#76B900"),
    TextColumn("{task.description}"),
    BarColumn(
        bar_width=32,
        style="#1F2933",
        complete_style="#76B900",
        finished_style="#76B900",
    ),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)

with progress:
    task = progress.add_task("Batch evaluation", total=len(batches))
    for batch in batches:
        evaluate(batch)
        progress.update(task, advance=1, refresh=True)
```

The calculation advances the task after each completed unit. Short cells should show their result directly.
For device comparisons, put the shared operation and workload in one heading,
for example `Energy + forces · 2,048 molecules`. Keep task labels stable:
`CPU · 2,048 single-molecule calls` and `GPU · 1 batch of 2,048 molecules`.
Only the active task spins; pending tasks show a quiet dot and completed tasks
show a green check. Refresh repeated one-molecule work every 100 completed
structures. A single batched model call advances once when that call completes.
Put the shared elapsed timer on a separate line above the task rows. Timing
figures use decimal seconds on a linear axis with elapsed values above the bars.
Size the completed/total column to its widest value and right-align it so
`2,048 / 2,048` and `1 / 1` end at the same position.
