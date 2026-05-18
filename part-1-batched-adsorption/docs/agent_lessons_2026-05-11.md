# Agent Lessons for Part 1 Tutorial Finalization

Generated: 2026-05-11

This document is for future Codex/Claude/agent sessions working on
`part-1-batched-adsorption/alchemi-mace-adsorption-search.ipynb`. It records the current
decisions and traps so another agent can work efficiently without re-opening
old arguments or undoing live notebook edits.

## Current Reader-Facing Scope

- The main notebook is **Toolkit-only for now**.
- Do not reintroduce `BACKEND = "bgr_nim"`, BGR server setup cells, Docker
  service instructions, or endpoint checks into the reader-facing notebook.
- NIM may remain only as high-level ALCHEMI platform context. It is not an
  executable path in this tutorial version.
- The runnable path is: ALCHEMI Toolkit + MACE-MPA-0 + batched relaxation on a
  CUDA GPU.
- D3(BJ) is available in Toolkit workflows, but it stays disabled here because
  the OC20Dense validation data follows the non-D3 OC20 convention.

## Narrative Order

Keep the learner-facing flow:

1. Chemical discovery as a search problem.
2. Where ALCHEMI fits: Toolkit, Toolkit-Ops, GPU-native batching, pluggable
   MLIPs.
3. Why configuration search matters.
4. Choose a surface-chemistry-capable model, then verify it.
5. H2O hello-world batching and speedup.
6. OC20Dense validation checkpoint.
7. Official Toolkit API mini-guide.
8. Build adsorption examples, relax/rank/inspect.
9. Interpretation and scope limits.

The H2O speedup example comes before accuracy discussion. Validation should not
read like a separate reviewer appendix; it is part of choosing the right tool.

## User Style Preferences

- Preserve the user's wording and narrative order as anchors. Polish rough
  half-sentences; do not heavily rewrite the flow unless it is technically
  wrong.
- If a user sentence is wrong or conflicts with the notebook, flag it instead
  of silently rewriting around it.
- Keep prose user-facing. Avoid text that sounds like notes to tutorial
  creators, reviewers, or agents.
- Avoid apologetic down-selling. This is a teaching tutorial and a glimpse into
  real R&D-style workflow, not a claim of a full discovery cycle.
- Keep TODO anchors visible. They are intentional review handles, not clutter.

## Required TODO Markers

The main notebook should keep exactly these visible review markers:

- `TODO - VISUAL REVIEW`
- `TODO - REFERENCE REVIEW`
- `TODO - HUMAN REVIEW`

Do not delete them during cleanup. Do not add duplicate markers unless the user
explicitly asks for another visible review anchor.

## Notebook Editing Rules

- Use the live notebook MCP bridge for notebook edits.
- Do not raw-patch the `.ipynb` while the user may be editing in VS Code or
  Jupyter.
- Re-read the target cell immediately before editing and use the current cell
  hash/document version.
- If MCP is unavailable, say so and stop before editing the notebook.
- Safe MCP endpoint observed in this session: `http://127.0.0.1:39271/mcp`.
- Useful MCP tools: `read_notebook`, `read_cell`, `apply_text_edits`,
  `insert_cell`, `delete_cell`, `save_notebook`, `run_cell`.
- Keep reusable notebook UI in helpers. Progress strips live in
  `helpers/visualization.py` and are imported with
  `make_notebook_progress`; long-running tutorial cells should call that helper
  instead of defining inline HTML or relying on noisy `tqdm` output.
- Progress counts should be deterministic: update after a Toolkit batch,
  validation stage, or other completed unit returns. Make batching explicit in
  the progress message, but keep rendering logic out of the notebook cell.

## Validation Conventions

- `USE_PRECOMPUTED_ACCURACY_BENCHMARK = False` is the default. That means run
  the OC20Dense validation pipeline from source structures.
- `USE_PRECOMPUTED_ACCURACY_BENCHMARK = True` means inspect saved validation
  outputs.
- Do not call CSV/table loading alone a validation check when the user asks for
  validation. It is only an inspection path.
- The OC20Dense validation slice is closed-shell: H2O, NH3, and N2. It is
  chosen because neutral gas references are unambiguous.
- The active teaching panel uses CO/H2O/CH3OH on Cu(111), alpha-Al2O3(0001),
  and TiO2(110). Keep this distinction clear.
- DFT trajectory arithmetic must reproduce released OC20Dense targets before
  using the numbers in the tutorial.
- MACE adsorption energies use the explicit convention:
  `E_ads = E(adslab) - E(surface) - E(gas)`.
- Strong chemical claims still require matching reference data, DFT,
  experiment, or expert review. Keep this as a review boundary, not as a
  generic disclaimer.

## Cell Interactivity

Long cells make the notebook feel less interactive. Prefer short cells with
one purpose:

- choose settings;
- define helpers;
- run the expensive calculation;
- validate/check outputs;
- display the figure/table.

The H2O speedup, validation, clean-slab relaxation, hello-world, and grid
generation sections were split in this session so readers can execute and
inspect incrementally.

Use short inline comments to orient readers, especially when a cell is mostly
setup or display code:

- label helper cells plainly, for example `# Plotting utility...` or
  `# Validation runner...`;
- call out the core Toolkit API cells explicitly, for example
  `# ALCHEMI Toolkit API: ASE Atoms -> AtomicData -> Batch -> FIRE2...`;
- do not turn comments into internal implementation notes for tutorial
  creators;
- avoid hidden state patterns such as `globals().get(...)` in learner-facing
  cells unless there is a strong reason. Prefer explicit sequential notebook
  variables.
- do not invent aliases or wrapper names that look like official Toolkit APIs.
  Keep official Toolkit imports and class names unchanged, and name notebook
  helpers plainly, for example `atoms_to_atomic_data`, `run_h2o_batch`,
  and `relax_all_adsorption_pairs`.
- keep run-scope controls reader-facing. Use only `RUN_SCOPE = "short"` or
  `RUN_SCOPE = "full"` rather than boolean names such as `SMALL_PANEL_MODE`,
  and print how many examples and starting geometries are being run.

The active adsorption grid is intentionally not the old pure-metal-heavy panel.
Current teaching scope is one pure metal plus two oxides: Cu(111),
Al2O3(0001), and TiO2(110). Avoid adding a second pure metal unless the user
explicitly asks for a metal comparison.

The H2O batch-size comments should frame the sweep as a VRAM/throughput
calibration. Do not describe the 800-structure default as tied to one local
workstation or as a universal saturation point; larger GPUs such as cloud H100s
should be invited to extend the sweep.

For adsorption batching, do not chase full VRAM. The 2026-05-12 ws-loc
benchmark showed real Al2O3/H2O throughput plateauing around batch 24 while
VRAM kept rising linearly. Keep homogeneous chunks by adsorbate/surface pair
because Toolkit graph batching avoids coordinate padding, but FIRE2 convergence
is still batch-level and a hard structure can hold a chunk open.

Low-value plotting implementation belongs in helpers, not in long notebook
cells. Keep the notebook calls explicit with keyword arguments, for example
`plot_adsorption_energy_spread(pair_results=..., output_path=...)`, so readers
can see what data is being plotted without reading Matplotlib styling code.
Do not hide the scientific transformations: Toolkit API calls, structure/grid
construction, adsorption-energy arithmetic, ranking, and validation checks
should remain visible in the notebook.

## Helper Surface Warning

`part-1-batched-adsorption/helpers/__init__.py` still exports a broad legacy surface,
including BGR/NIM client models and routes. That is useful for older scripts and
tests, but it is too noisy for the current Toolkit-only notebook.

For the main notebook:

- import only the curated Toolkit-facing symbols needed by the tutorial;
- do not import `check_endpoint`, `run_bgr*`, `BGRAtomicData`, `BGRReply`,
  `BGRRequest`, or `BgrNimBackend`;
- do not expose BGR/NIM as an executable reader choice.

Future cleanup should consider a smaller Toolkit-only facade, for example
`helpers/tutorial_toolkit.py`, instead of exporting every legacy helper from
`helpers/__init__.py`. Do not remove legacy exports blindly: tests and support
scripts still use them.

## Current Cleanup Log

- Removed the reader-facing BGR/NIM backend setup from the main notebook.
- Replaced the backend toggle with a Toolkit-only execution label.
- Replaced the old cache wording with two explicit precomputed-output toggles:
  `USE_PRECOMPUTED_TUTORIAL_RESULTS` for the main tutorial workflow and
  `USE_PRECOMPUTED_ACCURACY_BENCHMARK` for the OC20Dense accuracy benchmark.
- Reworded the control panel and Toolkit check to describe the current path.
- Converted the accidentally-markdown notebook-housekeeping cell back to code.
- Split several long cells into smaller cells for interactivity.
- Kept ALCHEMI/NIM only as platform context in the ALCHEMI intro.

## Verification Rules

Before declaring the notebook production-ready:

- read the live notebook through MCP after edits;
- validate JSON and `nbformat`;
- parse code cells after IPython transforms, because notebook magics are
  present;
- verify the TODO marker set;
- search the main notebook for stale `BGR`, `bgr_nim`, `BACKEND`, and
  `BGR_SERVER` executable-path leaks;
- run `git diff --check`;
- if claiming runtime validation, run the actual Toolkit/OC20Dense path or
  state clearly that only static checks were run.
