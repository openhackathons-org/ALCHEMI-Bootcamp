# Part 1 Tutorial Review — 2026-05-11 (Delta Pass)

Author: multi-agent delta review (4 parallel agents, Opus 4.7)
Scope: re-verification of the morning's findings in `docs/tutorial_review_2026-05-11.md` after a same-day fix pass. **No source files were edited.**

Framing carried over: "teaching but realistic GPU-batched simulations, with the ability to reproduce energies as a load-bearing setup for future MD."

---

## 1. Bottom line

The morning report listed 10 ranked fixes. **Six landed clean. Two are partial. Two are untouched.** Several additional 2026-05-07-baseline items also resolved in the same pass. The cross-document story is now substantially more coherent than at the start of the day — a first-time reader will get a consistent Toolkit-first / D3-off / closed-shell-validation-slice narrative from root README → Part 1 README → main notebook → OC20Dense notebook → contract, without hitting a contradiction.

**Two regressions and one new mislabel were introduced** during the fix pass:
- Cell 17 of the main notebook lost its one-sentence definitions of `AtomicData` and `Batch` (the morning §3.1 finding *was* addressed and then un-addressed by a separate edit).
- The MAD-envelope caption now calls the 0.38 eV number "D3-free MACE-MP-0/MPA-0 OC157 MAD", but `references/manual_checks.md:26-27` and `helpers/references.py:612-614` both record that 0.38 eV is the MACE-**MP-0b3**+**D3** value. The swap fixed one problem and introduced a different one in the same caption.
- Cell 5 ("Verification and quantitative uncertainty") expanded into a ~400-word text wall, the densest in the notebook, arriving at cell 5 of 58 — over-correction relative to the morning ask.

These are all sub-15-minute fixes. There is no structural regression.

---

## 2. Morning fix list — status

| # | Morning fix | Status | Evidence |
|---|---|---|---|
| 1 | Parity scatter + residual histogram in OC20Dense notebook | **Clean** | `oc20dense-accuracy-reproducibility-check.ipynb` cell ~338. Two-panel figure, y=x line, ±MAD band, per-system color + per-layer marker shape, axis labels with units. Saved to `outputs/oc20dense_known_examples/plots/oc20dense_mace_eads_parity.png` (238 KB). |
| 2 | Cell-4 reproducibility numbers wired + strip `<mark>TODO</mark>` | **Partial / Untouched** | Companion notebook is now linked from cell 5 (and cell 53). Numbers reframed as "workflow evidence" with caveat, **but**: the `<mark>TODO - REFERENCE REVIEW</mark>` tag still ships at line 144 of the .ipynb. No cell loads the 3/3 / 2/3 numbers from the companion CSV. |
| 3 | Scope-drift acknowledgment (disjoint system list) | **Clean** | Both notebooks. Main notebook line 138: "uses a closed-shell OC20Dense slice... H2O, NH3, and N2 on OC20Dense slabs. That set is **intentionally separate** from the active CO/H2O/CH3OH teaching panel." Companion notebook cell 0 mirrors. |
| 4 | Tools-at-a-glance table | **Clean** | New cell 2 of the main notebook: 5-row markdown table covering MACE-MPA-0 / AdsorbML / ASE / pymatgen / OVITO. |
| 5 | D3 envelope reconciliation | **Clean numerically, mislabeled in caption** | `MACE_MP0B3_OC157_MAD_EV = 0.38` is imported and used in main notebook lines 367, 1463, 1474, 1679-1680; in `run_toolkit_full_panel.py:27,396`; in `run_toolkit_step_diagnostics.py:26,323-324`; in `summarize_pair_validation` default at `analysis.py:387`. Caption mislabel — see §3 below. |
| 6 | Embed OVITO contact sheet | **Half-clean** | Embedded in the OC20Dense notebook at cell 16 (`outputs/ovito_dft_toolkit_pairs/dft_toolkit_selected_contact_sheet.png`) with graceful "not found" fallback. **Not** embedded in the main notebook. |
| 7 | Sync root README; resolve D3 contradiction in Part 1 README | **Clean** | Root README line 9 now reads "ALCHEMI Toolkit with MACE-MPA-0… runnable notebook keeps D3 disabled to match the non-D3 OC20Dense reference convention." Part 1 README line 80 carries the same message; the BGR table at line 87 now annotates `ALCHEMI_NIM_DFT3_ENABLED=true` as a service default to record per-dataset (caveated, not removed). |
| 8 | Move duplicated OC20Dense constants to a common module | **Clean (constants only)** | `scripts/_oc20dense_common.py` (15 LOC) holds `DEFAULT_CLOSED_SHELL_SYSTEMS`, `DEFAULT_SYSTEMS`, `CLOSED_SHELL_ADSORBATE_REFERENCES`, `MACE_RANK_BASIS`, `MACE_EADS_REFERENCE_STATUS`. All five OC20Dense scripts import from it. The drift-detection tests at `tests/test_oc20dense_benchmark.py:60-88` triangulate against the common module. **Shared utility functions** (`_safe`, `_ensure_dirs`, `_result_to_json`, `_load_result`, `_write_result`, `_env_bool`, `_file_md5`, `_command_output`, `_package_versions`) are **still duplicated across 4–6 scripts** — not in the new common module. |
| 9 | Tighten test guardrail | **Clean — went further than asked** | `tests/test_oc20dense_benchmark.py:152-159` asserts per-layer top-1 (initial=3/3, DFT-final=3/3, Toolkit=2/3), top-3, the exact 0.155246 eV *NH3 gap (162-167), and the 62/88, 42/92, 6/42 fmax-convergence rates (168-170). |
| 10 | 20-line scientific hello-world | **Partial** | New section cell 29 ("Scientific hello-world: two starts, one adsorption question") + cell 30. Two-start CO/Pd(111) calc with `assert hello_df["E_bind (eV)"].min() < 0.0`. **But** cell 30 presupposes `HOST_RELAXED`, `E_HOST`, `E_H2O_gas`, `RELAXATION_BACKEND`, plus eight helpers — it is a bridge into the panel, not a standalone primer. The cell returns a 6-column dataframe instead of one summary print. |

**Runners-up from the morning report** — also addressed:

- Cell-36 site-marker legend → **clean** (added as cell 39 markdown: ``o` top or Al-top, `s` bridge, `^` fcc, `v` hcp, `D` O-top, `P` hollow`).
- Cell-49 companion-notebook clickable link → **clean** (cells 5 and 53).
- Image-manifest cleanup → **clean** (`manifest.md:9-14` now lists exactly the four assets the notebooks embed; orphan v1–v9 trees deleted from git; survivors confined to `_review_candidates/`).
- `KCAL_MOL_TO_EV` precision → **clean** (`helpers/constants.py:11` → `0.04336411530`).
- `_tilt_angle_deg` dissociation guard → **patched downstream, not at source.** `analysis.py:353-356` now nulls `tilt_deg` when `geometry_status != "adsorbed"`. The function itself (`analysis.py:149-183`) still computes from `symbols.index("O")` — fine in practice, defensible as a layered fix.
- OC20Dense kernelspec rename → **still `python3`** (untouched).
- OC20Dense hard-coded `/home/nfedik/...` fallback in cell 2 → **still present**.

**Additionally resolved (not on morning top-10 but verified clean):**

- OVITO `Tab()` wiring bug (morning §3.3) — fixed at OC20Dense cell 19: `initial_panel = make_ovito_widget(row["initial_structure"], ...)` is now distinct from the two DFT-final panels.
- `_adsorbate_integrity_status` upgraded to all-pairs covalent-radius bond detection (`analysis.py:195-238`); cutoff `1.25·(r_i + r_j)`, dissociation if `d_final > 1.5·d_initial`. Logic correctly handles 1-atom and 2+-atom adsorbates. The morning §3.5 / 2026-05-07 §1.4 "string-indexing fragility" item is closed at the source.
- `reference_scope` and `validation_status` plumbed into per-config rows (`analysis.py:351-352`) and consumed by the notebook (`alchemi-mace-adsorption-search.ipynb:1478, 1484, 1673`); test asserts presence at `tests/test_adsorbml_analysis.py:230-231`.
- `host` and `adsorbate` are now primary columns matching `contract.py:120-139` (asdict ordering at `analysis.py:354-357`).
- `OptimizationResult` legacy field acceptance — `helpers/models.py:166-172` pre-validator. Precedence is correct: when both `optimizer_nsteps` and `num_optimization_steps` are present, the new field wins.

---

## 3. New issues introduced by the recent edits

### 3.1 Caption mislabel: "D3-free MACE-MP-0/MPA-0 OC157 MAD"

Notebook lines 1463 ("uncertainty bars only for matched references… scaled by the 0.38 eV **D3-free** MACE-MP-0/MPA-0 OC157 relative-energy MAD") and 1686 ("D3-free uncertainty bars only for matched references").

Two factual issues:

- `references/manual_checks.md:26-27` explicitly records: *"Cached arXiv v3 foundation-model supplement reports MACE-MP-**0b3+D3** at 0.38 eV MAD and 126/157 correct lowest-DFT-configuration identifications."*
- `helpers/references.py:612-614` docstring still says the value's lineage is "+D3".

So the 0.38 number is **not** D3-free; it's the MACE-MP-0b3 +D3 number used as the closer comparator for a D3-off run. That's a defensible choice. Labeling it "D3-free" contradicts the project's own pinned source.

The second mislabel — "MACE-MP-0/MPA-0" — conflates two distinct model families. The 0.38 eV value is **MACE-MP-0b3**; `MACE_MPA0_OC157_MAD_EV = 0.28` (`references.py:615`) is MPA-0. The morning fix swapped the constant correctly; the caption now blurs the model distinction the manual check preserves.

**Fix:** rewrite the caption to *"MACE-MP-0b3+D3 OC157 MAD used as the closer envelope for this D3-off run"*. Same number, honest provenance. 15 min.

### 3.2 Regression: cell 17 lost the `AtomicData` / `Batch` definitions

The 2026-05-07 §3.1 review flagged that `AtomicData` and `Batch` appear in code (cell 15) before any narrative. A fix was applied earlier but the recent edit pass trimmed those sentences. Cell 17 ("Official Toolkit API mini-guide") now opens with "The clean boundary is `ASE Atoms -> AtomicData -> Batch`" and dives into code with no definition of either term. **Regression.** 5-minute fix: restore one sentence ("`AtomicData` is the Toolkit representation of one structure ready for the model; `Batch` packs many `AtomicData` objects so the GPU evaluates them together").

### 3.3 Cell 5 over-correction: 5-layer benchmark stack stacks too much before code runs

Cell 5 ("Verification and quantitative uncertainty") is now ~400 words and stacks five concepts in sequence: five-layer benchmark stack → companion-notebook scope → 3/3 / 2/3 numbers → "workflow evidence" caveat → reference-matching TODO. This is the densest text wall in the notebook and arrives at cell 5 of 58. The morning report asked for verification clarity; the new section over-corrects. Consider splitting the five-layer enumeration into its own subsection placed *after* the hello-world, where the reader has run something concrete.

### 3.4 Cell 29 → cell 30 hello-world contract mismatch

Cell 29 promises "if we try **two** starting geometries under the same slab, gas reference, model, optimizer, and constraints, do we already see why a single start can be misleading?" Cell 30 in fact runs **one** orientation (`["C-down"]`), one rotation, one height, with `sites_filter=["top", "fcc"]` — consistent with "two starts" but a reader expecting to see two distinct starting geometries side by side would benefit from one summary print: `print(f"top: {E_top:.3f} eV; fcc-hollow: {E_fcc:.3f} eV; gap: {E_top-E_fcc:+.3f} eV")` rather than a 6-column dataframe dump. The `assert E_bind < 0` is silent on success.

### 3.5 Cell 51 vs cell 53 "central measurement" framing collision

Cell 51 calls the top-site-vs-batch comparison "**the tutorial's central measurement**." Cell 53 then frames the takeaways as a three-way split: "Search effect / Verification effect / …". Either cell 51 is the central measurement or cell 53's three-way split is the takeaway; readers will not know which.

### 3.6 BGR detail leak in cell 4

Cell 4 ("Toolkit path and BGR NIM path") now explains the BGR service image, `/v1/infer`, and the four `ALCHEMI_NIM_*` knobs **before** any Toolkit code has run, even though cell 6 then says "If `BACKEND = "toolkit"`, skip this setup section." A reader on the default Toolkit path reads three paragraphs about a service they will not start. Move the BGR knobs to a collapsible aside, or to cell 6 itself.

### 3.7 New script `benchmark_h2o_saturation.py` — three minor issues

193 LOC, developer-only utility (picks a max batch size for the notebook H2O hello-world):

- `:82` — `getattr(optimizer, "step_count", n_steps)` silently falls back to the configured budget if the optimizer doesn't expose `step_count`. Misreports "everything converged at the budget" when the attribute is missing. Replace with an explicit assertion or a recorded flag. 5 min.
- `:100` — default `--output` path is `/tmp/h2o_saturation_ws_loc.json`. CLAUDE.md says scripts should use `$TMPDIR`, not `/tmp` directly. Filename suffix `_ws_loc` is opaque.
- `:151-161` — OOM event writes one row and `return 0`; calling shell sees success. Return a distinct nonzero (e.g. 2).
- No type hints, no per-function docstrings — inconsistent with the rest of `scripts/`.

No P0 issues. No unit slips, no sign errors, no hardcoded user paths.

### 3.8 OC20Dense parity-plot polish

Plot is real and tells the headline story. Three small improvements would convert it from "evidence that something is happening" into "evidence of what is happening":

- The per-system biases (*OH2 underbinds ≈ 0.5 eV; *NH3 overbinds ≈ 0.4 eV; *N2 near-parity) are visible from cluster positions but **not annotated**. Three short text labels at the cluster centroids would close this. Inputs already computed in `recomputed_eads_summary`.
- Legend overlaps the *NH3 / *N2 high-energy points at `loc="best"`, fontsize 7. Move to `loc="lower right"` or shrink to fontsize 6.
- `ax_scatter.set_aspect("equal")` is missing; the y=x diagonal renders at ~35° instead of 45°, which subtly degrades parity readability.

---

## 4. Items still untouched from morning report

- **`<mark>TODO - REFERENCE REVIEW</mark>` at .ipynb line 144** (cell 5). Five-minute fix. Flagged twice in one day. The single highest-impact "feels finished" edit remaining.
- **Cell 14 vs cell 15 near-duplicate prose** — both open "Before … start with…" with the same H2O batch content. Merge.
- **OC20Dense kernelspec is `python3`** — violates the project's named-ipykernel discipline (per `MEMORY.md`).
- **Hard-coded `/home/nfedik/projects/tutorials/part-1-nim`** fallback in OC20Dense cell 2.
- **OVITO contact sheet not embedded in the main notebook** (only in companion).
- **Duplicated utility functions across scripts** (`_safe`, `_ensure_dirs`, `_result_to_json`, `_load_result`, `_write_result`, `_env_bool`, `_file_md5`, `_command_output`, `_package_versions`) — 4-6 byte-identical copies each. Constants got the common-module treatment; utilities didn't.
- **`tutorial_status_and_plan.md` dated 2026-05-04** with a false claim ("the root README and `part-1-nim/README.md` still describe the older atmospheric water harvesting tutorial") that no longer holds. Either refresh or move to `_archive/`.
- **`references/manifest.yml:13` `last_reviewed: 2026-05-04`** — pre-dates the D3 swap and the Toolkit-first pivot. Bump or annotate.
- **`references/manual_checks.md:15-21`** AdsorbML 50% / 87% reliability rows still `[ ]` despite the 2026-05-07 audit recording the citations verbatim. Restricted-PDF row 29-34 and geometry checks 96-103 also `[ ]`.
- **Contract gap:** `shared/adsorption_tutorial/contract.py:103-110` keeps NH3 as `active=False` while the companion notebook validates against `*NH3`. `panel.yml:42-45` mirrors. `N2` is absent from `contract.py` and `panel.yml` entirely, yet is a benchmark row in the companion.
- **Duplicate `E_bind_eV` / `E_bind (eV)` columns** at `helpers/analysis.py:349-350` — intentional dual-publishing (contract field + plot-friendly alias). Defensible but document.
- **God module `run_oc20dense_known_examples.py`** grew to 1164 LOC (+7 from the import block and one elif branch logging the LD-reexec skip). No data/reporting split. **Recommended priority drop** — the constants are now deduplicated, tests are real, and the bulk is honest report-generation. Keep on the P2 radar; don't gate a deliverable on it.

---

## 5. Consolidated priority list — delta-aware

Ranked by impact-per-effort across all four delta agents. Rows marked **R** are regressions or new mislabels from the recent fix pass; **U** are untouched morning items.

| # | Fix | Why | Effort |
|---|---|---|---|
| 1 **U** | Remove `<mark>TODO - REFERENCE REVIEW</mark>` from .ipynb line 144 (cell 5). | Flagged twice; single visible TODO in shipped path; disproportionate "feels finished" impact. | 1 min |
| 2 **R** | Rewrite cell 41 / cell 50 caption: "D3-free MACE-MP-0/MPA-0 OC157 MAD" → "MACE-MP-0b3+D3 OC157 MAD used as the closer envelope for this D3-off run". Also at line 1686. | The current caption contradicts `references/manual_checks.md:26-27` and `helpers/references.py:612-614`. Honest provenance, same number. | 15 min |
| 3 **R** | Restore one-sentence definitions of `AtomicData` and `Batch` to cell 17. | Regression from a separate edit. Names appear in code before any narrative. | 5 min |
| 4 | Trim cell 30 hello-world to a single summary `print(...)` of the two-start comparison; replace dataframe with `top: X.XXX eV; fcc-hollow: Y.YYY eV; gap: Z.ZZZ eV`. | Cell promises "two starts"; current output is a 6-column dataframe. Converts the section from "bridge into panel" to the anchor calculation the 2026-05-07 §1.2 asked for. | 20 min |
| 5 | Annotate per-adsorbate biases on the OC20Dense parity plot ("*OH2 underbinds ≈0.5 eV" etc.) + `ax_scatter.set_aspect("equal")` + tighten legend placement. | Turns the plot from "evidence that something is happening" into "evidence of what is happening". Inputs already computed. | 15 min |
| 6 | Three small fixes in `benchmark_h2o_saturation.py`: silent-fallback (:82), `/tmp` → `$TMPDIR` (:100), nonzero exit on OOM (:151). | New script with three P1-ish issues. | 10 min |
| 7 **U** | Rename OC20Dense kernelspec to a named project ipykernel; remove the `/home/nfedik/...` fallback in cell 2. | Reproducibility blockers per `MEMORY.md`. | 5 min |
| 8 | Reconcile contract: add N2 as a benchmark adsorbate in `panel.yml` and `contract.py`; flip NH3 to a clearly-labeled "benchmark-only" tier (not `active=False` while the companion validates against it). | Closes the residual scope drift between contract and OC20Dense check. | 20 min |
| 9 **U** | Move shared script utilities to `scripts/_common.py`: `_safe`, `_ensure_dirs`, `_result_to_json`, `_load_result`, `_write_result`, `_env_bool`, `_file_md5`, `_command_output`, `_package_versions`. | Constants split has proven the pattern; utilities pass next. | 1 h |
| 10 **U** | Refresh `tutorial_status_and_plan.md` (or move to `_archive/`) and bump `references/manifest.yml.last_reviewed`. Tick the AdsorbML 50%/87% rows in `manual_checks.md`. | Three governance documents now stale relative to source state. | 15 min |
| 11 | Embed `outputs/ovito_dft_toolkit_pairs/dft_toolkit_selected_contact_sheet.png` in the main notebook (already in companion). | Strongest visual evidence in the repo is only in one of the two notebooks. | 5 min |
| 12 | Restructure cell 5 to split the five-layer benchmark stack off into a later subsection. Move BGR knobs out of cell 4 into cell 6's setup block. | Cell 5 / cell 4 are dense text walls before the reader has run code. | 30 min |
| 13 | Collapse cells 14 / 15 near-duplicate prose into one markdown intro to the H2O batch. | Minor narration friction; flagged on 2026-05-11 morning. | 5 min |
| 14 | Resolve cell 51 vs cell 53 "central measurement" framing collision. | Two competing phrasings of the same takeaway. | 5 min |

**Summary timing:** items 1, 3, 6, 7, 11, 13, 14 are all sub-15-minute edits that close most of the remaining surface drift. Items 2, 4, 5, 8, 10, 12 are 15–30-minute edits with disproportionate scientific honesty / clarity impact. Item 9 is the largest remaining engineering item but has no end-user impact and can be done independently.

---

## 6. Appendix — file references (delta)

Most-touched in the next fix pass:

- `/home/nfedik/projects/tutorials/part-1-nim/alchemi-mace-adsorption-search.ipynb` — cells 4, 5 (line 144 `<mark>TODO</mark>`), 14/15, 17, 29/30, 41 (line 1463), 50 (line 1686), 51, 53.
- `/home/nfedik/projects/tutorials/part-1-nim/oc20dense-accuracy-reproducibility-check.ipynb` — cell 2 (kernelspec, hardcoded path), parity plot annotations.
- `/home/nfedik/projects/tutorials/part-1-nim/scripts/benchmark_h2o_saturation.py:82, 100, 151`.
- `/home/nfedik/projects/tutorials/part-1-nim/scripts/_common.py` — to create.
- `/home/nfedik/projects/tutorials/shared/adsorption_tutorial/contract.py:103-110` and `panel.yml:42-45` — NH3 / N2 reconciliation.
- `/home/nfedik/projects/tutorials/part-1-nim/references/manifest.yml:13` (`last_reviewed`).
- `/home/nfedik/projects/tutorials/part-1-nim/references/manual_checks.md:15-21, 26-27, 96-109`.
- `/home/nfedik/projects/tutorials/tutorial_status_and_plan.md` — refresh or move to `_archive/`.

End of delta report.
