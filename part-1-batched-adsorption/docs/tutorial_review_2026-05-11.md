# Part 1 Tutorial Review — 2026-05-11

Author: multi-agent review pass (5 parallel agents, Opus 4.7)
Scope: Part 1 only (`part-1-nim/`), with cross-references to root README, `shared/adsorption_tutorial/`, and `tutorial_status_and_plan.md`.
Complements (and updates) `docs/tutorial_review_2026-05-07.md`. **No source files were edited.**

Framing applied throughout: the user has stated the tutorial is "teaching but realistic GPU-batched simulations, with the ability to reproduce energies as a load-bearing setup for future MD." Findings are graded against that goal, not a generic rubric.

---

## 1. Executive summary

**What's working now (since 2026-05-07):**

- Headline MAD constant drift is resolved. `helpers/references.py:615` reads `MACE_MPA0_OC157_MAD_EV = 0.28` (was 0.42), parity-plot error bars are correct, and `manual_checks.md:23–27` retracts the old value.
- The `strict_parity_subset` guard is now wired (main notebook cell 44).
- Adsorbate dissociation detection now uses covalent radii (`helpers/analysis.py:216–236`) instead of plain symbol-string indexing.
- A real chemistry-bearing "H2O Toolkit batch" warm-up section (main notebook cells 13–19) replaced the prior throughput-only sweep.
- The reference manifest discipline (`context` / `near-strict` / `strict`) is honest, unusually disciplined, and continues to hold.

**What blocks "feels finished":**

1. **The reproducibility story is asserted, not demonstrated.** Main notebook cell 4 quotes "top-1 within 0.10 eV for 3/3 systems" but no cell in that notebook computes the number. The companion `oc20dense-accuracy-reproducibility-check.ipynb` produces **zero plots** — every output is a pandas dataframe. Given the user's stated goal ("if the reader can't trust the energies, they can't trust the MD"), this is the highest-impact gap in the repo.
2. **Scope drift between the main panel and the reproducibility check, unacknowledged.** Active panel: CO/H2O/CH3OH × Cu(111)/Pd(111)/α-Al2O3(0001). OC20Dense check: *OH2/*NH3/*N2 on OC20 slabs — **disjoint** systems. The OC20Dense set was also explicitly gated to the *unambiguous-reference closed-shell slice* (`run_oc20dense_known_examples.py:390–423`), excluding every hard case (*COH, *CH3, *NO). The notebook calls this the "strict benchmark set" without naming the selection bias.
3. **A reader-facing `<mark>TODO - REFERENCE REVIEW</mark>` HTML tag still renders in cell 4** of the main notebook. Five-minute fix, disproportionate "feels finished" impact.
4. **The MAD constant is from a D3-enabled evaluation applied to a D3-off run.** 0.28 eV (MACE-MPA-0+D3, OC157) is the correct *number* but the wrong *envelope* for a D3-disabled tutorial. `MACE_MP0B3_OC157_MAD_EV = 0.38` is already imported but unused.
5. **The strongest visual evidence (`outputs/ovito_dft_toolkit_pairs/`) is unembedded** in either notebook. A side-by-side DFT-vs-Toolkit contact sheet exists on disk and is never shown to the reader.

**Verdict:** structurally close to a high-quality tutorial. Methodology is sound; the helper layer is clean. Blocking items are (a) wiring the reproducibility evidence into both notebooks visually, (b) acknowledging the OC20Dense scope and selection bias, and (c) reconciling the D3 / no-D3 envelope choice. None require new infrastructure — only deliberate edits to existing cells and one parity plot.

**Top-3 highest-leverage edits** (drawn from across the angle reviews; ranked impact-per-effort):

1. **Add one parity scatter + residual histogram to `oc20dense-accuracy-reproducibility-check.ipynb`.** All inputs already exist (`eads["dft_adsorption_energy_target_eV"]`, `recalc_mace_dft_final_eads_eV`, `recalc_mace_relaxed_eads_eV`). Single scatter (y=x, ±MAD band) + one histogram converts the notebook from "table audit" to visible evidence and directly serves the MD-trust framing. **~30 min.**
2. **Replace cell 4's asserted "3/3 / 2/3" numbers with either a loader from the companion notebook's saved CSV or a clickable `[oc20dense-accuracy-reproducibility-check.ipynb](...)` link plus one sentence framing the scope** (closed-shell, OC20Dense slabs, disjoint from the active panel). Strip the `<mark>TODO</mark>` tag in the same edit. **~20 min.**
3. **Add a 5-row tools-at-a-glance markdown table between cells 1 and 2 of the main notebook** covering MACE-MPA-0, AdsorbML, ASE, pymatgen, OVITO — one sentence each. Cell 1 already does this for the ALCHEMI surface; the science tools still appear in code before introduction. **~30 min.**

---

## 2. Status of prior-review (2026-05-07) findings

| Item | Status | Evidence |
|---|---|---|
| §1.1 MAD `0.42 → 0.28` drift | **Resolved** | `helpers/references.py:615`; main notebook cell at line 1344. |
| §1.2 No scientific hello-world | **Open** | `SMALL_PANEL_MODE=True` runs the panel slice on one pair; not a 20-line one-pair primer. |
| §1.3 Tools-at-a-glance | **Partial** | ALCHEMI surface introduced (cell 1); MACE-MPA-0 / AdsorbML / ASE / pymatgen / OVITO still un-introduced. |
| §1.4 Adsorbate-chemistry inference from symbol strings | **Partial** | `_adsorbate_integrity_status` now uses covalent radii (`analysis.py:216–236`). But `_tilt_angle_deg` (`analysis.py:149–183`) still does `symbols.index("O")` without consulting the dissociation flag. |
| §1.5 `strict_parity_subset` guard unused | **Resolved** | Called in main notebook cell 44. |
| Industrial-adoption claims (§11/12) | **Apparently addressed** | Offending paragraphs not found in current prose. But `references/manual_checks.md:15–21` still lists AdsorbML 50/87 rows as `[ ]` despite the 2026-05-07 audit recording p.1, p.3, p.5 verbatim. |

`tutorial_status_and_plan.md` itself is dated 2026-05-04 and is now obsolete in non-trivial parts (line 25 still describes both READMEs as carrying the older atmospheric-water-harvesting pivot).

---

## 3. Findings by angle

### 3.1 Pedagogy and scientific narration (main notebook)

`/home/nfedik/projects/tutorials/part-1-nim/alchemi-mace-adsorption-search.ipynb` (54 cells)

**Works well**
- Cell 0 framing: "chemical discovery as a search problem" puts the model inside a workflow before any code.
- Cell 4 ("Verification and quantitative uncertainty") names a five-layer benchmark stack (Generated panel / Initial-coordinate SP / DFT-final SP / Toolkit relaxation / Defined MACE adsorption energies). That scaffold is exactly what the reproducibility-for-future-MD argument requires.
- Cell 5 gates the BGR setup behind the backend choice — eliminates a stumbling block for the default Toolkit reader.
- Cell 53 (Scope limits) is one of the strongest closing markdown blocks in the repo.

**Open items**

- **Cell 4 quotes reproducibility numbers ("top-1 within 0.10 eV for 3/3 systems; Toolkit relaxation top-1 within 0.10 eV for 2/3 systems and top-3 for 3/3 systems") without a runnable cell, table reference, or link to the companion notebook.** This is the highest-impact narrative weakness.
- **Cell 4 still renders `<mark>TODO - REFERENCE REVIEW</mark>`** — visible to the reader as a yellow-highlighted HTML tag.
- **No tools-at-a-glance for MACE-MPA-0 / AdsorbML / ASE / pymatgen / OVITO** before they appear in code (cells 1–16).
- **Cell 16 introduces "ASE Atoms → AtomicData → Batch" as "the clean boundary" without conceptually defining `AtomicData` or `Batch`** — first-time readers see these names in code (cell 15) before any narrative.
- **Cell 36 site-marker scheme (`top: o, bridge: s, fcc: ^, hcp: v, al-top: o, o-top: D, hollow: P`) has no key in markdown.** Readers without FCC(111)-nomenclature familiarity cannot decode the dot plot.
- **Cell 41 prose says "scaled by the 0.28 eV MACE-MPA-0+D3 OC157 relative-energy MAD"** while cell 6 explained D3 is disabled. Needs a one-clause caveat or a different envelope (see §3.5).
- **Cell 49 references the companion OC20Dense notebook generically** without a clickable link.
- **No scientific hello-world before the panel.** The H2O batched-speedup section (cells 13–19) is real H2O chemistry but teaches batching mechanics, not one scientific adsorption end-to-end. A 20-line "CO on Pd(111) at top + fcc-hollow; assert E_bind<0" cell before cell 28 would close §1.2 of the 2026-05-07 review.

Minor narration friction: cells 13 and 14 markdown are near-duplicates ("Before building surfaces..." / "Before the adsorption search..."); cell 19 is a floating paragraph after two code cells; cell 22 is a one-line heading with no context.

### 3.2 OC20Dense reproducibility — scientific rigor

`/home/nfedik/projects/tutorials/part-1-nim/oc20dense-accuracy-reproducibility-check.ipynb` (19 cells) + 6 supporting scripts.

**The check is mechanically sound:**
- Sign convention `E_ads = E(adslab) - E(surface) - E(gas)` is consistent end-to-end (`run_oc20dense_mace_adsorption_energies.py:514–523`).
- Surface reference choice is methodologically correct — MACE single-point on the official OC20Dense DFT-final clean-slab trajectory (`run_oc20dense_mace_adsorption_energies.py:289–300`); script warns against the alternative at lines 13–16, 379–381, 467–471.
- Frozen-layer scheme reads tags directly from the OC20Dense LMDB (`run_oc20dense_known_examples.py:293–305`) — no silent mismatch.
- No D3, matching OC20's PBE convention.
- DFT reference arithmetic round-trips to ≤ 1e-12 eV (`tests/test_oc20dense_benchmark.py:190–204`).

**Methodologically weaker than the framing implies:**
- **N = 3 systems** dominates everything. Aggregate top-1 success is 2/3 or 3/3 — coin-flip resolution.
- **Closed-shell selection bias.** Selection enforces `{*OH2, *NH3, *N2}` (`run_oc20dense_known_examples.py:390–423`), exactly the OC20Dense entries where the reference convention is unambiguous *and* where MACE-class foundation models do best. Hard cases (radical *COH, dissociative *CH3, charge-transfer *NO) are explicitly excluded. The notebook calls this "strict benchmark set" without acknowledging that it is the unambiguous-reference slice, not a representative slice.
- **Empirical biases tell a story.** `mace_adsorption_energy_summary.csv`: *OH2 MAE / bias = 0.518 / −0.518 eV; *NH3 = 0.404 / +0.404 eV; *N2 = 0.064 / −0.063 eV. Biases are large *and not the same sign*, so the aggregate MAE hides everything. No residual or parity plot surfaces this.
- **Backend convergence is poor and not flagged.** `relaxed_backend_converged`: 62/88 (*OH2), 42/92 (*NH3), **6/42 (*N2)** at the 0.05 eV/Å fmax / 200-step budget (`accuracy_layer_summary.csv`). Only 14 % of *N2 configs converge.
- **Test guardrail is set to the always-passing bar.** `tests/test_oc20dense_benchmark.py:127–128` asserts `top3_success_0p10eV == "3/3"` only. Actual Toolkit top-1 is **2/3** (*NH3, gap 0.155 eV). The top-1 success threshold is never asserted.
- **"Strict benchmark set" wording contradicts the rigor taxonomy** — the underlying MACE manifest entry is `status: context` (`references/manifest.yml:21`).
- **No bridge to MD.** The user's stated goal ("trust the energies → trust the MD") is articulated nowhere. The notebook does *not* license off-equilibrium MD on the same potential, but the framing implies it does.

**Reproducibility of the check itself:**
- Hard-coded `/home/nfedik/projects/tutorials/part-1-nim` fallback in cell 2.
- Kernelspec is `python3`, violating the project's named-ipykernel discipline.
- No auto-fetch of the 15–20 GB OC20Dense archives. URLs in `OFFICIAL_SOURCES` (`run_oc20dense_known_examples.py:112–123`) are metadata only.
- Cell 3 asserts all saved tables exist — the notebook is a "review the saved artifacts" workflow, not a "reproduce from zero" workflow. The introduction does not state this.
- **Misleading script name:** `scripts/run_oc20dense_dft_final_single_points.py` does **not** run DFT — it runs MACE single points on DFT-final geometries. Docstring is honest; filename invites misreading.
- The **live subset recompute (cells 11–13)** with explicit tolerance dict is the most credible reproducibility argument in the notebook and should be elevated narratively.

### 3.3 Plots and figures

**Inventory.** Only **4 static images** ship across the two notebooks (`banner_adsorbml_bgr.svg`, `v0_core/discovery_funnel.png`, `v0_core/alchemi_toolkit_community_ops.png`, `v0_core/workflow_adsorbml_bgr_ovito_v2.png`). All of `assets/images/v1/`, `v2_split/`, `v3_illustration/`, `v4_icon_style/`, `v5/`, `v6/`, `v7/`, `v8/`, `v9/` are orphan (~40 PNGs). `assets/images/manifest.md:16` claims `discovery_funnel_v2.png` is "current" but the v1 ships — manifest is stale.

**Inline-plot issues in the main notebook:**
- `h2o_toolkit_batch_speedup.png` (cell 15): speedup ≈ 0.7 at batch=2 and ≈ 1.0 at batch=4 — both below the "ideal linear" reference. Prose says "batching amortizes fixed work"; plot disagrees at small batches with no caption explaining the dip. `suptitle` and right-panel title collide.
- `binding_distribution.png` (cell 36): legends use `bbox_to_anchor=(1.01, 0.5)` with `tight_layout()`; PNG at dpi=150 clips legend text — the site-marker shape encoding is unreachable.
- `discovery_plot.png` (cell 50): caption advertises "uncertainty bars only for matched references" but **no bar renders** in the entire figure, because no `helpers/references.py` row has `reference_scope ∈ {"strict","near-strict"}`. The MAD-band code path is currently dead.
- `adsorbml_bias.png` (cell 48): headline message ("batching reveals lower-energy states") contradicts the data — 6 of 9 pairs show Δ ≈ 0 meV. Only 3 pairs (CO/Pd, CO/Al2O3, CH3OH/Pd) show a non-trivial bias. The figure works against its own narrative.

**OC20Dense notebook: zero plots.** Every output is a pandas dataframe or an `ipywidgets.Output`. For the MD-trust framing this is the single weakest link in the repo.

**OVITO renders:** the side-by-side DFT-vs-Toolkit contact sheet (`outputs/ovito_dft_toolkit_pairs/dft_toolkit_selected_contact_sheet.png` + per-pair PNGs) is *strong* visual evidence — readable annotations (`DFT_rank 1 | gap 0.0000 eV | active RMSD 0.275 A | ads RMSD 0.246 A`). It is **never embedded** in either notebook.

**Bug:** in `oc20dense-accuracy-reproducibility-check.ipynb` cell 16, `dft_panel_initial` and `dft_panel_relaxed` are both built from `row["dft_final_structure_path"]` — the "Initial vs DFT" tab actually shows DFT-final twice. Labels happen to read right by accident, but the data wiring is wrong.

### 3.4 Helpers and scripts — code quality

**Helper layer (`helpers/`, ~3,490 LOC) is in good shape for a tutorial.** Docstring coverage 96 %+ on public funcs; type hints widely present; no circular imports; no mutable defaults; energy sign convention centralized in one well-documented function (`helpers/analysis.py:73`); silent-fallback patterns confined to one async density helper that warns (`helpers/api_client.py:410`). Layering is clean (constants → models → cache/analysis/api_client/surfaces → config_search/oxide_slabs/metal_slabs).

**Concrete issues, severity-ranked:**

P0 — correctness risk
- **Module-level constants duplicated across four scripts.** `DEFAULT_CLOSED_SHELL_SYSTEMS`, `DEFAULT_SYSTEMS`, `CLOSED_SHELL_ADSORBATE_REFERENCES`, `MACE_RANK_BASIS`, `MACE_EADS_REFERENCE_STATUS` defined identically in `scripts/run_oc20dense_known_examples.py:100–110` and `scripts/oc20dense_dft_reference_checks.py:39–47`, re-imported by two more. `tests/test_oc20dense_benchmark.py:39–60` exists specifically to detect drift — the team already knows this is fragile. Fix: move to `scripts/_oc20dense_common.py`.
- **`KCAL_MOL_TO_EV = 0.0434` is imprecise** (`helpers/constants.py:11`). Correct value 0.04336411530. ~0.1 % drift, ~1 meV per binding energy. Currently unused in OC20Dense flow but exported as public API.
- **`os.execv` re-exec gated by `OC20DENSE_LD_REEXEC`** (`run_oc20dense_known_examples.py:73–89`) skips the LD path fix silently if the env var leaks across invocations. Tests set this on purpose (line 18 of the test file). At minimum, log when the re-exec is skipped.
- **Hardcoded user-specific conda paths** in `scripts/render_ovito_molecule_references.py:22–24` (`/home/nfedik/miniconda3/...`).

P1 — maintainability
- **`run_oc20dense_known_examples.py` is a god module** at 1,157 LOC, `main()` is 188 lines with 5-level nesting at 1000–1100. `_write_summary_tables` (127 lines), `_write_report` (143 lines). Suggested split: `oc20dense_loader.py` (LMDB+selection) / `oc20dense_compute.py` (relax/SP chunks) / `oc20dense_reporting.py` (tables+report+metadata) / thin driver.
- **`run_oc20dense_mace_adsorption_energies.py` `main()`** also 188 lines; `_surface_references` (74 lines) does three things.
- **Duplicated utility functions** (`_safe`, `_ensure_dirs`, `_result_to_json`, `_load_result`, `_write_result`, `_env_bool`, `_file_md5`, `_command_output`, `_package_versions`) byte-identical across four scripts. Fix: `scripts/_common.py`.
- **Missing docstrings on `ToolkitBackend` private methods** in `helpers/relaxation_backends.py:391–562` — exactly where a learner spends the most time.
- **Inconsistent imports in `helpers/surfaces.py`** — line 10 uses `from helpers.models import ...` while line 405 uses `from .models import ...`.

**Test coverage gaps:**
- No direct unit test for `_relax_chunk` / `_single_point_chunk` (the compute paths).
- No test for the *sign* of `mace_dft_final_eads_eV` against a known DFT example. The round-trip test verifies internal arithmetic; a (slab + gas) − total bug would still pass.
- No adversarial test for `_member_matches` (prefix collisions like "rand4" vs "rand42", trailing slashes).

### 3.5 Cross-cutting consistency

**Root README is out of step with Part 1.** Root `README.md:9`: "BGR NIM with MACE-MPA-0 and DFT-D3(BJ) dispersion." `part-1-nim/README.md:9–13`: "MACE-MPA-0 through the Toolkit. DFT-D3(BJ) … disabled here." Notebook cell 7 confirms `BACKEND = "toolkit"`, `TOOLKIT_REQUIRE_D3BJ = False`.

**D3 mixed signal inside Part 1.** `part-1-nim/README.md:83` BGR table lists `ALCHEMI_NIM_DFT3_ENABLED = true` while §13 and notebook cells 3/6/7 disable D3 to match OC20Dense. The Docker-stack table is the BGR service default; the runnable Toolkit path overrides; the two paragraphs never reconcile.

**MAD constant is from a D3-enabled evaluation, applied to a D3-off run.** Cell 41 reads "scaled by the 0.28 eV MACE-MPA-0+D3 OC157 relative-energy MAD." The constant is correct against arXiv v3 p.122 — but the run disables D3. `MACE_MP0B3_OC157_MAD_EV = 0.38` is already imported (`helpers/references.py:616`, main notebook cell at line 347) and is the closer D3-free comparator. Either swap the envelope or add a one-clause caveat in cell 41.

**Scope drift: OC20Dense check operates on a disjoint system list.** Main panel: CO/H2O/CH3OH × Cu(111)/Pd(111)/α-Al2O3(0001) (`contract.py:113–117`). OC20Dense set: *OH2/*NH3/*N2 on OC20 slabs.
- H2O appears in both but on **different slabs**.
- NH3 is `active=False` in `contract.py:103–110` ("optional first-binding context only") yet is a benchmark row in the OC20Dense check.
- N2 is not in the contract at all.
- CO and CH3OH (main-panel adsorbates) are absent from the OC20Dense set; cell 4 of the OC20Dense notebook even asserts `not contains("COH|CH3")`.

No prose anywhere states this. The OC20Dense check is framed as additional validation of the active panel; it is actually validation of adjacent closed-shell chemistry on different surfaces.

**Schema drift between contract and helper:** `contract.py:120–139` requires `host`, `adsorbate`, `E_bind_eV`, `reference_scope`, `validation_status` as top-level columns. `helpers/analysis.py:336–352` emits `pair = "{adsorbate}/{host}"` instead, plumbs `host` / `adsorbate` only via `asdict(site)`, has duplicate `E_bind_eV` and `E_bind (eV)` columns, and omits `reference_scope` / `validation_status` from the per-config row. Field-name drift for the same physical quantity: `E_ads` (contract `backends.md:54`), `E_bind_eV` (contract `contract.py:128` and helper), `E_bind (eV)` (notebook plots).

**`_tilt_angle_deg` still computes from `symbols.index("O")` without a dissociation guard** (`helpers/analysis.py:149–183`) — the dissociation flag now exists in integrity status but is not consulted by tilt.

**Stale `manual_checks.md`:** AdsorbML 50 % / 87.36 % rows still `[ ]` (lines 15–21) despite the 2026-05-07 audit recording verbatim citations. Restricted-PDF and geometry sub-rows also untouched (97–109).

---

## 4. Consolidated priority list (top 10 fixes)

Ranked by impact-per-effort across all five review angles. Same items may appear in multiple sections above.

| # | Fix | Effort | Why it matters |
|---|---|---|---|
| 1 | Add a parity scatter (DFT vs MACE, y=x, ±MAD band) and a residual histogram to `oc20dense-accuracy-reproducibility-check.ipynb`. All inputs already computed. | 30 min | Converts the notebook from "table audit" to visible evidence. Directly serves the MD-trust framing. |
| 2 | Replace cell-4 asserted numbers in the main notebook with a loader from the companion CSV or a clickable link + one-sentence scope framing. Strip `<mark>TODO - REFERENCE REVIEW</mark>` in the same edit. | 20 min | Eliminates the "asserted but not shown" gap and the visible-TODO. |
| 3 | Add a paragraph to both notebooks acknowledging the OC20Dense system list is **disjoint** from the active panel and is the **unambiguous-reference closed-shell slice**, not a representative slice. Rename "strict benchmark set" → "closed-shell benchmark slice" (the manifest entry is `status: context`). | 15 min | Stops overpromising and stops scope drift from corrupting the validation narrative. |
| 4 | Add a 5-row tools-at-a-glance markdown table between cells 1 and 2 of the main notebook: MACE-MPA-0, AdsorbML, ASE, pymatgen, OVITO — one sentence each. | 30 min | Closes the 2026-05-07 §1.3 gap that is still partially open. |
| 5 | Reconcile the D3-on MAD with the D3-off run. Either swap to `MACE_MP0B3_OC157_MAD_EV = 0.38` (already imported, never used), or add a one-clause caveat in cell 41 / cell 50 captions explaining the envelope is from a D3-enabled MPA-0 evaluation. | 15 min | Removes a hidden methodological mismatch in the headline uncertainty bar. |
| 6 | Embed `outputs/ovito_dft_toolkit_pairs/dft_toolkit_selected_contact_sheet.png` (or one or two per-pair PNGs) in the OC20Dense notebook as static fallback visual evidence. Fix the cell-16 OVITO Tab() bug (`dft_panel_initial` uses the wrong path). | 20 min | The strongest visual evidence in the repo is currently unembedded; the widget that *is* shown has a wiring bug. |
| 7 | Sync root `/home/nfedik/projects/tutorials/README.md` with the Toolkit-first / D3-off pivot. Resolve the `ALCHEMI_NIM_DFT3_ENABLED=true` vs "D3 disabled" contradiction in `part-1-nim/README.md`. | 20 min | Anyone landing on the root README sees an outdated backend story. |
| 8 | Move duplicated OC20Dense module-level constants and shared script utilities to `scripts/_oc20dense_common.py` / `scripts/_common.py`. Remove the regression tests that exist only to detect the drift. | 1–2 h | Eliminates a known-fragile pattern in the newest pipeline; reduces the 1,157-line god module. |
| 9 | Tighten `tests/test_oc20dense_benchmark.py:127–128`. Also assert each layer's top-1 explicitly, the actual *NH3 relaxation gap, and the fmax-convergence rates (especially the 6/42 *N2 number). | 30 min | Current guard is set to the always-passing bar. |
| 10 | Add a 20-line scientific hello-world (CO on Pd(111), top + fcc-hollow, `assert E_bind < 0`, print value) between cells 26 and 28 of the main notebook. Closes 2026-05-07 §1.2. | 30 min | Gives the reader a single anchor scientific calculation before the 9-pair panel. |

**Runners-up (lower priority, mention in same pass):** cell-36 site-marker legend, cell-49 clickable companion-notebook link, cell-15 H2O speedup figure cleanup, `KCAL_MOL_TO_EV` precision fix, `_tilt_angle_deg` dissociation guard, kernelspec rename in the OC20Dense notebook, hardcoded `/home/nfedik` fallback in cell 2 of the OC20Dense notebook, archive the orphan image trees (v1–v9) into `assets/images/_archive_explore/`.

---

## 5. Appendix — file references

Most-edited files in any subsequent fix pass:

- `/home/nfedik/projects/tutorials/part-1-nim/alchemi-mace-adsorption-search.ipynb` — cells 1, 4, 26–28, 36, 41, 48, 49, 50.
- `/home/nfedik/projects/tutorials/part-1-nim/oc20dense-accuracy-reproducibility-check.ipynb` — cells 0, 1, 4, 7, 11–13, 16; kernelspec.
- `/home/nfedik/projects/tutorials/part-1-nim/README.md` — backend table (line ~83), D3 statement (line ~13).
- `/home/nfedik/projects/tutorials/README.md` — Part 1 description line ~9.
- `/home/nfedik/projects/tutorials/part-1-nim/helpers/references.py` — lines 615–616 (MAD constants).
- `/home/nfedik/projects/tutorials/part-1-nim/helpers/analysis.py` — lines 149–183 (`_tilt_angle_deg`), 336–352 (per-config row schema vs contract).
- `/home/nfedik/projects/tutorials/part-1-nim/helpers/constants.py:11` — `KCAL_MOL_TO_EV`.
- `/home/nfedik/projects/tutorials/part-1-nim/scripts/run_oc20dense_known_examples.py` — lines 73–89 (re-exec), 100–110 (constants), 390–423 (closed-shell gate), splits suggested.
- `/home/nfedik/projects/tutorials/part-1-nim/scripts/run_oc20dense_mace_adsorption_energies.py:514–523` — sign convention reference.
- `/home/nfedik/projects/tutorials/part-1-nim/scripts/oc20dense_dft_reference_checks.py:39–47` — duplicated constants.
- `/home/nfedik/projects/tutorials/part-1-nim/scripts/run_oc20dense_dft_final_single_points.py` — script name + docstring clarification.
- `/home/nfedik/projects/tutorials/part-1-nim/tests/test_oc20dense_benchmark.py:127–128` — guardrail.
- `/home/nfedik/projects/tutorials/part-1-nim/references/manifest.yml:16–37` — pin OC20 / OC20Dense MAD anchor.
- `/home/nfedik/projects/tutorials/part-1-nim/references/manual_checks.md:15–21, 97–109` — refresh.
- `/home/nfedik/projects/tutorials/shared/adsorption_tutorial/contract.py:120–139` — schema reconcile with helper.
- `/home/nfedik/projects/tutorials/part-1-nim/assets/images/manifest.md:16` — funnel version mismatch.
- `/home/nfedik/projects/tutorials/part-1-nim/outputs/ovito_dft_toolkit_pairs/` — visual evidence to embed.
- `/home/nfedik/projects/tutorials/tutorial_status_and_plan.md` — retire or move to `_archive/`.

End of report.
