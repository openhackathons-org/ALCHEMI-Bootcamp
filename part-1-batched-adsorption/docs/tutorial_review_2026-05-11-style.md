# Part 1 Tutorial Review — 2026-05-11 (Style & Coherence Pass)

Author: multi-agent style/coherence review (3 parallel agents, Opus 4.7)
Anchor: `part-1-nim/docs/current_tutorial_status_and_flow.md` (the user's same-day source-of-truth document) is treated as the canonical style guide. **No source files were edited.**

This pass complements the morning's full review (`tutorial_review_2026-05-11.md`) and the afternoon delta (`tutorial_review_2026-05-11-delta.md`). The anchor was written *after* the delta report and supersedes some of its recommendations.

---

## 1. Bottom line

**The repository is now editorially coherent.** A reader can move root README → Part 1 README → main notebook → companion notebook → shared contract → references → output reports without hitting a contradiction in title, backend default, D3 framing, or adsorption-energy sign convention. The big narrative pivot the anchor codifies — "**batched atomistic simulation** is the headline; **adsorption configuration search** is the worked example" — has landed across every reader-facing document.

The remaining work is **mostly polish and one structural decision**:

- **One structural decision still pending:** the `E_ads` vs `E_bind` column-name split. The anchor mandates `E_ads`. Reader-facing prose and every plot axis already use `E_ads`. But `helpers/analysis.py:349-350` still emits both `E_bind_eV` *and* `E_bind (eV)` columns, and the notebook patches the difference at presentation time with explicit renames. This is the single largest hidden inconsistency in the corpus: invisible to a reader who only reads the docs and figures, very visible to anyone who reads the helper code or a CSV.

- **The anchor itself overrides one prior recommendation.** The delta and morning reviews both flagged the `<mark>TODO - REFERENCE REVIEW</mark>` tag at notebook cell 5 line 144 as a "feels finished" blocker. The anchor at lines 173-183 ("Review Anchors To Keep") explicitly designates `<mark>TODO - {REFERENCE,VISUAL,HUMAN} REVIEW</mark>` as load-bearing review surfaces that should *not* be stripped. This advice is now superseded; the single shipped `<mark>` marker is at a genuine human-judgment item (promoting literature comparisons to strict parity) and is correct to keep.

- **Numbers cross-check.** All nine numeric claims in the anchor (252 structures, 252/252 converged, 222 OC20Dense records, 3/3 and 2/3 top-1 success, 3481 s panel runtime, 415.8 s notebook runtime) verify exactly against `run_metadata.json`, `full_run_metadata.json`, and the auto-generated reports.

**Top three remaining fixes** (ranked impact × effort):

1. Resolve the `E_ads` / `E_bind` column-name split — either rename the helper columns or write a one-line note in the anchor (and `shared/adsorption_tutorial/contract.py`) blessing the dual-publishing pattern.
2. Convert first-person "we" / "our" to third-person scoped voice in main notebook cells 0, 3, 21 — the only cells that drift from the anchor's voice contract.
3. Add a generation-stamp header to the three auto-generated reports that currently have no `Generated:` / `Backend:` / `Model:` metadata (`oc20dense_accuracy_comparison_report.md`, `dft_reference_check_report.md`, `mace_adsorption_energy_report.md`).

---

## 2. What the style anchor codifies

`current_tutorial_status_and_flow.md` is structured in eight sections: Goal, Audience, Current Execution Contract, Reader Flow, Documentation Coverage, Current Computed Evidence, Current Verification State, Review Anchors To Keep, Remaining Presentation Work, Definition Of Done For Presentation.

Canonical claims (verbatim or paraphrased):

- **Tutorial title:** "Batched Atomistic Simulation with NVIDIA ALCHEMI". Adsorption is the *worked example*, not the headline.
- **Framing of ALCHEMI:** "the enabling layer that connects familiar structure workflows to GPU throughput." ASE / pymatgen / MACE-MPA-0 build and supply energies; ALCHEMI provides batching, GPU execution, optimizers, constraints, metadata, and the service path.
- **Research question** (verbatim): *"How many starting geometries do we need before we can trust that we found the lowest-energy adsorption structure?"*
- **Execution contract:** default backend `toolkit`; default model MACE-MPA-0; default device CUDA GPU on `ws-loc`; D3(BJ) disabled; service path `bgr_nim` modular. `E_ads = E(adslab) - E(clean slab) - E(gas adsorbate)`; negative = exothermic. Programmatic structure generation only.
- **Panel:** CO/H2O/CH3OH × Cu(111)/Pd(111)/α-Al2O3(0001) = 9 examples, 252 starting structures.
- **OC20Dense companion:** closed-shell H2O/NH3/N2, intentionally separate from the active panel.
- **Voice:** third-person, expository, scoped. Literature MAD values are *orientation guides*, not run-specific error bars.
- **Mark-TODO discipline:** `<mark>TODO - REFERENCE REVIEW</mark>`, `<mark>TODO - VISUAL REVIEW</mark>`, `<mark>TODO - HUMAN REVIEW</mark>` are first-class review markers that **should be kept visible** where they represent real human work.
- **Kernel:** `alchemi-toolkit`.

This anchor changes how some prior review items should be read:
- Recommendations to **strip TODO markers** are overridden.
- Recommendations to **drop the BGR NIM path** are not supported — BGR remains modular as the service route.
- Recommendations on **scope-drift acknowledgment** are honored (anchor explicitly states the companion is "intentionally separate").

---

## 3. Coherence status table

Synthesized from agent 1's findings, verified against the actual file states.

| File | Title framing | Backend default | D3 framing | E_ads token in prose | MACE naming | Notes |
|---|---|---|---|---|---|---|
| `/home/nfedik/projects/tutorials/README.md` | ✓ "Batched Atomistic Simulation" (line 7) | ✓ Toolkit first (line 9) | ✓ "keeps D3 disabled" | ✓ no `E_bind` leak | ✓ | One ~140-word wall paragraph at line 9 |
| `/home/nfedik/projects/tutorials/part-1-nim/README.md` | ✓ retitled (line 7) | ✓ Toolkit-first (lines 101-105) | ✓ caveated (lines 13, 83) | ✓ no `E_bind` leak | ✓ | BGR table at line 87 still has `ALCHEMI_NIM_DFT3_ENABLED=true` annotated as service default |
| `shared/adsorption_tutorial/README.md` | ✓ | ✓ Toolkit-first (lines 8-11) | – (correctly silent at contract level) | ⚠ "adsorption energy in eV" (line 65) — not the `E_ads` token | – | New "Benchmark-Only Adsorbates" section closes NH3/N2 scope drift |
| `shared/adsorption_tutorial/backends.md` | ✓ | ✓ `toolkit` first (lines 4-8) | ⚠ line 48 reads as if D3-on is a parity requirement (see drift #6 below) | ✓ canonical formula at lines 55-56 | – | – |
| `part-1-nim/assets/images/manifest.md` | – | – | – | – | – | ⚠ Does not list `assets/banner_adsorbml_toolkit.svg` that cell 1 embeds |
| `part-1-nim/references/manual_checks.md` | – | – | ✓ explicit (lines 29-30) | – | ✓ MPA-0 vs MP-0b3 distinguished | Provenance for 0.38 eV is clean |
| `part-1-nim/references/manifest.yml` | – | – | – | ✓ canonical formula (line 35) | – | `last_reviewed: 2026-05-11` ✓; five `notes:` lines stamp event date "2026-05-04" |
| `alchemi-mace-adsorption-search.ipynb` | ✓ (cell 0) | ✓ (cells 4, 6) | ✓ (cell 5 + cell 45 caption) | ✓ in prose and axis labels (cells 40, 52, 54); ⚠ DataFrame columns still `E_bind (eV)` at cell 30, 38 | ✓ | First-person "we" / "our" leaks in cells 0, 3, 21 |
| `oc20dense-accuracy-reproducibility-check.ipynb` | ✓ | ✓ | ✓ | ✓ (axis labels) | ✓ | Third-person voice fully consistent |

---

## 4. Confirmed resolved since the 2026-05-11 delta report

The morning + delta reviews together flagged ~30 items. The anchor and the same-day fix pass have closed many of them:

- **Title pivot** to "Batched Atomistic Simulation with NVIDIA ALCHEMI" — landed across root README, Part 1 README, main notebook cell 0.
- **Cell 17 `AtomicData` / `Batch` definitions** — three explicit one-liners now appear at .ipynb lines 137-139 (regression closed).
- **Cell 5 text wall** — reduced from ~400 words to 224 words across 7 paragraphs.
- **MAD envelope provenance** — captions at cell 45 / cell 50 / companion cell 8 now read *"literature OC157 relative-energy MAD used only for orientation; it is not an error bar for this MACE-MPA-0, D3-disabled run"*. The earlier "D3-free MACE-MP-0/MPA-0" mislabel is gone. Upstream provenance documented at `references.py:616-618` and `manual_checks.md:23-30`.
- **Kernelspec** — both notebooks now ship as `alchemi-toolkit` (was `python3`). Delta finding closed.
- **`E_ads` adoption in reader-facing surfaces** — all axis labels (cells 40, 52, 54), pivot-table column rename (cell 38 line 1290), helper docstrings, contract formulas, and manifest YAML use `E_ads`.
- **NH3 / N2 contract reconciliation** — `shared/adsorption_tutorial/README.md` has a new "Benchmark-Only Adsorbates" section that lists both as benchmark-only (closes delta #8).
- **`tutorial_status_and_plan.md` retired** — moved to `part-1-nim/_archive/reports/tutorial_status_and_plan_2026-05-04.md` (closes delta #10).
- **`references/manifest.yml` `last_reviewed`** bumped to 2026-05-11.
- **Image manifest pruning** — `assets/images/manifest.md` reduced from 65 lines listing v0_core/v1/v2_split/v3_illustration/v4_icon_style trees to 17 lines listing the three core figures the notebook actually embeds. Orphan PNGs deleted from git.
- **Mark-TODO discipline** — exactly one `<mark>` marker survives in the main notebook (cell 5 line 144, REFERENCE REVIEW). Companion has zero. Both within the anchor's keep-where-load-bearing rule.
- **Companion scope statement** — both notebooks now explicitly say "intentionally separate from the active CO/H2O/CH3OH teaching panel".
- **Cell 30 summary print** — the two-start hello-world now ends with `print(f"top start: {top_e:.3f} eV; fcc-hollow start: {fcc_e:.3f} eV; gap (top - fcc): {top_e - fcc_e:+.3f} eV")`, giving the at-a-glance answer the delta asked for. (Still returns a DataFrame as well — see drift #2.)
- **Numeric claims** — all nine numbers in the anchor verified against `run_metadata.json`, `full_run_metadata.json`, and the six auto-reports.

---

## 5. Drift findings (ranked)

### CRITICAL — single largest hidden inconsistency

**D1. `E_ads` vs `E_bind` column-name split.** The anchor mandates `E_ads`. Reader-facing prose and every plot axis use `E_ads`. But the **code path** still emits and consumes `E_bind`:

- `helpers/analysis.py:349-350` writes **both** `"E_bind_eV"` and `"E_bind (eV)"` as duplicate columns of the per-config row.
- `alchemi-mace-adsorption-search.ipynb:1066` — `.sort_values("E_bind (eV)")`.
- `:1287` — `.pivot_table(index="pair", values="E_bind (eV)", …)`.
- `:1290` — `summary_table.rename(columns={"E_bind (eV)": "E_ads (eV)"}, level=1)`.
- `outputs/reports/toolkit_full_panel_report.md:22` — table column header is `E_bind (eV)`.

The notebook keeps patching the difference at presentation time with explicit renames. The 2026-05-11-delta report flagged this as intentional dual-publishing; the anchor does not bless the dual name. **Fix:** either rename the helper columns to `E_ads_eV` / `E_ads (eV)` and remove the rename at line 1290, or add a one-line "internal name `E_bind_eV`, display name `E_ads (eV)`" note to `shared/adsorption_tutorial/contract.py` and the style anchor.

### HIGH — voice and standalone-ness

**D2. First-person "we" in cells 0, 3, 21 of the main notebook.** Anchor mandates third-person expository voice. Examples:

- Cell 0 line 21: *"we can now run large sets of MLIP relaxations…"*
- Cell 0 line 25: *"Once we decide to study a molecule on a surface, we still have to choose…"*
- Cell 3 line 79: *"We generate a controlled set of plausible starts…"*
- Cell 21 line 164: *"We use a curated set of small probe molecules…"*

Most of the notebook is third-person; these four cells are the only systematic drift. Low-risk wording polish.

**D3. Cell 30 hello-world still returns a DataFrame.** Two `print()` summary lines were added (good), but the cell also ends with `.sort_values("E_bind (eV)")` returning a 6-column DataFrame that displays under the prints. The cell promises "two starts, one adsorption question" and now answers that question textually — but the visual signal is still "look at this dataframe". Either suppress the DataFrame (`_ = hello_df.sort_values(...)`) or retitle the section to "two starts under the full panel setup" so the panel-helper dependency becomes part of the promise.

### MEDIUM — auto-generated reports

**D4. Three of six auto-generated reports have no `Generated:` / `Backend:` / `Model:` header**:

- `outputs/oc20dense_known_examples/reports/oc20dense_accuracy_comparison_report.md`
- `outputs/oc20dense_known_examples/dft_reference_checks/dft_reference_check_report.md`
- `outputs/oc20dense_known_examples/mace_adsorption_energy/reports/mace_adsorption_energy_report.md`

The other three (`toolkit_full_panel_report.md`, `oc20dense_known_examples_report.md`, `dft_final_single_point_report.md`) include `Generated: <UTC>` timestamps. The anchor's Definition Of Done implies explicit metadata; the auto-report writers should emit a standard 3-line header.

### MEDIUM — single-source-of-truth drift

**D5. `shared/adsorption_tutorial/backends.md:47-49`** is the only place that reads as if D3-on is a Toolkit-adapter *requirement*: *"BGR parity requires explicit DFT-D3(BJ) damping parameters from verified runtime metadata."* Every other doc says D3-off is the default. The sentence is true for a parity claim against a D3-enabled BGR reference run, but it lands badly without context. **Fix:** *"If the BGR service is run with D3(BJ) enabled, parity with that run requires the Toolkit adapter to use matching explicit damping parameters; otherwise both paths run D3-disabled."*

**D6. `shared/adsorption_tutorial/README.md:65`** reads "adsorption energy in eV". Every other authoritative surface uses the `E_ads` token. Replace with the token to match the style anchor and the backends contract.

### LOW — minor surface drift

**D7. `assets/images/manifest.md`** doesn't list `assets/banner_adsorbml_toolkit.svg` that the main notebook's cell 1 embeds. Either add it or scope the manifest explicitly to `assets/images/` (one sentence).

**D8. `references/manifest.yml` notes lines (`:83`, `:106`, `:128`, `:150`, `:216`)** end with "Official … PDF download returned … on 2026-05-04". Historically accurate but a quick reader could mistake them for stale review. Reword as event timestamps: "on the 2026-05-04 download attempt".

**D9. Cell 51 vs cell 53/56 framing collision** is now softer but not unified. Cell 51 says *"the measurement that makes the starting-point assumption visible and testable"* (was "the tutorial's central measurement"). Cell 53 uses a two-bucket frame (Search effect / Verification effect); cell 56 uses a three-bucket frame (Reference check / Search effect / Needs review). Pick one bucket scheme and use it in both cells.

**D10. H2O framing repeats across cells 14, 15, 17, 20** — no longer strict duplicate ("Before building surfaces…" / "Before the adsorption search…") but the H2O batch is still introduced four times. Cell 14 (problem motivation), cell 15 (figure setup), cell 17 (API mini-guide), cell 20 (transition into surfaces) — each can keep one paragraph if the cells are reordered, but the current state reads as repetition.

### Style anchor self-consistency

**D11. "Review Anchors To Keep" lists three TODO marker categories** (`REFERENCE`, `VISUAL`, `HUMAN`). Only `REFERENCE` is actually used in the shipped notebooks. Either delete the two unused entries or add the missing markers at appropriate cells (e.g., `TODO - VISUAL REVIEW` next to the workflow-graphics placeholder per Remaining Work item 3 is a natural fit).

**D12. "Documentation Coverage" omits**: `current_tutorial_status_and_flow.md` itself (self-reference, but curious), `assets/images/manifest.md`, `references/domain_expert_fact_check.md`, the six auto-generated `outputs/.../*_report.md` files, the six new untracked scripts under `scripts/`. A reader following the coverage list will miss content that the anchor depends on.

**D13. Anchor is silent on most engineering-hygiene items the delta review surfaced**: cell-30 standalone-ness, the OVITO contact sheet not embedded in the main notebook (only in the companion), duplicated utility functions across scripts, three `benchmark_h2o_saturation.py` issues, the parity-plot annotations. These can be inherited or explicitly waived in a "delta items intentionally deferred" subsection.

**D14. Definition Of Done item "every plotted energy is labeled `E_ads`"** is satisfied in the figures but contradicted by the `E_bind (eV)` column header in `outputs/reports/toolkit_full_panel_report.md:22`. If the report counts as "plotted" the DoD is violated; if it doesn't, say so in the anchor.

---

## 6. Top consolidated fixes

Ranked across all three agents, with the anchor's policies and the delta-review's overrides factored in.

| # | Fix | Where | Effort |
|---|---|---|---|
| 1 | Resolve the `E_ads` / `E_bind` column-name split. Either rename helper columns to `E_ads_eV` / `E_ads (eV)` and drop the rename in the notebook, or add a one-line dual-publishing note to `contract.py` + the style anchor. | `helpers/analysis.py:349-350`, `alchemi-mace-adsorption-search.ipynb:1066,1287,1290`, `outputs/reports/toolkit_full_panel_report.md:22`, `contract.py`, anchor | 30-45 min |
| 2 | Convert first-person "we" / "our" to third-person scoped voice. | `alchemi-mace-adsorption-search.ipynb` cells 0, 3, 21 | 15 min |
| 3 | Add standard 3-line `Generated: <UTC>` / `Backend:` / `Model:` header to the three reports that lack it. Update the script writers, not the report files. | `scripts/run_oc20dense_*` writers; reports listed in D4 | 20 min |
| 4 | Decide on cell 30 hello-world: either suppress the trailing DataFrame, or retitle to acknowledge dependency on panel setup. | `alchemi-mace-adsorption-search.ipynb` cell 30 | 5 min |
| 5 | Rewrite `backends.md:47-49` so D3 is described conditionally on the BGR service config, not as a flat parity requirement. | `shared/adsorption_tutorial/backends.md:47-49` | 5 min |
| 6 | Replace "adsorption energy in eV" with the `E_ads` token in `shared/adsorption_tutorial/README.md:65`. | one-line edit | 1 min |
| 7 | Add `banner_adsorbml_toolkit.svg` to `assets/images/manifest.md`, or scope the manifest to "files under `assets/images/`". | `assets/images/manifest.md` | 3 min |
| 8 | Reconcile anchor's three-marker TODO policy with the one shipped marker: either delete `VISUAL REVIEW` and `HUMAN REVIEW` from the anchor or add them to appropriate notebook cells. | `current_tutorial_status_and_flow.md:178-183` | 5 min |
| 9 | Add `manifest.md`, `domain_expert_fact_check.md`, the auto-reports, and the six new scripts to anchor's "Documentation Coverage" list, or scope the list explicitly. | `current_tutorial_status_and_flow.md:99-118` | 10 min |
| 10 | Rephrase the five "2026-05-04" event notes in `references/manifest.yml` to read as event timestamps. | `manifest.yml:83,106,128,150,216` | 5 min |
| 11 | Pick one bucket scheme (two-bucket cell 53 or three-bucket cell 56) and use it consistently. | `alchemi-mace-adsorption-search.ipynb` cells 51, 53, 56 | 10 min |
| 12 | Tighten the H2O introduction so the chemistry is set up in one place. | `alchemi-mace-adsorption-search.ipynb` cells 14, 15, 17, 20 | 15 min |

**Summary timing:** items 1, 2, 3, 4, 5, 6, 7, 8 are all sub-30-minute edits that close the remaining surface drift. Items 9, 10, 11, 12 are polish items that depend on the user's editorial preference more than on factual error.

---

## 7. Appendix — file references (style pass)

Files most-touched in the next polish pass:

- `/home/nfedik/projects/tutorials/part-1-nim/alchemi-mace-adsorption-search.ipynb` — cells 0, 3, 21 (voice), 30 (hello-world), 51/53/56 (bucket), 14/15/17/20 (H2O framing), 1066/1287/1290 (column rename).
- `/home/nfedik/projects/tutorials/part-1-nim/helpers/analysis.py:349-350` — dual `E_bind_eV` / `E_bind (eV)` columns.
- `/home/nfedik/projects/tutorials/part-1-nim/docs/current_tutorial_status_and_flow.md:99-118` (Documentation Coverage), `:178-183` (Review Anchors).
- `/home/nfedik/projects/tutorials/shared/adsorption_tutorial/README.md:65`.
- `/home/nfedik/projects/tutorials/shared/adsorption_tutorial/backends.md:47-49`.
- `/home/nfedik/projects/tutorials/part-1-nim/assets/images/manifest.md`.
- `/home/nfedik/projects/tutorials/part-1-nim/references/manifest.yml:83,106,128,150,216`.
- `/home/nfedik/projects/tutorials/part-1-nim/outputs/oc20dense_known_examples/reports/oc20dense_accuracy_comparison_report.md`, `…/dft_reference_check_report.md`, `…/mace_adsorption_energy_report.md`.
- `/home/nfedik/projects/tutorials/part-1-nim/scripts/run_oc20dense_*.py` — report writers that need the standard 3-line header.

End of style/coherence report.
