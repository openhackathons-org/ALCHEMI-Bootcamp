# Part 1 Tutorial Review — 2026-05-07

Author: review pass (Claude Opus 4.7, 1M context)
Scope: Part 1 only (`part-1-nim/`)
Approach: pedagogy + scientific-rigour review against established
scientific-software tutorial patterns and against the cited primary
literature. Recommendations only; no code diffs.

This review is *complementary* to the existing
`/home/nfedik/projects/tutorials/tutorial_status_and_plan.md`
(2026-05-04). That document is an execution plan; this one is a
tutorial-quality review focused on intro/tool-intro/hello-world,
non-tokenization scientific validation, and a per-claim audit of
references against the actual cited papers.

---

## 1. Executive summary

**Three things Part 1 already does well**

1. The reference manifest at `part-1-nim/references/manifest.yml`
   separates `context` / `near-strict` / `strict` rigorously and is
   an unusual-but-correct discipline for a tutorial.
2. Slab construction in `helpers/surfaces.py` /
   `helpers/metal_slabs.py` / `helpers/oxide_slabs.py` uses
   `pymatgen.core.surface.SlabGenerator` with orthogonalization.
   Site finding (`find_fcc_sites`, `find_al2o3_0001_sites`) is
   geometric — not text-based.
3. The throughput sweep (notebook cells 14–18) is a strong sanity
   path for the BGR endpoint and amortization curve.

**Five gaps worth addressing before public release**

1. **Stale-version drift on the headline accuracy claim, propagating
   across files.** Root cause: `helpers/references.py:615` defines
   `MACE_MPA0_OC157_MAD_EV = 0.42` and the value is exported from
   `helpers/__init__.py:54,227`, imported by `helpers/analysis.py:23,369`
   as the default `mad_ev` for parity-plot error bars, and quoted in
   notebook cell 3 ("0.42 eV ... 121 of 157"), cell 1126 (axis
   label), cell 1284 (figure caption / error bars), and asserted as
   "verified" in `references/manual_checks.md:24`. The number is real
   — but it is from MACE **arXiv:2401.00096 v2** (page 100), where it
   describes the v2 model called "MACE-MP-0". When v3 was released
   (Sep 2025) the v2 model was renamed MACE-MP-0b3, the new
   MACE-MPA-0 model was added (which is what the BGR NIM actually
   deploys per cell 2), and the v2 number 0.42 eV / 121-of-157 was
   updated *for the same model* to 0.38 eV / 126-of-157 (v3 page 119),
   while the new MPA-0 number is 0.28 eV (v3 page 122). The cached
   PDF in `references/pdfs/` is v3 — so the helper constant, the
   manual-check entry, and the notebook prose are all out of step
   with the manifest's own pinned source. Net effect: parity-plot
   error bars are ≈ 1.5× too wide and the constant name (`...MPA0...`)
   does not match its value. See section 11.1 row 4 for the verbatim
   v2 vs v3 quotes.
2. **No scientific hello-world.** A single-pair "build CO + Pd(111),
   relax, print E_bind, assert sign" cell does not exist before the
   nine-pair panel. The throughput sweep is a *throughput* test, not a
   pedagogical first calculation.
3. **No tools-at-a-glance section.** Six tools (MACE-MPA-0, DFT-D3(BJ),
   BGR NIM, ASE, pymatgen, AdsorbML protocol) are referenced before
   they are introduced. Only the BGR NIM gets a brief description
   (cell 2).
4. **Several places infer adsorbate chemistry from element-symbol
   string matching with no connectivity check.** Most importantly,
   `helpers/analysis.py:_tilt_angle_deg` (lines 149–183) computes
   bond-axis vectors from `symbols.index("O")` etc. without verifying
   the named atoms are bonded. A relaxation that dissociates the
   molecule will silently report a tilt angle. The AdsorbML paper
   (cached) explicitly invalidates dissociated geometries — so the
   tutorial's protocol is in tension with its own primary reference.
5. **The reference-scope guard `helpers/analysis.py:strict_parity_subset`
   exists but is not called.** The notebook does not invoke it before
   parity plots, so context rows could in principle be plotted as if
   they were validated parity. (At the moment all rows are `context`
   and plotting code currently uses different filters, so no immediate
   visible damage — but the guard is the correct discipline and it is
   leaving safety on the table.)

**Verdict on tutorial readiness**

The notebook is structurally close to a high-quality tutorial. The
methodology is sound. The reference layer is more honest than typical.
The blocking items are (a) the factual MAD/count numbers in the intro,
(b) the missing hello-world, and (c) the tokenization-style adsorbate
chemistry inference. None of these require new helpers — only careful
edits using the chemistry libraries already imported.

---

## 2. Best-practice scoring against established scientific tutorials

Patterns confirmed by direct fetch on 2026-05-07 unless noted:

- **fairchem / Open Catalyst** —
  `https://github.com/FAIR-Chem/fairchem/blob/main/docs/core/quickstart.md`
  fetched successfully. Their `Relax an adsorbate on a catalytic
  surface` example is a 20-line block (Cu(100)/CO/bridge) before any
  other content. This is the closest-analog reference for Part 1's
  hello-world.
- **ACEsuit/mace** —
  `https://github.com/ACEsuit/mace/blob/main/README.md` fetched
  successfully. Pretrained-model section opens with a small ASE-side
  energy-and-forces snippet. Foundation-model variants (MACE-MP, MACE-OFF,
  MACE-Polar, MACE-MPA-0) are listed with explicit accuracy/coverage notes.
- **HuggingFace transformers** —
  `https://github.com/huggingface/transformers/blob/main/docs/source/en/quicktour.md`
  fetched. Pattern: a 3-row class table introducing `PreTrainedConfig`,
  `PreTrainedModel`, and `Preprocessor` is shown *before* any code.
  This is the canonical "what is each tool" pattern.
- **scikit-learn** — `getting_started.html` fetched (the `tutorial.rst`
  path drifted; the canonical landing page is now
  `scikit-learn.org/stable/getting_started.html`). Pattern: each
  example is "Description / Code / Output / Discussion."
- **fast.ai course notebooks** — top-down "show the result first" remains
  canonical (knowledge cite, not fetched in this session).
- **AiiDA / atomate2** — manifest-cell pattern (versions, commits,
  hardware printed at notebook top) is canonical (knowledge cite, not
  fetched in this session).

### Scoring axes

| Axis | Current state in Part 1 | Target pattern (source) |
|------|--------------------------|---------------------------|
| General intro: problem, audience, prerequisites, learning outcomes | Cells 0–4 cover problem and uncertainty scope, but never list audience or prerequisites explicitly. No "you will learn" bullet list. | All four stated in the first markdown block (fairchem quickstart, HF quicktour). |
| Tool intro before first use | BGR NIM described in cell 2; others (MACE-MPA-0, DFT-D3(BJ), ASE, pymatgen, AdsorbML, OVITO) appear in code before any explanation. | A "tools at a glance" markdown table with one paragraph per tool (HF quicktour 3-row class table; MACE README pretrained-models table). |
| Hello-world example | Throughput sweep (cells 14–18) is a *system/endpoint* smoke test. There is no scientific one-pair end-to-end run. | A 20-ish-line single-pair worked example before the panel (fairchem quickstart "Relax an adsorbate on a catalytic surface"). |
| Helper API surface explained | `helpers/*` imported as a wall in cell 7 with no per-helper explanation. | A short "what each helper does" row table (HF AutoClass discussion). |
| Reproducibility receipt at top | Endpoint metadata fetched in cell 11; no consolidated cell for git commit, package versions, GPU model, BGR service tag. | One preflight cell that prints all of these and stores them in a dict for the run summary (AiiDA/atomate2 pattern). |
| Failure-mode framing per cell | Most analysis cells say what success looks like; few say what failure looks like (e.g., "if all sites collapse to one minimum", "if any E_bind > 0", "if convergence > N steps"). | One short failure-signature line at the bottom of each major analysis cell (sklearn examples-gallery pattern). |

---

## 3. Hello-world / onboarding gap

**Recommendation**: insert a single-pair end-to-end section between
the current clean-slab cell (cell 25) and the configuration-grid
section (cell 27). Use CO on Pd(111) with three starting sites
(top, fcc-hollow, hcp-hollow). Reuse, in order:

- `helpers.metal_slabs.build_metal_slab("Pd(111)")`
- `helpers.config_search.build_co("C-down")`
- `helpers.config_search.fcc111_site_candidates(...)` — pick three
- `helpers.api_client._relax_pair(...)` (or the BGR client equivalent)
- `helpers.analysis.compute_adsorption_energy_ev(...)`
- One inline `assert e_bind_min < 0`

The success printout should show: configuration index, final
E_bind in eV, identified site name, geometric tilt. This converts the
existing batch path into a 30-second pedagogical demo. No new helper
code is needed.

The fairchem upstream pattern uses bridge site for CO on Cu(100)
(`docs/core/quickstart.md` lines 159–179). Pd(111) fcc-hollow vs top
is a stronger pedagogical example because it directly demonstrates
the "configuration search matters" thesis (cell 1) — the Hammer 1996
"CO/Pd puzzle."

---

## 4. Tool-intro gap

**Recommendation**: add a "Tools and conventions" markdown section
between the current cell 0 (title) and cell 1 (configuration-search
motivation). One row per tool, one short paragraph each, with link to
the corresponding helper module that wraps it. Cover at minimum:

- **MACE-MPA-0** (Batatia 2024, arXiv:2401.00096v3) — what a
  foundation MLIP is; training data MPtrj + sAlex (define both —
  MPtrj is the MP relaxation-trajectory dataset; sAlex is the
  curated 10.5M-structure subset of the Alexandria database, per the
  cached MACE paper page 122); state the empirically observed accuracy
  envelope on the OC157 supplement test set (see section 11.4 below
  for the numbers).
- **DFT-D3(BJ)** (Grimme 2010, JCP 132:154104; Grimme 2011, JCC
  32:1456) — what dispersion correction is, what BJ damping is, and
  that it is enabled server-side in this tutorial via
  `ALCHEMI_NIM_DFT3_ENABLED=true` (cell 2 already has this).
- **ALCHEMI BGR NIM** — extend the existing cell 2 with one sentence
  on what a NIM is more generally, plus a link to the NIM docs.
- **AdsorbML** (Lan 2023, npj Comp. Mat. 9:172) — what configuration
  search is, with the verified 87.36% / 50% framing from the cached
  paper (see section 11.4 below for the page reference).
- **ASE** + **pymatgen** — what each does (`ase.Atoms` / `ase.build`
  for adsorbates; `pymatgen.core.surface.SlabGenerator` /
  `pymatgen.analysis.adsorption` for slabs and site finding); link
  to `helpers/surfaces.py` and `helpers/config_search.py`.
- **OVITO** — used for publication renders only (not for scientific
  validation); link to `helpers/visualization.py`.

Each entry should explicitly point readers at the helper module that
wraps it so a reader can jump from prose to code.

### 4.1 Native Toolkit API mini-guide — what to keep, what to add

The current mini-guide (notebook cell 13) covers `AtomicData`,
`Batch.from_data_list`, `MACEWrapper.from_checkpoint`, `FIRE2` +
`ConvergenceHook.from_fmax`, `FreezeAtomsHook`, `NaNDetectorHook`, and
`make_neighbor_hooks()`. That is a defensible minimum but it omits two
APIs that are central to *this specific* tutorial.

API surface confirmed against `nvalchemi` at the pinned commit
`7fe7756bd1b13580a619cff39b69742145d416e1` by reading the upstream
`nvalchemi/{data,models,dynamics,dynamics/hooks}/__init__.py` and the
upstream `README.md`. Cross-checked against actual usage in
`helpers/relaxation_backends.py` and `part-2-toolkit/melting-point-slc.ipynb`.

**APIs the mini-guide should add (high priority).** These are missing
from cell 13 but used by `helpers/relaxation_backends.py` to assemble
the model the BGR NIM actually deploys — without them the reader cannot
reproduce the tutorial's MACE+D3(BJ) stack:

- **`PipelineModelWrapper` + `PipelineGroup` + `PipelineStep`**
  (`nvalchemi.models.pipeline`) — the upstream README calls compositional
  pipelines a headline feature. Used by `relaxation_backends.py:408–432`
  as `PipelineModelWrapper(groups=[PipelineGroup(steps=[mace]),
  PipelineGroup(steps=[d3])])`. The mini-guide should show this exact
  three-line construction so a reader who only reads the markdown can
  produce a model that matches the deployed one.
- **`DFTD3ModelWrapper`** (`nvalchemi.models.dftd3`) — the BGR NIM has
  D3(BJ) enabled (cell 2 cites `ALCHEMI_NIM_DFT3_ENABLED=true`), and
  `relaxation_backends.py:411–423` constructs this with explicit
  `a1`/`a2`/`s8`/`cutoff` parameters from `ToolkitD3BJConfig`. The
  current mini-guide says nothing about how dispersion is applied. This
  is the single most important addition.
- **`model.model_config.active_outputs = {"energy", "forces"}`** — the
  "capability negotiation" pattern the upstream README highlights as
  feature #1. Used at `relaxation_backends.py:433`. Worth one sentence
  in the mini-guide: "you opt into the outputs you need; this avoids
  paying for forces / charges / stress when you do not."

**APIs worth promoting from one-liners to short examples (medium
priority):**

- **`FIRE2VariableCell`** (`nvalchemi.dynamics.optimizers.fire2`) — the
  cell-relaxation variant. Used by `relaxation_backends.py:577` for
  the `cellopt=True` path. Add one line: "use `FIRE2VariableCell`
  instead of `FIRE2` when the cell shape should also relax (e.g.,
  clean-slab full relaxations); pair with `AlignCellHook` from
  `nvalchemi.dynamics.hooks`."
- **`model.make_neighbor_hooks()`** is already shown but unexplained.
  Add one sentence on *why*: neighbor lists need periodic rebuilds
  during a relaxation or MD run; the model emits the right hooks for
  its cutoff and the user just registers them.
- **`LoggingHook`** (`nvalchemi.dynamics.hooks.logging`) — diagnostic
  visibility on long relaxations. The melting-point notebook uses it.
  For the adsorption tutorial it is optional but useful as a "how to
  see what is happening" primitive.

**Worth a single forward-pointer paragraph (low priority for Part 1,
core for Part 2):**

- The composability operators `+` (FusedStage, single-GPU) and `|`
  (DistributedPipeline, multi-GPU) — `nvalchemi.dynamics.{FusedStage,
  DistributedPipeline}`. The Part 2 melting-point notebook uses
  `DynamicsStage` directly; Part 1 does not need to demonstrate them
  but a one-line "Toolkit pipelines compose with `+` and `|`; see
  Part 2 for an MD example" closes the loop with the upstream README
  framing.

**Out of scope for the Part 1 mini-guide (skip):**

- Integrators (`NPT`, `NVE`, `NVTLangevin`, `NVTNoseHoover`) — these
  are MD, not relaxation. Belongs in Part 2.
- Data pipeline (`AtomicDataZarrReader`, `Dataset`, `DataLoader`) —
  Zarr-backed datasets are a production-scale concern; the tutorial
  uses cached JSON instead.
- Other model wrappers (`AIMNet2Wrapper`, `EwaldModelWrapper`,
  `PMEModelWrapper`, `LennardJonesModelWrapper`) — not used in
  Part 1's adsorption-search workflow.
- `SizeAwareSampler` (inflight batching) — large-screen primitive;
  Part 2 territory.

**Recommended cell-13 ordering (after these additions):**

1. The five-step list (already present) — but expanded to seven steps:
   build `Atoms` → convert to `AtomicData` → `Batch.from_data_list`
   → load `MACEWrapper` → wrap with `DFTD3ModelWrapper` →
   compose with `PipelineModelWrapper` + `PipelineGroup`s →
   run `FIRE2` (or `FIRE2VariableCell`) with `ConvergenceHook` and
   `FreezeAtomsHook` / `NaNDetectorHook`.
2. A single ~30-line code block showing the full assembly. Keep the
   existing `to_atomic_data` helper; insert the D3 + pipeline lines
   after the `MACEWrapper.from_checkpoint(...)` call and before the
   `FIRE2(...)` call.
3. The wrapper-helper second example (already present) — keep it.
4. One forward-pointer sentence to Part 2 for the `+` / `|`
   composability and integrators.

This shape matches how `helpers/relaxation_backends.py` actually
constructs the toolkit-side model (lines 391–434). A reader who
follows the mini-guide will be able to read the helper without
guessing.

**Upstream sources verified on 2026-05-07:**

- `https://github.com/NVIDIA/nvalchemi-toolkit` README at the pinned
  commit `7fe7756bd1b13580a619cff39b69742145d416e1`.
- Per-module `__init__.py` exports for `nvalchemi.{data, models,
  dynamics, dynamics.hooks}` at the same pin (used to confirm the
  public surface, not internal helpers).
- Hosted docs at `https://nvidia.github.io/nvalchemi-toolkit/` (linked
  from the upstream README; not all subpages were fetched in this
  session).

---

## 5. Tokenization-only inference: line-by-line audit

This is the section where the user's "do not rely solely on
tokenization for scientific tasks" principle is operationalized.
Every site below was confirmed by direct file read.

### 5.1 Confirmed fragility sites

**`part-1-nim/helpers/analysis.py:126–146` — `_binding_atom_offset`**

The function selects the binding atom for a given adsorbate by
indexing on element symbols (`np.where(np.isin(symbols, ["C", "O"]))`
for CH3OH and CO; `np.argmin(z)` fallback for NH3 and unknowns).

The geometric `argmin(z)` is defensible. The fragility is that there
is no pre-check that the adsorbate has the *expected stoichiometry*.
If a relaxation fragments CH3OH into CH3* and OH*, the function still
returns a "binding atom offset," which downstream code interprets as a
valid result. The AdsorbML paper (cached, p. 1 and p. 3) explicitly
invalidates such configurations.

**Recommendation**: add a thin helper
`_assert_adsorbate_intact(atoms, formula)` that compares
`Atoms.get_chemical_formula()` against the expected formula
(`"CO"`, `"H2O"`, `"CH3OH"`, `"NH3"`) and raises a clear error.
Call it once at the top of `_binding_atom_offset` and once at the top
of `_tilt_angle_deg`.

**`part-1-nim/helpers/analysis.py:149–183` — `_tilt_angle_deg`**

```python
if adsorbate == "CO" and "C" in symbols and "O" in symbols:
    vec = positions[symbols.index("O")] - positions[symbols.index("C")]
elif adsorbate == "H2O" and "O" in symbols:
    o_idx = symbols.index("O")
    h_idx = [i for i, s in enumerate(symbols) if s == "H"]
    if h_idx:
        vec = positions[h_idx].mean(axis=0) - positions[o_idx]
elif adsorbate == "CH3OH" and "C" in symbols and "O" in symbols:
    vec = positions[symbols.index("C")] - positions[symbols.index("O")]
```

This computes axis vectors assuming the named C/O/N atoms are *bonded*
to each other. For a fragmented or partially-dissociated final
geometry, the named atoms may be many ångström apart. The function
will still return a numeric tilt angle and downstream code will treat
it as valid.

**Recommendation**: build a chemistry-aware neighbor list with
`ase.neighborlist.natural_cutoffs(atoms)` (already an ASE primitive
the codebase uses for slabs), and assert the named C–O / O–H / N–H
pair is within the bond cutoff before computing the tilt. If not,
return `None` *and* set an explicit `dissociated` flag in the
integrity status. That flag should propagate to the per-pair
result row so dissociated geometries are excluded from binding-energy
plots — which is exactly what AdsorbML page 1 and page 3 require.

**`part-1-nim/helpers/config_search.py:127–150` — `build_methanol`**

```python
m = molecule("CH3OH")
symbols = m.get_chemical_symbols()
o_idx = symbols.index("O")
c_idx = symbols.index("C")
```

This is *safe* because the source is `ase.build.molecule("CH3OH")`,
which has well-defined atomic ordering. The fragility risk is
extension: if a future contributor changes the source to a parsed
XYZ file or to a relaxed geometry, the same `.index("O")` call
becomes unsafe. Recommendation: a one-line comment
documenting that this function only operates on the canonical
`ase.build.molecule(...)` output, and that a stoichiometry assert
should be added if the source ever changes.

**`part-1-nim/helpers/config_search.py:153–180` — `build_nh3`**

Same pattern, same recommendation.

### 5.2 Sites where chemistry libraries *are* used correctly

For positive reinforcement and to give future contributors a pattern
to copy:

- `helpers/surfaces.py`, `helpers/metal_slabs.py`,
  `helpers/oxide_slabs.py` — slab construction via
  `pymatgen.core.surface.SlabGenerator` with orthogonalization. Slab
  identity is defined by the SlabGenerator's Miller index, not by
  string parsing.
- `helpers/config_search.py:find_fcc_sites` and
  `find_al2o3_0001_sites` — geometric site finding via z-layer
  analysis and atomic-number filters (`numbers == 13` for Al,
  `numbers == 8` for O), not symbol string matching.
- `helpers/analysis.py:_adsorbate_integrity_status` (lines ~195–215)
  — uses positions and slab atom counts to detect missing or floated
  adsorbates. No symbol parsing.
- `helpers/analysis.py:compute_adsorption_energy_ev` (lines 73–79) —
  pure numeric formula, sign convention documented and unit-tested
  in `tests/test_adsorbml_analysis.py`.
- `helpers/analysis.py:strict_parity_subset` (lines ~416–420) — the
  reference-scope guard. Correct shape; just not called yet by the
  notebook.

### 5.3 Recommended new tests

Three small tests would lock the chemistry-aware checks in:

1. `tests/test_adsorbate_integrity.py::test_dissociated_adsorbate_flagged`
   — feed a deliberately fragmented CH3OH (C and O 4 Å apart) and
   assert the integrity status returns a `dissociated`-style flag and
   that `_tilt_angle_deg` returns `None`.
2. `tests/test_adsorbate_integrity.py::test_wrong_stoichiometry_raises`
   — pass a 1C/2O `Atoms` object as `"CO"` and assert a clear failure
   from `_binding_atom_offset`.
3. `tests/test_geometry_validation.py::test_neighbor_list_cutoff_choice`
   — confirm that `ase.neighborlist.natural_cutoffs(atoms)` is what
   the bond-cutoff pipeline relies on, and that no hardcoded magic
   numbers are used downstream of the natural-cutoffs call.

---

## 6. Scientific-soundness review

This section walks through every quantitative claim in the notebook
intro / helper docstrings / pivot brief and gives a verified verdict.
The detailed audit table is in section 11; this is the narrative
summary.

- **`MACE_MP0B3_OC157_MAD_EV = 0.38`** in
  `helpers/references.py:616` and the pivot brief line 46 — **VERIFIED**
  against MACE arXiv 2401.00096v3, page 119: "an MAD of 0.38 eV"
  for MACE-MP-0b3+D3 on the OC157 supplement test set.
- **MACE-MPA-0 OC157 MAD ≈ 0.28 eV** (pivot brief line 56,
  "0.28–0.38 eV per system") — **VERIFIED** against MACE arXiv page 122:
  "MPA-0+D3 achieves a significantly lower MAD (0.28 eV vs. 0.38 eV)".
- **Notebook cell 3: "0.42 eV MAD on OC157, 121 of 157" — INCORRECT.**
  The cached MACE paper (page 119) says 0.38 eV and 126 of 157 for
  MACE-MP-0b3+D3, or 0.28 eV (figure not stating count) for MACE-MPA-0
  (the deployed model). Neither 0.42 eV nor 121/157 appears in the
  cached source. This is the highest-priority correction.
- **AdsorbML "single-start vs batched" reliability** — cell 1's
  thesis (configuration search matters because single-start often
  misses the global minimum) is **VERIFIED** against the AdsorbML
  paper, page 1 (87.36% balanced option), page 3 (Table 2 baseline ML
  success rates 46–56%, with GemNet-OC-MD = 50.05%), page 5 (Fig. 3
  caption explicitly states "balanced option reported in the abstract
  — a 87.36% success rate and 2290× speedup"). The framing is sound;
  the precise sentence "raises ML-only success from ~50% baseline to
  87.36% (balanced ML+SP, k=...)" is well supported. The current
  notebook should make the baseline-vs-AdsorbML framing explicit so
  it does not read as if 50% is from the single-start *DFT* protocol.
- **AdsorbML constraint requirements** — page 1: "the adsorbate
  should not be desorbed... should not dissociate or break apart";
  page 3: "ML-driven relaxations are run on all initial
  configurations; systems not suitable for adsorption energy
  calculations due to physical constraints are removed, including
  dissociation, desorption, and surface mismatch." This **directly
  supports** the section 5 recommendation: the tutorial's analysis
  pipeline should detect dissociation rather than silently produce a
  binding energy on a fragmented system.
- **Sign convention** `E_ads = E(slab+ads) − E(clean_slab) − E(gas_ads)`
  in `helpers/analysis.py:73–79` — **VERIFIED** against AdsorbML
  page 1, Eq. (1): `ΔE_ads = E_sys − E_slab − E_gas`. Identical.
- **"0.5 eV decision threshold"** (pivot brief line 56) — appears as
  a rule-of-thumb claim with no specific citation. It is consistent
  with the AdsorbML paper's "0.1 eV success threshold" discussion
  (page 3, "an acceptable tolerance (0.1 eV in this work)") but the
  0.5 eV figure is not pinned to a primary source. Recommendation:
  either cite the section that justifies it or rephrase as "we use
  0.5 eV as a conservative screening threshold for this tutorial."
- **"100–1000× DFT speedup"** (pivot brief) — partially supported.
  AdsorbML page 1 reports "~2000× speedup"; the MACE paper does not
  give a single multiplicative number. Recommendation: cite AdsorbML
  page 1 for the 2000× figure and note the speedup is workflow-
  specific.
- **Industrial-user list** (pivot brief: BASF, ExxonMobil, Shell,
  Chevron, Siemens Energy, Ørsted, Topsoe) — **NOT VERIFIED** in any
  cached source. Recommendation: either move these to a "industrial
  context" appendix with public case-study citations, or rephrase as
  "representative industrially relevant problem classes." This is
  the kind of broad claim a careful reviewer will flag.
- **DFT-D3(BJ) damping function form** — Grimme 2011 JCC source is
  paywalled and has no arXiv preprint. The MACE paper page 16
  confirms the *application* in this tutorial: "The same parameters
  used in PBE-D3(BJ), i.e., DFT-D3 with a Becke-Johnson damping
  function (153), are used in the D3 correction to MACE-MP-0b3."
  Recommendation: keep Grimme as cited, mark its row as `context`
  (it already is), do not promote.

---

## 7. Patterns to import (concrete recommendations)

Six concrete patterns to apply, each tied to a precise notebook cell
or helper file. None of these require new helper code beyond the
chemistry-integrity helper described in section 5.

1. **Top-down "result first" intro** — new first code cell after the
   intro markdown: a 5-line ping/relax/print of CO on Pd(111).
   Source pattern: HuggingFace `pipeline()` one-liner; fairchem
   `quickstart.md` lines 159–179.
2. **Manifest cell** — new cell after imports (current cell 7) that
   prints versions, git commit, GPU model, BGR NIM service tag and
   model checkpoint, and stores the dict for the run summary.
   Source pattern: AiiDA / atomate2 quickstart.
3. **One-pair end-to-end before panel** — see section 3 of this
   review. Source pattern: fairchem quickstart.
4. **Per-cell expected-vs-failure note** — one line at the bottom of
   each major analysis markdown cell. Source pattern: scikit-learn
   examples-gallery "Discussion" subsection.
5. **Tools table at the top** — see section 4 of this review. Source
   pattern: HuggingFace quicktour 3-row class table; MACE README
   pretrained-models table.
6. **Inline `assert` smoke checks** — at least one inline `assert` in
   each major code cell (e.g., `assert df["e_bind_ev"].notna().all()`,
   `assert (df["e_bind_ev"] < 0).all()` for the smoke pair) so a
   broken kernel or stale cache fails loudly. Source pattern: PyTorch
   tutorials; sklearn user guide.

---

## 8. Concrete change list (recommendations only)

Numbered, ordered. Each item references the precise file or cell and
the helper that already supplies the functionality. No item invents
new modules; the heaviest new code is the
`_assert_adsorbate_intact` helper in `helpers/analysis.py`.

1. **Correct the 0.42 eV / 121-of-157 propagation across multiple
   files.** This is one logical fix that touches several places:
   - `helpers/references.py:615` — change `MACE_MPA0_OC157_MAD_EV = 0.42`
     to `MACE_MPA0_OC157_MAD_EV = 0.28` (matches MACE arXiv p.122,
     MPA-0+D3). The MP-0b3 constant on line 616 (`= 0.38`) is correct.
   - Notebook cell 3 prose — rewrite to cite MACE-MPA-0+D3 (the
     deployed model) with MAD 0.28 eV, and either remove the "X of 157"
     count (the paper does not state it for MPA-0) or substitute the
     MP-0b3 number 126/157 with a clear note about which model.
   - Notebook cell 1126 axis-label and cell 1284 figure-caption /
     error-bars — same correction. Error bars will narrow by ≈ 1.5×.
   - `references/manual_checks.md:24` — revert the "verified 0.42 eV /
     121 of 157" assertion; this entry is incorrect against the cached
     PDF. Re-mark as `[ ]` (not verified) and add the cached source
     verbatim quote when re-checked.
   - Cite arXiv:2401.00096v3 page 119 (MP-0b3 numbers) and page 122
     (MPA-0 numbers) in any new prose.
2. **Notebook cell 0–1 — add "Tools and conventions" markdown
   section.** Six rows, see section 4 above.
3. **Notebook between cells 25 and 27 — add hello-world section.**
   See section 3 above.
4. **Notebook cell 7 (imports) — add manifest cell.** See section 7
   pattern 2.
5. **Notebook every major analysis cell — add expected-vs-failure
   line.** See section 7 pattern 4.
6. **Notebook parity-plot cell — call `strict_parity_subset(...)`
   explicitly** before plotting, even though the result is currently
   empty. Wire the guard so it cannot be bypassed if a future
   contributor promotes a row to `near-strict`.
7. **`helpers/analysis.py` — add `_assert_adsorbate_intact` and call
   it at the top of `_binding_atom_offset` and `_tilt_angle_deg`.**
   See section 5.1.
8. **`helpers/analysis.py:_tilt_angle_deg` — replace
   `symbols.index("X")` with an ASE-neighbor-list-driven bond check.**
   See section 5.1.
9. **`helpers/config_search.py:build_methanol` and `build_nh3` — add
   one-line comment** stating the function is only safe on
   `ase.build.molecule(...)` output and that any other source
   requires a stoichiometry assert.
10. **`tests/` — add three new tests** described in section 5.3.
11. **Pivot brief / cell 4 — soften industrial claims** unless cited.
    See section 6.
12. **Pivot brief / cell 1 — sharpen the "50% → 87%" framing** to
    explicitly distinguish the AdsorbML balanced-option success rate
    from baseline ML success rates from Table 2 of the AdsorbML paper.
    See section 11.4.

---

## 9. Items already done well

Repeating from section 1 with one extra each:

- Reference manifest separation of `context` / `near-strict` /
  `strict` is rigorous (`references/manifest.yml`).
- Slab construction uses real chemistry libraries.
- Site finding is geometric, not text-based.
- The reference-scope guard `strict_parity_subset` is implemented
  correctly even if not yet wired into the notebook.
- Sign convention for adsorption energy is correctly documented and
  unit-tested (`helpers/analysis.py:73–79`,
  `tests/test_adsorbml_analysis.py`).
- Throughput sweep is a strong endpoint sanity path.
- The pivot brief at `mace_tutorial_adsorbml_pivot.md` already
  contains a well-organized one-page framing of the science the
  notebook should match.
- `helpers/analysis.py:_adsorbate_integrity_status` already detects
  some failure modes (missing adsorbate, floated adsorbate); it is
  the right place to add the dissociation check.

---

## 10. Cross-references

External tutorial-pattern sources fetched on 2026-05-07:

- fairchem quickstart —
  `https://github.com/FAIR-Chem/fairchem/blob/main/docs/core/quickstart.md`
- ACEsuit/mace README —
  `https://github.com/ACEsuit/mace/blob/main/README.md`
- HuggingFace transformers quicktour —
  `https://github.com/huggingface/transformers/blob/main/docs/source/en/quicktour.md`
- scikit-learn getting-started —
  `https://scikit-learn.org/stable/getting_started.html`

External tutorial-pattern sources cited from training knowledge (not
fetched in this session because URL paths drifted or were too narrow
to fetch usefully):

- ASE tutorials (canonical structure: build → calculator → run →
  analyze, smallest system first) —
  `https://wiki.fysik.dtu.dk/ase/`
- fast.ai course notebooks (top-down "show the result first") —
  `https://docs.fast.ai/`
- AiiDA / atomate2 manifest-cell pattern —
  `https://aiida.readthedocs.io/`,
  `https://atomate2.readthedocs.io/`
- PyTorch tutorials inline-assert pattern —
  `https://pytorch.org/tutorials/`

Internal cross-references:

- Existing project plan with execution roadmap:
  `/home/nfedik/projects/tutorials/tutorial_status_and_plan.md`
- Reference manifest:
  `/home/nfedik/projects/tutorials/part-1-nim/references/manifest.yml`
- Manual-checks tracker:
  `/home/nfedik/projects/tutorials/part-1-nim/references/manual_checks.md`
- Domain-expert fact-check packet:
  `/home/nfedik/projects/tutorials/part-1-nim/references/domain_expert_fact_check.md`
- Pivot brief (canonical scope):
  `/home/nfedik/projects/tutorials/mace_tutorial_adsorbml_pivot.md`

---

## 11. Full reference audit

Per-claim verification against the cited primary sources, followed by
a promotion-recommendation summary. Verbatim quotes from cached PDFs
are given so a reader can cross-check without re-fetching.

### 11.0 Up-front reality check

Of the nine references in `references/manifest.yml` and the additional
quantitative claims in `mace_tutorial_adsorbml_pivot.md` and the
notebook intro, two papers are cached locally and fully verifiable
in this session:

- `mace_foundation_model_arxiv_2401.00096.pdf` (153 pages)
- `adsorbml_2023_npj.pdf` (9 pages)

Two more were obtainable in this session via arXiv preprint:

- OC20 (Chanussot 2021) → arXiv:2010.09990 (37 pages, fetched)
- OC22 (Tran 2023) → arXiv:2206.08917 (50 pages, fetched)

Five remain Tier B (paywalled, no usable arXiv preprint in this
session): Hammer/Morikawa/Norskov 1996 PRL, Feibelman 2002 Science,
Greeley/Mavrikakis 2002 J. Catal., Grimme 2011 JCC, Stukowski 2010
MSMSE. Pre-arXiv-era physics journals or older Elsevier/IOP/Science/
APS journals systematically lack a public preprint. The audit records
each as `unverifiable_in_session` rather than inventing a verification.

### 11.1 Per-claim audit table

Columns: claim → source pointer → fetch outcome → exact location →
verdict. The "exact location" column gives page numbers from the
cached PDFs so a domain expert can confirm directly.

| # | Claim | Source pointer | Fetch | Exact location | Verdict |
|---|-------|----------------|-------|----------------|---------|
| 1 | "MACE-MP-0b3+D3 OC157 MAD = 0.38 eV" (helper constant `MACE_MP0B3_OC157_MAD_EV` and pivot brief line 46) | Batatia 2024, arXiv:2401.00096v3 | cached_pdf | p.119: "we observe a strong correlation … providing a Pearson correlation coefficient of 0.86 and an MAD of 0.38 eV" | **verified-strict** |
| 2 | "MACE-MPA-0+D3 OC157 MAD = 0.28 eV" (pivot brief line 56 range) | Batatia 2024, arXiv:2401.00096v3 | cached_pdf | p.122: "MPA-0+D3 achieves a significantly lower MAD (0.28 eV vs. 0.38 eV), RMSD (0.37 eV vs. 0.52 eV), and higher Pearson correlation (0.92 vs. 0.86)" | **verified-strict** |
| 3 | "MACE-MP-0b3+D3 lowest-DFT-energy correctly identified in 126 of 157 systems" | Batatia 2024, arXiv:2401.00096v3 | cached_pdf | p.119: "out of the 157 molecule-surface combinations, the lowest DFT energy configuration was correctly identified by MACE-MP-0b3+D3 for 126 of the surfaces. Fine-tuning further increases this number to 132 for MACE-MP-0b3+D3 FT" | **verified-strict** |
| 4 | `MACE_MPA0_OC157_MAD_EV = 0.42` (helpers/references.py:615) and propagated to notebook cell 3 prose, cell 1126 axis label, cell 1284 figure caption / error bars, and to `manual_checks.md:24` as a "verified" entry. | Batatia 2024, arXiv:2401.00096 | cached_pdf (v3 in manifest) **+ separately fetched v2 in this session for cross-check** | v3 page 119 (MP-0b3, MAD 0.38, 126/157) and page 122 (MPA-0, MAD 0.28). v2 page 100 (MP-0, MAD **0.42**, **121/157**). | **stale-version drift, not fabrication.** The numbers 0.42 eV / 121 of 157 are correctly from arXiv v2 page 100 (the "MACE-MP-0" model, which v3 renamed to MACE-MP-0b3). When v3 added the new MACE-MPA-0 model, the helper constant was *renamed* `MACE_MPA0_OC157_MAD_EV` but the *value* was not re-checked against the new MPA-0 row, which is 0.28 eV (v3 p.122). Net effect: the constant is currently mis-named, mis-valued, and out of date relative to the manifest's pinned arXiv version. Fix: set `MACE_MPA0_OC157_MAD_EV = 0.28` (cite v3 p.122) and either delete `MACE_MP0B3_OC157_MAD_EV = 0.38` or update its v2-v3 numerical bump (0.42 → 0.38) at the same time. The "manual_check: verified" annotation in `manual_checks.md:24` should be re-marked unverified and re-verified against v3. |
| 5 | "AdsorbML balanced option achieves 87.36% success" | Lan 2023, npj Comp. Mat. 9:172 | cached_pdf | p.1 (abstract): "one balanced option finding the lowest energy configuration 87.36% of the time"; p.5 (Fig. 3 caption): "The point highlighted in teal corresponds to the balanced option reported in the abstract — a 87.36% success rate and 2290× speedup" | **verified-strict** |
| 6 | "Baseline ML adsorbate-placement success rate ~50%" | Lan 2023, npj Comp. Mat. 9:172 | cached_pdf | p.3 Table 2: GemNet-OC-MD = 50.05%, GemNet-OC = 46.51%, GemNet-OC-MD-Large = 48.03%, SCN-MD-Large = 51.87%, eSCN-MD-Large = 56.52%; p.4: "a success rate of ~50% could result in a substantial waste of time" | **verified-strict** |
| 7 | Sign convention "ΔE_ads = E_sys − E_slab − E_gas" (helpers/analysis.py:73–79) | Lan 2023, npj Comp. Mat. 9:172 | cached_pdf | p.1, Eq. (1): "ΔE_ads = E_sys − E_slab − E_gas" | **verified-strict** |
| 8 | AdsorbML invalidates dissociated/desorbed/surface-mismatched configs (motivates the section 5 dissociation check) | Lan 2023, npj Comp. Mat. 9:172 | cached_pdf | p.1: "the adsorbate should not be desorbed, i.e., float away ... if the adsorbate has multiple atoms it should not dissociate or break apart"; p.3: "systems not suitable for adsorption energy calculations due to physical constraints are removed, including dissociation, desorption, and surface mismatch" | **verified-strict** |
| 9 | OC20 dataset size = 1,281,040 DFT relaxations | Chanussot 2021, ACS Catal. 11:6059 | arxiv_preprint_substitute (arXiv:2010.09990) | p.1 abstract states 1,281,040 DFT relaxations (extracted verbatim: "consisting of 1,281,040 Density Functional ..."; line break truncated the trailing words but the figure is unambiguous) | **verified-near-strict** (arXiv preprint, not the published ACS version; matches per common practice but the journal-version sha256 is not in the manifest) |
| 10 | OC22 dataset size = 62,331 oxide DFT relaxations (~9.85M single points) | Tran 2023, ACS Catal. 13:3066 | arxiv_preprint_substitute (arXiv:2206.08917) | p.1 abstract: "consisting of 62,331 Density Functional Theory (DFT) relaxations (∼9,854,504 single point calculations) across a range of oxide materials" | **verified-strict** |
| 11 | DFT-D3(BJ) Becke-Johnson damping is what the BGR NIM applies | Grimme 2011 JCC + MACE paper application | cached_pdf (MACE) | MACE p.16: "DFT-D3 with a Becke-Johnson damping function (153), are used in the D3 correction to MACE-MP-0b3" | **verified-context** (the *application* is verified from the MACE paper; the Grimme 2011 primary source is itself unverifiable in session) |
| 12 | MACE training data is MPtrj (and sAlex for MPA-0) | Batatia 2024, arXiv:2401.00096v3 | cached_pdf | p.16: "MACE-MP-0b3 model was trained on the MPtrj dataset which was compiled originally for CHGNet"; p.122: "MPA-0 ... trained on an expanded dataset combining MPtraj and sAlex … sAlex dataset comprises 10.5M structures extracted from the original Alexandria dataset" | **verified-strict** |
| 13 | "0.5 eV decision threshold" (pivot brief line 56) | unattributed | n/a | not in cached MACE or AdsorbML | **unverified** — recommend rephrase as tutorial convention |
| 14 | "100–1000× DFT speedup" (pivot brief) | partial; AdsorbML reports 2290× | cached_pdf | AdsorbML p.5 Fig. 3 caption: "2290× speedup" | **partial-mismatch** — recommend cite AdsorbML 2290× and note workflow-specific |
| 15 | Industrial-user list (BASF, ExxonMobil, Shell, Chevron, Siemens Energy, Ørsted, Topsoe; pivot brief) | n/a | n/a | n/a | **unverified** — not in any cached source; recommend remove or cite public case studies |
| 16 | Hammer 1996 CO/Pd(111) fcc-hollow puzzle | Hammer/Morikawa/Norskov 1996 PRL 76:2141 | publisher_403 (APS) | n/a | **unverifiable_in_session** — pre-arXiv-era; escalate |
| 17 | Feibelman 2002 water/Ru(0001) partial dissociation | Feibelman 2002 Science 295:99 | publisher_403 (Science) | n/a | **unverifiable_in_session**. Existing manual_checks.md already flags this is a Ru reference, *not* a Cu/Pd reference — important. |
| 18 | Greeley/Mavrikakis 2002 CH3OH/Cu(111) | Greeley/Mavrikakis 2002 J Catal | publisher_403 (Elsevier) | n/a | **unverifiable_in_session** — escalate |
| 19 | Stukowski 2010 OVITO software paper | Stukowski 2010 MSMSE | publisher_403 (IOP) | n/a | **unverifiable_in_session** — software citation; not load-bearing for science |

### 11.2 Pair-level adsorption-energy audit (the 9 active pairs)

The notebook's per-pair `context` rows in `manifest.yml` carry
`parity_requirements: TODO`. To convert any of these to `near-strict`,
the manifest needs five fields confirmed from a primary source: slab
geometry, coverage, functional, dispersion convention, sign
convention. None of the nine can be promoted from cached evidence
alone — primary sources are paywalled. The audit records what blocks
each promotion.

| Pair | Best-known primary-source candidate | Required modeling-detail match | Verifiable in session? | Block |
|------|--------------------------------------|---------------------------------|------------------------|-------|
| CO / Cu(111) | Need a low-coverage, top-site-preferred PBE+D3 reference. The Hammer/Morikawa/Norskov 1996 PRL discusses Pd, not Cu. Bagus/Pacchioni line in current manifest needs confirmation. | (1/8) coverage; PBE-D3(BJ); ≥4-layer; relaxed top 2 layers; sign convention | No | primary source is paywalled or not yet identified. Escalate to domain expert. |
| CO / Pd(111) | Hammer/Morikawa/Norskov 1996 PRL 76:2141 (already in manifest) is the canonical fcc-hollow vs top reference. | Same fields; APS PDF paywalled | No | publisher_403; escalate. |
| CO / α-Al2O3(0001) | No specific reference in current manifest. OC22 (verified) covers oxides broadly but each (oxide, adsorbate) row in OC22 requires identifying the specific record ID, not just the dataset. | Same fields plus termination (Al-vs-O); coverage; functional | No | needs OC22 record-ID lookup or separate primary source. |
| H2O / Cu(111) | Manifest currently uses Feibelman 2002 (Ru(0001)) as context only, which is correct since Ru ≠ Cu. A Cu-specific reference is needed. | Same fields; PBE-D3(BJ); single H2O molecule low-coverage | No | no Cu-specific primary source identified. |
| H2O / Pd(111) | Same caveat as Cu — Feibelman is Ru. | Same fields | No | no Pd-specific primary source identified. |
| H2O / α-Al2O3(0001) | Water-on-alumina often reports dissociative adsorption; this complicates parity. | Termination, dissociation pathway, functional | No | needs domain-expert review of which surface termination matches the tutorial slab. |
| CH3OH / Cu(111) | Greeley/Mavrikakis 2002 J Catal (in manifest as context) is a candidate. | Same fields; coverage; PBE-D3(BJ) | No | publisher_403 (Elsevier); escalate. |
| CH3OH / Pd(111) | No specific Pd-CH3OH reference in current manifest. | Same fields | No | needs primary source identification. |
| CH3OH / α-Al2O3(0001) | No specific reference in current manifest. | Same fields plus termination | No | needs primary source identification. |

### 11.3 Promotion recommendations

| Citation key | Current status | Recommended status | Reason |
|--------------|----------------|--------------------|--------|
| `mace_foundation_model` | context | **strict** for the OC157 MAD/count claims (rows 1, 2, 3, 12 above). | Cached PDF, exact page references confirmed (p.119, p.122, p.16). |
| `adsorbml_2023` | context | **strict** for rows 5, 6, 7, 8 above. | Cached PDF, exact page references confirmed (p.1, p.3, p.5). |
| `oc20_dataset` | context | **near-strict** for dataset-size and methodology claims only. | arXiv preprint substitute confirmed; tutorial does not yet pin specific OC20 record IDs to per-pair pairs, so cannot promote to strict for adsorption energies. |
| `oc22_dataset` | context | **near-strict** with same caveat as OC20. | Same. |
| `co_pd111_hammer_1996` | context | leave **context** | Unverifiable in session; promotion requires manual fetch via library access. |
| `water_feibelman_2002` | context | leave **context** AND add explicit caveat in the manifest that this is a Ru(0001) reference, not a Cu/Pd reference. | Already noted in `tutorial_status_and_plan.md`; should be enforced in manifest's `parity_requirements` field. |
| `methanol_cu_greeley_2002` | context | leave **context** | Unverifiable in session. |
| `d3_bj_grimme_2011` | context | leave **context** | The *application* is verified via the MACE paper; the Grimme 2011 primary is paywalled but is a methods citation, not a parity reference. |
| `ovito_stukowski_2010` | context | leave **context** | Software citation; not load-bearing. |

**New citations recommended for addition** (currently missing rows
that the audit identified):

- A primary or near-primary source for the "0.5 eV decision threshold"
  if it is to remain in the prose. Otherwise, rephrase as a tutorial
  convention.
- A specific OC20 / OC22 record-ID per (host, adsorbate) pair if any
  pair is ever to be promoted to `near-strict`.
- Direct Cu(111) and Pd(111) low-coverage water references, since
  Feibelman is not appropriate for those pairs.

### 11.4 Honest unknowns (to escalate)

The following claims could not be verified in this session and should
be added to `references/manual_checks.md` as a domain-expert checklist
(this report does not edit `manual_checks.md` directly; that is a
follow-up):

1. ~~The exact origin of the "0.42 eV MAD on OC157" / "121 of 157" claim~~
   **Resolved during this session:** the numbers come from MACE
   arXiv v2 page 100 (verified by separately fetching v2:
   `arxiv.org/pdf/2401.00096v2`, 119 pages, abstract block on p.100
   reads `"Pearson correlation coefficient of 0.83 and an MAD of
   0.42 eV. ... lowest DFT energy configuration was correctly
   identified by MACE-MP-0 for 121 of the surfaces"`). The drift
   pattern is "v2 → v3 update, downstream artifacts didn't follow."
2. The arXiv-v3 vs published-journal-version question for MACE.
   arXiv:2401.00096v3 is dated 4 Sep 2025 — the cached file. If the
   journal version (Nature Computational Science or Nature
   Communications) appears with different numbers, the manifest row
   should pin the version explicitly.
3. Per-pair primary sources for all nine adsorption pairs (section
   11.2 above).
4. The industrial-user list (row 15 above).
5. The "0.5 eV decision threshold" provenance (row 13 above).

### 11.5 Out of scope for this audit

- Re-running DFT or MLIP calculations to verify numbers
  experimentally.
- Domain-expert judgement on whether a slab model in a published
  paper is "close enough" to the tutorial's slab for `near-strict`
  status.
- Audit of Part 2 references (per the user's narrowing).
- Editing `manual_checks.md` or `manifest.yml` directly (this report
  is a recommendations document; manifest edits are a follow-up
  task).
