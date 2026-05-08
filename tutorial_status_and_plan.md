# Tutorial Status Review And Consolidated Plan

Date reviewed: 2026-05-04 16:41 MDT

## Scope Reviewed

- Root docs: `README.md`
- Part 1 NIM tutorial: `part-1-nim/alchemi-mace-adsorption-search.ipynb`
- Part 1 helpers/tests/assets: `part-1-nim/helpers/`, `part-1-nim/tests/`, `part-1-nim/assets/`
- Canonical pivot brief: `mace_tutorial_adsorbml_pivot.md`
- Part 2 toolkit notebooks/docs: `part-2-toolkit/alchemi-toolkit-sandbox.ipynb`, `part-2-toolkit/melting-point-slc.ipynb`, `part-2-toolkit/OLED-melting-point-case-study.md`

Tooling notes:

- Jupyter MCP notebook reading works with the absolute notebook path.
- Jupyter MCP `open_file` timed out in this session, so direct UI-open status is not reliable yet.
- The local shell base environment is not the tutorial environment: `ase`, `ovito`, and `pytest` are missing, and the declared `alchemi-playbook` conda env is not installed locally.
- A Claude/Opus plugin was not exposed by tool discovery. Two read-only planning agents were spawned with the available Codex agent runtime instead.
- Web checks were done for the major primary-source claims. The open MACE arXiv PDF and AdsorbML PDF were downloaded into `part-1-nim/references/pdfs/`; restricted or non-PDF responses are listed in `part-1-nim/references/manual_checks.md`.

## Current Status

### 1. Canonical Theme

The active Part 1 notebook is now an AdsorbML-style catalyst surface adsorption configuration-search tutorial. The root README and `part-1-nim/README.md` still describe the older atmospheric water harvesting tutorial, so public documentation is stale and conflicts with the notebook.

Decision needed: keep AdsorbML as canonical, and mark AWH/OER as archived context. This matches `mace_tutorial_adsorbml_pivot.md`.

### 2. Part 1 Notebook Structure

The Part 1 notebook is structurally close to the target flow:

1. Motivation and scope.
2. Control panel.
3. Endpoint and NIM metadata check.
4. Gas-phase H2O and throughput sweep.
5. Host slab construction.
6. Adsorbate construction.
7. Clean-slab relaxation.
8. Configuration grid generation.
9. Batch relaxation.
10. Binding-energy distributions.
11. Site/reference comparison.
12. AdsorbML bias plot.
13. Discovery plot.
14. Narrative conclusion and limitations.

Main gaps:

- Academic language needs a pass. Some claims are too broad or too promotional for a scientific tutorial.
- There is a stray direct-edit marker and `111` literal in the control-panel cell.
- Durable conceptual visuals are absent. Current figures are generated analysis outputs, not tutorial assets.
- The active notebook uses CO, H2O, and CH3OH, while `helpers/references.py` also contains NH3 rows. Tests still expect exactly 9 AdsorbML rows, so the reference table and notebook panel are inconsistent.
- No active cached BGR response JSONs are present under `part-1-nim/cached_responses/`.

### 3. References And Evidence

The code correctly separates context references from strict parity references. At present, all AdsorbML panel rows are `context`; there are zero strict or near-strict parity rows.

That is scientifically correct for the current evidence level. It also means the notebook must not present any MACE-vs-DFT energy comparison as validated parity until exact OC20/OC22/AdsorbML records are pinned with matching slab, coverage, functional, dispersion, and sign convention.

Major source checks already confirmed:

- AdsorbML DOI: https://doi.org/10.1038/s41524-023-01121-5
- OC20 DOI: https://doi.org/10.1021/acscatal.0c04525
- OC22 DOI: https://doi.org/10.1021/acscatal.2c05426
- MACE foundation model arXiv DOI: https://doi.org/10.48550/arXiv.2401.00096
- CO/Pd reference DOI: https://doi.org/10.1103/PhysRevLett.76.2141
- Feibelman water/Ru reference DOI: https://doi.org/10.1126/science.1065483
- Methanol/Cu reference DOI found via search: `10.1006/jcat.2002.3586`

Downloaded PDFs:

- `part-1-nim/references/pdfs/mace_foundation_model_arxiv_2401.00096.pdf`
- `part-1-nim/references/pdfs/adsorbml_2023_npj.pdf`

Official direct PDF downloads that did not succeed in this environment:

- ACS OC20 and OC22 returned HTTP 403.
- APS CO/Pd returned HTTP 403.
- Science water/Ru returned HTTP 403.
- IOP OVITO returned HTML rather than a PDF.

Important caveat: Feibelman 2002 is a water/Ru(0001) partial-dissociation reference, not a strict single-H2O-on-Cu/Pd parity source. It can motivate non-intuitive water configurations, but direct Cu/Pd water references are still required for strict pair checks.

### 4. Scientific Verification

Promising foundations already exist:

- Slabs are generated with `pymatgen.core.surface.SlabGenerator`.
- Slab c axes are orthogonalized.
- Bottom-half `active_mask` logic exists.
- Final sites are classified geometrically instead of trusting starting labels.
- Context references are kept out of strict statistics.

Missing verification:

- Miller-index correctness for Cu(111), Pd(111), and Al2O3(0001).
- Layer count and termination checks.
- Vacuum thickness and no-image-interaction checks.
- Active-mask tests proving the bottom layers are frozen and adsorbates are relaxable.
- fcc/hcp/bridge/top site correctness, especially second-layer stacking for fcc vs hcp.
- Al2O3 site finder must not silently fall back to generic sites without a visible failure.
- Gas-phase adsorbate orientation and binding atom checks.
- Reference-energy sign convention checks.
- NIM metadata capture for model, checkpoint, D3 setting, PBC, and optimizer preset.

### 5. Visual Assets

Current committed Part 1 assets:

- `part-1-nim/assets/images/logos/nvidia-logo.png`
- `part-1-nim/assets/images/logos/ovito_logo.png`

Missing durable assets:

- Banner image.
- Workflow image.
- Phenomenon image explaining local minima from different starting configurations.
- Four separate NV-style phenomenon icons.
- Programmatic canvas that composes icons, labels, workflow arrows, and notebook section anchors.

Generated notebook figures such as `throughput_scaling.png`, `binding_distribution.png`, `site_agreement_heatmap.png`, `adsorbml_bias.png`, and `discovery_plot.png` should remain analysis outputs under `part-1-nim/assets/images/plots/`. They should not replace the conceptual banner/workflow visuals.

### 6. Part 2 Toolkit Status

Part 2 has two different artifacts:

- `alchemi-toolkit-sandbox.ipynb`: almost empty, just a title and `import nvalchemi`.
- `melting-point-slc.ipynb`: substantial code for a solid-liquid coexistence melting-point workflow, but essentially no explanatory markdown.

The toolkit Dockerfile installs `nvalchemi-toolkit` from GitHub without a tag or commit. This creates drift risk. A pinned toolkit version or commit is required before claiming reproducibility.

## Recommended Scientific Scope

Keep the active Part 1 tutorial centered on AdsorbML-style configuration search for molecular adsorption on surfaces.

Use water harvesting, OER catalysis, and NH3 synthesis as examples of the broader "first sorption/adsorption step" motif only if the language stays precise:

- Water harvesting: first H2O binding to a hydrophilic site is relevant, but flat slabs do not model pore filling, capillary condensation, or full AWH selectivity.
- OER catalysis: H2O/OH adsorption on oxide surfaces is relevant, but full OER needs electrochemical free energies, solvent, proton-electron transfer, and often radical/open-shell intermediates.
- NH3 synthesis: N2 or NH3 adsorption is relevant, but N2 dissociation and Fe/Ru catalyst chemistry involve barriers, coverage effects, and sometimes magnetic/open-shell complications.
- Heterogeneous catalysis: CO/Pd(111) top vs hollow competition is the cleanest active-tutorial example because it directly demonstrates why configuration search matters.

## Full Execution Plan

### Phase 0: Freeze Scope And Baseline

Deliverables:

- A short scope note in the root README: Part 1 is AdsorbML/NIM; older AWH and OER notebooks are archived.
- A worktree status snapshot before edits.
- A list of user/unrelated uncommitted files that will not be touched.

Acceptance criteria:

- Root README and Part 1 README no longer contradict the active notebook.
- The tutorial name, notebook title, and deployment docs describe the same scientific task.

### Phase 1: Academic Language Pass

Actions:

- Rewrite introductory markdown in a human academic style: direct, precise, cited, and modest.
- Replace phrases such as "production catalyst workflow", "standard workflow", and "one sentence to memorise" with discipline-appropriate language.
- Add a short interpretive sentence before every major code block: what question the cell answers, what valid output means, and what failure means.
- Soften industry claims unless directly cited.
- Keep the scientific uncertainty hierarchy, but distinguish published model-level uncertainty from tutorial-specific validation.

Acceptance criteria:

- No broad industrial adoption claim appears without citation or hedging.
- Every figure/table has an interpretation paragraph.
- Context-only references are described as context, not validation.

### Phase 2: Visual System And Programmatic Canvas

Asset structure:

- `part-1-nim/assets/icons/icon_configuration_search.png`
- `part-1-nim/assets/icons/icon_water_first_binding.png`
- `part-1-nim/assets/icons/icon_oer_first_adsorption.png`
- `part-1-nim/assets/icons/icon_nh3_surface_binding.png`
- `part-1-nim/assets/images/v0_core/banner_adsorbml_bgr.png`
- `part-1-nim/assets/images/v0_core/workflow_adsorbml_bgr.png`
- `part-1-nim/assets/images/v0_core/phenomenon_local_minima.png`
- `part-1-nim/scripts/build_visual_assets.py`

Icon concepts:

1. Catalysis/configuration search: CO above Pd(111), with top/bridge/hollow candidates and one lower-energy minimum.
2. Water harvesting first sorption step: H2O binding to an isolated hydrophilic site, explicitly labelled as first binding, not full pore condensation.
3. OER first adsorption step: H2O or OH approaching an oxide active site, with a limitation note that electrochemical free energies are out of scope.
4. NH3 synthesis/small-molecule activation: N2/NH3 near a metal surface, framed as adsorption/binding only, not N2 dissociation kinetics.

Canvas plan:

- Generate icons separately.
- Programmatically compose one horizontal workflow canvas:
  `surface + adsorbate panel -> configuration grid -> BGR batch relaxation -> energy/site ranking -> reference/manual validation`.
- Use a restrained NV visual style: black/graphite background, NVIDIA green accents, white labels, minimal line art, no decorative gradients.
- Store the script and final PNG so the canvas can be regenerated.

Acceptance criteria:

- Banner appears before the first technical markdown section.
- Workflow canvas appears before the control panel.
- Phenomenon image appears before the configuration-grid section.
- Image files are deterministic artifacts or have prompts/scripts recorded.

### Phase 3: Reference PDF And Statement Verification

Create:

- `part-1-nim/references/manifest.yml`
- `part-1-nim/references/pdfs/`
- `part-1-nim/references/manual_checks.md`
- `part-1-nim/references/link_check_ignore.txt`

Each manifest row must include:

- Citation key.
- DOI or official URL.
- PDF/SI path.
- SHA256 checksum.
- Exact table, figure, page, or dataset row.
- Claim supported.
- Whether it supports context, near-strict parity, or strict parity.
- Slab/facet.
- Layers.
- Supercell/coverage.
- Functional.
- Dispersion.
- Frozen-layer convention.
- Sign convention.
- Manual reviewer initials/date.

Manual checks required before strict parity:

- AdsorbML: verify the single-start reliability and batch-search reliability numbers from the paper/SI, including the exact meaning of "random", "heuristic", and "ML-relaxed" configurations.
- MACE: verify OC157 MAD values for MACE-MPA-0 and MACE-MP-0b3 from the exact version used in the notebook. Note that arXiv v3 and the 2025 journal version may differ.
- OC20: verify dataset size, adsorbate/surface record availability, and whether the tutorial pair geometries have exact matching records.
- OC22: verify oxide dataset size and whether clean Al2O3(0001) with each adsorbate has a matching row.
- CO/Pd(111): verify fcc-hollow site and energy from Hammer/Morikawa/Norskov or a matching OC20 record.
- CO/Cu(111): find a direct low-coverage, same-facet source for top-site preference and energy. The current Bagus/Pacchioni citation needs exact bibliographic confirmation.
- H2O/Cu(111) and H2O/Pd(111): replace Feibelman-as-context with direct single-water low-coverage sources if these pairs remain in the strict panel.
- CH3OH/Cu(111) and CH3OH/Pd(111): verify exact low-coverage adsorption mode, energy, and functional/coverage convention.
- Al2O3(0001): verify termination and molecular-vs-dissociative adsorption for CO, H2O, CH3OH, and NH3 if used.
- Industrial-user claims: either cite public case studies or rewrite as "representative industrially relevant problem classes."

Acceptance criteria:

- Link checker passes for public links, with localhost ignored.
- Every quantitative claim in markdown maps to a manifest row.
- No reference row is promoted beyond `context` without manifest evidence.

### Phase 4: Geometry And Chemistry Verification

Add tests, preferably under `part-1-nim/tests/test_geometry_validation.py`:

- Cu and Pd slab Miller index is (111) using pymatgen slab metadata or an independent plane-normal check.
- Al2O3 slab Miller index is (0001) and termination is reported explicitly.
- Slab layer count and bottom/frozen layer count match notebook claims.
- Vacuum thickness is at least the target after adsorbate placement.
- `active_mask` freezes only intended slab atoms and never freezes adsorbate atoms.
- fcc, hcp, bridge, and top sites are geometrically distinct and central.
- fcc/hcp labels are checked against second-layer stacking, not just nearest-neighbor heuristics.
- Starting configurations have no unphysical overlaps.
- Adsorbate binding atoms match intended orientation: C-down CO, O-down H2O/CH3OH, N-down NH3.
- Adsorption energy sign convention is tested against a simple fixture.

Optional API/library checks:

- `pymatgen.core.surface.SlabGenerator`
- `pymatgen.analysis.adsorption.AdsorbateSiteFinder`
- `ase.build.fcc111` and `ase.build.add_adsorbate` as independent fcc site references.
- `spglib` or pymatgen symmetry tools for slab orientation sanity checks.
- RDKit/OpenBabel for future SMILES/XYZ formula and connectivity checks.

Acceptance criteria:

- Helper tests run in the tutorial environment.
- Hard-to-tokenize chemical assertions have executable checks or are flagged in `manual_checks.md`.

### Phase 5: Notebook Execution Through MCP

Execution policy:

- Do not run the full notebook until references, cache behavior, and BGR endpoint state are controlled.
- First run notebook cells in small-panel mode with `USE_CACHED_RESPONSES=True` after caches exist.
- Then run small-panel live mode against BGR.
- Only then run the full 9-pair live panel.

MCP procedure:

1. Read notebook cells with `read_notebook_cells`.
2. Patch cells with `edit_cell` only after backing up the intent in git status.
3. Run individual cells with `run_cell` for imports/control/geometry checks.
4. Use `run_all_cells` only for controlled cached smoke execution.
5. Re-read outputs with `read_notebook_cells`.

Execution artifacts:

- `part-1-nim/outputs/notebook_run_summary.json`
- Generated analysis figures under `part-1-nim/assets/generated/` or `part-1-nim/outputs/figures/`
- Cached BGR responses under `part-1-nim/cached_responses/adsorption-search/`

Acceptance criteria:

- Small-panel cached run completes.
- Small-panel live BGR run completes and records NIM metadata.
- Full-panel run either completes or has a documented failure mode with the exact cell and error.

### Phase 6: Code Cleanup And Test Alignment

Actions:

- Remove the control-cell direct-edit marker and `111` literal.
- Decide whether NH3 is part of the active panel or only an icon/context example.
- If NH3 remains in `ADSORBML_REFERENCES`, update tests and notebook panel logic to handle 12 rows.
- If NH3 is not active, move NH3 rows into a separate optional-reference table.
- Add a `reference_scope` guard so plots cannot accidentally display context values as strict parity.
- Keep notebooks thin and move reusable logic into tested Python modules.

Acceptance criteria:

- `python -m pytest -q` passes in the tutorial environment.
- Notebook import cell imports only maintained public helper APIs.
- No stale AWH code path is used by the active AdsorbML notebook.

### Phase 7: Modularity And Toolkit/NIM Sync

Goal: keep the NIM version and future toolkit version synchronized without manually copying notebook logic.

Recommended structure:

- `shared/adsorption_tutorial/panel.py`: hosts, adsorbates, configuration-grid declarations.
- `shared/adsorption_tutorial/references.py`: manifest-backed reference loading.
- `shared/adsorption_tutorial/geometry_checks.py`: Miller, site, layer, active-mask checks.
- `shared/adsorption_tutorial/analysis.py`: adsorption energy, site classification, summary tables.
- `shared/adsorption_tutorial/visuals.py`: shared plotting helpers.
- `part-1-nim/helpers/backend_bgr.py`: BGR/NIM execution adapter.
- `part-2-toolkit/helpers/backend_toolkit.py`: toolkit/ASE execution adapter.

Notebook policy:

- Notebooks should orchestrate, explain, and display; tested modules should compute.
- Use one reference manifest for both versions.
- Use one visual-generation script for both versions where possible.
- Consider Jupytext or a notebook build script only after the shared module boundary is stable.

Toolkit-specific actions:

- Pin `nvalchemi-toolkit` to a tag or commit in `part-2-toolkit/Dockerfile`.
- Print and assert toolkit version/commit in the first notebook code cell.
- Add a lightweight toolkit import/API smoke test.
- Decide whether Part 2 remains the melting-point SLC tutorial or becomes the toolkit counterpart of the AdsorbML adsorption tutorial.

Acceptance criteria:

- Shared tests verify both backends consume the same panel/reference definitions.
- A change to the panel or references updates both notebooks automatically.
- Toolkit image builds reproducibly from a pinned dependency.

### Phase 8: Part 2 Tutorial Completion

If Part 2 remains the SLC melting-point tutorial:

- Convert `OLED-melting-point-case-study.md` into academic notebook markdown cells.
- Add banner, workflow canvas, and solid-liquid interface phenomenon diagram.
- Start with naphthalene as the validation case, not unvalidated UDC OLED candidates.
- Mark UDC molecule thermal data as an unresolved data gap unless cached evidence supports the search claim.
- Add version pinning, toolkit smoke tests, and short dry-run settings.
- Add full-run settings separately from smoke settings.

If Part 2 becomes the toolkit version of Part 1:

- Reuse the AdsorbML panel and reference manifest.
- Replace BGR execution with toolkit calculator execution.
- Keep identical analysis cells and compare NIM vs toolkit outputs where scientifically meaningful.

Decision required before major Part 2 edits.

## Immediate Next Steps

1. Update README files to match AdsorbML canonical scope.
2. Remove obvious notebook cleanup artifacts.
3. Create the reference manifest skeleton and manual-check list.
4. Add geometry verification tests before changing more scientific content.
5. Generate the four separate NV-style icons and a programmatic workflow canvas.
6. Run notebook smoke execution through MCP in cached small-panel mode after cache fixtures exist.
7. Decide whether NH3 is active tutorial chemistry or visual/context-only chemistry.
8. Pin toolkit dependency and decide the Part 2 role.

## Done Criteria For A Publishable Tutorial

- Public docs, notebook title, and deployment instructions agree.
- Academic prose is precise and cited.
- Conceptual banner/workflow/phenomenon visuals are present and reproducible.
- Link checker passes.
- Reference PDFs/SI files and checksums are present where legally downloadable.
- Every quantitative claim maps to a manifest row.
- Strict parity plots contain only strict/near-strict rows.
- Geometry/site/Miller-index tests pass.
- Small-panel cached notebook run passes.
- Small-panel live BGR run passes.
- Full-panel run is completed or explicitly marked as pending with a reproducible blocker.
- Toolkit dependency is pinned.
- Shared modules prevent drift between NIM and toolkit versions.
