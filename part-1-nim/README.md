<p align="center">
  <img src="assets/images/logos/nvidia-logo.png" alt="NVIDIA" height="55"/>
</p>

---

# Catalyst Adsorption Configuration Search with NVIDIA ALCHEMI

**Interactive Jupyter tutorial.** This notebook demonstrates an AdsorbML-style configuration search for molecular adsorption on catalyst surfaces. The scientific task is deliberately narrow: generate several plausible starting geometries for an adsorbate on a surface, relax them with the NVIDIA ALCHEMI Batch Geometry Relaxation (BGR) NIM, and compare the lowest-energy relaxed structures with literature or Open Catalyst reference context.

The active panel is CO, H2O, and CH3OH on Cu(111), Pd(111), and alpha-Al2O3(0001). These systems are useful because they expose the central methodological issue in adsorption screening: a single chemically reasonable starting geometry can relax into a local minimum, while a batched configuration search can reveal a lower-energy site.

The notebook uses the **MACE-MPA-0** foundation model with **DFT-D3(BJ)** dispersion through the BGR NIM. Quantitative claims are intentionally scoped. Published model-level errors provide an uncertainty envelope, but strict tutorial-level validation requires exact matching reference records for slab model, coverage, functional, dispersion convention, frozen layers, and energy sign convention. Until those records are pinned in the reference manifest, per-pair literature values are treated as contextual checkpoints rather than strict parity data.

Archived context:

- The earlier atmospheric-water-harvesting pivot materials are under `_archive/awh-pivot-sources/`.
- The earlier OER catalyst-screening tutorial is under `_archive/oer-catalyst-screening/`.

## Docker Deployment

### Prerequisites

| Requirement | Details |
|-------------|---------|
| NGC API key | Get one at [build.nvidia.com](https://build.nvidia.com) |
| Docker + Compose | `docker compose` v2 plugin |
| GPU | NVIDIA GPU; the full panel is intended for a data-center GPU such as A100/H100/B200 or comparable workstation hardware |

### Setup

Configure your NGC API key locally:

```bash
cp .env.example .env
# Edit .env and set NGC_API_KEY=<your-key>
```

SSH into your login host and allocate a GPU node from the repository root:

```bash
ssh <login-host>
./start
# Note the compute node hostname, for example dgx-node-42.
```

Then deploy the full Part 1 stack:

```bash
cd part-1-nim
./scripts/deploy.sh setup <login-host> <compute-node>
```

Access JupyterLab at `http://localhost:8888` and Grafana at `http://localhost:3000` (admin/admin).

### Management

```bash
./scripts/deploy.sh status       # Jupyter URL, BGR health, Grafana
./scripts/deploy.sh restart      # Sync local changes, restart stack
./scripts/deploy.sh pull-changes # Sync remote Jupyter edits to local
./scripts/deploy.sh stop         # Tear down stack, close tunnels
```

## BGR NIM Configuration

The Docker Compose stack configures the BGR NIM with:

| Setting | Variable | Value |
|---------|----------|-------|
| Model family | `ALCHEMI_NIM_MODEL_TYPE` | `mace` |
| Boundary conditions | `ALCHEMI_NIM_PBC` | `true` |
| Optimizer preset | `ALCHEMI_NIM_BGR_OPTIMIZER_PRESET` | `materials` |
| Dispersion corrections | `ALCHEMI_NIM_DFT3_ENABLED` | `true` |
| Shared memory | `--shm-size` | `8g` |

The notebook queries the NIM runtime metadata before the scientific workflow. Do not hard-code the deployed checkpoint or service metadata in the prose.

## Runtime Modes

`SMALL_PANEL_MODE = True` runs a one-pair smoke case suitable for debugging notebook mechanics. `SMALL_PANEL_MODE = False` runs the full AdsorbML panel.

`USE_CACHED_RESPONSES = True` replays cached BGR responses once cache fixtures have been created. `USE_CACHED_RESPONSES = False` calls the live BGR endpoint.

`BACKEND = "bgr_nim"` runs the NIM/cache route. `BACKEND = "toolkit"` runs the
native Toolkit route through `AtomicData`, `Batch.from_data_list`,
`MACEWrapper`, `PipelineModelWrapper`, and `FIRE2`. Toolkit selection is not a
fallback: if the native package/API, checkpoint, or explicit D3(BJ) parity
configuration is missing, the notebook fails before any BGR call is attempted.

## Scientific Scope

The tutorial is about adsorption configuration search, not a complete catalyst-discovery workflow. The calculations report electronic adsorption energies for isolated adsorbates at low coverage. They do not compute activation barriers, electrochemical free energies, explicit solvent effects, coverage-dependent lateral interactions, temperature/entropy corrections, or magnetic/open-shell chemistry.

The reusable backend-neutral science contract lives in
[`../shared/adsorption_tutorial`](../shared/adsorption_tutorial/). Part 1 is the
BGR NIM implementation of that contract; the toolkit implementation should emit
the same result schema before the two versions are compared.

The reference layer is deliberately conservative:

- `context` rows support interpretation and plotting.
- `near-strict` rows may support limited quantitative comparison when only minor modeling details differ.
- `strict` rows require an exact manifest-backed match to the tutorial's slab, adsorbate, coverage, functional, dispersion, frozen-layer convention, and sign convention.

## References To Verify

Primary references currently used by the notebook or plan:

1. Batatia, I. et al. "A foundation model for atomistic materials chemistry." arXiv:2401.00096.
2. Lan, J. et al. "AdsorbML: a leap in efficiency for adsorption energy calculations using generalizable machine learning potentials." *npj Computational Materials* 9, 172 (2023).
3. Chanussot, L. et al. "Open Catalyst 2020 (OC20) Dataset and Community Challenges." *ACS Catalysis* 11, 6059 (2021).
4. Tran, R. et al. "The Open Catalyst 2022 (OC22) Dataset and Challenges for Oxide Electrocatalysts." *ACS Catalysis* 13, 3066 (2023).
5. Hammer, B., Morikawa, Y. and Norskov, J. K. "CO chemisorption at metal surfaces and overlayers." *Physical Review Letters* 76, 2141 (1996).
6. Grimme, S. et al. "Effect of the damping function in dispersion corrected density functional theory." *Journal of Computational Chemistry* 32, 1456 (2011).
7. Stukowski, A. "Visualization and analysis of atomistic simulation data with OVITO." *Modelling and Simulation in Materials Science and Engineering* 18, 015012 (2010).

See `references/manual_checks.md` and `references/manifest.yml` for the verification state before promoting any contextual number into strict validation.

## License

Apache 2.0 -- see [LICENSE](../LICENSE).
