<p align="center">
  <img src="assets/nvidia-logo.png" alt="NVIDIA" height="55"/>
</p>

---

# Atmospheric Water Harvesting — MACE-MP-0 Screening with NVIDIA ALCHEMI

**Interactive Jupyter tutorial** — screen six inorganic sorbent frameworks (zeolite chabazite, SAPO-34, MFI silicalite, α-Al₂O₃, rutile TiO₂, monoclinic ZrO₂) for H₂O adsorption strength using GPU-accelerated machine-learning interatomic potentials.

Built on the **NVIDIA ALCHEMI Batch Geometry Relaxation (BGR) NIM** running the **MACE-MPA-0** foundation model with DFT-D3(BJ) dispersion corrections, the notebook walks learners through a scientifically honest discovery workflow: validate against published DFT benchmarks (Batatia 2024 S24, Plessow 2024 CCSD(T)/CBS for H-MFI, Fischer 2015 CP2K for H-SAPO-34), then extend to a host without published reference data (ZrO₂) and state the uncertainty explicitly.

Beyond the science, the notebook doubles as a hands-on introduction to the ALCHEMI NIM: wire protocol, batch relaxation, active-mask constraints, `FAST_DEMO` cached replay, and Prometheus/Grafana observability.

## Docker Deployment

### Prerequisites

| Requirement | Details |
|-------------|---------|
| NGC API key | Get one at [build.nvidia.com](https://build.nvidia.com) |
| Docker + Compose | `docker compose` (v2 plugin) — used to build and run the container environment |
| GPU | NVIDIA GPU (tested: A100, H100, B200, L40S, RTX 6000 Ada) |

### Setup

Configure your NGC API key locally:

```bash
cp .env.example .env
# Edit .env and set NGC_API_KEY=<your-key>
```

SSH into your login host and allocate a GPU node (see `./start` for an example SLURM script):

```bash
ssh <login-host>
./start
# Note the compute node hostname (e.g., dgx-node-42)
```

Then from your local machine, deploy the full stack:

```bash
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

### BGR NIM Configuration

The Docker Compose stack configures the BGR NIM with:

| Setting | Variable | Value |
|---------|----------|-------|
| Model | `ALCHEMI_NIM_MODEL_TYPE` | `mace` (MACE-MPA-0) |
| Boundary conditions | `ALCHEMI_NIM_PBC` | `true` |
| Optimizer preset | `ALCHEMI_NIM_BGR_OPTIMIZER_PRESET` | `materials` |
| Dispersion corrections | `ALCHEMI_NIM_DFT3_ENABLED` | `true` (DFT-D3(BJ)) |
| Shared memory | `--shm-size` | `8g` |

The notebook queries the NIM's runtime metadata in cell 5 and prints the deployed model version + checkpoint — do not hard-code assumptions.

### Monitoring

Prometheus scrapes BGR metrics at `/v1/metrics`. View them in Grafana at `localhost:3000` (datasource auto-provisioned). The BGR status endpoint is also available at `localhost:8000/v1/status`.

## FAST_DEMO Mode

The notebook defaults to `FAST_DEMO = False` — it will call a **live BGR endpoint**. Set `FAST_DEMO = True` in the control panel cell to use pre-cached JSON responses in `cached_responses/water-sorbents/` for fully offline operation. Recommended for workshop environments without GPU access.

## Scope and caveats

MACE-MP-0 has known limitations that the tutorial explicitly surfaces at the points where they matter:

- **12 Å receptive field** — no long-range dispersion beyond that cutoff.
- **No explicit spin** — every system in the panel is closed-shell singlet (no magnetic 3d transition metals, no reducible cations, no f electrons).
- **MPtrj training-set gaps** — the training data contains no gas-phase molecules, no surface slabs, and no MOFs (which is why MOF-based AWH champions are deliberately out of scope).
- **No free energy / entropy** — E_ads reported here is electronic-energy-only. Thermal and configurational corrections require MD + thermodynamic integration (out of tutorial scope).

The notebook validates against published DFT/CC reference data wherever available:

- Tier 1 (H-CHA, α-Al₂O₃, TiO₂) — direct S24 PBE-D3(BJ) checkpoints from Batatia 2024.
- Tier 2 (H-MFI) — CCSD(T)/CBS from Plessow 2024; independent MACE benchmark in Anderson 2025.
- Tier 3 (H-SAPO-34) — CP2K PBE-D3 from Fischer 2015.
- Tier 4 (ZrO₂) — no published reference; flagged as candidate for DFT/experimental follow-up with MAD-derived uncertainty band.

## References

1. Batatia, I. *et al.* "A foundation model for atomistic materials chemistry." arXiv:2401.00096v3 (2024).
2. Plessow, P. N. "Ab initio calculations on the adsorption of water in zeolites." *J. Phys. Chem. C* (2024).
3. Anderson, A. *et al.* "MACE-MP-0 for zeolite/water systems." *Phys. Chem. Chem. Phys.* (2025).
4. Fischer, M. "Structure and water adsorption of AlPO-based chabazite and SAPO-34: a DFT study." *J. Phys. Chem. C* (2015).
5. Furukawa, H. *et al.* "Water adsorption in porous metal-organic frameworks and related materials." *J. Am. Chem. Soc.* **136**, 4369–4381 (2014).
6. Kim, H. *et al.* "Water harvesting from air with metal-organic frameworks powered by natural sunlight." *Science* **356**, 430–434 (2017).
7. Grimme, S. *et al.* "Effect of the damping function in dispersion corrected density functional theory." *J. Comput. Chem.* **32**, 1456–1465 (2011).
8. Stukowski, A. "Visualization and analysis of atomistic simulation data with OVITO." *Model. Simul. Mater. Sci. Eng.* **18**, 015012 (2010).

> The previous iteration of this tutorial (OER catalyst screening on rutile oxides) has been archived under [`_archive/oer-catalyst-screening/`](_archive/oer-catalyst-screening/) and remains runnable by copying the files back to the tutorial root.

## License

Apache 2.0 — see [LICENSE](../LICENSE).
