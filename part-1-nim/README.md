<p align="center">
  <img src="assets/eneos_logo.png" alt="ENEOS" height="50"/>
  &nbsp;&nbsp;&nbsp;
  <img src="assets/matlantis_logo.png" alt="Matlantis" height="35"/>
  &nbsp;&nbsp;&nbsp;
  <img src="assets/nvidia-logo.png" alt="NVIDIA" height="55"/>
</p>

---

# Part 1: OER Catalyst Screening with NVIDIA ALCHEMI

**75-minute interactive workshop** — screen oxide catalyst surfaces for oxygen-evolution activity using GPU-accelerated machine-learning interatomic potentials.

Built on the **NVIDIA ALCHEMI BGR (Batch Geometry Relaxation) NIM** and the **MACE-MP-0** foundation model, this notebook walks you through a complete computational catalyst screening workflow — from bulk crystal construction through adsorption energy ranking — at 10,000× the speed of conventional DFT.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| NGC API key | Get one at [build.nvidia.com](https://build.nvidia.com) |
| Docker | Docker Engine with NVIDIA GPU support (`nvidia-container-toolkit`) and the `docker compose` v2 plugin |
| GPU | NVIDIA GPU (tested: A100, H100, B200, L40S, RTX 6000 Ada) |

## Quick Start

Configure your NGC API key:

```bash
cp .env.example .env
# Edit .env and set NGC_API_KEY=<your-key>
```

Build and launch the full stack (BGR NIM + JupyterLab + Prometheus + Grafana):

```bash
docker compose up
```

Open the JupyterLab URL printed in the terminal (`http://localhost:8891/lab?token=...`) and run `alchemi-oer-catalyst-screening.ipynb` top to bottom. Grafana is also available at `http://localhost:3000` (admin/admin) for live BGR metrics.

## BGR NIM Configuration

The Docker Compose stack configures the BGR NIM with:

| Setting | Variable | Value |
|---------|----------|-------|
| Model | `ALCHEMI_NIM_MODEL_TYPE` | `mace` (MACE-MPA-0) |
| Boundary conditions | `ALCHEMI_NIM_PBC` | `true` |
| Optimizer preset | `ALCHEMI_NIM_BGR_OPTIMIZER_PRESET` | `materials` |
| Dispersion corrections | `ALCHEMI_NIM_DFT3_ENABLED` | `true` (DFT-D3(BJ)) |
| Shared memory | `--shm-size` | `8g` |

Prometheus scrapes BGR metrics at `localhost:8000/v1/metrics`; the BGR readiness endpoint is at `localhost:8000/v1/health/ready`.

## FAST_DEMO Mode

The notebook defaults to `FAST_DEMO = False` and calls a **live BGR endpoint**. Set `FAST_DEMO = True` in the control-panel cell to replay pre-cached JSON responses from `cached_responses/oer-catalyst-screening/` instead. This is recommended for workshop environments without GPU access.

## What's inside the notebook

The notebook is organised in 11 sections:

1. **Environment setup** — control panel and library imports
2. **Endpoint connectivity** — verify the BGR NIM is reachable; inspect health and metadata
3. **The Oxygen Evolution Reaction** — review the four-step OER mechanism and thermodynamic framework
4. **Oxide catalyst surfaces** — build rutile bulks, optimise lattice parameters, cleave (110) slabs
5. **Clean slab + gas-phase references** — relax pristine slabs and isolated adsorbate molecules
6. **Adsorbate placement** — position H₂O, OH, O, OOH at catalytically active CUS sites on each slab
7. **Batched geometry relaxation** — submit 30+ slab–adsorbate structures to the BGR NIM in one async batch
8. **Relaxation quality control** — classify outcomes (converged, migrated, dissociated, desorbed) and visualise surface displacement
9. **Adsorption energies** — compute ΔE_ads for each intermediate and inspect metal–adsorbate bond lengths
10. **Screening results + material ranking** — 3-D adsorption-energy scatter, energy ladders, and ranking by proximity to the ideal OER catalyst
11. **Extensions + scaling** — multi-site sampling, larger material libraries, refinements

The notebook ends with *Discussion* and *References* sections that interpret results, compare to literature, and discuss limitations of the structural-screening approach.

## Tutorial Preview

By the end of the workshop, you will have screened three rutile (110) catalysts (IrO₂, RuO₂, TiO₂) against the Sabatier ideal for the four-step oxygen-evolution mechanism, producing comparisons like these:

<table>
  <tr>
    <td align="center"><img src="assets/figures/oer_material_comparison.png" alt="OER binding-energy ladder vs. ideal catalyst" width="100%"/></td>
    <td align="center"><img src="assets/figures/eads_barchart.png" alt="Lowest adsorption energy per (Material, Adsorbate)" width="100%"/></td>
    <td align="center"><img src="assets/figures/oer_3d_scatter.png" alt="3-D adsorption-energy space vs. ideal target" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><sub><i>Free-energy ladder vs. the ideal-catalyst reference (1.23 eV per electrochemical step).</i></sub></td>
    <td align="center"><sub><i>Lowest adsorption energy per (material, adsorbate); labels show the (tilt, site) of the best configuration.</i></sub></td>
    <td align="center"><sub><i>Materials in adsorption-energy space against the CHE ideal target (gold star); legend reports distance-to-target.</i></sub></td>
  </tr>
</table>

## References

1. Batatia, I. *et al.* "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields." *NeurIPS* (2022).
2. Rossmeisl, J. *et al.* "Electrolysis of water on oxide surfaces." *J. Electroanal. Chem.* **607**, 83–89 (2007).
3. Ping, Y., Nielsen, R. J. & Goddard, W. A. "The Reaction Mechanism with Free Energy Barriers at Constant Potentials for the Oxygen Evolution Reaction at the IrO<sub>2</sub>(110) Surface." *J. Am. Chem. Soc.* **139**, 149–155 (2017).
4. Dickens, C. F., Kirk, C. & Norskov, J. K. "Insights into the Electrochemical Oxygen Evolution Reaction with ab Initio Calculations and Microkinetic Modeling." *J. Phys. Chem. C* **123**, 18960–18977 (2019).
5. Stukowski, A. "Visualization and analysis of atomistic simulation data with OVITO." *Model. Simul. Mater. Sci. Eng.* **18**, 015012 (2010).
6. U.S. Geological Survey. "2022 Final List of Critical Minerals." *Federal Register* **87**, 10381 (2022).
7. ENEOS Holdings, Matlantis, and NVIDIA. Collaboration on GPU-accelerated atomistic simulation for catalyst discovery.

## License

Apache 2.0 — see the top-level [LICENSE](../LICENSE).
