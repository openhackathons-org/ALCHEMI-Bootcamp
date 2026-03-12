# ALCHEMI BMD Playbooks

Two interactive playbooks demonstrating NVIDIA ALCHEMI's GPU-accelerated **Batched Molecular Dynamics (BMD)** and **Batch Geometry Relaxation (BGR)** NIM endpoints.

## Playbooks

### Materials Science (`alchemi-playbook-nims.ipynb`)

End-to-end crystalline materials workflow — from unit cell construction through thermodynamic property extraction using NaCl as a benchmark.

**You will learn to:**
- Connect to the ALCHEMI BMD/BGR endpoints and interpret request/response schemas
- Build crystalline supercells programmatically with pymatgen
- Run NVT and NPT molecular dynamics simulations via the REST API
- Extract thermodynamic properties: density, RDF, MSD, diffusion, thermal expansion
- Batch simulations for temperature sweeps
- Switch between MLIP models with a single parameter change

### Conformer Stability (`alchemi-conformer-stability.ipynb`)

UDC-inspired conformer stability scoring workflow — generate conformers, run batched NVT MD at elevated temperature, and rank thermal stability from energy statistics.

**You will learn to:**
- Generate conformers for a candidate organic molecule (naphthalene)
- Build gas-phase simulation cells with periodic boundary conditions
- Run batched NVT MD at elevated temperature via the REST API
- Score conformer stability using energy statistics (variance, drift)
- Compare conformers and identify the most thermally stable geometry

## Prerequisites

- NVIDIA ALCHEMI BMD NIM running on `localhost:{BMD_PORT}` (or cached responses) — port is configurable in each notebook's control panel (default `8000`)
- NVIDIA ALCHEMI BGR NIM running on `localhost:{BGR_PORT}` (or cached responses) — materials playbook only (default `8890`)
- Python 3.11+ with conda environment (see below)

## Quick Start

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate alchemi-bmd-materials

# Launch a playbook
jupyter notebook alchemi-playbook-nims.ipynb           # Materials science
jupyter notebook alchemi-conformer-stability.ipynb      # Conformer stability
```

### FAST_DEMO Mode

Set `FAST_DEMO = True` (default) to use cached responses and shorter simulations.
Set `FAST_DEMO = False` for full production-length runs against a live endpoint.

## Directory Structure

```
alchemi-bmd-materials-playbook/
├── alchemi-playbook-nims.ipynb           # Materials science playbook (NaCl MD)
├── alchemi-conformer-stability.ipynb     # Conformer stability playbook (UDC/naphthalene)
├── helpers/                              # Shared Python helper modules
│   ├── models.py                         # Pydantic models for BMD/BGR
│   ├── api_client.py                     # Endpoint client with caching
│   ├── analysis.py                       # RDF, MSD, thermo extraction
│   ├── visualization.py                  # OVITO rendering
│   └── cache.py                          # FAST_DEMO cache logic
├── tests/                                # pytest test suite
├── cached_responses/                     # Populated on first live run
├── structures/                           # Input structure files
│   └── naphthalene.xyz
├── assets/                               # Static images
│   └── udc_workflow_diagram.png
├── outputs/                              # Generated plots (gitignored)
├── environment.yml                       # Conda environment spec
├── requirements.txt                      # Runtime dependencies
└── requirements-dev.txt                  # Dev dependencies (pytest, ruff)
```

## Notebook Sections

### Materials Science Playbook

| # | Section | Key Output |
|---|---------|------------|
| 1 | Environment & Control Panel | All tunable parameters |
| 2 | Endpoint Connectivity | Health check, H2 hello world |
| 3 | Unit Cell to Supercell | NaCl supercell, OVITO render |
| 4 | NVT → NPT Simulation | Equilibrated trajectory |
| 5 | Thermodynamic Properties | Density, RDF, MSD, thermal expansion |
| 6 | Summary | Recap & next steps |

### Conformer Stability Playbook

| # | Section | Key Output |
|---|---------|------------|
| 1 | Environment & Control Panel | UDC workflow parameters |
| 2 | Endpoint Connectivity | Health check, H2 hello world |
| 3 | Conformer Generation | Naphthalene conformers |
| 4 | Batched NVT MD | Elevated-temperature trajectories |
| 5 | Stability Scoring & Comparison | Ranked conformer table, energy plot |
| 6 | Scaling to Production | UDC production pipeline |
| 7 | Summary | Recap & next steps |

## License

MIT — see [LICENSE.txt](LICENSE.txt).
