# ALCHEMI BMD Materials Science Playbook

End-to-end materials science workflow using NVIDIA ALCHEMI Batched Molecular Dynamics (BMD) and Batch Geometry Relaxation (BGR) NIM endpoints — from unit cell construction through thermodynamic property extraction.

## What You Will Learn

- Connect to the ALCHEMI BMD/BGR endpoints and interpret request/response schemas
- Build crystalline supercells programmatically with pymatgen
- Run NVT and NPT molecular dynamics simulations via the REST API
- Extract thermodynamic properties: density, RDF, MSD, diffusion, thermal expansion
- Batch simulations for temperature sweeps
- Reproduce a UDC-inspired conformer stability scoring workflow
- Switch between MLIP models with a single parameter change

## Prerequisites

- NVIDIA ALCHEMI BMD NIM running on `localhost:8000` (or cached responses)
- Python 3.11+ with conda environment (see below)

## Quick Start

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate alchemi-bmd-materials

# Launch the notebook
jupyter notebook alchemi_bmd_materials.ipynb
```

### FAST_DEMO Mode

Set `FAST_DEMO = True` (default) to use cached responses and shorter simulations.
Set `FAST_DEMO = False` for full production-length runs against a live endpoint.

## Directory Structure

```
alchemi-bmd-materials-playbook/
├── alchemi_bmd_materials.ipynb       # Main notebook
├── helpers/                          # Python helper modules
│   ├── models.py                     # Pydantic models for BMD/BGR
│   ├── api_client.py                 # Endpoint client with caching
│   ├── analysis.py                   # RDF, MSD, thermo extraction
│   ├── visualization.py              # OVITO rendering
│   └── cache.py                      # FAST_DEMO cache logic
├── cached_responses/                 # Populated on first live run
├── structures/                       # Input structure files
│   └── naphthalene.xyz
├── outputs/                          # Generated plots (gitignored)
├── environment.yml                   # Conda environment spec
└── requirements.txt                  # pip fallback
```

## Notebook Sections

| # | Section | Key Output |
|---|---------|------------|
| 0 | Title & Overview | Learning objectives |
| 1 | Environment & Control Panel | All tunable parameters |
| 2 | Endpoint Connectivity | Health check, H2 hello world |
| 3 | Unit Cell to Supercell | NaCl 2x2x2, OVITO render |
| 4 | NVT → NPT Simulation | Equilibrated trajectory |
| 5 | Thermodynamic Properties | Density, RDF, MSD, thermal expansion |
| 6 | UDC Conformer Stability | Naphthalene batched MD scoring |
| 7 | Summary | Recap & next steps |

## License

MIT — see [LICENSE.txt](LICENSE.txt).
