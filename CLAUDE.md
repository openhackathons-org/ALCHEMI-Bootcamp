# BMD Materials NIM API Playbook

Interactive playbook demonstrating the NVIDIA Biomolecular Dynamics (BMD) NIM API for materials science molecular dynamics simulations (NaCl, naphthalene UDC workflows).

## Environment Setup

```bash
conda env create -f environment.yml
conda activate alchemi-bmd-materials
uv pip install -r requirements.txt        # runtime deps
uv pip install -r requirements-dev.txt    # adds pytest, ruff
```

- **Conda env**: `alchemi-bmd-materials`
- **Python**: >=3.11
- **Package manager**: `uv` (inside the conda env)

## Commands

```bash
# Tests
pytest tests/ -v

# Lint & format
ruff check .
ruff format .
```

## Directory Structure

```
./                                # Repo root
├── alchemi_bmd_materials.ipynb   # Main playbook notebook
├── helpers/                      # Python package
│   ├── api_client.py             # BMD NIM API client
│   ├── models.py                 # Pydantic request/response models
│   ├── analysis.py               # MD trajectory analysis utilities
│   ├── visualization.py          # Plotting helpers
│   └── cache.py                  # Response caching logic
├── tests/                        # pytest test suite (90 tests)
├── structures/                   # Input structure files (.xyz, .cif)
├── cached_responses/             # Saved API JSON responses
├── outputs/                      # Generated plots and structures (gitignored)
├── environment.yml               # Conda environment spec
├── requirements.txt              # Runtime dependencies
└── requirements-dev.txt          # Dev dependencies (pytest, ruff)
```

## Git Remote

```
origin  git@github.com:Ryan-Reese/alchemi-playbooks.git
```

## Key Dependencies

numpy, ase, pymatgen, pydantic, requests, matplotlib, pandas, rdkit, ovito

## Gotchas

- **BMD endpoint requires PBC**: All structures sent to the `/molecular-dynamics` endpoint must have `cell` and `pbc=[True, True, True]`. Non-periodic molecules need a vacuum box (e.g. 50 Å).
- **OVITO must be conda-installed**: `pip install ovito` breaks in conda envs. Use `pip uninstall -y ovito PySide6` then `conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.15.0`.
- **pymatgen has no `__version__`**: pymatgen 2025+ removed the attribute. Use `importlib.metadata.version('pymatgen')` instead.
- **FAST_DEMO mode**: The notebook uses `FAST_DEMO = True` with cached JSON responses in `cached_responses/` for offline operation. Set `FAST_DEMO = False` to hit the live endpoint.
- **90 pytest tests**: Tests cover models, structures, caching, analysis, UDC workflow, visualization, and endpoint (endpoint tests skip if server is down).
