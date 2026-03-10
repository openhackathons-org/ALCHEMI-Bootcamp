# BMD Materials NIM API Playbook

Interactive playbook demonstrating the NVIDIA Biomolecular Dynamics (BMD) and Batch Geometry Relaxation (BGR) NIM APIs for materials science molecular dynamics simulations (NaCl, naphthalene UDC workflows).

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
./                                    # Repo root
├── alchemi-playbook-nims.ipynb       # Main playbook notebook
├── helpers/                          # Python package
│   ├── api_client.py                 # BMD/BGR NIM API client
│   ├── models.py                     # Pydantic request/response models
│   ├── analysis.py                   # MD trajectory analysis utilities
│   ├── visualization.py              # Plotting helpers
│   └── cache.py                      # Response caching logic
├── tests/                            # pytest test suite
├── structures/                       # Input structure files (.xyz, .cif)
├── cached_responses/                 # Saved API JSON responses
├── outputs/                          # Generated plots and structures (gitignored)
├── environment.yml                   # Conda environment spec
├── requirements.txt                  # Runtime dependencies
└── requirements-dev.txt              # Dev dependencies (pytest, ruff)
```

## Git Remote

```
origin  git@github.com:Ryan-Reese/alchemi-playbooks.git
```

## Key Dependencies

numpy, ase, pymatgen, pydantic, requests, matplotlib, pandas, rdkit, ovito

## Gotchas

- **BMD endpoint on port 8000**: The BMD NIM runs on `localhost:8000`. All structures sent to `/infer` must have `cell` and `pbc=[True, True, True]`. Non-periodic molecules need a vacuum box (e.g. 50 Å).
- **BGR endpoint on port 8890**: The BGR NIM runs on `localhost:8890`, separate from BMD. Use `BGR_SERVER_URL` (not `SERVER_URL`) for BGR calls.
- **OVITO must be conda-installed**: `pip install ovito` breaks in conda envs. Use `pip uninstall -y ovito PySide6` then `conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.15.0`.
- **pymatgen has no `__version__`**: pymatgen 2025+ removed the attribute. Use `importlib.metadata.version('pymatgen')` instead.
- **FAST_DEMO mode**: The notebook uses `FAST_DEMO = True` with cached JSON responses in `cached_responses/` for offline operation. Set `FAST_DEMO = False` to hit the live endpoint.
