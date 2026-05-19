# Part 2: ALCHEMI Toolkit Sandbox And Adsorption Backend

Interactive Jupyter environment for exploring the [NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) Python library.

Current status: this part is the toolkit execution area, but it is not yet the
full toolkit counterpart of the Part 1 AdsorbML tutorial.
`alchemi-toolkit-sandbox.ipynb` is a minimal setup check that verifies the
Toolkit imports work, while `melting-point-slc.ipynb` is a separate
solid-liquid coexistence melting-point workflow.

The reusable AdsorbML science contract now lives in
[`../shared/adsorption_tutorial`](../shared/adsorption_tutorial/):

- `contract.py` defines the canonical Cu(111), Pd(111), alpha-Al2O3(0001) panel.
- `backends.md` defines the output format (column names, units, error
  handling) that all relaxation engines must produce.
- `domain_expert_fact_check.md` in Part 1 references records the energy
  fact-check packet.

Before treating this as equivalent to the BGR NIM tutorial, add a matching
Toolkit notebook that produces the shared output format for at least the small
CO/Cu(111) setup check, then compare the slab, gas, clean-slab, final-site, and
adsorption-energy outputs against Part 1.

The Docker image is fixed to a specific Toolkit version (`7fe7756bd1b13580a619cff39b69742145d416e1`) via the `NVALCHEMI_TOOLKIT_REF` environment variable, so results are reproducible across runs.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Docker | Docker Engine with GPU support |
| GPU | NVIDIA GPU with drivers installed |

No NGC API key or enterprise licence is required.

## Reproducibility

The toolkit dependency is installed from GitHub at a fixed version:

```dockerfile
ARG NVALCHEMI_TOOLKIT_REF=7fe7756bd1b13580a619cff39b69742145d416e1
```

Override this build argument only when intentionally testing a toolkit upgrade, then rerun the import/API smoke cells before using the notebooks for scientific results.

## Quick Start

Allocate a GPU node (from the repo root):

```bash
./start
```

Then deploy:

```bash
scripts/deploy.sh setup <login-host> <compute-node>
```

Access JupyterLab at `http://localhost:8890`.

## Management

```bash
scripts/deploy.sh restart        # Push local changes and rebuild
scripts/deploy.sh pull-changes   # Pull remote Jupyter edits back locally
scripts/deploy.sh status         # Show Jupyter URL and tunnel state
scripts/deploy.sh stop           # Tear down container and close tunnel
```
