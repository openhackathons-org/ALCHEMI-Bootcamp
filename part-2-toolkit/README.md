# Part 2: ALCHEMI Toolkit Sandbox

Interactive Jupyter environment for exploring the [NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) Python library.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Docker | Docker Engine with GPU support |
| GPU | NVIDIA GPU with drivers installed |

No NGC API key or enterprise licence is required.

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
