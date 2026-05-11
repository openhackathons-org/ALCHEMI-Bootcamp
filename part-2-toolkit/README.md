# Part 2: ALCHEMI Toolkit Sandbox

Interactive Jupyter environment for exploring the [NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) Python library. Walks through a melting-point validation of naphthalene using the Solid-Liquid Coexistence (SLC) pipeline with AIMNet2.

## Deployment

This notebook runs from the unified compose stack at the repo root (`dev/`) alongside Part 1. See [`dev/README.md`](../README.md) for setup, port table, and HPC instructions.

Part 2 is standalone — no NGC API key required. If you only want to run Part 2, skip the BGR sidecar:

```bash
cd dev
docker compose up jupyter prometheus grafana
```

The notebook is reachable at `http://localhost:8888/lab` (`part-2-toolkit/melting-point-slc.ipynb`).
