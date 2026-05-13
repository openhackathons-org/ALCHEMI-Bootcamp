# Part 2: OLED Melting Point predictions with ALCHEMI Toolkit

Interactive Jupyter notebook that predicts the melting point of a molecular crystal using the [NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) Python library and the AIMNet2 neural network potential. Naphthalene serves as the model OLED-material system, walked through the Solid-Liquid Coexistence (SLC) pipeline end-to-end.

## Deployment

This notebook runs from the unified compose stack at the repo root alongside Part 1. See [the repo README](../README.md) for setup and port table.

Part 2 is standalone — no NGC API key required. If you only want to run Part 2, skip the BGR sidecar:

```bash
cd build
docker compose up jupyter prometheus grafana
```

The notebook is reachable at `http://localhost:8888/lab` (`part-2-toolkit/melting-point-slc.ipynb`).
