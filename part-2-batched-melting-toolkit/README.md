# Part 2: OLED Melting Point predictions with ALCHEMI Toolkit

Interactive Jupyter notebook that predicts the melting point of a molecular crystal using the [NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) Python library and the Orb-v3 (OMol) machine-learned interatomic potential. Naphthalene serves as the model OLED-material system, walked through the Solid-Liquid Coexistence (SLC) pipeline end-to-end.

## Deployment

This notebook runs from the unified compose stack at the repo root alongside Part 1 — `cd build && docker compose up`. No NGC API key is required. See [the repo README](../README.md) for full setup.

The notebook is reachable at `http://localhost:8888/lab`
(`part-2-batched-melting-toolkit/melting-point-slc.ipynb`).

## Crystal input

The naphthalene input is
[COD entry 2311088](https://www.crystallography.net/2311088.cif), distributed
under CC0. See [Sources and licenses](../SOURCES_AND_LICENSES.md) for its
original publication, checksum, and a source-temperature metadata note.

External pre-computed trajectories, logs, and the generated result image are
not included. They will be regenerated from COD 2311088 when this tutorial is
updated. Until then, the notebook's saved-result mode and long cached stages
are unavailable.
