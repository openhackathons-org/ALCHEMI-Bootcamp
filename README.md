<!-- markdownlint-disable MD033 -->

<img src="shared/alchemi-banner-left.png" alt="NVIDIA ALCHEMI: AI for Chemistry and Materials Science" width="100%">

# NVIDIA ALCHEMI Toolkit tutorials

Build GPU-accelerated atomistic simulation workflows with NVIDIA ALCHEMI
Toolkit.

**[Start with Module 00: Toolkit quick tour](notebooks/00-core-playbook/alchemi-core-00-quick-tour.ipynb)**

[📦 Toolkit docs](https://nvidia.github.io/nvalchemi-toolkit/) ·
[📄 Toolkit examples](https://nvidia.github.io/nvalchemi-toolkit/examples/) ·
[⚙️ Toolkit-Ops docs](https://nvidia.github.io/nvalchemi-toolkit-ops/)

## Why Core?

These tutorials teach the core concepts and workflows of NVIDIA ALCHEMI
Toolkit. Future additions may cover more topics, deeper lessons, and simulation
workflows inspired by real research and development use cases.

## Tutorials

| Module | What you will do | Time |
|---|---|---:|
| [00 · Toolkit quick tour](notebooks/00-core-playbook/alchemi-core-00-quick-tour.ipynb) | Choose research systems and run a complete batched relaxation. | 5–10 min |
| [01 · Toolkit fundamentals and data structures](notebooks/00-core-playbook/alchemi-core-01-data-and-batching.ipynb) | Create `AtomicData`, pack several structures into a `Batch`, and work with Zarr data. | 30 min |
| [02 · Simulate and compose](notebooks/00-core-playbook/alchemi-core-02-models-and-simulation.ipynb) | Compose a model and control molecular dynamics with a bias hook; optionally debug a failed update. | 40 min |
| [03 · Scale a study](notebooks/00-core-playbook/alchemi-core-03-scale-a-study.ipynb) | Build fixed-stage, `FusedStage`, and inflight routes for one molecular study. | 15–20 min |

## Prerequisites

### Background

- basic Python and NumPy or PyTorch experience
- some familiarity with basic computational chemistry

### Hardware

- a CUDA-capable NVIDIA GPU

## Running the tutorials

### Launch on NVIDIA Brev

**[Launch the ALCHEMI tutorials on NVIDIA Brev](https://bit.ly/learn-alchemi)**

### Run the tutorials locally

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), clone
this repository, and run these commands from the repository root.

Create `.venv` from the exact packages recorded in `uv.lock`:

```bash
uv sync --locked --no-default-groups --group kernel --group lab
```

This installs the tutorial runtime, Jupyter kernel support, and JupyterLab. It
leaves contributor test tools out of the learner environment.

Download and verify the model files used by the tutorials:

```bash
uv run --locked --no-sync --env-file environment/runtime.env \
  python environment/prewarm_assets.py
```

Start JupyterLab:

```bash
uv run --locked --no-sync --env-file environment/runtime.env jupyter lab
```

Open the URL printed by JupyterLab and begin with Module 00.

For VS Code, open a notebook and select `.venv/bin/python` under **Select
Kernel > Python Environments**.

`pyproject.toml` declares the packages, and [`uv.lock`](uv.lock) records their
exact resolved versions. [`environment/runtime-pins.toml`](environment/runtime-pins.toml)
records the Toolkit commits and model checksums.

## Contributors

- [Ryan Reese](https://github.com/Ryan-Reese) created the ALCHEMI Toolkit
  melting-point study.
- [Anoushka Bhutani](https://github.com/anoushka2000) developed companion
  tutorial challenges.

## License

NVIDIA-authored source and course diagrams are licensed under
[Apache 2.0](LICENSE). The NVIDIA course banner and NVIDIA marks are excluded
from that grant.

Third-party packages, model checkpoints, CUDA components, data, and viewer code
use their own terms. NCI Atlas data are distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with attribution and
change notices. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[SOURCES_AND_LICENSES.md](SOURCES_AND_LICENSES.md).
