# UV Environment Setup

This challenge can be used in two modes:

- model-free grading and static checks;
- full ALCHEMI Toolkit notebook execution on a CUDA-capable machine.

The full notebook path requires an NVIDIA GPU, CUDA-compatible drivers, and
network access to install the pinned ALCHEMI Toolkit dependency from GitHub.

## Model-Free Grading And Tests

From this folder:

```bash
uv sync --extra dev
uv run pytest tests
uv run python scripts/grade_submission.py outputs/challenge_submission.csv
```

This mode does not install ALCHEMI Toolkit and does not run model inference.

## Full Notebook Runtime

Use this on a GPU node or CUDA-capable workstation:

```bash
uv sync --extra alchemi --extra dev
uv run python -m ipykernel install --user --name alchemi-sei-challenge --display-name "ALCHEMI SEI Challenge"
uv run jupyter lab
```

Then open `sei-pareto-challenge.ipynb` and select the
`ALCHEMI SEI Challenge` kernel.

## Reproducibility

The `alchemi` extra pins `nvalchemi-toolkit` to the same Git commit used by the
repository Dockerfile:

```text
7fe7756bd1b13580a619cff39b69742145d416e1
```

PyTorch is resolved from the CUDA 13.0 PyTorch wheel index declared in
`pyproject.toml`. Override these only when intentionally testing a Toolkit or
CUDA stack upgrade.
