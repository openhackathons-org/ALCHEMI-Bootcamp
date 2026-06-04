# SEI Pareto Challenge

This folder contains a challenge problem for the Part 1 batched adsorption tutorial.
Participants use the ALCHEMI Toolkit workflow pattern to relax small molecule/surface
systems, compute binding energies, convert them into two challenge scores, and choose
the additive that gives the largest Pareto hypervolume improvement. The bundled
molecules are a starter panel, not a closed set: participants may add electrolyte
or additive molecules from the literature and submit results for those molecules too.

The grader is intentionally model-free. It reads `outputs/challenge_submission.csv`
and optionally `outputs/raw_component_energies.csv`; it does not import ALCHEMI,
Torch, ASE, or any MLIP package.

## Files

- `sei-pareto-challenge.ipynb` - participant notebook with fill-in code blocks.
- `sei-pareto-challenge-solution.ipynb` - one completed solution path for instructors.
- `pyproject.toml` and `UV_SETUP.md` - local `uv` environment setup.
- `data/molecule_manifest.csv` - starter molecules and roles.
- `data/custom_molecule_manifest_template.csv` - template for optional literature molecules.
- `data/surface_manifest.csv` - reactive/passivating surface proxies.
- `data/class_surface_lookup.csv` - molecule-class to SEI proxy mapping.
- `challenge_utils/pareto.py` - shared Pareto-front and hypervolume helpers.
- `scripts/grade_submission.py` - model-free grader.

The bundled structures are compact teaching inputs for a bootcamp exercise. They are
not production-quality battery interface models and should not be reported as
scientific predictions.

## Adding Literature Molecules

To extend the challenge, create `data/custom_molecule_manifest.csv` with the same
columns as `data/molecule_manifest.csv`, place the corresponding structures under
`data/molecules/`, and record the literature/source provenance. Use molecule
classes from `data/class_surface_lookup.csv` so the notebook can select the
passivating SEI proxy automatically. The grader does not require a fixed candidate
list; it evaluates the rows present in the submitted CSV.

The notebook follows the public ALCHEMI Toolkit workflow documented at
https://nvidia.github.io/nvalchemi-toolkit/: atomistic structures are represented
as Toolkit data objects, packed into batches, evaluated by model wrappers, and
relaxed with Toolkit dynamics/optimizer components.

## Environment

This folder includes a local `uv` setup. For model-free grading and tests:

```bash
cd challenge-sei
uv sync --extra dev
uv run pytest tests
```

For full notebook execution with ALCHEMI Toolkit on a CUDA-capable GPU machine:

```bash
cd challenge-sei
uv sync --extra alchemi --extra dev
uv run python -m ipykernel install --user --name alchemi-sei-challenge --display-name "ALCHEMI SEI Challenge"
uv run jupyter lab
```

See `UV_SETUP.md` for details.

## Grading

From this folder:

```bash
python3 scripts/grade_submission.py outputs/challenge_submission.csv
```

If `outputs/raw_component_energies.csv` exists, the grader also checks that the
submitted binding energies match the raw component energies.
