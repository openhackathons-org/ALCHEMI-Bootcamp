# SEI Pareto Challenge

This folder contains a challenge problem for the Part 1 batched adsorption tutorial.
Participants use the ALCHEMI Toolkit workflow pattern to relax small molecule/surface
systems, compute binding energies, convert them into two challenge scores, and choose
the additive that gives the largest Pareto hypervolume improvement.

The grader is intentionally model-free. It reads `outputs/challenge_submission.csv`
and optionally `outputs/raw_component_energies.csv`; it does not import ALCHEMI,
Torch, ASE, or any MLIP package.

## Files

- `sei-pareto-challenge.ipynb` - participant notebook with fill-in code blocks.
- `data/molecule_manifest.csv` - candidate molecules and roles.
- `data/surface_manifest.csv` - reactive/passivating surface proxies.
- `data/class_surface_lookup.csv` - molecule-class to SEI proxy mapping.
- `scripts/grade_submission.py` - model-free grader.

The bundled structures are compact teaching inputs for a bootcamp exercise. They are
not production-quality battery interface models and should not be reported as
scientific predictions.

The notebook follows the public ALCHEMI Toolkit workflow documented at
https://nvidia.github.io/nvalchemi-toolkit/: atomistic structures are represented
as Toolkit data objects, packed into batches, evaluated by model wrappers, and
relaxed with Toolkit dynamics/optimizer components.

## Grading

From this folder:

```bash
python3 scripts/grade_submission.py outputs/challenge_submission.csv
```

If `outputs/raw_component_energies.csv` exists, the grader also checks that the
submitted binding energies match the raw component energies.
