# SEI Pareto Challenge

This folder contains a challenge problem for the Part 1 batched adsorption tutorial.
Participants use the ALCHEMI Toolkit workflow pattern to relax small molecule/surface systems, compute binding energies, convert them into two challenge scores, and choose the additive that gives the largest Pareto hypervolume improvement. 
The bundled molecules are a starter panel, not a closed set: participants may add electrolyte or additive molecules from the literature and submit results for those molecules too.

The grader is intentionally model-free. It reads `outputs/challenge_submission.csv` and optionally `outputs/raw_component_energies.csv`; it does not import ALCHEMI, Torch, ASE, or any MLIP package.

## Files

- `sei-pareto-challenge.ipynb` - participant notebook with fill-in code blocks.
- `sei-pareto-challenge-solution.ipynb` - one completed solution path for instructors.
- `pyproject.toml` and `UV_SETUP.md` - local `uv` environment setup.
- `data/molecule_manifest.csv` - starter molecules and roles.
- `data/custom_molecule_manifest_template.csv` - template for optional literature molecules.
- `data/surface_manifest.csv` - reactive/passivating surface metadata.
- `data/class_surface_lookup.csv` - molecule-class to SEI proxy mapping.
- `challenge_utils/pareto.py` - shared Pareto-front and hypervolume helpers.
- `challenge_utils/rewards.py` - shared SEI reward functions and rubric constants.
- `scripts/grade_submission.py` - model-free grader.

The bundled molecule structures and surface metadata are compact teaching inputs for a bootcamp exercise. Surface slabs should be built from bulk structures in the notebooks, not loaded from toy proxy files. 
These models should not be reported as scientific predictions.

## Adding Literature Molecules

Create `data/custom_molecule_manifest.csv` with the same columns as `data/molecule_manifest.csv`, place the corresponding structures under `data/molecules/`. 
Use molecule classes from `data/class_surface_lookup.csv` so the notebook can select the passivating SEI proxy automatically. 

The notebook follows the public ALCHEMI Toolkit workflow documented at https://nvidia.github.io/nvalchemi-toolkit/: atomistic structures are represented as Toolkit data objects, packed into batches, evaluated by model wrappers, and relaxed with Toolkit dynamics/optimizer components.

## Reward Rubric

The challenge scores are deterministic reward functions in `challenge_utils/rewards.py`. 
They use adsorption strength, `max(0, -E_bind)`, rather than a single arbitrary target binding energy.

- `seeding_score` rewards moderate Li-metal adsorption: full reward for strengths from 0.8 to 1.5 eV, tapering to zero below 0.5 eV and above 2.0 eV.
- `passivation_score` rewards weak adsorption on the passivating SEI proxy: full reward for strengths at or below 0.3 eV, tapering to zero by 0.8 eV.

Its qualitative basis is that useful electrolyte additives can preferentially form protective films, useful SEI layers should passivate further electrolyte reduction, and surface activity often follows a Sabatier-style "neither too weak nor too strong" adsorption tradeoff.

Useful background references:

- ["Electrolyte additives for improved lithium-ion battery performance and
  overcharge protection"](https://www.sciencedirect.com/science/article/pii/S2451910320300089),
  2020.
- ["Review on modeling of the anode solid electrolyte interphase (SEI) for
  lithium-ion batteries"](https://www.nature.com/articles/s41524-018-0064-0),
  2018.
- ["The Sabatier Principle in
  Electrocatalysis"](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2021.654460/full),
  2021.
- ["Determination of thermodynamic parameters in adsorption studies: a
  review"](https://link.springer.com/article/10.1007/s11696-025-04218-x),
  2025.
