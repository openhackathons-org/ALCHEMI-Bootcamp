# SEI Pareto Challenge

This folder contains a challenge problem for the Part 1 batched adsorption tutorial.
Participants use the ALCHEMI Toolkit workflow pattern to relax small molecule/surface systems, compute binding energies, convert them into two challenge scores, and choose the additive that gives the largest Pareto hypervolume improvement. 
The bundled molecules are a starter panel, not a closed set: participants may add electrolyte or additive molecules from the literature and submit results for those molecules too.

## Relevant Files

- `sei-pareto-challenge.ipynb` - participant notebook with fill-in code blocks.
- `data/molecule_manifest.csv` - starter molecules and roles (metadata only).
- `data/custom_molecule_manifest_template.csv` - metadata template for literature molecules.
- `custom_molecules_template.py` - template for registering literature geometries in code.
- `data/surface_manifest.csv` - reactive/passivating surface metadata.
- `data/class_surface_lookup.csv` - molecule-class to SEI proxy mapping.
- `challenge_utils/molecules.py` - SMILES-only starter panel; `build_molecule` generates each geometry on demand with RDKit (`geometry_from_smiles`), plus the molecule registry.
- `challenge_utils/relaxation_engine.py` - the step-by-step Toolkit relaxation engine.
- `challenge_utils/pareto.py` - shared Pareto-front and hypervolume helpers.
- `challenge_utils/rewards.py` - shared SEI reward functions and rubric constants.


## Adding Literature Molecules

1. Copy `custom_molecules_template.py` to `custom_molecules.py` and register each
   molecule as an `ase.Atoms` (`challenge_utils.molecules.geometry_from_smiles("<SMILES>")`,
   inline coordinates from a cited source, `ase.build.molecule` for G2 species, or a
   pymatgen `Molecule`) under its `candidate_id` via
   `challenge_utils.molecules.register_molecule`.
2. Create `data/custom_molecule_manifest.csv` with the same columns as
   `data/molecule_manifest.csv` (copy the template) using the same `candidate_id`.

Use molecule classes from `data/class_surface_lookup.csv` so the notebook can select the passivating SEI proxy automatically. 

The notebook follows the public ALCHEMI Toolkit workflow documented at https://nvidia.github.io/nvalchemi-toolkit/: atomistic structures are represented as Toolkit data objects, packed into batches, evaluated by model wrappers, and relaxed with Toolkit dynamics/optimizer components.

## Reward Rubric

The challenge scores are deterministic reward functions in `challenge_utils/rewards.py`. 
They use adsorption strength, `max(0, -E_bind)`, rather than a single arbitrary target binding energy.

- `seeding_score` rewards moderate Li-metal adsorption: full reward for strengths from 0.8 to 1.5 eV, tapering to zero below 0.5 eV and above 2.0 eV.
- `passivation_score` rewards weak adsorption on the passivating SEI proxy: full reward for strengths at or below 0.3 eV, tapering to zero by 0.8 eV.

Its qualitative basis is that useful electrolyte additives can preferentially form protective films, useful SEI layers should passivate further electrolyte reduction, and surface activity often follows a Sabatier-style "neither too weak nor too strong" adsorption tradeoff.

Useful background references:

- ["Electrolyte additives for improved lithium-ion battery performance and overcharge protection"](https://www.sciencedirect.com/science/article/pii/S2451910320300089), 2020.
- ["Review on modeling of the anode solid electrolyte interphase (SEI) for lithium-ion batteries"](https://www.nature.com/articles/s41524-018-0064-0) 2018.
- ["The Sabatier Principle in Electrocatalysis"](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2021.654460/full),
  2021.
- ["Determination of thermodynamic parameters in adsorption studies: a review"](https://link.springer.com/article/10.1007/s11696-025-04218-x),
  2025.