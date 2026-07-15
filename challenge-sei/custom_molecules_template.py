"""Template for adding literature molecules.

Copy this file to ``custom_molecules.py`` (next to the notebook), fill in your
molecules, and add matching metadata rows to ``data/custom_molecule_manifest.csv``
(copy ``data/custom_molecule_manifest_template.csv``). The notebook imports
``custom_molecules`` automatically if it exists, which registers your geometries.

Build each geometry as an ``ase.Atoms`` in code — from a SMILES via
``geometry_from_smiles`` (the same RDKit path the starter panel uses), inline
coordinates from a cited source, ``ase.build.molecule`` for G2 species, or a pymatgen
``Molecule`` — then register it under the same ``candidate_id`` you used in the
manifest. The notebook MLIP-relaxes every molecule before any energy is used, so an
idealized starting geometry is fine.
"""

from ase import Atoms

from challenge_utils.molecules import geometry_from_smiles, register_molecule

# Example: build a geometry from SMILES (the canonical, structure-file-free path).
# register_molecule("my_literature_additive", geometry_from_smiles("O=C(OC)OC"))

# Example: register an inline geometry (replace with your literature molecule).
# register_molecule(
#     "my_inline_additive",
#     Atoms(
#         symbols=["C", "O", "H", "H"],   # cited source for the geometry goes in the manifest
#         positions=[(0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (-0.6, 0.9, 0.0), (-0.6, -0.9, 0.0)],
#     ),
# )

# Example: a G2-database species via ASE's built-in builder.
# from ase.build import molecule
# register_molecule("my_g2_species", molecule("CH3CN"))
