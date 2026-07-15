"""Starter-molecule geometries for the SEI Pareto challenge."""

from __future__ import annotations

from collections import Counter

# candidate_id -> SMILES. Geometries are generated from these by build_molecule().
_STARTER_SMILES: dict[str, str] = {
    "EC": "C1COC(=O)O1",            # ethylene carbonate       -> C3H4O3
    "EMC": "CCOC(=O)OC",            # ethyl methyl carbonate   -> C4H8O3
    "FEC": "FC1COC(=O)O1",          # fluoroethylene carbonate -> C3H3FO3
    "VC": "C1=COC(=O)O1",           # vinylene carbonate       -> C3H2O3
    "TMP": "COP(=O)(OC)OC",         # trimethyl phosphate      -> C3H9O4P
    "succinonitrile": "N#CCCC#N",   # succinonitrile           -> C4H4N2
}

# Fixed embedding seed so a given SMILES builds the same starting geometry every
# run (the downstream MLIP relaxation removes any residual dependence on it).
_EMBED_SEED = 2026

# Runtime registry: anything added via register_molecule().
_REGISTRY: dict[str, object] = {}


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "RDKit is required to build starter geometries from SMILES "
            "(`pip install rdkit`). Alternatively, register a prebuilt ase.Atoms "
            "with challenge_utils.molecules.register_molecule()."
        ) from exc
    return Chem, AllChem


def geometry_from_smiles(smiles: str, *, seed: int = _EMBED_SEED):
    """Build a centroid-centred ``ase.Atoms`` 3D geometry from ``smiles``.

    Parses the SMILES, adds explicit hydrogens, embeds a conformer with ETKDGv3
    (deterministic ``seed``), and relaxes it with MMFF94 (falling back to UFF if
    MMFF lacks parameters). Reusable for literature molecules registered by SMILES.
    """
    from ase import Atoms

    Chem, AllChem = _require_rdkit()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}.")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"RDKit ETKDGv3 embedding failed for SMILES {smiles!r}.")

    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol)
    else:  # pragma: no cover - starter panel all have MMFF params
        AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = [
        (p.x, p.y, p.z)
        for i in range(mol.GetNumAtoms())
        for p in (conf.GetAtomPosition(i),)
    ]

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.positions -= atoms.positions.mean(axis=0)  # centroid at the origin
    return atoms


def register_molecule(candidate_id: str, atoms) -> None:
    """Register an ``ase.Atoms`` geometry (or zero-arg builder) for ``candidate_id``.

    Use this for literature molecules instead of shipping structure files: build the
    ``Atoms`` in code (:func:`geometry_from_smiles` from a SMILES, inline coordinates,
    ``ase.build.molecule``, or a pymatgen ``Molecule``) and register it before calling
    :func:`build_molecule`. Re-registering an id overwrites it.
    """
    _REGISTRY[str(candidate_id)] = atoms


def known_molecules() -> tuple[str, ...]:
    """All candidate_ids that build_molecule() can currently construct."""
    return tuple(sorted(set(_STARTER_SMILES) | set(_REGISTRY)))


def build_molecule(candidate_id: str):
    """Construct the molecule for ``candidate_id`` as a fresh ``ase.Atoms``.

    Starter-panel molecules are generated from their SMILES via
    :func:`geometry_from_smiles`; custom molecules must first be added with
    :func:`register_molecule`.
    """
    if candidate_id in _REGISTRY:
        entry = _REGISTRY[candidate_id]
        return entry() if callable(entry) else entry.copy()
    if candidate_id in _STARTER_SMILES:
        return geometry_from_smiles(_STARTER_SMILES[candidate_id])
    raise KeyError(
        f"Unknown molecule {candidate_id!r}. Known ids: {', '.join(known_molecules())}. "
        "For literature molecules, register an ase.Atoms geometry with "
        "challenge_utils.molecules.register_molecule(candidate_id, atoms) "
        "(e.g. from a custom_molecules.py next to the notebook)."
    )


def _symbols_for(candidate_id: str):
    if candidate_id in _REGISTRY:
        entry = _REGISTRY[candidate_id]
        atoms = entry() if callable(entry) else entry
        return atoms.get_chemical_symbols()
    if candidate_id in _STARTER_SMILES:
        # Count atoms straight from the (hydrogen-added) graph -- no 3D embedding needed.
        Chem, _ = _require_rdkit()
        mol = Chem.AddHs(Chem.MolFromSmiles(_STARTER_SMILES[candidate_id]))
        return [atom.GetSymbol() for atom in mol.GetAtoms()]
    raise KeyError(candidate_id)


def molecule_formula(candidate_id: str) -> str:
    """Hill-style formula (C, H, then alphabetical) for cross-checking the manifest."""
    counts = Counter(_symbols_for(candidate_id))
    order = ["C", "H"] + sorted(k for k in counts if k not in ("C", "H"))
    return "".join(f"{el}{counts[el] if counts[el] > 1 else ''}" for el in order if el in counts)
