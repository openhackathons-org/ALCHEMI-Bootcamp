# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checked structures and deterministic SMILES preparation for the Core course."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, fcc111, graphene, molecule
from ase.io import read

_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
_MOF74_MANIFEST = _DATA_ROOT / "mof74" / "manifest.json"
_SMILES_SEED = 0xC0FFEE
_SMILES_MAX_ITERATIONS = 200
_MACE_ALIAS = "medium-mpa-0"
_MACE_SHA256 = "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structure_from_smiles(
    smiles: str,
    *,
    seed: int = _SMILES_SEED,
    max_iterations: int = _SMILES_MAX_ITERATIONS,
) -> Atoms:
    """Build a deterministic, explicit-hydrogen, neutral 3D ASE structure."""

    from rdkit import Chem
    from rdkit.Chem import AllChem

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES preparation requires a non-empty string")
    if not 0 <= seed <= 0x7FFFFFFF:
        raise ValueError("SMILES embedding seed must fit a signed 32-bit integer")
    if max_iterations < 1:
        raise ValueError("SMILES optimization requires at least one iteration")

    molecule_2d = Chem.MolFromSmiles(smiles)
    if molecule_2d is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    formal_charge = int(Chem.GetFormalCharge(molecule_2d))
    if formal_charge != 0:
        raise ValueError(
            f"Core SMILES preparation expects a neutral structure; found charge "
            f"{formal_charge:+d} for {smiles!r}"
        )

    molecule_3d = Chem.AddHs(molecule_2d)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    parameters.numThreads = 1
    if AllChem.EmbedMolecule(molecule_3d, parameters) != 0:
        raise ValueError(f"RDKit could not embed a 3D conformer for {smiles!r}")

    if AllChem.MMFFHasAllMoleculeParams(molecule_3d):
        force_field = "MMFF94"
        status = AllChem.MMFFOptimizeMolecule(
            molecule_3d,
            mmffVariant=force_field,
            maxIters=max_iterations,
        )
    elif AllChem.UFFHasAllMoleculeParams(molecule_3d):
        force_field = "UFF"
        status = AllChem.UFFOptimizeMolecule(
            molecule_3d,
            maxIters=max_iterations,
        )
    else:
        raise ValueError(f"RDKit has no MMFF or UFF parameters for {smiles!r}")
    if status < 0:
        raise ValueError(f"RDKit {force_field} setup failed for {smiles!r}")

    conformer = molecule_3d.GetConformer()
    structure = Atoms(
        numbers=[atom.GetAtomicNum() for atom in molecule_3d.GetAtoms()],
        positions=conformer.GetPositions(),
        pbc=False,
    )
    structure.center(vacuum=4.0)
    structure.info.update(
        {
            "source": "RDKit",
            "canonical_smiles": Chem.MolToSmiles(molecule_2d, canonical=True),
            "embedding": "ETKDGv3",
            "embedding_seed": seed,
            "force_field": force_field,
            "force_field_converged": status == 0,
        }
    )
    return structure


def mace_checkpoint_path(alias: str = _MACE_ALIAS) -> Path:
    """Resolve the workshop MACE-MPA-0 checkpoint and verify its exact bytes."""

    if alias != _MACE_ALIAS:
        raise ValueError(
            f"The Core runtime pins {_MACE_ALIAS!r}; received checkpoint alias {alias!r}"
        )
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    path = Path(download_mace_mp_checkpoint(alias)).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"MACE checkpoint resolver returned a missing file: {path}")
    actual = _sha256_file(path)
    if actual != _MACE_SHA256:
        raise RuntimeError(
            f"MACE checkpoint SHA-256 mismatch: expected {_MACE_SHA256}, found {actual}"
        )
    return path


def load_mof74(
    path: str | Path | None = None,
    *,
    manifest_path: str | Path = _MOF74_MANIFEST,
) -> Atoms:
    """Load the pinned 54-atom Mg-MOF-74 primitive cell."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read the MOF-74 manifest: {manifest_file}") from error

    local = manifest.get("local_file", {})
    structure_path = (
        Path(path).expanduser().resolve()
        if path is not None
        else manifest_file.parent / str(local.get("path", ""))
    )
    if not structure_path.is_file():
        raise RuntimeError(f"Pinned MOF-74 structure is missing: {structure_path}")
    expected = str(local.get("sha256", ""))
    actual = _sha256_file(structure_path)
    if not expected or actual != expected:
        raise RuntimeError(
            f"MOF-74 SHA-256 mismatch: expected {expected or '<missing>'}, found {actual}"
        )

    structure = read(structure_path)
    if not isinstance(structure, Atoms):
        raise TypeError(
            f"MOF-74 source did not contain one ASE Atoms object: {structure_path}"
        )
    expected_atoms = int(manifest["primitive_check"]["primitive_atoms"])
    if len(structure) != expected_atoms or not bool(structure.pbc.all()):
        raise RuntimeError(
            f"MOF-74 structure check failed: expected {expected_atoms} periodic atoms"
        )
    structure.info.update(
        {
            "display_name": manifest["display_name"],
            "source": manifest["source"]["repository"],
            "license": manifest["source"]["license"],
        }
    )
    return structure


def _smiles_builder(smiles: str) -> Callable[[], Atoms]:
    return lambda: structure_from_smiles(smiles)


def _c60() -> Atoms:
    structure = molecule("C60")
    structure.center(vacuum=4.0)
    structure.info["source"] = "ASE molecule('C60')"
    return structure


def _ammonia() -> Atoms:
    structure = molecule("NH3")
    structure.center(vacuum=4.0)
    structure.info["source"] = "ASE molecule('NH3')"
    return structure


def _surface_with_adsorbate(adsorbate: str) -> Atoms:
    slab = fcc111("Pt", size=(2, 2, 3), vacuum=7.0)
    adsorbate_atoms = molecule(adsorbate)
    anchor = "C" if adsorbate == "CO" else "N"
    anchor_index = adsorbate_atoms.get_chemical_symbols().index(anchor)
    height = 1.85 if adsorbate == "CO" else 2.10
    add_adsorbate(
        slab,
        adsorbate_atoms,
        height=height,
        position="ontop",
        mol_index=anchor_index,
    )
    slab.info["source"] = f"ASE fcc111 Pt slab with {adsorbate} adsorbate"
    return slab


def _benchmark_surface_with_adsorbate(
    metal: str,
    adsorbate: str,
    *,
    anchor: str,
    height: float,
) -> Atoms:
    """Build a 48-atom (111) slab plus one compact adsorbate."""

    slab = fcc111(metal, size=(4, 4, 3), vacuum=7.0)
    adsorbate_atoms = molecule(adsorbate)
    anchor_index = adsorbate_atoms.get_chemical_symbols().index(anchor)

    # Place the adsorbate on the top-layer atom nearest the center of the slab.
    cell_center_xy = 0.5 * np.asarray(slab.cell[:2, :2], dtype=float).sum(axis=0)
    top_layer = np.flatnonzero(
        np.isclose(slab.positions[:, 2], slab.positions[:, 2].max())
    )
    center_site = slab.positions[
        top_layer[
            np.argmin(
                np.linalg.norm(
                    slab.positions[top_layer, :2] - cell_center_xy,
                    axis=1,
                )
            )
        ],
        :2,
    ]
    add_adsorbate(
        slab,
        adsorbate_atoms,
        height=height,
        position=tuple(center_site),
        mol_index=anchor_index,
    )
    slab.info.update(
        {
            "source": f"ASE fcc111 {metal} slab with {adsorbate} adsorbate",
            "frozen_layers": 2,
        }
    )
    return slab


def _graphene() -> Atoms:
    structure = graphene(size=(2, 2, 1), vacuum=7.0)
    structure.info["source"] = "ASE graphene"
    return structure


@dataclass(frozen=True, slots=True)
class MoleculeSpec:
    """Explorer metadata plus a factory that returns a fresh ASE structure."""

    key: str
    display_name: str
    domain: str
    purpose: str
    formula: str
    atom_count: int
    periodic: bool
    source: str
    model_scope: str
    is_default: bool
    _builder: Callable[[], Atoms] = field(repr=False, compare=False)

    def build(self) -> Atoms:
        """Build and check a fresh structure for this catalog entry."""

        structure = self._builder()
        formula = structure.get_chemical_formula(mode="hill")
        if (
            formula != self.formula
            or len(structure) != self.atom_count
            or bool(structure.pbc.any()) != self.periodic
        ):
            raise RuntimeError(
                f"Catalog entry {self.key!r} changed: got {formula}, "
                f"{len(structure)} atoms, periodic={bool(structure.pbc.any())}"
            )
        structure.info.setdefault("display_name", self.display_name)
        structure.info.setdefault("catalog_key", self.key)
        return structure


def core_structure_catalog(*, include_alternates: bool = True) -> dict[str, MoleculeSpec]:
    """Return the legacy small-system catalog."""

    scope = (
        "MACE-MPA-0 materials-domain preview; inspect force changes and convergence "
        "instead of treating the prediction as molecular reference data"
    )
    entries = (
        MoleculeSpec(
            "ammonia",
            "Ammonia",
            "Agricultural chemistry",
            "Fertilizer feedstock and compact molecular example",
            "H3N",
            4,
            False,
            "ASE molecule('NH3')",
            scope,
            True,
            _ammonia,
        ),
        MoleculeSpec(
            "caffeine",
            "Caffeine",
            "Biochemistry and pharma",
            "Familiar bioactive molecule",
            "C8H10N4O2",
            24,
            False,
            "RDKit from canonical SMILES",
            scope,
            True,
            _smiles_builder("Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
        ),
        MoleculeSpec(
            "c60",
            "C₆₀",
            "Nanomaterials",
            "Molecular nanocarbon",
            "C60",
            60,
            False,
            "ASE molecule('C60')",
            "MACE-MPA-0 contains carbon in materials environments",
            True,
            _c60,
        ),
        MoleculeSpec(
            "co_pt111",
            "CO on Pt(111)",
            "Catalysis",
            "Surface adsorption motif",
            "COPt12",
            14,
            True,
            "ASE fcc111 slab and CO adsorbate",
            "MACE-MPA-0 materials-domain preview",
            True,
            lambda: _surface_with_adsorbate("CO"),
        ),
        MoleculeSpec(
            "atrazine",
            "Atrazine",
            "Agricultural chemistry",
            "Herbicide alternate",
            "C8H14ClN5",
            28,
            False,
            "RDKit from SMILES",
            scope,
            False,
            _smiles_builder("CCNC1=NC(=NC(=N1)Cl)NC(C)C"),
        ),
        MoleculeSpec(
            "aspirin",
            "Aspirin",
            "Biochemistry and pharma",
            "Small-molecule drug alternate",
            "C9H8O4",
            21,
            False,
            "RDKit from SMILES",
            scope,
            False,
            _smiles_builder("CC(=O)Oc1ccccc1C(=O)O"),
        ),
        MoleculeSpec(
            "graphene",
            "Graphene",
            "Nanomaterials",
            "Periodic carbon alternate",
            "C8",
            8,
            True,
            "ASE graphene 2×2 cell",
            "MACE-MPA-0 materials-domain preview",
            False,
            _graphene,
        ),
        MoleculeSpec(
            "nh3_pt111",
            "NH₃ on Pt(111)",
            "Catalysis",
            "Adsorption alternate",
            "H3NPt12",
            16,
            True,
            "ASE fcc111 slab and NH3 adsorbate",
            "MACE-MPA-0 materials-domain preview",
            False,
            lambda: _surface_with_adsorbate("NH3"),
        ),
        MoleculeSpec(
            "mof_linker",
            "Terephthalic acid",
            "Porous materials",
            "Framework linker alternate",
            "C8H6O4",
            18,
            False,
            "RDKit from SMILES",
            scope,
            False,
            _smiles_builder("O=C(O)c1ccc(C(=O)O)cc1"),
        ),
        MoleculeSpec(
            "ethanol",
            "Ethanol",
            "General chemistry",
            "Compact custom-SMILES example",
            "C2H6O",
            9,
            False,
            "RDKit from SMILES CCO",
            scope,
            False,
            _smiles_builder("CCO"),
        ),
        MoleculeSpec(
            "ibuprofen",
            "Ibuprofen",
            "Biochemistry and pharma",
            "Larger drug alternate",
            "C13H18O2",
            33,
            False,
            "RDKit from SMILES",
            scope,
            False,
            _smiles_builder("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"),
        ),
    )
    if not include_alternates:
        entries = tuple(entry for entry in entries if entry.is_default)
    return {entry.key: entry for entry in entries}


def opening_performance_catalog() -> dict[str, MoleculeSpec]:
    """Return the twelve 33-60 atom systems shown in the Core picker."""

    scope = (
        "MACE-MPA-0 materials-domain preview; compare execution routes without "
        "treating these predictions as reference chemistry"
    )
    entries = (
        MoleculeSpec(
            "ibuprofen", "Ibuprofen", "Pharmaceutical R&D",
            "Analgesic and anti-inflammatory medicine", "C13H18O2", 33,
            False, "PubChem CID 3672", scope, True,
            _smiles_builder("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"),
        ),
        MoleculeSpec(
            "azoxystrobin", "Azoxystrobin", "Crop protection",
            "Broad-spectrum agricultural fungicide",
            "C22H17N3O5", 47, False, "PubChem CID 3034285", scope, True,
            _smiles_builder("COC=C(C1=CC=CC=C1OC2=NC=NC(=C2)OC3=CC=CC=C3C#N)C(=O)OC"),
        ),
        MoleculeSpec(
            "permethrin", "Permethrin", "Pest control",
            "Pyrethroid insecticide used in agriculture and public health",
            "C21H20Cl2O3", 46, False, "PubChem CID 40326", scope, True,
            _smiles_builder("CC1(C(C1C(=O)OCC2=CC(=CC=C2)OC3=CC=CC=C3)C=C(Cl)Cl)C"),
        ),
        MoleculeSpec(
            "capsaicin", "Capsaicin", "Food and sensory chemistry",
            "Pungent natural product and TRPV1 agonist",
            "C18H27NO3", 49, False, "PubChem CID 1548943", scope, True,
            _smiles_builder("CC(C)C=CCCCCC(=O)NCC1=CC(=C(C=C1)O)OC"),
        ),
        MoleculeSpec(
            "c60", "Fullerene (C₆₀)", "Nanomaterials",
            "Molecular carbon nanomaterial", "C60", 60, False,
            "ASE molecule('C60')", "MACE-MPA-0 contains carbon in materials environments",
            True, _c60,
        ),
        MoleculeSpec(
            "retinol", "Retinol", "Nutrition and skin science",
            "Vitamin A alcohol used in biological and cosmetic research",
            "C20H30O", 51, False, "PubChem CID 445354", scope, True,
            _smiles_builder("CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CCO)C)C"),
        ),
        MoleculeSpec(
            "avobenzone", "Avobenzone", "Sunscreen chemistry", "Organic UVA filter",
            "C20H22O3", 45, False, "PubChem CID 51040", scope, True,
            _smiles_builder("CC(C)(C)C1=CC=C(C=C1)C(=O)CC(=O)C2=CC=C(C=C2)OC"),
        ),
        MoleculeSpec(
            "nile_red", "Nile red", "Fluorescence imaging",
            "Solvatochromic dye for hydrophobic environments", "C20H18N2O2", 42,
            False, "PubChem CID 65182", scope, True,
            _smiles_builder("CCN(CC)C1=CC2=C(C=C1)N=C3C4=CC=CC=C4C(=O)C=C3O2"),
        ),
        MoleculeSpec(
            "dibutyl_phthalate", "Dibutyl phthalate", "Polymers and plastics",
            "Plasticizer used in PVC and flexible polymers", "C16H22O4", 42,
            False, "PubChem CID 3026", scope, True,
            _smiles_builder("CCCCOC(=O)C1=CC=CC=C1C(=O)OCCCC"),
        ),
        MoleculeSpec(
            "co_pt111_4x4", "CO on Pt(111)", "Heterogeneous catalysis",
            "CO adsorption on a platinum catalyst surface",
            "COPt48", 50, True, "ASE 4x4x3 fcc111 slab", scope, True,
            lambda: _benchmark_surface_with_adsorbate(
                "Pt", "CO", anchor="C", height=1.85
            ),
        ),
        MoleculeSpec(
            "h2o_cu111_4x4", "H₂O on Cu(111)", "Corrosion and electrochemistry",
            "Water adsorption on a copper surface",
            "H2Cu48O", 51, True, "ASE 4x4x3 fcc111 slab", scope, True,
            lambda: _benchmark_surface_with_adsorbate(
                "Cu", "H2O", anchor="O", height=2.20
            ),
        ),
        MoleculeSpec(
            "mg_mof74", "Mg-MOF-74", "Carbon capture",
            "Porous adsorbent with open Mg sites for CO₂ capture",
            "C24H6Mg6O18", 54, True,
            "RASPA2 MgMOF-74 primitive cell", scope, True, load_mof74,
        ),
    )
    return {entry.key: entry for entry in entries}
