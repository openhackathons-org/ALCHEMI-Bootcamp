"""Deterministic structures and result handling for the adsorption lesson.

This module deliberately stops at ASE ``Atoms`` objects and plain result
tables. Toolkit ``AtomicData`` conversion, batching, model composition, and
model execution remain visible in the notebook.

The checked-in structures are generated starting geometries. They are not
pre-relaxed structures and must not be presented as converged adsorption
geometries.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import add_adsorbate, fcc111, molecule
from ase.io import read, write


ADSORBATES: tuple[str, ...] = ("CO", "CO2", "NH3", "CH3OH")
CLEAN_SLAB_KEY = "clean_cu111"
ADSLAB_KEYS: dict[str, str] = {
    "CO": "co_on_cu111",
    "CO2": "co2_on_cu111",
    "NH3": "nh3_on_cu111",
    "CH3OH": "ch3oh_on_cu111",
}
GAS_KEYS: dict[str, str] = {
    "CO": "co_gas",
    "CO2": "co2_gas",
    "NH3": "nh3_gas",
    "CH3OH": "ch3oh_gas",
}
PERIODIC_KEYS: tuple[str, ...] = (
    CLEAN_SLAB_KEY,
    *(ADSLAB_KEYS[name] for name in ADSORBATES),
)
FINITE_KEYS: tuple[str, ...] = tuple(GAS_KEYS[name] for name in ADSORBATES)
STRUCTURE_KEYS: tuple[str, ...] = PERIODIC_KEYS + FINITE_KEYS

METHODOLOGY_SCHEMA = "alchemi.adsorption-methodology.v1"
STRUCTURE_MANIFEST_SCHEMA = "alchemi.adsorption-structures.v1"
METHODOLOGY_FILENAME = "methodology.json"
STRUCTURE_MANIFEST_FILENAME = "manifest.json"
GEOMETRY_STATUS = "ASE-generated initial placements; not model-relaxed"

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "adsorption"
    / "cu111-important-molecules-v1"
)

_EXPECTED_GAS_COMPOSITIONS: dict[str, Counter[str]] = {
    "CO": Counter({"C": 1, "O": 1}),
    "CO2": Counter({"C": 1, "O": 2}),
    "NH3": Counter({"N": 1, "H": 3}),
    "CH3OH": Counter({"C": 1, "O": 1, "H": 4}),
}
_EXPECTED_SLAB_ATOMS = 3 * 3 * 4


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _finite_scalar(value: Any, *, name: str) -> float:
    array = _as_numpy(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain exactly one value")
    result = float(array.reshape(()))
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_methodology(methodology: Mapping[str, Any]) -> None:
    if methodology.get("schema") != METHODOLOGY_SCHEMA:
        raise ValueError(f"methodology schema must be {METHODOLOGY_SCHEMA}")
    if methodology.get("geometry_status") != GEOMETRY_STATUS:
        raise ValueError(
            "methodology geometry_status must identify structures as initial and "
            "not model-relaxed"
        )

    slab = methodology.get("slab")
    if not isinstance(slab, Mapping):
        raise ValueError("methodology slab must be a mapping")
    if slab.get("element") != "Cu" or slab.get("ase_builder") != "ase.build.fcc111":
        raise ValueError("the lesson methodology requires an ASE-built Cu(111) slab")
    if tuple(slab.get("size", ())) != (3, 3, 4):
        raise ValueError("the lesson methodology requires a 3x3 four-layer slab")
    if tuple(slab.get("pbc", ())) != (True, True, False):
        raise ValueError("the slab methodology requires PBC only in x and y")
    for field in ("lattice_constant_angstrom", "vacuum_angstrom"):
        value = slab.get(field)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"slab {field} must be a real number")
        if not isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"slab {field} must be positive and finite")

    placements = methodology.get("adsorbates")
    if not isinstance(placements, list):
        raise ValueError("methodology adsorbates must be a list")
    names = tuple(item.get("name") for item in placements if isinstance(item, Mapping))
    if names != ADSORBATES or len(placements) != len(ADSORBATES):
        raise ValueError(
            "methodology adsorbates must be CO, CO2, NH3, and CH3OH in order"
        )
    for placement in placements:
        if not isinstance(placement, Mapping):
            raise ValueError("each adsorbate placement must be a mapping")
        if placement.get("ase_molecule") != placement.get("name"):
            raise ValueError("adsorbate names must match their ASE molecule names")
        height = placement.get("height_angstrom")
        if isinstance(height, bool) or not isinstance(height, Real):
            raise ValueError("adsorbate height_angstrom must be a real number")
        if not isfinite(float(height)) or float(height) <= 0.0:
            raise ValueError("adsorbate height_angstrom must be positive and finite")
        if placement.get("site") not in {"ontop", "bridge", "fcc", "hcp"}:
            raise ValueError("adsorbate site is not supported by ASE fcc111")


def load_adsorption_methodology(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the versioned structure-generation methodology."""

    methodology_path = (
        DEFAULT_DATA_DIR / METHODOLOGY_FILENAME if path is None else Path(path)
    )
    methodology = json.loads(methodology_path.read_text(encoding="utf-8"))
    if not isinstance(methodology, dict):
        raise ValueError("adsorption methodology must be a JSON object")
    _validate_methodology(methodology)
    return methodology


def _single_element_index(atoms: Atoms, symbol: str) -> int:
    indices = [index for index, atom in enumerate(atoms) if atom.symbol == symbol]
    if len(indices) != 1:
        raise ValueError(f"expected exactly one {symbol} anchor atom")
    return indices[0]


def _rotate_vector_to(
    atoms: Atoms,
    vector: np.ndarray,
    target: Sequence[float],
    *,
    center_index: int,
) -> None:
    vector = np.asarray(vector, dtype=float)
    target_vector = np.asarray(target, dtype=float)
    if np.linalg.norm(vector) <= 1.0e-12:
        raise ValueError("cannot orient a molecule from a zero-length vector")
    atoms.rotate(
        vector,
        target_vector,
        center=atoms.positions[center_index],
        rotate_cell=False,
    )


def _orient_adsorbate(atoms: Atoms, placement: Mapping[str, Any]) -> int:
    name = str(placement["name"])
    anchor = _single_element_index(atoms, str(placement["anchor_element"]))
    anchor_position = atoms.positions[anchor]
    orientation = placement["orientation"]

    if orientation == "anchor_partner_away_from_surface":
        partner = next(index for index in range(len(atoms)) if index != anchor)
        _rotate_vector_to(
            atoms,
            atoms.positions[partner] - anchor_position,
            (0.0, 0.0, 1.0),
            center_index=anchor,
        )
    elif orientation == "molecular_axis_parallel_to_surface":
        oxygen_indices = [
            index for index, atom in enumerate(atoms) if atom.symbol == "O"
        ]
        _rotate_vector_to(
            atoms,
            atoms.positions[oxygen_indices[1]] - atoms.positions[oxygen_indices[0]],
            (1.0, 0.0, 0.0),
            center_index=anchor,
        )
    elif orientation == "substituents_away_from_surface":
        substituents = np.delete(atoms.positions, anchor, axis=0)
        _rotate_vector_to(
            atoms,
            substituents.mean(axis=0) - anchor_position,
            (0.0, 0.0, 1.0),
            center_index=anchor,
        )
    elif orientation == "heavy_atom_and_hydroxyl_bonds_away_from_surface":
        carbon = _single_element_index(atoms, "C")
        hydrogen_indices = [
            index for index, atom in enumerate(atoms) if atom.symbol == "H"
        ]
        hydroxyl_hydrogen = min(
            hydrogen_indices,
            key=lambda index: np.linalg.norm(atoms.positions[index] - anchor_position),
        )
        bond_vectors = np.stack(
            (
                atoms.positions[carbon] - anchor_position,
                atoms.positions[hydroxyl_hydrogen] - anchor_position,
            )
        )
        bond_vectors /= np.linalg.norm(bond_vectors, axis=1)[:, None]
        _rotate_vector_to(
            atoms,
            bond_vectors.sum(axis=0),
            (0.0, 0.0, 1.0),
            center_index=anchor,
        )
    else:
        raise ValueError(f"unsupported orientation for {name}: {orientation}")
    return anchor


def _structure_info(*, key: str, role: str, adsorbate: str | None) -> dict[str, Any]:
    return {
        "structure_key": key,
        "role": role,
        "adsorbate": "none" if adsorbate is None else adsorbate,
        "surface": "none" if role == "gas" else "Cu(111)",
        "geometry_status": GEOMETRY_STATUS,
        "relaxed": False,
    }


def build_initial_structure_set(
    methodology: Mapping[str, Any] | None = None,
) -> dict[str, Atoms]:
    """Build the clean slab, four adslabs, and four finite gas molecules.

    The returned insertion order is the order used by the two tutorial
    batches: the clean slab and four adslabs first, followed by four gas-phase
    molecules.
    """

    resolved = (
        load_adsorption_methodology() if methodology is None else dict(methodology)
    )
    _validate_methodology(resolved)
    slab_settings = resolved["slab"]
    slab = fcc111(
        "Cu",
        size=tuple(int(value) for value in slab_settings["size"]),
        a=float(slab_settings["lattice_constant_angstrom"]),
        vacuum=float(slab_settings["vacuum_angstrom"]),
        orthogonal=False,
    )
    slab.set_pbc(tuple(bool(value) for value in slab_settings["pbc"]))
    slab.new_array("is_adsorbate", np.zeros(len(slab), dtype=bool))
    slab.info.update(
        _structure_info(key=CLEAN_SLAB_KEY, role="clean_slab", adsorbate=None)
    )

    structures: dict[str, Atoms] = {CLEAN_SLAB_KEY: slab}
    gas_structures: dict[str, Atoms] = {}
    for placement in resolved["adsorbates"]:
        name = str(placement["name"])
        gas = molecule(str(placement["ase_molecule"]))
        gas.center(about=(0.0, 0.0, 0.0))
        gas.set_cell(np.zeros((3, 3)))
        gas.set_pbc(False)
        gas.new_array("is_adsorbate", np.ones(len(gas), dtype=bool))
        gas.info.update(_structure_info(key=GAS_KEYS[name], role="gas", adsorbate=name))
        gas_structures[GAS_KEYS[name]] = gas

        adsorbate = molecule(str(placement["ase_molecule"]))
        anchor = _orient_adsorbate(adsorbate, placement)
        adslab = slab.copy()
        del adslab.arrays["is_adsorbate"]
        add_adsorbate(
            adslab,
            adsorbate,
            height=float(placement["height_angstrom"]),
            position=str(placement["site"]),
            mol_index=anchor,
        )
        mask = np.zeros(len(adslab), dtype=bool)
        mask[_EXPECTED_SLAB_ATOMS:] = True
        adslab.new_array("is_adsorbate", mask)
        adslab.info.clear()
        adslab.info.update(
            _structure_info(key=ADSLAB_KEYS[name], role="adslab", adsorbate=name)
        )
        adslab.info.update(
            adsorption_site=str(placement["site"]),
            initial_height_angstrom=float(placement["height_angstrom"]),
            initial_orientation=str(placement["orientation"]),
        )
        structures[ADSLAB_KEYS[name]] = adslab

    slab.info.pop("adsorbate_info", None)
    structures.update(gas_structures)
    validate_structure_set(structures)
    return structures


def _composition(atoms: Atoms) -> Counter[str]:
    return Counter(atoms.get_chemical_symbols())


def validate_structure_set(structures: Mapping[str, Atoms]) -> None:
    """Validate identities, compositions, atom counts, cells, and PBC."""

    if len(structures) != len(STRUCTURE_KEYS) or set(structures) != set(STRUCTURE_KEYS):
        raise ValueError(
            "structure set keys must contain exactly: " + ", ".join(STRUCTURE_KEYS)
        )
    for key, atoms in structures.items():
        if not isinstance(atoms, Atoms):
            raise TypeError(f"{key} must be an ASE Atoms object")
        if (
            atoms.positions.shape != (len(atoms), 3)
            or not np.isfinite(atoms.positions).all()
        ):
            raise ValueError(f"{key} positions must be finite with shape (atoms, 3)")
        if atoms.info.get("structure_key") != key:
            raise ValueError(f"{key} structure_key metadata does not match")
        if atoms.info.get("geometry_status") != GEOMETRY_STATUS:
            raise ValueError(f"{key} must be labeled as an unrelaxed initial geometry")
        if bool(atoms.info.get("relaxed", True)):
            raise ValueError(f"{key} must not be labeled as relaxed")
        mask = np.asarray(atoms.arrays.get("is_adsorbate"))
        if mask.shape != (len(atoms),) or mask.dtype.kind != "b":
            raise ValueError(f"{key} must contain a boolean is_adsorbate array")

    clean = structures[CLEAN_SLAB_KEY]
    expected_slab = Counter({"Cu": _EXPECTED_SLAB_ATOMS})
    if len(clean) != _EXPECTED_SLAB_ATOMS or _composition(clean) != expected_slab:
        raise ValueError("clean_cu111 must contain exactly 36 Cu atoms")
    if tuple(bool(value) for value in clean.pbc) != (True, True, False):
        raise ValueError("clean_cu111 must be periodic only in x and y")
    if not np.isfinite(clean.cell.array).all() or clean.cell.volume <= 0.0:
        raise ValueError("clean_cu111 must have a finite, non-zero cell")
    if np.asarray(clean.arrays["is_adsorbate"]).any():
        raise ValueError("clean_cu111 cannot mark any atoms as adsorbate atoms")

    for name in ADSORBATES:
        gas = structures[GAS_KEYS[name]]
        adslab = structures[ADSLAB_KEYS[name]]
        gas_composition = _EXPECTED_GAS_COMPOSITIONS[name]
        if gas.pbc.any():
            raise ValueError(f"{GAS_KEYS[name]} must be finite, with PBC disabled")
        if _composition(gas) != gas_composition:
            raise ValueError(f"{GAS_KEYS[name]} has the wrong molecular composition")
        if not np.asarray(gas.arrays["is_adsorbate"]).all():
            raise ValueError(f"{GAS_KEYS[name]} must mark every atom as molecular")

        expected_adslab = expected_slab + gas_composition
        if _composition(adslab) != expected_adslab:
            raise ValueError(f"{ADSLAB_KEYS[name]} has the wrong composition")
        if tuple(bool(value) for value in adslab.pbc) != (True, True, False):
            raise ValueError(f"{ADSLAB_KEYS[name]} must be periodic only in x and y")
        if not np.allclose(adslab.cell.array, clean.cell.array, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{ADSLAB_KEYS[name]} cell must match the clean slab")
        mask = np.asarray(adslab.arrays["is_adsorbate"])
        if mask.sum() != len(gas) or mask[:_EXPECTED_SLAB_ATOMS].any():
            raise ValueError(f"{ADSLAB_KEYS[name]} has an invalid adsorbate mask")


def split_for_batches(
    structures: Mapping[str, Atoms],
) -> tuple[dict[str, Atoms], dict[str, Atoms]]:
    """Return ordered periodic and finite mappings for two Toolkit batches."""

    validate_structure_set(structures)
    periodic = {key: structures[key].copy() for key in PERIODIC_KEYS}
    finite = {key: structures[key].copy() for key in FINITE_KEYS}
    return periodic, finite


def build_structure_inventory_table(
    structures: Mapping[str, Atoms],
) -> pd.DataFrame:
    """Summarize the fixed structures without changing their order."""

    validate_structure_set(structures)
    return pd.DataFrame(
        [
            {
                "structure": key,
                "role": atoms.info["role"],
                "formula": atoms.get_chemical_formula(),
                "atoms": len(atoms),
                "pbc": tuple(bool(value) for value in atoms.pbc),
                "geometry": atoms.info["geometry_status"],
            }
            for key, atoms in structures.items()
        ]
    )


def build_placement_table(methodology: Mapping[str, Any]) -> pd.DataFrame:
    """Return the four documented starting placements as a compact table."""

    _validate_methodology(methodology)
    return pd.DataFrame(methodology["adsorbates"]).rename(
        columns={
            "name": "molecule",
            "anchor_element": "anchor",
            "height_angstrom": "height_A",
            "orientation": "starting_orientation",
        }
    )[["molecule", "site", "anchor", "height_A", "starting_orientation"]]


def _structure_filename(key: str) -> str:
    return f"{key}_initial.extxyz"


def write_initial_structure_set(
    output_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    methodology_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate inspectable initial structures and a deterministic manifest."""

    output_dir = Path(output_dir)
    source_methodology_path = (
        DEFAULT_DATA_DIR / METHODOLOGY_FILENAME
        if methodology_path is None
        else Path(methodology_path)
    )
    methodology = load_adsorption_methodology(source_methodology_path)
    structures = build_initial_structure_set(methodology)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / STRUCTURE_MANIFEST_FILENAME
    targets = [output_dir / _structure_filename(key) for key in STRUCTURE_KEYS]
    targets.append(manifest_path)
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to replace existing adsorption files: "
            + ", ".join(path.name for path in existing)
        )

    records: list[dict[str, Any]] = []
    for key in STRUCTURE_KEYS:
        path = output_dir / _structure_filename(key)
        write(path, structures[key], format="extxyz")
        records.append(
            {
                "key": key,
                "file": path.name,
                "sha256": _sha256_file(path),
                "atoms": len(structures[key]),
                "formula": structures[key].get_chemical_formula(),
                "pbc": [bool(value) for value in structures[key].pbc],
                "role": str(structures[key].info["role"]),
            }
        )

    import ase

    manifest: dict[str, Any] = {
        "schema": STRUCTURE_MANIFEST_SCHEMA,
        "method_id": str(methodology["method_id"]),
        "methodology_file": source_methodology_path.name,
        "methodology_sha256": _sha256_file(source_methodology_path),
        "geometry_status": GEOMETRY_STATUS,
        "generated_with": {"library": "ASE", "version": ase.__version__},
        "structures": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_initial_structure_set(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    verify_hashes: bool = True,
) -> dict[str, Atoms]:
    """Load checked-in initial structures and verify their manifest."""

    data_dir = Path(data_dir)
    manifest_path = data_dir / STRUCTURE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != STRUCTURE_MANIFEST_SCHEMA:
        raise ValueError(
            f"structure manifest schema must be {STRUCTURE_MANIFEST_SCHEMA}"
        )
    if manifest.get("geometry_status") != GEOMETRY_STATUS:
        raise ValueError("structure manifest does not describe initial geometries")
    records = manifest.get("structures")
    if not isinstance(records, list):
        raise ValueError("structure manifest must contain a structures list")
    by_key = {
        str(record["key"]): record for record in records if isinstance(record, Mapping)
    }
    if len(records) != len(STRUCTURE_KEYS) or tuple(by_key) != STRUCTURE_KEYS:
        raise ValueError("structure manifest keys or order do not match the lesson")

    structures: dict[str, Atoms] = {}
    for key in STRUCTURE_KEYS:
        record = by_key[key]
        path = data_dir / str(record["file"])
        if verify_hashes and _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {path.name}")
        structures[key] = read(path, format="extxyz")
    validate_structure_set(structures)
    return structures


def adsorption_energy(
    adslab_energy_eV: Any,
    clean_slab_energy_eV: Any,
    gas_energy_eV: Any,
) -> float:
    """Return ``E(adslab) - E(clean slab) - E(gas)`` in eV.

    A negative value means the fixed adslab electronic energy is below the
    clean-slab-plus-gas electronic energy. It is not a thermochemical result.
    """

    return (
        _finite_scalar(adslab_energy_eV, name="adslab_energy_eV")
        - _finite_scalar(clean_slab_energy_eV, name="clean_slab_energy_eV")
        - _finite_scalar(gas_energy_eV, name="gas_energy_eV")
    )


def _energy_mapping(energies: Mapping[str, Any], *, component: str) -> dict[str, float]:
    missing = [key for key in STRUCTURE_KEYS if key not in energies]
    extra = [key for key in energies if key not in STRUCTURE_KEYS]
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError(f"{component} energies: {'; '.join(details)}")
    return {
        key: _finite_scalar(energies[key], name=f"{component}[{key}]")
        for key in STRUCTURE_KEYS
    }


def force_statistics(
    forces_eV_A: Any,
    *,
    expected_atoms: int | None = None,
    mask: Any | None = None,
) -> tuple[float, float, int]:
    """Return maximum norm, RMS norm, and number of included atoms."""

    forces = np.asarray(_as_numpy(forces_eV_A), dtype=float)
    if forces.ndim != 2 or forces.shape[1] != 3:
        raise ValueError("forces_eV_A must have shape (atoms, 3)")
    if forces.shape[0] == 0:
        raise ValueError("forces_eV_A must contain at least one atom")
    if expected_atoms is not None and forces.shape[0] != expected_atoms:
        raise ValueError(
            f"forces_eV_A contains {forces.shape[0]} atoms; expected {expected_atoms}"
        )
    if not np.isfinite(forces).all():
        raise ValueError("forces_eV_A contains non-finite values")
    if mask is not None:
        selected = np.asarray(_as_numpy(mask))
        if selected.shape != (forces.shape[0],) or selected.dtype.kind != "b":
            raise ValueError("force mask must be boolean with shape (atoms,)")
        if not selected.any():
            raise ValueError("force mask must include at least one atom")
        forces = forces[selected]
    norms = np.linalg.norm(forces, axis=1)
    return float(norms.max()), float(np.sqrt(np.mean(norms**2))), len(norms)


def assemble_adsorption_results(
    *,
    model_energies_eV: Mapping[str, Any],
    d3_energies_eV: Mapping[str, Any],
    combined_forces_eV_A: Mapping[str, Any],
    force_masks: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Assemble component adsorption energies and combined-force statistics."""

    model = _energy_mapping(model_energies_eV, component="model")
    d3 = _energy_mapping(d3_energies_eV, component="D3")
    expected_force_keys = tuple(ADSLAB_KEYS[name] for name in ADSORBATES)
    if len(combined_forces_eV_A) != len(expected_force_keys) or set(
        combined_forces_eV_A
    ) != set(expected_force_keys):
        raise ValueError(
            "combined forces must contain exactly the four adslabs: "
            + ", ".join(expected_force_keys)
        )
    if force_masks is not None and (
        len(force_masks) != len(expected_force_keys)
        or set(force_masks) != set(expected_force_keys)
    ):
        raise ValueError("force_masks keys must match the four adslabs")

    rows: list[dict[str, Any]] = []
    for name in ADSORBATES:
        adslab_key = ADSLAB_KEYS[name]
        gas_key = GAS_KEYS[name]
        model_adsorption = adsorption_energy(
            model[adslab_key], model[CLEAN_SLAB_KEY], model[gas_key]
        )
        d3_adsorption = adsorption_energy(
            d3[adslab_key], d3[CLEAN_SLAB_KEY], d3[gas_key]
        )
        combined_adslab = model[adslab_key] + d3[adslab_key]
        combined_clean = model[CLEAN_SLAB_KEY] + d3[CLEAN_SLAB_KEY]
        combined_gas = model[gas_key] + d3[gas_key]
        combined_adsorption = adsorption_energy(
            combined_adslab, combined_clean, combined_gas
        )
        mask = None if force_masks is None else force_masks[adslab_key]
        expected_atoms = _EXPECTED_SLAB_ATOMS + sum(
            _EXPECTED_GAS_COMPOSITIONS[name].values()
        )
        fmax, force_rms, force_atoms = force_statistics(
            combined_forces_eV_A[adslab_key],
            expected_atoms=expected_atoms,
            mask=mask,
        )
        rows.append(
            {
                "molecule": name,
                "model_adslab_energy_eV": model[adslab_key],
                "model_clean_slab_energy_eV": model[CLEAN_SLAB_KEY],
                "model_gas_energy_eV": model[gas_key],
                "model_adsorption_energy_eV": model_adsorption,
                "d3_adslab_energy_eV": d3[adslab_key],
                "d3_clean_slab_energy_eV": d3[CLEAN_SLAB_KEY],
                "d3_gas_energy_eV": d3[gas_key],
                "d3_adsorption_energy_eV": d3_adsorption,
                "combined_adslab_energy_eV": combined_adslab,
                "combined_clean_slab_energy_eV": combined_clean,
                "combined_gas_energy_eV": combined_gas,
                "adsorption_energy_eV": combined_adsorption,
                "fmax_eV_A": fmax,
                "force_rms_eV_A": force_rms,
                "force_atoms": force_atoms,
            }
        )
    table = pd.DataFrame(rows)
    if not np.allclose(
        table["adsorption_energy_eV"],
        table["model_adsorption_energy_eV"] + table["d3_adsorption_energy_eV"],
        rtol=1.0e-10,
        atol=1.0e-8,
    ):
        raise RuntimeError("combined adsorption energies do not match their components")
    return table


def build_full_force_table(
    structures: Mapping[str, Atoms],
    forces_eV_A: Mapping[str, Any],
) -> pd.DataFrame:
    """Return one inspectable row per atom for every evaluated structure."""

    unknown = [key for key in forces_eV_A if key not in structures]
    if unknown:
        raise ValueError("forces contain unknown structures: " + ", ".join(unknown))
    if len(forces_eV_A) != len(structures) or set(forces_eV_A) != set(structures):
        raise ValueError("forces must contain every structure exactly once")
    rows: list[dict[str, Any]] = []
    for key, atoms in structures.items():
        forces = np.asarray(_as_numpy(forces_eV_A[key]), dtype=float)
        force_statistics(forces, expected_atoms=len(atoms))
        molecular_mask = np.asarray(atoms.arrays["is_adsorbate"], dtype=bool)
        for index, (atom, position, force, is_adsorbate) in enumerate(
            zip(atoms, atoms.positions, forces, molecular_mask, strict=True)
        ):
            rows.append(
                {
                    "structure": key,
                    "role": str(atoms.info["role"]),
                    "atom_index": index,
                    "element": atom.symbol,
                    "is_adsorbate": bool(is_adsorbate),
                    "x_angstrom": float(position[0]),
                    "y_angstrom": float(position[1]),
                    "z_angstrom": float(position[2]),
                    "fx_eV_A": float(force[0]),
                    "fy_eV_A": float(force[1]),
                    "fz_eV_A": float(force[2]),
                    "force_norm_eV_A": float(np.linalg.norm(force)),
                }
            )
    return pd.DataFrame(rows)


def summarize_adslab_force_regions(force_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize adsorbate and Cu-slab force norms for each adslab."""

    required = {
        "structure",
        "role",
        "atom_index",
        "element",
        "is_adsorbate",
        "force_norm_eV_A",
    }
    missing = sorted(required.difference(force_table.columns))
    if missing:
        raise ValueError("force table is missing columns: " + ", ".join(missing))
    adslabs = force_table.loc[force_table["role"] == "adslab"].copy()
    if adslabs.empty:
        raise ValueError("force table does not contain any adslabs")
    if adslabs["is_adsorbate"].dtype.kind != "b":
        raise ValueError("is_adsorbate must be boolean")

    rows: list[dict[str, Any]] = []
    for structure in adslabs["structure"].drop_duplicates():
        structure_rows = adslabs.loc[adslabs["structure"] == structure]
        for region, selected in (
            ("Cu slab", ~structure_rows["is_adsorbate"]),
            ("adsorbate", structure_rows["is_adsorbate"]),
        ):
            region_rows = structure_rows.loc[selected]
            if region_rows.empty:
                raise ValueError(f"{structure} has no atoms in the {region} region")
            force_norms = region_rows["force_norm_eV_A"].to_numpy(dtype=float)
            if not np.isfinite(force_norms).all() or np.any(force_norms < 0.0):
                raise ValueError("force norms must be finite and nonnegative")
            maximum_row = region_rows.loc[region_rows["force_norm_eV_A"].idxmax()]
            rows.append(
                {
                    "structure": structure,
                    "region": region,
                    "atoms": len(region_rows),
                    "fmax_eV_A": float(force_norms.max()),
                    "force_rms_eV_A": float(np.sqrt(np.mean(force_norms**2))),
                    "max_force_atom_index": int(maximum_row["atom_index"]),
                    "max_force_element": str(maximum_row["element"]),
                }
            )
    return pd.DataFrame(rows)


def export_full_forces(
    path: str | Path,
    *,
    structures: Mapping[str, Atoms],
    forces_eV_A: Mapping[str, Any],
) -> dict[str, Any]:
    """Write all atom-wise coordinates and forces to one tidy CSV file."""

    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("full force export path must end in .csv")
    table = build_full_force_table(structures, forces_eV_A)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.12g")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "rows": len(table),
        "structures": len(structures),
    }
