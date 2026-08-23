# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate the pinned NCIA250 campaign dataset."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import pandas as pd
import torch
from ase import Atoms
from ase.data import chemical_symbols

from .model_support import model_checkpoint

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_NCIA250_ARCHIVE = (
    _PROJECT_ROOT
    / "notebooks"
    / "00-core-playbook"
    / "data"
    / "nci_atlas"
    / "NCIA250.zip"
)
_NCIA250_SHA256 = "34e3c2cec763344dd9be41aa008672c7d052e50db57abe1abc59873d3935c433"
_NCI_KEYS = ("subset", "system_id", "system_name", "interaction_class", "scale")


def _checkpoint_species() -> tuple[int, ...]:
    """Read the elements implemented by the verified AIMNet2 checkpoint."""

    checkpoint = torch.load(model_checkpoint(), map_location="cpu", weights_only=True)
    values = checkpoint.get("implemented_species")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("AIMNet2 checkpoint does not list implemented species")
    species = tuple(int(value) for value in values)
    if not species or len(species) != len(set(species)):
        raise ValueError("AIMNet2 checkpoint contains an invalid element list")
    return species


def _selection(value: str, natoms: int, *, label: str) -> tuple[int, ...]:
    """Convert a one-based inclusive XYZ selection to zero-based indices."""

    parts = value.split("-")
    if len(parts) == 1:
        parts *= 2
    if len(parts) != 2:
        raise ValueError(f"{label} must be one index or one inclusive range")
    try:
        first, last = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"{label} must contain integer atom indices") from error
    if first < 1 or last < first or last > natoms:
        raise ValueError(f"{label} lies outside 1 through {natoms}")
    return tuple(range(first - 1, last))


def _read_ncia250_member(
    archive: ZipFile, member: str
) -> tuple[Atoms, Atoms, Atoms, dict[str, Any]]:
    """Read one NCIA250 equilibrium complex and its two fragments."""

    label = Path(member).name
    lines = archive.read(member).decode("utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"{label} is missing XYZ content")
    try:
        natoms = int(lines[0])
    except ValueError as error:
        raise ValueError(f"{label} has an invalid atom count") from error
    if natoms < 1 or len(lines) != natoms + 2:
        raise ValueError(f"{label} does not contain its declared atom count")

    metadata: dict[str, str] = {}
    for token in lines[1].split():
        if "=" not in token:
            raise ValueError(f"{label} contains malformed metadata")
        key, value = token.split("=", 1)
        metadata[key] = value
    required = {
        "charge",
        "charge_a",
        "charge_b",
        "selection_a",
        "selection_b",
        "scaling",
        "benchmark_Eint",
        "benchmark_unit",
        "group",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"{label} metadata is missing {sorted(missing)!r}")

    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in lines[2:]:
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"{label} contains an invalid atom row")
        symbols.append(fields[0])
        positions.append([float(value) for value in fields[1:]])
    complex_atoms = Atoms(symbols=symbols, positions=positions, pbc=False)

    selection_a = _selection(
        metadata["selection_a"], natoms, label=f"{label} fragment A"
    )
    selection_b = _selection(
        metadata["selection_b"], natoms, label=f"{label} fragment B"
    )
    if sorted((*selection_a, *selection_b)) != list(range(natoms)):
        raise ValueError(f"{label} fragment selections must cover every atom once")
    fragment_a = complex_atoms[list(selection_a)]
    fragment_b = complex_atoms[list(selection_b)]

    charge = int(metadata["charge"])
    charge_a = int(metadata["charge_a"])
    charge_b = int(metadata["charge_b"])
    if charge != charge_a + charge_b:
        raise ValueError(f"{label} fragment charges do not sum to the complex charge")
    scale = float(metadata["scaling"])
    reference = float(metadata["benchmark_Eint"])
    if metadata["benchmark_unit"] != "kcal/mol":
        raise ValueError(f"{label} benchmark energy must use kcal/mol")

    name_parts = Path(member).stem.split("_")
    if len(name_parts) != 4 or name_parts[0] != "NCIA" or name_parts[-1] != "100":
        raise ValueError(f"{label} does not follow the NCIA250 naming scheme")
    subset, system_id = name_parts[1:3]
    common = {
        "subset": subset,
        "system_id": system_id,
        "system_name": f"{subset} {system_id}",
        "interaction_class": metadata["group"],
        "scale": scale,
        "source_geometry_file": member,
    }
    for fragment, atoms, fragment_charge in (
        ("AB", complex_atoms, charge),
        ("A", fragment_a, charge_a),
        ("B", fragment_b, charge_b),
    ):
        atoms.info.update(common)
        atoms.info["fragment"] = fragment
        atoms.info["charge"] = fragment_charge

    numbers = tuple(int(value) for value in complex_atoms.numbers)
    record = {
        **common,
        "charge": charge,
        "charge_a": charge_a,
        "charge_b": charge_b,
        "natoms": natoms,
        "atomic_numbers": numbers,
        "ccsd_t_cbs_kcal_mol": reference,
    }
    return complex_atoms, fragment_a, fragment_b, record


def load_ncia250(
    path: str | Path | None = None,
    *,
    supported_atomic_numbers: Sequence[int] | None = None,
) -> tuple[list[Atoms], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load all NCIA250 records and select the checkpoint-compatible complexes."""

    archive_path = _NCIA250_ARCHIVE if path is None else Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"NCIA250 archive is missing: {archive_path}")
    observed = sha256(archive_path.read_bytes()).hexdigest()
    if observed != _NCIA250_SHA256:
        raise ValueError(f"NCIA250 archive SHA-256 mismatch: {observed}")

    supported = tuple(
        _checkpoint_species()
        if supported_atomic_numbers is None
        else (int(value) for value in supported_atomic_numbers)
    )
    if not supported or len(supported) != len(set(supported)):
        raise ValueError("supported_atomic_numbers must contain unique elements")
    supported_set = set(supported)

    structures: list[Atoms] = []
    graph_records: list[dict[str, Any]] = []
    reference_records: list[dict[str, Any]] = []
    inventory_records: list[dict[str, Any]] = []
    try:
        with ZipFile(archive_path) as archive:
            members = sorted(
                name for name in archive.namelist() if name.endswith(".xyz")
            )
            if len(members) != 250:
                raise ValueError("NCIA250 must contain 250 XYZ files")
            for member in members:
                complex_atoms, fragment_a, fragment_b, record = _read_ncia250_member(
                    archive, member
                )
                unsupported = tuple(
                    sorted(set(record["atomic_numbers"]) - supported_set)
                )
                inventory_records.append(
                    {
                        **record,
                        "model_compatible": not unsupported,
                        "unsupported_elements": " ".join(
                            chemical_symbols[value] for value in unsupported
                        ),
                    }
                )
                if unsupported:
                    continue
                for fragment, atoms, fragment_charge in (
                    ("AB", complex_atoms, record["charge"]),
                    ("A", fragment_a, record["charge_a"]),
                    ("B", fragment_b, record["charge_b"]),
                ):
                    graph_records.append(
                        {
                            "graph_index": len(structures),
                            **{key: record[key] for key in _NCI_KEYS},
                            "fragment": fragment,
                            "charge": int(fragment_charge),
                            "natoms": len(atoms),
                            "source_geometry_file": record["source_geometry_file"],
                        }
                    )
                    structures.append(atoms)
                reference_records.append(
                    {
                        **{key: record[key] for key in _NCI_KEYS},
                        "ccsd_t_cbs_kcal_mol": record["ccsd_t_cbs_kcal_mol"],
                    }
                )
    except BadZipFile as error:
        raise ValueError(
            f"NCIA250 is not a valid ZIP archive: {archive_path}"
        ) from error

    inventory = pd.DataFrame(inventory_records)
    graph_rows = pd.DataFrame(graph_records)
    references = pd.DataFrame(reference_records)
    compatible = int(inventory["model_compatible"].sum())
    if len(inventory) != 250 or len(structures) != 3 * compatible:
        raise ValueError("NCIA250 records are incomplete")
    excluded_elements = sorted(
        {
            symbol
            for values in inventory.loc[
                ~inventory["model_compatible"], "unsupported_elements"
            ]
            for symbol in str(values).split()
        }
    )
    stats = {
        "total_complexes": len(inventory),
        "compatible_complexes": compatible,
        "excluded_complexes": len(inventory) - compatible,
        "compatible_structures": len(structures),
        "compatible_atom_rows": int(sum(len(atoms) for atoms in structures)),
        "excluded_elements": tuple(excluded_elements),
    }
    return structures, graph_rows, references, inventory, stats
