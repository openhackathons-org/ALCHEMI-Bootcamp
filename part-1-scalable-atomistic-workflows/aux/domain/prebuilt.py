"""Load the checked, prebuilt molecular box used by the Part 1 domain lesson.

The learner notebook should not spend class time running Packmol or rendering a
large structure.  This loader checks the shipped files, reconstructs the
declared NCI-derived construction plan, and verifies the saved structure before
it reaches Toolkit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
from typing import Any

from ase import Atoms
from ase.io import read as ase_read
import numpy as np
import pandas as pd

from ..artifacts import sha256_file
from ..nci_atlas import NCI_ATLAS_SUBSET_SHA256
from .config import DOMAIN_METHODOLOGY
from .packing import (
    COMPONENT_NAMES,
    MolecularBoxPlan,
    MolecularBoxValidation,
    plan_nci_molecular_box,
    validate_molecular_box,
)


PREBUILT_DOMAIN_BOX_SCHEMA = "alchemi.part1-domain-base-box.v1"
PREBUILT_DOMAIN_BOX_FILES = frozenset(
    {"manifest.json", "structure.extxyz", "preview.png", "SHA256SUMS"}
)
_INDEXED_FILES = frozenset(PREBUILT_DOMAIN_BOX_FILES - {"SHA256SUMS"})
_REQUIRED_ATOM_ARRAYS = (
    "source_atom_id",
    "molecule_id",
    "molecule_component",
    "molecule_kind",
    "template_atom_index",
)
_BASE_ATOM_ARRAYS = frozenset({"numbers", "positions"})
_HEX_DIGITS = frozenset("0123456789abcdef")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PrebuiltDomainBoxError(ValueError):
    """Raised when the shipped domain-box bundle is incomplete or inconsistent."""


@dataclass(frozen=True)
class PrebuiltDomainBoxBundle:
    """One checked prebuilt box and the plan from which it was constructed."""

    bundle_dir: Path
    plan: MolecularBoxPlan
    atoms: Atoms
    validation: MolecularBoxValidation
    manifest: dict[str, Any]
    preview_path: Path


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    context: str,
) -> None:
    observed = set(value)
    if observed == set(expected):
        return
    missing = sorted(set(expected) - observed)
    unexpected = sorted(observed - set(expected))
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise PrebuiltDomainBoxError(
        f"{context} has the wrong keys ({'; '.join(details)})"
    )


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PrebuiltDomainBoxError(f"{context} must be an object")
    return value


def _require_int(
    value: Any,
    *,
    context: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise PrebuiltDomainBoxError(f"{context} must be an integer")
    try:
        integer = int(value)
        unchanged = float(value) == integer
    except (TypeError, ValueError, OverflowError) as exc:
        raise PrebuiltDomainBoxError(f"{context} must be an integer") from exc
    if not unchanged or (minimum is not None and integer < minimum):
        qualifier = f" at least {minimum}" if minimum is not None else ""
        raise PrebuiltDomainBoxError(f"{context} must be an integer{qualifier}")
    return integer


def _require_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise PrebuiltDomainBoxError(f"{context} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PrebuiltDomainBoxError(f"{context} must be finite") from exc
    if not math.isfinite(number):
        raise PrebuiltDomainBoxError(f"{context} must be finite")
    return number


def _require_sha256(value: Any, *, context: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or not set(digest) <= _HEX_DIGITS:
        raise PrebuiltDomainBoxError(f"{context} must be a SHA-256 digest")
    return digest


def _require_nonempty_text(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PrebuiltDomainBoxError(f"{context} must be nonempty text")
    return value.strip()


def _parse_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise PrebuiltDomainBoxError(
            f"{path.name} contains the non-finite JSON value {value}"
        )

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrebuiltDomainBoxError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise PrebuiltDomainBoxError("manifest.json must contain one object")
    return value


def _load_checksum_index(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise PrebuiltDomainBoxError(f"cannot read SHA256SUMS: {exc}") from exc
    checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) != 2:
            raise PrebuiltDomainBoxError(
                f"malformed SHA256SUMS line {line_number}"
            )
        digest = _require_sha256(
            fields[0],
            context=f"SHA256SUMS line {line_number}",
        )
        filename = fields[1].removeprefix("*")
        if filename not in _INDEXED_FILES:
            raise PrebuiltDomainBoxError(
                f"SHA256SUMS line {line_number} names an unexpected file"
            )
        if filename in checksums:
            raise PrebuiltDomainBoxError(
                f"duplicate SHA256SUMS entry for {filename}"
            )
        checksums[filename] = digest
    if set(checksums) != set(_INDEXED_FILES):
        raise PrebuiltDomainBoxError(
            "SHA256SUMS must cover exactly manifest.json, structure.extxyz, "
            "and preview.png"
        )
    return checksums


def _validate_bundle_files(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(f"prebuilt domain-box directory not found: {root}")
    entries = {path.name: path for path in root.iterdir()}
    if set(entries) != set(PREBUILT_DOMAIN_BOX_FILES):
        missing = sorted(set(PREBUILT_DOMAIN_BOX_FILES) - set(entries))
        unexpected = sorted(set(entries) - set(PREBUILT_DOMAIN_BOX_FILES))
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise PrebuiltDomainBoxError(
            "prebuilt domain-box directory has the wrong files "
            f"({'; '.join(details)})"
        )
    for name, path in entries.items():
        if path.is_symlink() or not path.is_file():
            raise PrebuiltDomainBoxError(f"{name} must be a regular file")

    checksums = _load_checksum_index(entries["SHA256SUMS"])
    for filename, expected in checksums.items():
        observed = sha256_file(entries[filename])
        if observed != expected:
            raise PrebuiltDomainBoxError(f"SHA-256 mismatch for {filename}")
    return checksums


def _load_one_structure(path: Path) -> Atoms:
    try:
        frames = ase_read(path, index=":", format="extxyz")
    except Exception as exc:
        raise PrebuiltDomainBoxError(f"cannot read structure.extxyz: {exc}") from exc
    if not isinstance(frames, list) or len(frames) != 1:
        raise PrebuiltDomainBoxError(
            "structure.extxyz must contain exactly one structure"
        )
    atoms = frames[0]
    if not isinstance(atoms, Atoms):
        raise PrebuiltDomainBoxError("structure.extxyz did not produce ASE Atoms")
    return atoms


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return sha256(contiguous.tobytes()).hexdigest()


def _expected_source_arrays(plan: MolecularBoxPlan) -> dict[str, np.ndarray]:
    molecule_ids: list[np.ndarray] = []
    component_ids: list[np.ndarray] = []
    template_indices: list[np.ndarray] = []
    next_molecule_id = 0
    for component_id, (template, count) in enumerate(
        zip(plan.templates, plan.molecule_counts, strict=True)
    ):
        molecule_ids.append(
            np.repeat(
                np.arange(
                    next_molecule_id,
                    next_molecule_id + count,
                    dtype=np.int64,
                ),
                template.atom_count,
            )
        )
        component_ids.append(
            np.full(count * template.atom_count, component_id, dtype=np.int64)
        )
        template_indices.append(
            np.tile(np.arange(template.atom_count, dtype=np.int64), count)
        )
        next_molecule_id += count

    components = np.concatenate(component_ids)
    return {
        "source_atom_id": np.arange(plan.atom_count, dtype=np.int64),
        "molecule_id": np.concatenate(molecule_ids),
        "molecule_component": components,
        "molecule_kind": components,
        "template_atom_index": np.concatenate(template_indices),
    }


def _validate_arrays(
    atoms: Atoms,
    plan: MolecularBoxPlan,
    records: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        records,
        set(_REQUIRED_ATOM_ARRAYS),
        context="manifest.structure.arrays",
    )
    observed_names = set(atoms.arrays)
    expected_names = set(_BASE_ATOM_ARRAYS) | set(_REQUIRED_ATOM_ARRAYS)
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise PrebuiltDomainBoxError(
            "structure.extxyz has the wrong atom arrays "
            f"({'; '.join(details)})"
        )

    expected_arrays = _expected_source_arrays(plan)
    for name in _REQUIRED_ATOM_ARRAYS:
        value = np.asarray(atoms.arrays[name])
        record = _require_mapping(
            records[name],
            context=f"manifest.structure.arrays.{name}",
        )
        _require_exact_keys(
            record,
            {"dtype", "shape", "sha256"},
            context=f"manifest.structure.arrays.{name}",
        )
        shape = record["shape"]
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in shape
        ):
            raise PrebuiltDomainBoxError(
                f"manifest.structure.arrays.{name}.shape must be a list of integers"
            )
        if shape != list(value.shape):
            raise PrebuiltDomainBoxError(f"{name} shape does not match its manifest")
        if str(record["dtype"]) != str(value.dtype):
            raise PrebuiltDomainBoxError(f"{name} dtype does not match its manifest")
        digest = _require_sha256(
            record["sha256"],
            context=f"manifest.structure.arrays.{name}.sha256",
        )
        if digest != _array_sha256(value):
            raise PrebuiltDomainBoxError(
                f"{name} SHA-256 does not match its manifest"
            )
        if not np.issubdtype(value.dtype, np.integer):
            raise PrebuiltDomainBoxError(f"{name} must contain integers")
        if not np.array_equal(value.astype(np.int64), expected_arrays[name]):
            raise PrebuiltDomainBoxError(
                f"{name} does not match the NCI-derived molecule layout"
            )


def _validate_methodology(value: Any) -> None:
    record = _require_mapping(value, context="manifest.methodology")
    _require_exact_keys(
        record,
        {"schema", "name", "version"},
        context="manifest.methodology",
    )
    expected = {
        "schema": DOMAIN_METHODOLOGY.schema,
        "name": DOMAIN_METHODOLOGY.name,
        "version": DOMAIN_METHODOLOGY.version,
    }
    if dict(record) != expected:
        raise PrebuiltDomainBoxError(
            "manifest.methodology does not match DOMAIN_METHODOLOGY"
        )


def _validate_source(value: Any) -> None:
    source = _require_mapping(value, context="manifest.source")
    _require_exact_keys(
        source,
        {
            "nci_subset_file",
            "nci_subset_sha256",
            "nci_system_id",
            "nci_scale",
            "molecule_names",
            "molecule_counts",
            "packmol",
        },
        context="manifest.source",
    )
    if (
        str(source["nci_subset_file"])
        != "../../nci_atlas/nci-atlas-curves.csv.gz"
    ):
        raise PrebuiltDomainBoxError(
            "manifest.source.nci_subset_file is not the packaged NCI subset"
        )
    if (
        _require_sha256(
            source["nci_subset_sha256"],
            context="manifest.source.nci_subset_sha256",
        )
        != NCI_ATLAS_SUBSET_SHA256
    ):
        raise PrebuiltDomainBoxError(
            "manifest.source.nci_subset_sha256 does not match the packaged NCI subset"
        )
    if str(source["nci_system_id"]) != DOMAIN_METHODOLOGY.nci_system_id:
        raise PrebuiltDomainBoxError("manifest.source.nci_system_id is incorrect")
    if not math.isclose(
        _require_float(source["nci_scale"], context="manifest.source.nci_scale"),
        DOMAIN_METHODOLOGY.nci_scale,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise PrebuiltDomainBoxError("manifest.source.nci_scale is incorrect")
    if source["molecule_names"] != list(COMPONENT_NAMES):
        raise PrebuiltDomainBoxError("manifest.source.molecule_names is incorrect")
    expected_counts = {
        name: DOMAIN_METHODOLOGY.live_molecules_per_species
        for name in COMPONENT_NAMES
    }
    if source["molecule_counts"] != expected_counts:
        raise PrebuiltDomainBoxError("manifest.source.molecule_counts is incorrect")

    packmol = _require_mapping(
        source["packmol"],
        context="manifest.source.packmol",
    )
    _require_exact_keys(
        packmol,
        {
            "version",
            "seed",
            "tolerance_a",
            "precision_a",
            "input_sha256",
            "stdout_sha256",
            "stderr_sha256",
            "raw_output_sha256",
        },
        context="manifest.source.packmol",
    )
    _require_nonempty_text(
        packmol["version"],
        context="manifest.source.packmol.version",
    )
    if (
        _require_int(packmol["seed"], context="manifest.source.packmol.seed")
        != DOMAIN_METHODOLOGY.packmol_seed
    ):
        raise PrebuiltDomainBoxError("manifest.source.packmol.seed is incorrect")
    for key, expected in (
        ("tolerance_a", DOMAIN_METHODOLOGY.packmol_tolerance_a),
        ("precision_a", DOMAIN_METHODOLOGY.packmol_precision_a),
    ):
        if not math.isclose(
            _require_float(
                packmol[key],
                context=f"manifest.source.packmol.{key}",
            ),
            expected,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise PrebuiltDomainBoxError(
                f"manifest.source.packmol.{key} is incorrect"
            )
    for key in (
        "input_sha256",
        "stdout_sha256",
        "stderr_sha256",
        "raw_output_sha256",
    ):
        _require_sha256(
            packmol[key],
            context=f"manifest.source.packmol.{key}",
        )


def _as_float_array(value: Any, *, context: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PrebuiltDomainBoxError(f"{context} must be numeric") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise PrebuiltDomainBoxError(
            f"{context} must have finite shape {shape}"
        )
    return array


def _same_float(
    value: Any,
    expected: float,
    *,
    context: str,
    atol: float = 1.0e-10,
) -> None:
    observed = _require_float(value, context=context)
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=atol):
        raise PrebuiltDomainBoxError(f"{context} does not match the checked structure")


def _validate_structure_record(
    value: Any,
    *,
    atoms: Atoms,
    plan: MolecularBoxPlan,
    validation: MolecularBoxValidation,
    structure_path: Path,
    structure_sha256: str,
) -> None:
    record = _require_mapping(value, context="manifest.structure")
    _require_exact_keys(
        record,
        {
            "file",
            "sha256",
            "bytes",
            "atom_count",
            "molecule_count",
            "molecules_per_species",
            "cell_a",
            "pbc",
            "construction_density_g_cm3",
            "density_from_mass_and_cell_g_cm3",
            "periodic_min_distance_a",
            "min_distance_required_a",
            "arrays",
        },
        context="manifest.structure",
    )
    if record["file"] != "structure.extxyz":
        raise PrebuiltDomainBoxError(
            "manifest.structure.file must be structure.extxyz"
        )
    if (
        _require_sha256(record["sha256"], context="manifest.structure.sha256")
        != structure_sha256
    ):
        raise PrebuiltDomainBoxError(
            "manifest.structure.sha256 does not match structure.extxyz"
        )
    if (
        _require_int(
            record["bytes"],
            context="manifest.structure.bytes",
            minimum=1,
        )
        != structure_path.stat().st_size
    ):
        raise PrebuiltDomainBoxError(
            "manifest.structure.bytes does not match structure.extxyz"
        )
    for key, expected in (
        ("atom_count", plan.atom_count),
        ("molecule_count", plan.molecule_count),
        ("molecules_per_species", plan.molecules_per_species),
    ):
        if _require_int(record[key], context=f"manifest.structure.{key}") != expected:
            raise PrebuiltDomainBoxError(f"manifest.structure.{key} is incorrect")
    if plan.atom_count != 3_200 or plan.molecules_per_species != 128:
        raise PrebuiltDomainBoxError(
            "the prebuilt lesson input must contain 3,200 atoms and "
            "128 molecules per species"
        )

    manifest_cell = _as_float_array(
        record["cell_a"],
        context="manifest.structure.cell_a",
        shape=(3, 3),
    )
    actual_cell = np.asarray(atoms.cell, dtype=float)
    if not np.allclose(manifest_cell, actual_cell, rtol=0.0, atol=1.0e-10):
        raise PrebuiltDomainBoxError(
            "manifest.structure.cell_a does not match structure.extxyz"
        )
    if record["pbc"] != [True, True, True] or not np.asarray(
        atoms.pbc, dtype=bool
    ).all():
        raise PrebuiltDomainBoxError(
            "the prebuilt structure must be periodic in x, y, and z"
        )
    _same_float(
        record["construction_density_g_cm3"],
        plan.construction_density_g_cm3,
        context="manifest.structure.construction_density_g_cm3",
    )
    _same_float(
        record["density_from_mass_and_cell_g_cm3"],
        validation.density_from_mass_and_cell_g_cm3,
        context="manifest.structure.density_from_mass_and_cell_g_cm3",
    )
    if validation.periodic_min_distance_a is None:
        raise PrebuiltDomainBoxError(
            "the prebuilt structure has no measurable intermolecular contact"
        )
    _same_float(
        record["periodic_min_distance_a"],
        validation.periodic_min_distance_a,
        context="manifest.structure.periodic_min_distance_a",
        atol=1.0e-8,
    )
    _same_float(
        record["min_distance_required_a"],
        validation.min_distance_required_a,
        context="manifest.structure.min_distance_required_a",
    )


def _png_dimensions(path: Path) -> tuple[int, int]:
    try:
        with path.open("rb") as stream:
            header = stream.read(24)
    except OSError as exc:
        raise PrebuiltDomainBoxError(f"cannot read preview.png: {exc}") from exc
    if (
        len(header) != 24
        or header[:8] != _PNG_SIGNATURE
        or header[12:16] != b"IHDR"
    ):
        raise PrebuiltDomainBoxError("preview.png is not a valid PNG image")
    width, height = struct.unpack(">II", header[16:24])
    if width < 1 or height < 1:
        raise PrebuiltDomainBoxError("preview.png has invalid dimensions")
    return width, height


def _validate_render(
    value: Any,
    *,
    preview_path: Path,
    preview_sha256: str,
    structure_sha256: str,
    atom_count: int,
) -> None:
    record = _require_mapping(value, context="manifest.render")
    _require_exact_keys(
        record,
        {
            "schema",
            "renderer",
            "ovito_version",
            "source",
            "source_sha256",
            "output",
            "output_sha256",
            "bytes",
            "width_px",
            "height_px",
            "particle_count",
            "bond_count",
            "camera_direction",
            "camera_distance_margin",
            "background_rgb",
        },
        context="manifest.render",
    )
    if record["schema"] != "alchemi.part1-domain-box-render.v1":
        raise PrebuiltDomainBoxError("manifest.render has an unknown schema")
    if record["renderer"] != "OVITO ANARI renderer":
        raise PrebuiltDomainBoxError("manifest.render.renderer is incorrect")
    _require_nonempty_text(
        record["ovito_version"],
        context="manifest.render.ovito_version",
    )
    if record["source"] != "structure.extxyz":
        raise PrebuiltDomainBoxError(
            "manifest.render.source must be structure.extxyz"
        )
    if (
        _require_sha256(
            record["source_sha256"],
            context="manifest.render.source_sha256",
        )
        != structure_sha256
    ):
        raise PrebuiltDomainBoxError(
            "manifest.render.source_sha256 does not match structure.extxyz"
        )
    if record["output"] != "preview.png":
        raise PrebuiltDomainBoxError("manifest.render.output must be preview.png")
    if (
        _require_sha256(
            record["output_sha256"],
            context="manifest.render.output_sha256",
        )
        != preview_sha256
    ):
        raise PrebuiltDomainBoxError(
            "manifest.render.output_sha256 does not match preview.png"
        )
    if (
        _require_int(
            record["bytes"],
            context="manifest.render.bytes",
            minimum=1,
        )
        != preview_path.stat().st_size
    ):
        raise PrebuiltDomainBoxError(
            "manifest.render.bytes does not match preview.png"
        )
    width, height = _png_dimensions(preview_path)
    if (
        _require_int(
            record["width_px"],
            context="manifest.render.width_px",
            minimum=1,
        )
        != width
        or _require_int(
            record["height_px"],
            context="manifest.render.height_px",
            minimum=1,
        )
        != height
    ):
        raise PrebuiltDomainBoxError(
            "manifest.render dimensions do not match preview.png"
        )
    if (
        _require_int(
            record["particle_count"],
            context="manifest.render.particle_count",
            minimum=1,
        )
        != atom_count
    ):
        raise PrebuiltDomainBoxError(
            "manifest.render.particle_count does not match structure.extxyz"
        )
    _require_int(
        record["bond_count"],
        context="manifest.render.bond_count",
        minimum=0,
    )
    _as_float_array(
        record["camera_direction"],
        context="manifest.render.camera_direction",
        shape=(3,),
    )
    if (
        _require_float(
            record["camera_distance_margin"],
            context="manifest.render.camera_distance_margin",
        )
        <= 0.0
    ):
        raise PrebuiltDomainBoxError(
            "manifest.render.camera_distance_margin must be positive"
        )
    background = _as_float_array(
        record["background_rgb"],
        context="manifest.render.background_rgb",
        shape=(3,),
    )
    if np.any((background < 0.0) | (background > 1.0)):
        raise PrebuiltDomainBoxError(
            "manifest.render.background_rgb values must be between 0 and 1"
        )


def _validate_structure_info(atoms: Atoms, plan: MolecularBoxPlan) -> None:
    expected: dict[str, Any] = {
        "system": "phenol + N-methylacetamide",
        "pair_count": plan.molecules_per_species,
        "molecules_per_species": plan.molecules_per_species,
        "charge": plan.net_charge_e,
        "construction_density_g_cm3": plan.construction_density_g_cm3,
        "packmol_seed": plan.packmol_seed,
        "packmol_tolerance_a": plan.packmol_tolerance_a,
        "packmol_precision_a": plan.packmol_precision_a,
        "nci_system_id": plan.system_id,
        "nci_scale": plan.scale,
    }
    missing = set(expected) - set(atoms.info)
    if missing:
        raise PrebuiltDomainBoxError(
            "structure.extxyz is missing metadata: " + ", ".join(sorted(missing))
        )
    for key, expected_value in expected.items():
        observed = atoms.info[key]
        if key == "nci_system_id":
            if str(observed) != str(expected_value):
                raise PrebuiltDomainBoxError(
                    "structure.info.nci_system_id does not match the "
                    "construction plan"
                )
        elif isinstance(expected_value, float):
            if not math.isclose(
                _require_float(observed, context=f"structure.info.{key}"),
                expected_value,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise PrebuiltDomainBoxError(
                    f"structure.info.{key} does not match the construction plan"
                )
        elif observed != expected_value:
            raise PrebuiltDomainBoxError(
                f"structure.info.{key} does not match the construction plan"
            )


def load_prebuilt_domain_box(
    bundle_dir: str | Path,
    nci_table: pd.DataFrame,
) -> PrebuiltDomainBoxBundle:
    """Load and verify the shipped 3,200-atom Part 1 domain input.

    ``nci_table`` is the already loaded, checked NCI Atlas tutorial subset used
    earlier in the notebook.  It is used to reconstruct the phenol and
    N-methylacetamide templates and the exact molecular-box plan.  Packmol is
    not run by this function.
    """

    root = Path(bundle_dir).resolve()
    checksums = _validate_bundle_files(root)
    manifest = _parse_json(root / "manifest.json")
    _require_exact_keys(
        manifest,
        {
            "schema",
            "bundle_id",
            "methodology",
            "source",
            "structure",
            "render",
            "interpretation",
        },
        context="manifest",
    )
    if manifest["schema"] != PREBUILT_DOMAIN_BOX_SCHEMA:
        raise PrebuiltDomainBoxError("manifest has an unknown schema")

    _validate_methodology(manifest["methodology"])
    _validate_source(manifest["source"])
    source = _require_mapping(manifest["source"], context="manifest.source")
    plan = plan_nci_molecular_box(
        nci_table,
        nci_system_id=str(source["nci_system_id"]),
        nci_scale=_require_float(
            source["nci_scale"],
            context="manifest.source.nci_scale",
        ),
        molecules_per_species=DOMAIN_METHODOLOGY.live_molecules_per_species,
        construction_density_g_cm3=(
            DOMAIN_METHODOLOGY.construction_density_g_cm3
        ),
        packmol_tolerance_a=DOMAIN_METHODOLOGY.packmol_tolerance_a,
        packmol_precision_a=DOMAIN_METHODOLOGY.packmol_precision_a,
        packmol_seed=DOMAIN_METHODOLOGY.packmol_seed,
    )

    structure_path = root / "structure.extxyz"
    atoms = _load_one_structure(structure_path)
    structure_record = _require_mapping(
        manifest["structure"],
        context="manifest.structure",
    )
    arrays_record = _require_mapping(
        structure_record.get("arrays"),
        context="manifest.structure.arrays",
    )
    _validate_arrays(atoms, plan, arrays_record)
    _validate_structure_info(atoms, plan)

    try:
        validation = validate_molecular_box(plan, atoms.copy())
    except (TypeError, ValueError) as exc:
        raise PrebuiltDomainBoxError(
            f"structure.extxyz does not match the construction plan: {exc}"
        ) from exc

    structure_sha256 = checksums["structure.extxyz"]
    _validate_structure_record(
        structure_record,
        atoms=atoms,
        plan=plan,
        validation=validation,
        structure_path=structure_path,
        structure_sha256=structure_sha256,
    )
    if manifest["bundle_id"] != "part1-phenol-nma-domain-base-128-v1":
        raise PrebuiltDomainBoxError("manifest.bundle_id is incorrect")

    preview_path = root / "preview.png"
    _validate_render(
        manifest["render"],
        preview_path=preview_path,
        preview_sha256=checksums["preview.png"],
        structure_sha256=structure_sha256,
        atom_count=plan.atom_count,
    )
    interpretation = _require_mapping(
        manifest["interpretation"],
        context="manifest.interpretation",
    )
    _require_exact_keys(
        interpretation,
        {"construction", "scaling"},
        context="manifest.interpretation",
    )
    for key in ("construction", "scaling"):
        _require_nonempty_text(
            interpretation[key],
            context=f"manifest.interpretation.{key}",
        )

    return PrebuiltDomainBoxBundle(
        bundle_dir=root,
        plan=plan,
        atoms=atoms,
        validation=validation,
        manifest=manifest,
        preview_path=preview_path,
    )


__all__ = (
    "PREBUILT_DOMAIN_BOX_FILES",
    "PREBUILT_DOMAIN_BOX_SCHEMA",
    "PrebuiltDomainBoxBundle",
    "PrebuiltDomainBoxError",
    "load_prebuilt_domain_box",
)
