#!/usr/bin/env python3
"""Generate a frozen-monomer B97-3c water-dimer interaction scan.

The eight geometries come directly from
``aux.structures.make_water_dimer_scan``.  For every O--O separation this
driver performs three independent Psi4 single points at the *same Cartesian
coordinates* and reports

    E_int = E(AB) - E(A) - E(B).

All three energies are full canonical B97-3c/def2-mTZVP endpoint energies.
They therefore include Psi4's integrated D3(BJ)-ATM and geometrical
counterpoise/short-range basis correction (gCP).  No Boys--Bernardi ghost
basis counterpoise is applied, and no B97-3c component is subtracted or
relabelled as an AIMNet residual.

The command has no reduced or sampled mode.  A successful run always contains
the declared eight distances and 24 single-point calculations::

    python water_dimer_b97_3c.py \
      --threads 8 \
      --memory "16 GB"

The output directory must be new or empty.  The generated structures are new
deterministic tutorial geometries, not a redistributed benchmark dataset.
See ``license_and_data_boundary`` in the JSON manifest for the dependency and
redistribution boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


METHOD = "b97-3c"
BASIS = "def2-mtzvp"
MODEL_CHEMISTRY = f"{METHOD}/{BASIS}"
PSI4_REQUIRED_MAJOR_MINOR = (1, 11)

# Dense sampling around the hydrogen-bonded well and a declared long-range
# tail.  These points are fixed: the CLI intentionally has no smaller mode.
DEFAULT_OO_DISTANCES_ANGSTROM = (
    2.50,
    2.70,
    2.90,
    3.20,
    3.50,
    3.90,
    4.40,
    5.00,
)

HARTREE_TO_KCAL_MOL = 627.5094740631
HARTREE_TO_KJ_MOL = HARTREE_TO_KCAL_MOL * 4.184
GENERATOR_SOURCE = Path(__file__).resolve()
TUTORIAL_DIR = GENERATOR_SOURCE.parents[1]
STRUCTURE_SOURCE = TUTORIAL_DIR / "aux" / "structures.py"
ENVIRONMENT_SOURCE = GENERATOR_SOURCE.with_name("environment.yml")
LICENSE_SOURCE = TUTORIAL_DIR.parents[0] / "LICENSE"
DEFAULT_OUTPUT_DIR = GENERATOR_SOURCE.parent / "artifacts" / "water_dimer_b97_3c"


@dataclass(frozen=True)
class Geometry:
    """One finite, ordered Cartesian structure used in a single point."""

    symbols: tuple[str, ...]
    positions_angstrom: np.ndarray
    label: str
    role: str

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_angstrom, dtype=np.float64)
        if positions.shape != (len(self.symbols), 3):
            raise ValueError(
                f"{self.label}: positions must have shape "
                f"({len(self.symbols)}, 3), got {positions.shape}"
            )
        if not np.isfinite(positions).all():
            raise ValueError(f"{self.label}: positions contain non-finite values")
        copied = np.ascontiguousarray(positions).copy()
        copied.setflags(write=False)
        object.__setattr__(self, "positions_angstrom", copied)


@dataclass(frozen=True)
class ScanGeometry:
    """The dimer and its two exact frozen Cartesian monomer slices."""

    index: int
    requested_oo_distance_angstrom: float
    measured_oo_distance_angstrom: float
    ab: Geometry
    a: Geometry
    b: Geometry


@dataclass(frozen=True)
class EnergyRecord:
    """Full endpoint energies and the resulting interaction energy."""

    point: ScanGeometry
    energy_ab_Eh: float
    energy_a_Eh: float
    energy_b_Eh: float

    @property
    def interaction_Eh(self) -> float:
        return self.energy_ab_Eh - self.energy_a_Eh - self.energy_b_Eh


def _float_text(value: float) -> str:
    """Round-trip-safe decimal representation for stored float64 geometry."""

    return format(float(value), ".17g")


def canonical_geometry_payload(geometry: Geometry) -> dict[str, Any]:
    """Return the exact textual geometry representation used for hashing."""

    return {
        "symbols": list(geometry.symbols),
        "positions_angstrom_decimal": [
            [_float_text(value) for value in row]
            for row in geometry.positions_angstrom
        ],
    }


def geometry_sha256(geometry: Geometry) -> str:
    payload = json.dumps(
        canonical_geometry_payload(geometry),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atoms_positions(atoms: Any) -> np.ndarray:
    getter = getattr(atoms, "get_positions", None)
    values = getter() if callable(getter) else atoms.positions
    return np.asarray(values, dtype=np.float64)


def _atoms_symbols(atoms: Any) -> tuple[str, ...]:
    getter = getattr(atoms, "get_chemical_symbols", None)
    values = getter() if callable(getter) else atoms.symbols
    return tuple(str(symbol) for symbol in values)


def scan_geometry_from_atoms(
    atoms: Any, *, index: int, requested_distance_angstrom: float
) -> ScanGeometry:
    """Validate one ASE-like dimer and retain exact ``OHH | OHH`` slices."""

    symbols = _atoms_symbols(atoms)
    positions = _atoms_positions(atoms)
    if symbols != ("O", "H", "H", "O", "H", "H"):
        raise ValueError(
            f"scan point {index}: expected OHH|OHH atom order, got {symbols}"
        )
    if positions.shape != (6, 3) or not np.isfinite(positions).all():
        raise ValueError(f"scan point {index}: invalid 6-by-3 Cartesian geometry")

    pbc = np.asarray(getattr(atoms, "pbc", False), dtype=bool)
    if pbc.any():
        raise ValueError(f"scan point {index}: reference structures must be nonperiodic")

    info = dict(getattr(atoms, "info", {}))
    if "oo_distance_angstrom" not in info:
        raise ValueError(
            f"scan point {index}: missing builder oo_distance_angstrom metadata"
        )
    declared = float(info["oo_distance_angstrom"])
    requested = float(requested_distance_angstrom)
    if not np.isclose(declared, requested, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"scan point {index}: builder declared {declared}, requested {requested}"
        )

    measured = float(np.linalg.norm(positions[3] - positions[0]))
    if not np.isclose(measured, requested, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"scan point {index}: measured O-O distance {measured} does not "
            f"match requested {requested}"
        )

    stem = f"water-dimer-{index:02d}-{requested:.4f}A"
    ab = Geometry(symbols, positions, f"{stem}-AB", "AB")
    a = Geometry(symbols[:3], positions[:3], f"{stem}-A", "A")
    b = Geometry(symbols[3:], positions[3:], f"{stem}-B", "B")

    # This is intentionally stricter than an allclose check.  The monomers
    # used for subtraction are exact copies of the parent dimer coordinates.
    if not np.array_equal(a.positions_angstrom, ab.positions_angstrom[:3]):
        raise AssertionError("monomer A is not an exact frozen dimer slice")
    if not np.array_equal(b.positions_angstrom, ab.positions_angstrom[3:]):
        raise AssertionError("monomer B is not an exact frozen dimer slice")

    return ScanGeometry(index, requested, measured, ab, a, b)


def load_scan_geometries() -> list[ScanGeometry]:
    """Load the tutorial builder lazily, keeping Psi4 out of module import."""

    tutorial_text = str(TUTORIAL_DIR)
    if tutorial_text not in sys.path:
        sys.path.insert(0, tutorial_text)
    try:
        from aux.structures import make_water_dimer_scan
    except ImportError as exc:
        raise RuntimeError(
            "The tutorial structure builder could not be imported. Run in an "
            "environment containing ASE and NumPy, with the tutorial tree intact."
        ) from exc

    atoms_list = make_water_dimer_scan(DEFAULT_OO_DISTANCES_ANGSTROM)
    if len(atoms_list) != len(DEFAULT_OO_DISTANCES_ANGSTROM):
        raise RuntimeError("structure builder returned an incomplete scan")
    return [
        scan_geometry_from_atoms(atoms, index=index, requested_distance_angstrom=distance)
        for index, (atoms, distance) in enumerate(
            zip(atoms_list, DEFAULT_OO_DISTANCES_ANGSTROM, strict=True)
        )
    ]


def iter_structures(
    points: Sequence[ScanGeometry],
) -> Iterable[tuple[ScanGeometry, Geometry]]:
    for point in points:
        yield point, point.ab
        yield point, point.a
        yield point, point.b


def write_structures_extxyz(path: str | Path, points: Sequence[ScanGeometry]) -> None:
    """Write all 24 exact AB/A/B inputs as standard extended XYZ frames."""

    blocks: list[str] = []
    for point, geometry in iter_structures(points):
        parent_hash = geometry_sha256(point.ab)
        comment = " ".join(
            [
                "Properties=species:S:1:pos:R:3",
                f"label={geometry.label}",
                f"scan_index={point.index}",
                f"role={geometry.role}",
                (
                    "requested_oo_distance_angstrom="
                    f"{_float_text(point.requested_oo_distance_angstrom)}"
                ),
                (
                    "measured_oo_distance_angstrom="
                    f"{_float_text(point.measured_oo_distance_angstrom)}"
                ),
                f"geometry_sha256={geometry_sha256(geometry)}",
                f"parent_dimer_geometry_sha256={parent_hash}",
                'pbc="F F F"',
            ]
        )
        lines = [str(len(geometry.symbols)), comment]
        for symbol, xyz in zip(
            geometry.symbols, geometry.positions_angstrom, strict=True
        ):
            lines.append(
                f"{symbol} {_float_text(xyz[0])} {_float_text(xyz[1])} "
                f"{_float_text(xyz[2])}"
            )
        blocks.append("\n".join(lines))
    Path(path).write_text("\n".join(blocks) + "\n", encoding="utf-8")


def make_psi4_molecule(psi4: Any, geometry: Geometry) -> Any:
    """Build a neutral singlet without recentering or reorientation."""

    lines = ["0 1"]
    for symbol, xyz in zip(
        geometry.symbols, geometry.positions_angstrom, strict=True
    ):
        lines.append(
            f"{symbol} {_float_text(xyz[0])} {_float_text(xyz[1])} "
            f"{_float_text(xyz[2])}"
        )
    lines.extend(["units angstrom", "symmetry c1", "no_com", "no_reorient"])
    molecule = psi4.geometry("\n".join(lines))
    if hasattr(molecule, "set_name"):
        molecule.set_name(geometry.label)
    if hasattr(molecule, "update_geometry"):
        molecule.update_geometry()
    return molecule


def evaluate_full_b97_3c_energy(psi4: Any, geometry: Geometry) -> float:
    """Evaluate one complete canonical B97-3c endpoint energy in hartree."""

    molecule = make_psi4_molecule(psi4, geometry)
    energy = float(psi4.energy(MODEL_CHEMISTRY, molecule=molecule))
    if not np.isfinite(energy):
        raise RuntimeError(f"{geometry.label}: Psi4 returned a non-finite energy")
    return energy


def evaluate_scan_point(psi4: Any, point: ScanGeometry) -> EnergyRecord:
    """Run the required AB, A, and B calculations without monomer caching."""

    energies: dict[str, float] = {}
    for geometry in (point.ab, point.a, point.b):
        print(
            f"[{point.index + 1:02d}/{len(DEFAULT_OO_DISTANCES_ANGSTROM):02d}] "
            f"{point.requested_oo_distance_angstrom:.4f} A  {geometry.role}",
            flush=True,
        )
        energies[geometry.role] = evaluate_full_b97_3c_energy(psi4, geometry)
    return EnergyRecord(point, energies["AB"], energies["A"], energies["B"])


CSV_FIELDS = (
    "scan_index",
    "requested_oo_distance_angstrom",
    "measured_oo_distance_angstrom",
    "energy_ab_Eh",
    "energy_a_Eh",
    "energy_b_Eh",
    "interaction_Eh",
    "interaction_kcal_mol",
    "interaction_kJ_mol",
    "ab_geometry_sha256",
    "a_geometry_sha256",
    "b_geometry_sha256",
)


def energy_record_dict(record: EnergyRecord) -> dict[str, Any]:
    point = record.point
    interaction = record.interaction_Eh
    return {
        "scan_index": point.index,
        "requested_oo_distance_angstrom": point.requested_oo_distance_angstrom,
        "measured_oo_distance_angstrom": point.measured_oo_distance_angstrom,
        "energy_ab_Eh": record.energy_ab_Eh,
        "energy_a_Eh": record.energy_a_Eh,
        "energy_b_Eh": record.energy_b_Eh,
        "interaction_Eh": interaction,
        "interaction_kcal_mol": interaction * HARTREE_TO_KCAL_MOL,
        "interaction_kJ_mol": interaction * HARTREE_TO_KJ_MOL,
        "ab_geometry_sha256": geometry_sha256(point.ab),
        "a_geometry_sha256": geometry_sha256(point.a),
        "b_geometry_sha256": geometry_sha256(point.b),
    }


def write_scan_csv(path: str | Path, records: Sequence[EnergyRecord]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            row = energy_record_dict(record)
            writer.writerow(
                {
                    key: (format(value, ".16e") if isinstance(value, float) else value)
                    for key, value in row.items()
                }
            )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_metadata(psi4: Any) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "psi4": str(psi4.__version__),
        "numpy": np.__version__,
        "ase": package_version("ase"),
        "qcengine": package_version("qcengine"),
        "qcelemental": package_version("qcelemental"),
        "dftd3_python": package_version("dftd3"),
        "gcp_correction": package_version("gcp-correction"),
        "s_dftd3_executable": shutil.which("s-dftd3"),
        "mctc_gcp_executable": shutil.which("mctc-gcp"),
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_CLUSTER_NAME",
                "SLURM_JOB_NUM_NODES",
                "SLURM_CPUS_PER_TASK",
            )
            if os.environ.get(key) is not None
        },
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def source_record(path: Path, relative_to: Path | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": (
            path.relative_to(relative_to).as_posix()
            if relative_to is not None and path.is_relative_to(relative_to)
            else str(path)
        ),
        "sha256": sha256_file(path),
    }
    return record


def prepare_output_directory(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def stable_output_files(output_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir).as_posix()
        if relative.startswith("psi4_scratch/"):
            continue
        if path.name in {"SHA256SUMS", "timer.dat"}:
            continue
        if path.name.startswith("psi.") and path.name.endswith(".clean"):
            continue
        files.append(path)
    return files


def file_inventory(output_dir: Path, *, excluded: Iterable[str] = ()) -> list[dict[str, Any]]:
    excluded_set = set(excluded)
    return [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in stable_output_files(output_dir)
        if path.relative_to(output_dir).as_posix() not in excluded_set
    ]


def write_sha256sums(output_dir: Path) -> None:
    rows = [
        f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in stable_output_files(output_dir)
    ]
    (output_dir / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def configure_psi4(psi4: Any, args: argparse.Namespace, output_dir: Path) -> Path:
    version = tuple(int(part) for part in str(psi4.__version__).split(".")[:2])
    if version != PSI4_REQUIRED_MAJOR_MINOR:
        required = ".".join(str(part) for part in PSI4_REQUIRED_MAJOR_MINOR)
        raise RuntimeError(
            f"this reference is pinned to Psi4 {required}; found {psi4.__version__}"
        )

    scratch_dir = output_dir / "psi4_scratch"
    scratch_dir.mkdir(parents=False, exist_ok=False)
    psi4.core.IOManager.shared_object().set_default_path(str(scratch_dir))
    psi4.core.set_output_file(str(output_dir / "psi4.out"), False)
    if args.memory is not None:
        psi4.set_memory(args.memory)
    if args.threads is not None:
        psi4.set_num_threads(args.threads)
    psi4.set_options(
        {
            # Explicitly naming the integrated basis avoids the Psi4 1.11
            # composite-driver `(auto)` basis issue.  def2-mTZVP is B97-3c's
            # canonical basis; this is not a change of method.
            "basis": BASIS,
            "scf_type": "pk",
            "reference": "rhf",
            "dft_radial_points": 99,
            "dft_spherical_points": 590,
            "e_convergence": 1.0e-10,
            "d_convergence": 1.0e-10,
            "scf_initial_accelerator": "NONE",
        }
    )
    return scratch_dir


def _source_provenance() -> dict[str, Any]:
    root = TUTORIAL_DIR.parents[0]
    records = {
        "generator": source_record(GENERATOR_SOURCE, root),
        "geometry_builder": source_record(STRUCTURE_SOURCE, root),
    }
    if ENVIRONMENT_SOURCE.is_file():
        records["environment_spec"] = source_record(ENVIRONMENT_SOURCE, root)
    if LICENSE_SOURCE.is_file():
        records["repository_license"] = source_record(LICENSE_SOURCE, root)
    return records


def method_contract() -> dict[str, Any]:
    return {
        "method": "B97-3c",
        "basis": "def2-mTZVP",
        "psi4_model_chemistry": MODEL_CHEMISTRY,
        "basis_is_explicit_psi4_1_11_workaround": True,
        "components": [
            "modified B97 generalized-gradient density functional",
            "canonical def2-mTZVP orbital basis",
            "integrated D3(BJ) dispersion including ATM three-body terms",
            "integrated geometrical counterpoise/short-range basis correction (gCP)",
        ],
        "interaction_energy_definition": "E(AB) - E(A) - E(B)",
        "geometry_policy": (
            "Unrelaxed supermolecular scan. A and B are exact Cartesian slices "
            "of AB at every distance; all three totals are recomputed."
        ),
        "counterpoise_policy": (
            "No Boys-Bernardi ghost-basis counterpoise. Canonical B97-3c gCP "
            "remains included in each full endpoint energy."
        ),
        "comparison_boundary": (
            "Full B97-3c endpoint reference only. The aggregate Psi4 correction "
            "is not an AIMNet residual and is not termwise equivalent to an "
            "external pairwise D3 layer."
        ),
        "charge": 0,
        "multiplicity": 1,
        "psi4_options": {
            "scf_type": "pk",
            "reference": "rhf",
            "dft_radial_points": 99,
            "dft_spherical_points": 590,
            "e_convergence": 1.0e-10,
            "d_convergence": 1.0e-10,
            "scf_initial_accelerator": "NONE",
            "symmetry": "c1",
            "no_com": True,
            "no_reorient": True,
        },
        "references": [
            {
                "citation": (
                    "Brandenburg, Bannwarth, Hansen, and Grimme, "
                    "J. Chem. Phys. 148, 064104 (2018)"
                ),
                "doi": "10.1063/1.5012601",
            },
            {
                "title": "Psi4 1.11 integrated 3c-method documentation",
                "url": "https://psicode.org/psi4manual/1.11.x/gcp.html",
            },
        ],
    }


def license_and_data_boundary() -> dict[str, Any]:
    return {
        "generator_and_repository": (
            "Apache-2.0; see the repository LICENSE recorded by SHA-256"
        ),
        "runtime_dependencies_not_redistributed": {
            "Psi4": "LGPL-3.0; verify installed package metadata",
            "ASE": "LGPL-2.1-or-later; verify installed package metadata",
            "dftd3-python_and_s-dftd3": (
                "LGPL-3.0-or-later; verify installed package metadata"
            ),
            "mctc-gcp": "LGPL-3.0-or-later; verify installed package metadata",
        },
        "generated_data": (
            "New numerical results and deterministic tutorial structures; no "
            "third-party benchmark dataset or dependency binary is bundled."
        ),
        "citation_request": (
            "Cite the original B97-3c publication and Psi4 when using the "
            "calculated reference values."
        ),
    }


def run_scan(args: argparse.Namespace) -> int:
    try:
        import psi4
    except ImportError as exc:
        raise RuntimeError(
            "Psi4 is not importable. Use the preserved reference environment "
            "described by reference/environment.yml."
        ) from exc

    points = load_scan_geometries()
    output_dir = prepare_output_directory(args.output)
    # Psi4 writes timer.dat to the process working directory at interpreter
    # shutdown. Keep it inside this new artifact directory.
    os.chdir(output_dir)
    scratch_dir = configure_psi4(psi4, args, output_dir)
    write_structures_extxyz(output_dir / "structures.extxyz", points)

    run_config = {
        "schema": "alchemi.b97-3c-water-dimer-run-config.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(output_dir),
        "method": method_contract(),
        "scan": {
            "builder": "aux.structures.make_water_dimer_scan",
            "distances_angstrom": list(DEFAULT_OO_DISTANCES_ANGSTROM),
            "distance_count": len(points),
            "single_point_count": 3 * len(points),
            "roles_per_distance": ["AB", "A", "B"],
            "no_reduced_path": True,
        },
        "runtime": {
            "threads": int(psi4.get_num_threads()),
            "memory_argument": args.memory,
            "scratch_directory": str(scratch_dir),
        },
        "environment": dependency_metadata(psi4),
        "sources": _source_provenance(),
        "license_and_data_boundary": license_and_data_boundary(),
    }
    write_json(output_dir / "run_config.json", run_config)

    records: list[EnergyRecord] = []
    try:
        for point in points:
            records.append(evaluate_scan_point(psi4, point))
    except Exception as exc:
        if records:
            write_scan_csv(output_dir / "partial_scan.csv", records)
        write_json(
            output_dir / "failure.json",
            {
                "schema": "alchemi.b97-3c-water-dimer-failure.v1",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "completed_distance_count": len(records),
                "required_distance_count": len(points),
                "status": "incomplete-not-a-reference",
            },
        )
        raise

    if len(records) != len(DEFAULT_OO_DISTANCES_ANGSTROM):
        raise AssertionError("completed scan does not contain all declared distances")
    if hasattr(psi4.core, "flush_outfile"):
        psi4.core.flush_outfile()

    csv_path = output_dir / "interaction_curve.csv"
    write_scan_csv(csv_path, records)
    csv_hash = sha256_file(csv_path)
    manifest = {
        "schema": "alchemi.b97-3c-water-dimer-scan.v1",
        "artifact_id": f"b97-3c-water-dimer-{csv_hash[:16]}",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": {"name": "Psi4", "version": str(psi4.__version__)},
        "method": method_contract(),
        "scan": {
            "builder": "aux.structures.make_water_dimer_scan",
            "distances_angstrom": list(DEFAULT_OO_DISTANCES_ANGSTROM),
            "distance_count": len(records),
            "single_point_count": 3 * len(records),
            "all_declared_points_present": True,
            "records": [energy_record_dict(record) for record in records],
        },
        "units": {
            "geometry": "angstrom",
            "energy": "hartree",
            "interaction_energy_presentations": ["kcal/mol", "kJ/mol"],
            "hartree_to_kcal_per_mol": HARTREE_TO_KCAL_MOL,
            "hartree_to_kJ_per_mol": HARTREE_TO_KJ_MOL,
        },
        "environment": dependency_metadata(psi4),
        "sources": _source_provenance(),
        "license_and_data_boundary": license_and_data_boundary(),
        "files": file_inventory(output_dir, excluded={"manifest.json"}),
        "integrity": {
            "geometry_hash_definition": (
                "SHA-256 of canonical compact JSON containing ordered symbols "
                "and round-trip-safe decimal float64 coordinates"
            ),
            "checksum_index": "SHA256SUMS",
            "checksum_index_covers_manifest": True,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_sha256sums(output_dir)

    print("\nO-O / A        E_int / kcal mol^-1", flush=True)
    for record in records:
        print(
            f"{record.point.requested_oo_distance_angstrom:8.4f}  "
            f"{record.interaction_Eh * HARTREE_TO_KCAL_MOL:20.8f}",
            flush=True,
        )
    print(f"artifacts: {output_dir}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the fixed eight-point, 24-single-point canonical B97-3c "
            "water-dimer interaction reference."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "new or empty output directory; default: "
            "reference/artifacts/water_dimer_b97_3c; existing artifacts are "
            "never overwritten"
        ),
    )
    parser.add_argument("--threads", type=int, help="Psi4 CPU thread count")
    parser.add_argument(
        "--memory", help='Psi4 memory string, for example "16 GB"'
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.threads is not None and args.threads <= 0:
        raise ValueError("--threads must be positive")
    return run_scan(args)


if __name__ == "__main__":
    raise SystemExit(main())
