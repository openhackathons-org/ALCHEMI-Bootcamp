"""Persistence helpers for restartable, inspectable IR tutorial artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms

    from .capture import IRTrajectory


TRAJECTORY_ARRAY_KEYS = (
    "dipoles_e_angstrom",
    "charge_sums_e",
    "kinetic_energies_eV",
    "total_energies_eV",
    "positions_angstrom",
    "atomic_numbers",
    "atomic_masses_u",
    "batch_idx",
    "batch_ptr",
)

WATER_RUN_MANIFEST_NAME = "water_run_manifest.json"
WATER_RUN_MANIFEST_SCHEMA = "alchemi.water-ir-run.v2"

ORBMOL_RELAXATION_STRUCTURE_FILENAMES = {
    ("(H2O)2", "initial"): "water_orbmol_dimer_initial.extxyz",
    ("(H2O)2", "final"): "water_orbmol_dimer_final.extxyz",
    ("(H2O)6", "initial"): "water_orbmol_hexamer_initial.extxyz",
    ("(H2O)6", "final"): "water_orbmol_hexamer_final.extxyz",
}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    """Normalize supported scientific and path values for strict JSON."""

    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _run_output_inventory(
    output_dir: Path, *, excluded_path: Path
) -> list[dict[str, Any]]:
    """Inventory every regular run output in stable relative-path order."""

    files = sorted(
        (
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path != excluded_path
        ),
        key=lambda path: path.relative_to(output_dir).as_posix(),
    )
    return [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def write_water_run_manifest(
    output_dir: str | Path,
    *,
    run_details: Mapping[str, Any],
    settings: Mapping[str, Any],
    checks: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a deterministic manifest for a complete water-IR run.

    The caller supplies the run details, simulation settings, and scientific
    checks. This function records those values without inventing a timestamp
    or status, then inventories every file beneath the run-specific output
    directory. The manifest excludes only itself, so repeated calls with
    unchanged inputs and artifacts are byte-identical.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / WATER_RUN_MANIFEST_NAME
    manifest = _json_ready(
        {
            "schema": WATER_RUN_MANIFEST_SCHEMA,
            "run_details": run_details,
            "settings": settings,
            "checks": checks,
            "files": _run_output_inventory(output_dir, excluded_path=manifest_path),
        }
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _validate_trajectory_arrays(
    arrays: Mapping[str, np.ndarray | float], labels: Sequence[str]
) -> None:
    dipoles = np.asarray(arrays["dipoles_e_angstrom"])
    charge_sums = np.asarray(arrays["charge_sums_e"])
    kinetic = np.asarray(arrays["kinetic_energies_eV"])
    total = np.asarray(arrays["total_energies_eV"])
    positions = np.asarray(arrays["positions_angstrom"])
    numbers = np.asarray(arrays["atomic_numbers"])
    masses = np.asarray(arrays["atomic_masses_u"])
    batch_idx = np.asarray(arrays["batch_idx"])
    batch_ptr = np.asarray(arrays["batch_ptr"])
    dt_fs = float(arrays["dt_fs"])

    if dipoles.ndim != 3 or dipoles.shape[2] != 3:
        raise ValueError("dipoles_e_angstrom must have shape (frames, graphs, 3)")
    frames, graphs, _ = dipoles.shape
    if charge_sums.shape != (frames, graphs):
        raise ValueError("charge_sums_e shape does not match dipole trajectory")
    if kinetic.shape != (frames, graphs) or total.shape != (frames, graphs):
        raise ValueError("energy arrays must have shape (frames, graphs)")
    if positions.ndim != 3 or positions.shape[0] != frames or positions.shape[2] != 3:
        raise ValueError("positions_angstrom must have shape (frames, atoms, 3)")
    atoms = positions.shape[1]
    if numbers.shape != (atoms,) or masses.shape != (atoms,):
        raise ValueError("atomic metadata does not match trajectory atom count")
    if batch_idx.shape != (atoms,) or batch_ptr.shape != (graphs + 1,):
        raise ValueError("batch metadata does not match trajectory dimensions")
    if batch_ptr[0] != 0 or batch_ptr[-1] != atoms or np.any(np.diff(batch_ptr) <= 0):
        raise ValueError("batch_ptr must partition every atom into nonempty graphs")
    expected_idx = np.repeat(np.arange(graphs), np.diff(batch_ptr))
    if not np.array_equal(batch_idx, expected_idx):
        raise ValueError("batch_idx and batch_ptr disagree")
    if len(labels) != graphs:
        raise ValueError("labels must contain one entry per graph")
    if not np.isfinite(dt_fs) or dt_fs <= 0.0:
        raise ValueError("dt_fs must be positive and finite")
    for key in (
        "dipoles_e_angstrom",
        "charge_sums_e",
        "kinetic_energies_eV",
        "total_energies_eV",
        "positions_angstrom",
        "atomic_masses_u",
    ):
        if not np.isfinite(np.asarray(arrays[key])).all():
            raise ValueError(f"{key} contains non-finite values")


def save_ir_trajectory(
    path: str | Path,
    trajectory: "IRTrajectory",
    labels: Sequence[str],
) -> dict[str, Any]:
    """Save the complete raw trajectory before scientific comparison checks."""

    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("IR trajectory path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray | float] = {
        key: np.asarray(getattr(trajectory, key)) for key in TRAJECTORY_ARRAY_KEYS
    }
    arrays["dt_fs"] = float(trajectory.dt_fs)
    _validate_trajectory_arrays(arrays, labels)
    np.savez_compressed(path, **arrays, labels=np.asarray(labels))
    return {
        "file": path.name,
        "path": str(path),
        "sha256": sha256_file(path),
        "frames": int(np.asarray(trajectory.positions_angstrom).shape[0]),
        "graphs": len(labels),
        "atoms": int(np.asarray(trajectory.atomic_numbers).shape[0]),
        "dt_fs": float(trajectory.dt_fs),
    }


def load_ir_trajectory_arrays(
    path: str | Path,
) -> tuple[dict[str, np.ndarray | float], list[str]]:
    """Load and validate raw arrays without importing Torch or Toolkit."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {*TRAJECTORY_ARRAY_KEYS, "dt_fs", "labels"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"IR trajectory archive is missing: {', '.join(missing)}")
        arrays: dict[str, np.ndarray | float] = {
            key: np.asarray(archive[key]) for key in TRAJECTORY_ARRAY_KEYS
        }
        arrays["dt_fs"] = float(np.asarray(archive["dt_fs"]).reshape(()))
        labels = [str(label) for label in np.asarray(archive["labels"]).tolist()]
    _validate_trajectory_arrays(arrays, labels)
    return arrays, labels


def load_ir_trajectory(path: str | Path) -> tuple["IRTrajectory", list[str]]:
    """Load a validated archive into the CPU trajectory dataclass."""

    from .capture import IRTrajectory

    arrays, labels = load_ir_trajectory_arrays(path)
    return IRTrajectory(**arrays), labels


def graph_atoms_from_batch(
    batch: Any,
    graph_index: int,
    label: str,
    *,
    reference_atoms: Atoms | None = None,
    include_results: bool = False,
) -> Atoms:
    """Convert one graph of an existing Toolkit batch to ASE.

    ``reference_atoms`` preserves non-Toolkit metadata such as molecule IDs.
    Cell and periodic-boundary data always come from the evaluated batch when
    they are present. ``include_results`` adds energy, forces, charges, and
    total charge using names understood by :meth:`AtomicData.from_atoms`.
    """

    from ase import Atoms

    batch_ptr = _to_numpy(batch.batch_ptr).astype(int)
    if not 0 <= graph_index < len(batch_ptr) - 1:
        raise IndexError("graph_index is outside the batch")
    start, stop = batch_ptr[graph_index : graph_index + 2]
    numbers = _to_numpy(batch.atomic_numbers)[start:stop]
    positions = _to_numpy(batch.positions)[start:stop]
    masses = _to_numpy(batch.atomic_masses)[start:stop]

    if reference_atoms is None:
        atoms = Atoms(numbers=numbers, positions=positions, masses=masses)
    else:
        if len(reference_atoms) != stop - start:
            raise ValueError("reference_atoms does not match the selected graph")
        if not np.array_equal(reference_atoms.numbers, numbers):
            raise ValueError("reference_atoms has different atomic numbers")
        atoms = reference_atoms.copy()
        atoms.positions = positions
        atoms.set_masses(masses)

    cell = getattr(batch, "cell", None)
    pbc = getattr(batch, "pbc", None)
    if cell is not None and pbc is not None:
        cells = _to_numpy(cell)
        periodic = _to_numpy(pbc).astype(bool)
        atoms.set_cell(cells[graph_index])
        atoms.set_pbc(periodic[graph_index])
    elif reference_atoms is None:
        atoms.set_pbc(False)

    atoms.info["label"] = str(label)
    if include_results:
        energy = getattr(batch, "energy", None)
        forces = getattr(batch, "forces", None)
        charges = getattr(batch, "charges", None)
        total_charge = getattr(batch, "charge", None)
        if energy is not None:
            atoms.info["energy"] = float(_to_numpy(energy)[graph_index].reshape(-1)[0])
        if forces is not None:
            atoms.set_array("forces", _to_numpy(forces)[start:stop])
        if charges is not None:
            atoms.set_array("charges", _to_numpy(charges)[start:stop])
        if total_charge is not None:
            atoms.info["charge"] = int(
                np.rint(_to_numpy(total_charge)[graph_index].reshape(-1)[0])
            )
    return atoms


def write_orbmol_relaxation_structures(
    output_dir: str | Path,
    *,
    initial_batch: Any,
    final_batch: Any,
) -> dict[str, dict[str, Any]]:
    """Save initial and final OrbMol-v2 dimer/hexamer structures as extxyz.

    Both batches must contain the water dimer followed by the water hexamer.
    The helper checks that the graph layout and atom identities did not change
    before writing one inspectable file for each system and state. Energies and
    forces remain in the separate relaxation table because they are graph-level
    results, not atomic extxyz properties.
    """

    from ase.io import write

    initial_ptr = _to_numpy(initial_batch.batch_ptr).astype(int)
    final_ptr = _to_numpy(final_batch.batch_ptr).astype(int)
    expected_ptr = np.array([0, 6, 24])
    if not np.array_equal(initial_ptr, expected_ptr) or not np.array_equal(
        final_ptr, expected_ptr
    ):
        raise ValueError(
            "OrbMol-v2 structure artifacts require a dimer/hexamer batch "
            "with 6 and 18 atoms"
        )

    unchanged_fields = ("atomic_numbers", "atomic_masses")
    for field in unchanged_fields:
        initial = _to_numpy(getattr(initial_batch, field))
        final = _to_numpy(getattr(final_batch, field))
        if not np.array_equal(initial, final):
            raise ValueError(
                f"OrbMol-v2 relaxation changed the batch {field} field"
            )

    for name, batch in (("initial", initial_batch), ("final", final_batch)):
        positions = _to_numpy(batch.positions)
        if positions.shape != (24, 3) or not np.isfinite(positions).all():
            raise ValueError(
                f"OrbMol-v2 {name} positions must have finite shape (24, 3)"
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for graph_index, (system, short_name) in enumerate(
        (("(H2O)2", "dimer"), ("(H2O)6", "hexamer"))
    ):
        for state, batch in (("initial", initial_batch), ("final", final_batch)):
            atoms = graph_atoms_from_batch(
                batch,
                graph_index,
                f"OrbMol-v2 {system} {state}",
            )
            atoms.info.update(
                system=system,
                relaxation_state=state,
                model="OrbMol-v2",
            )
            filename = ORBMOL_RELAXATION_STRUCTURE_FILENAMES[(system, state)]
            path = output_dir / filename
            write(path, atoms, format="extxyz")
            records[f"{short_name}_{state}"] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "system": system,
                "state": state,
                "atoms": len(atoms),
            }
    return records


def trajectory_graph_frames(
    trajectory: "IRTrajectory",
    graph_index: int,
    frame_indices: Iterable[int],
    *,
    label: str | None = None,
) -> list[Atoms]:
    """Convert selected frames of one graph to viewer-ready ASE objects."""

    from ase import Atoms

    batch_ptr = np.asarray(trajectory.batch_ptr, dtype=int)
    if not 0 <= graph_index < len(batch_ptr) - 1:
        raise IndexError("graph_index is outside the trajectory")
    start, stop = batch_ptr[graph_index : graph_index + 2]
    frame_count = len(trajectory.positions_angstrom)
    frames: list[Atoms] = []
    for frame_index in frame_indices:
        frame_index = int(frame_index)
        if not 0 <= frame_index < frame_count:
            raise IndexError(f"frame index {frame_index} is outside the trajectory")
        atoms = Atoms(
            numbers=trajectory.atomic_numbers[start:stop],
            positions=trajectory.positions_angstrom[frame_index, start:stop],
            masses=trajectory.atomic_masses_u[start:stop],
            pbc=False,
        )
        atoms.info.update(
            step=frame_index,
            time_fs=float(frame_index * trajectory.dt_fs),
        )
        if label is not None:
            atoms.info["label"] = str(label)
        frames.append(atoms)
    return frames


def write_structure_artifacts(
    output_dir: str | Path,
    *,
    seed_batch: Any,
    relaxed_batch: Any,
    trajectory: "IRTrajectory",
    graph_index: int = 2,
    graph_label: str = "(H2O)6",
    stride: int = 100,
) -> dict[str, dict[str, str]]:
    """Write seed, relaxed, and strided trajectory structures as extxyz."""

    from ase.io import write

    if stride <= 0:
        raise ValueError("stride must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "seed": output_dir / "water_hexamer_seed.extxyz",
        "relaxed": output_dir / "water_hexamer_relaxed.extxyz",
        "trajectory": output_dir / f"water_hexamer_trajectory_stride{stride}.extxyz",
    }
    write(
        paths["seed"],
        graph_atoms_from_batch(seed_batch, graph_index, f"cyclic {graph_label} seed"),
    )
    write(
        paths["relaxed"],
        graph_atoms_from_batch(relaxed_batch, graph_index, f"relaxed {graph_label}"),
    )
    last = len(trajectory.positions_angstrom) - 1
    frame_indices = np.unique(np.append(np.arange(0, last + 1, stride), last))
    write(
        paths["trajectory"],
        trajectory_graph_frames(
            trajectory, graph_index, frame_indices, label=graph_label
        ),
    )
    return {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }
