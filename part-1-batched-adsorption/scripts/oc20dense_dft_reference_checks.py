#!/usr/bin/env python3
"""Compare Toolkit relaxations against OC20Dense DFT trajectory references.

This script assumes the official OC20Dense mapping/target files and the local
Toolkit outputs produced by ``run_oc20dense_known_examples.py`` are present.
It can:

* list or extract matching DFT trajectory members from the official archive;
* compare exact ``system_id``/``config_id``/``sid`` matches;
* report raw coordinate RMSD plus DFT adsorption-energy consistency checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import pickle
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read as ase_read
from ase.io import write as ase_write


PART1 = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PART1 / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from _oc20dense_common import (  # noqa: E402
    CLOSED_SHELL_ADSORBATE_REFERENCES,
    DEFAULT_DATA_ROOT,
    DEFAULT_EXTRACT_DIR,
    DEFAULT_CLOSED_SHELL_SYSTEMS,
    DEFAULT_SYSTEMS,
    DEFAULT_TRAJECTORY_ARCHIVE,
    MACE_EADS_REFERENCE_STATUS,
    MACE_RANK_BASIS,
    FULL_DATA_NOTICE,
    oc20dense_mapping_file,
    require_precomputed_write_allowed,
)


DEFAULT_TOOLKIT_ROOT = (
    PART1
    / "outputs"
    / "precomputed"
    / "accuracy"
    / "oc20dense_closed_shell_trajectory_mace_mpa0"
)
DEFAULT_ARCHIVE = DEFAULT_TRAJECTORY_ARCHIVE
DEFAULT_OUTDIR = DEFAULT_TOOLKIT_ROOT / "dft_reference_checks"


@dataclass(frozen=True)
class TrajectoryKey:
    system_id: str
    config_id: str
    sid: int

    @property
    def label(self) -> str:
        return f"{self.system_id}_{self.config_id}_sid{self.sid}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--toolkit-root", type=Path, default=DEFAULT_TOOLKIT_ROOT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DEFAULT_SYSTEMS),
        help="OC20Dense system_id values to compare.",
    )
    parser.add_argument(
        "--mode",
        choices=["list", "extract", "compare"],
        default="compare",
        help="list archive matches, extract them, or compare extracted trajectories.",
    )
    parser.add_argument(
        "--scope",
        choices=["summary", "all"],
        default="all",
        help=(
            "summary extracts DFT-best, SP-best, and relaxed-best configs only; "
            "all extracts every per-config row for the selected systems."
        ),
    )
    parser.add_argument(
        "--max-members",
        type=int,
        default=0,
        help="Stop after this many extracted/listed matching members; 0 means no cap.",
    )
    return parser.parse_args()


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _safe(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("(", "_")
        .replace(")", "")
        .replace(",", "_")
        .replace("*", "star")
    )


def _load_requested_keys(args: argparse.Namespace) -> list[TrajectoryKey]:
    per_config_path = args.toolkit_root / "tables" / "per_config_results.csv"
    summary_path = args.toolkit_root / "tables" / "system_summary.csv"
    if not per_config_path.exists():
        raise FileNotFoundError(f"Missing per-config table: {per_config_path}")

    per_config = pd.read_csv(per_config_path)
    requested_systems = {str(system_id) for system_id in args.systems}
    available_systems = set(per_config["system_id"].astype(str).unique())
    missing_systems = sorted(requested_systems - available_systems)
    if missing_systems:
        raise FileNotFoundError(
            "Toolkit per-config results do not contain the requested systems: "
            + ", ".join(missing_systems)
            + ". Rerun run_oc20dense_known_examples.py with the same --systems "
            "before extracting or comparing DFT trajectories."
        )

    per_config = per_config[per_config["system_id"].astype(str).isin(requested_systems)]
    if args.scope == "all":
        rows = per_config[["system_id", "config_id", "sid"]].drop_duplicates()
    else:
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary table: {summary_path}")
        summary = pd.read_csv(summary_path)
        summary = summary[summary["system_id"].astype(str).isin(args.systems)]
        wanted: set[tuple[str, str]] = set()
        for row in summary.itertuples(index=False):
            system_id = str(row.system_id)
            for attr in ("dft_best_config", "sp_best_config", "ml_best_config"):
                wanted.add((system_id, str(getattr(row, attr))))
        rows = per_config[
            [
                (str(row.system_id), str(row.config_id)) in wanted
                for row in per_config.itertuples(index=False)
            ]
        ][["system_id", "config_id", "sid"]].drop_duplicates()

    keys = [
        TrajectoryKey(str(row.system_id), str(row.config_id), int(row.sid))
        for row in rows.itertuples(index=False)
    ]
    return sorted(keys, key=lambda item: (item.system_id, item.config_id, item.sid))


def _member_matches(member_name: str, keys: list[TrajectoryKey]) -> TrajectoryKey | None:
    path = PurePosixPath(member_name)
    stem = path.name
    for suffix in (".traj.xz", ".extxyz.xz", ".traj", ".extxyz"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    parent_system = path.parent.name
    for key in keys:
        if parent_system == key.system_id and stem == f"{key.system_id}_{key.config_id}":
            return key
        # Fallback for any locally exported files that include sid explicitly.
        if (
            stem == key.label
            or stem == f"{key.system_id}_{key.config_id}_sid{key.sid}"
            or stem == f"sid{key.sid}"
        ):
            return key
    return None


def _iter_matching_members(
    archive: Path,
    keys: list[TrajectoryKey],
    *,
    max_members: int,
):
    count = 0
    with tarfile.open(archive, mode="r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            key = _member_matches(member.name, keys)
            if key is None:
                continue
            yield key, member
            count += 1
            if max_members and count >= max_members:
                return


def _extract_matches(args: argparse.Namespace, keys: list[TrajectoryKey]) -> list[dict[str, Any]]:
    args.extract_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    with tarfile.open(args.archive, mode="r:*") as tar:
        count = 0
        for member in tar:
            if not member.isfile():
                continue
            key = _member_matches(member.name, keys)
            if key is None:
                continue
            suffix = "".join(Path(member.name).suffixes) or ".traj"
            out_path = args.extract_dir / f"{_safe(key.label)}{suffix}"
            source = tar.extractfile(member)
            if source is None:
                continue
            out_path.write_bytes(source.read())
            manifest_rows.append(
                {
                    "system_id": key.system_id,
                    "config_id": key.config_id,
                    "sid": key.sid,
                    "archive_member": member.name,
                    "extracted_path": str(out_path),
                    "size": int(member.size),
                }
            )
            count += 1
            if args.max_members and count >= args.max_members:
                break
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.outdir / "extracted_trajectory_manifest.csv", index=False)
    return manifest_rows


def _manifest_from_preextracted(
    args: argparse.Namespace,
    keys: list[TrajectoryKey],
) -> list[dict[str, Any]]:
    if not args.extract_dir.exists():
        return []

    remaining = {(key.system_id, key.config_id, key.sid): key for key in keys}
    manifest_rows: list[dict[str, Any]] = []
    suffixes = ("*.traj", "*.traj.xz", "*.extxyz", "*.extxyz.xz")
    for pattern in suffixes:
        for path in sorted(args.extract_dir.glob(pattern)):
            key = _member_matches(str(path), list(remaining.values()))
            if key is None:
                continue
            remaining.pop((key.system_id, key.config_id, key.sid), None)
            manifest_rows.append(
                {
                    "system_id": key.system_id,
                    "config_id": key.config_id,
                    "sid": key.sid,
                    "archive_member": f"preextracted:{path.name}",
                    "extracted_path": str(path),
                    "size": int(path.stat().st_size),
                }
            )
    if manifest_rows:
        args.outdir.mkdir(parents=True, exist_ok=True)
        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(args.outdir / "extracted_trajectory_manifest.csv", index=False)
    return manifest_rows


def _read_atoms_sequence(path: Path) -> list[Atoms]:
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".xz"):
        payload = lzma.decompress(path.read_bytes())
        tmp = path.with_suffix("")
        tmp.write_bytes(payload)
        try:
            frames = ase_read(tmp, ":")
        finally:
            tmp.unlink(missing_ok=True)
    else:
        frames = ase_read(path, ":")
    if isinstance(frames, Atoms):
        return [frames]
    return list(frames)


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _mic_rmsd(
    a: np.ndarray,
    b: np.ndarray,
    *,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> float:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mic_delta, _lengths = find_mic(delta, cell=cell, pbc=pbc)
    return float(np.sqrt(np.mean(np.sum(mic_delta * mic_delta, axis=1))))


def _energy(atoms: Atoms) -> float | None:
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        value = atoms.info.get("energy")
    return None if value is None else float(value)


def _forces(atoms: Atoms) -> np.ndarray | None:
    try:
        return np.asarray(atoms.get_forces(), dtype=float).reshape(-1, 3)
    except Exception:
        if "forces" in atoms.arrays:
            return np.asarray(atoms.arrays["forces"], dtype=float).reshape(-1, 3)
        value = atoms.info.get("forces")
        if value is None:
            return None
        return np.asarray(value, dtype=float).reshape(-1, 3)


def _write_trajectory_artifacts(
    *,
    frames: list[Atoms],
    label: str,
    trajectory_path: Path,
    log_path: Path,
) -> None:
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out_frames: list[Atoms] = []
    log_rows = ["step,structure_id,energy_eV,max_force_eV_A"]
    for step, frame in enumerate(frames):
        atoms = frame.copy()
        energy = _energy(atoms)
        forces = _forces(atoms)
        max_force = (
            float(np.linalg.norm(forces, axis=1).max())
            if forces is not None and len(forces)
            else float("nan")
        )
        atoms.info["structure_id"] = label
        atoms.info["dft_trajectory_step"] = int(step)
        if energy is not None:
            atoms.info["energy_eV"] = float(energy)
        atoms.info["max_force_eV_A"] = max_force
        if forces is not None and len(forces) == len(atoms):
            atoms.arrays["forces"] = forces
        out_frames.append(atoms)
        log_rows.append(
            f"{step},{label},{'' if energy is None else float(energy)},{max_force}"
        )
    ase_write(trajectory_path, out_frames, format="extxyz")
    log_path.write_text("\n".join(log_rows) + "\n", encoding="utf-8")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(args: argparse.Namespace, result: pd.DataFrame) -> None:
    summary = pd.read_csv(args.toolkit_root / "tables" / "system_summary.csv")
    selected_rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        if str(row.system_id) not in set(args.systems):
            continue
        cases = [
            ("DFT best", str(row.dft_best_config)),
            ("Exact-xyz SP best", str(row.sp_best_config)),
            ("Toolkit-relaxed best", str(row.ml_best_config)),
        ]
        for case, config_id in cases:
            hit = result[
                (result["system_id"].astype(str) == str(row.system_id))
                & (result["config_id"].astype(str) == config_id)
            ].iloc[0]
            selected_rows.append(
                {
                    "system_id": str(row.system_id),
                    "case": case,
                    "config_id": config_id,
                    "sid": int(hit.sid),
                    "adsorbate": str(hit.adsorbate),
                    "adsorbate_reference_species": str(hit.adsorbate_reference_species),
                    "dft_rank": int(hit.dft_rank),
                    "dft_gap_to_best_eV": float(hit.dft_gap_to_system_best_eV),
                    "active_rmsd_A": float(hit.raw_active_atom_rmsd_A),
                    "adsorbate_rmsd_A": float(hit.raw_adsorbate_rmsd_A),
                    "mic_active_rmsd_A": float(hit.mic_active_atom_rmsd_A),
                    "mic_adsorbate_rmsd_A": float(hit.mic_adsorbate_rmsd_A),
                    "start_active_rmsd_A": float(hit.start_active_atom_rmsd_A),
                    "start_adsorbate_rmsd_A": float(hit.start_adsorbate_rmsd_A),
                    "dft_traj_minus_target_eV": float(hit.dft_traj_minus_target_eV),
                    "n_dft_frames": int(hit.n_dft_frames),
                    "dft_trajectory_path": str(hit.dft_trajectory_path),
                    "dft_trajectory_extxyz_path": str(hit.dft_trajectory_extxyz_path),
                    "dft_trajectory_log_path": str(hit.dft_trajectory_log_path),
                    "initial_structure_path": str(hit.initial_structure_path),
                    "ml_relaxed_path": str(hit.ml_relaxed_path),
                }
            )
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(args.outdir / "selected_case_comparison.csv", index=False)
    start_frame = result[
        [
            "system_id",
            "config_id",
            "sid",
            "n_dft_frames",
            "natoms",
            "start_species_match",
            "start_all_atom_rmsd_A",
            "start_active_atom_rmsd_A",
            "start_adsorbate_rmsd_A",
            "dft_trajectory_path",
            "dft_trajectory_extxyz_path",
            "dft_trajectory_log_path",
            "initial_structure_path",
        ]
    ].copy()
    start_frame.to_csv(args.outdir / "start_frame_parity.csv", index=False)

    stats = (
        result.groupby("system_id")
        .agg(
            n_configs=("config_id", "size"),
            active_rmsd_min_A=("raw_active_atom_rmsd_A", "min"),
            active_rmsd_median_A=("raw_active_atom_rmsd_A", "median"),
            active_rmsd_max_A=("raw_active_atom_rmsd_A", "max"),
            adsorbate_rmsd_min_A=("raw_adsorbate_rmsd_A", "min"),
            adsorbate_rmsd_median_A=("raw_adsorbate_rmsd_A", "median"),
            adsorbate_rmsd_max_A=("raw_adsorbate_rmsd_A", "max"),
            mic_active_rmsd_min_A=("mic_active_atom_rmsd_A", "min"),
            mic_active_rmsd_median_A=("mic_active_atom_rmsd_A", "median"),
            mic_active_rmsd_max_A=("mic_active_atom_rmsd_A", "max"),
            mic_adsorbate_rmsd_min_A=("mic_adsorbate_rmsd_A", "min"),
            mic_adsorbate_rmsd_median_A=("mic_adsorbate_rmsd_A", "median"),
            mic_adsorbate_rmsd_max_A=("mic_adsorbate_rmsd_A", "max"),
            max_abs_traj_target_eV=(
                "dft_traj_minus_target_eV",
                lambda values: float(np.abs(values).max()),
            ),
            start_active_rmsd_max_A=("start_active_atom_rmsd_A", "max"),
            start_adsorbate_rmsd_max_A=("start_adsorbate_rmsd_A", "max"),
        )
        .reset_index()
    )
    stats.to_csv(args.outdir / "system_rmsd_summary.csv", index=False)

    official_archive_md5 = "ee937e5290f8f720c914dc9a56e0281f"
    if args.archive.exists():
        archive_md5 = _md5(args.archive)
    else:
        archive_md5 = ""
    if archive_md5 == official_archive_md5:
        archive_note = (
            f"- Downloaded trajectory archive MD5: `{archive_md5}` "
            "(matches the published OC20Dense value)."
        )
    elif not archive_md5:
        archive_note = (
            "- Full trajectory archive was not present for this run; comparison "
            f"used pre-extracted exact-match DFT trajectories from `{args.extract_dir}`."
        )
    else:
        archive_note = (
            f"- Downloaded trajectory archive MD5: `{archive_md5}` "
            f"(published value: `{official_archive_md5}`)."
        )
    lines = [
        "# OC20Dense DFT Trajectory Reference Check",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "Backend: official OC20Dense DFT trajectory/target files",
        "Model: DFT reference arithmetic only; MACE totals are reported only as fixed-system ranking signals",
        "",
        "## Scope",
        "",
        (
            f"Compared {len(result)} exact OC20Dense trajectory matches across "
            f"{result['system_id'].nunique()} systems."
        ),
        "",
        (
            "The default systems are the same closed-shell set used by the "
            "single-point and relaxation script: `*OH2`/H2O, `*NH3`/NH3, and "
            "`*N2`/N2. CH3-containing systems are not included in this strict "
            "benchmark pass."
        ),
        "",
        (
            "A match means the DFT trajectory archive member has the exact "
            "`system_id` and `config_id` used by the Toolkit run; the `sid` is "
            "taken from the official OC20Dense mapping."
        ),
        "",
        "## Apples-to-Apples Frame Protocol",
        "",
        (
            "Starting structures are checked programmatically: the first frame "
            "of each extracted DFT trajectory is compared with the corresponding "
            "Toolkit input structure saved from the official OC20Dense LMDB for "
            "the same `system_id`, `config_id`, and `sid`."
        ),
        "",
        (
            "Final structures are compared programmatically: the last frame of "
            "that same DFT trajectory is compared with the Toolkit-relaxed final "
            "`extxyz` for the same record. Atom counts and species/order are "
            "preserved by the exact record match."
        ),
        "",
        (
            "RMSDs use the shared atom order from the exact record match. Raw "
            "coordinate RMSD is reported for traceability; minimum-image RMSD "
            "is also reported for final-frame geometry because adsorbates can "
            "cross a periodic boundary. No manual alignment, hand-picked atom "
            "mapping, or visual selection is used."
        ),
        "",
        (
            f"- Max starting-frame active-atom RMSD: "
            f"`{result['start_active_atom_rmsd_A'].max():.6g}` A."
        ),
        (
            f"- Max starting-frame adsorbate RMSD: "
            f"`{result['start_adsorbate_rmsd_A'].max():.6g}` A."
        ),
        (
            f"- Starting-frame species/order mismatches: "
            f"`{int((~result['start_species_match']).sum())}`."
        ),
        "",
        "## Energy Consistency",
        "",
        (
            "The DFT adsorption energy is the released OC20Dense target from "
            "`oc20dense_targets.pkl`. It is independently checked here by "
            "reading the last frame of each extracted DFT trajectory and "
            "computing `dft_final_total_energy - "
            "oc20dense_ref_energies[system_id]`."
        ),
        "",
        (
            "The ML energies are MACE-MPA-0 total energies. "
            "`ml_initial_sp_total_energy_eV` is a single-point model energy on "
            "the exact OC20Dense starting coordinates; `ml_relaxed_total_energy_eV` "
            "is the final model energy after Toolkit relaxation. These ML total "
            "energies are used only to rank configurations within the same "
            "OC20Dense system, where composition and reference are fixed. They "
            "are not reported as adsorption energies."
        ),
        "",
        (
            "MACE adsorption energies are not reported in this trajectory "
            "check. The closed-shell gas-reference species are pinned, but the "
            "MACE-scale clean-slab references still need to be generated with "
            "the same model settings before adsorption-energy parity can be "
            "reported. `oc20dense_ref_energies.pkl` is a DFT reference offset "
            "and must not be subtracted from MACE totals."
        ),
        "",
        (
            "`dft_gap_to_best_eV` is computed from released DFT adsorption "
            "energies as `this_config_dft_adsorption_energy - "
            "minimum_dft_adsorption_energy_for_the_same_system`."
        ),
        "",
        f"- Max absolute trajectory-target difference: `{np.abs(result['dft_traj_minus_target_eV']).max():.6g}` eV.",
        archive_note,
        "",
        "## Selected Cases",
        "",
        selected[
            [
                "system_id",
                "case",
                "config_id",
                "sid",
                "adsorbate",
                "adsorbate_reference_species",
                "dft_rank",
                "dft_gap_to_best_eV",
                "active_rmsd_A",
                "adsorbate_rmsd_A",
                "mic_active_rmsd_A",
                "mic_adsorbate_rmsd_A",
                "start_active_rmsd_A",
                "start_adsorbate_rmsd_A",
                "dft_traj_minus_target_eV",
            ]
        ].to_markdown(index=False),
        "",
        "## RMSD Distribution",
        "",
        stats.to_markdown(index=False),
        "",
        "## Output Files",
        "",
        f"- Full comparison: `{args.outdir / 'dft_reference_comparison.csv'}`",
        f"- Selected cases: `{args.outdir / 'selected_case_comparison.csv'}`",
        f"- Starting-frame parity: `{args.outdir / 'start_frame_parity.csv'}`",
        f"- System RMSD summary: `{args.outdir / 'system_rmsd_summary.csv'}`",
        f"- Extracted DFT trajectories: `{args.extract_dir}`",
    ]
    (args.outdir / "dft_reference_check_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _compare(args: argparse.Namespace, keys: list[TrajectoryKey]) -> pd.DataFrame:
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.outdir / "extracted_trajectory_manifest.csv"
    if not manifest_path.exists():
        rows = _manifest_from_preextracted(args, keys)
        if len(rows) < len(keys):
            if not args.archive.exists():
                missing = len(keys) - len(rows)
                raise FileNotFoundError(
                    f"Missing {missing} DFT trajectory files in {args.extract_dir}, "
                    f"and full archive is not available at {args.archive}. "
                    f"{FULL_DATA_NOTICE}"
                )
            _extract_matches(args, keys)
    manifest = pd.read_csv(manifest_path)
    per_config = pd.read_csv(args.toolkit_root / "tables" / "per_config_results.csv")
    targets: dict[str, list[tuple[str, float]]] = _read_pickle(
        oc20dense_mapping_file(args.data_root, "oc20dense_targets.pkl")
    )
    ref_energies: dict[str, float] = _read_pickle(
        oc20dense_mapping_file(args.data_root, "oc20dense_ref_energies.pkl")
    )

    dft_best_by_system = {
        system_id: min(float(energy) for _config_id, energy in rows)
        for system_id, rows in targets.items()
    }
    target_by_pair = {
        (system_id, str(config_id)): float(energy)
        for system_id, rows in targets.items()
        for config_id, energy in rows
    }

    rows: list[dict[str, Any]] = []
    for item in manifest.itertuples(index=False):
        system_id = str(item.system_id)
        config_id = str(item.config_id)
        sid = int(item.sid)
        label = f"{system_id}_{config_id}_sid{sid}"
        dft_frames = _read_atoms_sequence(Path(item.extracted_path))
        dft_extxyz_path = args.outdir / "dft_trajectories_extxyz" / f"{_safe(label)}.extxyz"
        dft_log_path = args.outdir / "dft_trajectory_logs" / f"{_safe(label)}.csv"
        _write_trajectory_artifacts(
            frames=dft_frames,
            label=label,
            trajectory_path=dft_extxyz_path,
            log_path=dft_log_path,
        )
        dft_start = dft_frames[0]
        dft_final = dft_frames[-1]
        initial_path = (
            args.toolkit_root / "structures" / "initial" / f"{_safe(label)}.extxyz"
        )
        ml_path = args.toolkit_root / "structures" / "relaxed" / f"{_safe(label)}.extxyz"
        if not initial_path.exists():
            continue
        if not ml_path.exists():
            continue
        initial_atoms = ase_read(initial_path)
        ml_final = ase_read(ml_path)

        start_pos = initial_atoms.get_positions()
        dft_start_pos = dft_start.get_positions()
        dft_pos = dft_final.get_positions()
        ml_pos = ml_final.get_positions()
        tags = np.asarray(initial_atoms.get_tags(), dtype=int)
        active = tags != 0
        adsorbate = tags == 2
        cell = np.asarray(dft_final.cell.array, dtype=float)
        pbc = np.asarray(dft_final.pbc, dtype=bool)

        dft_raw_e = _energy(dft_final)
        ref_e = float(ref_energies[system_id])
        dft_ads_from_traj = None if dft_raw_e is None else dft_raw_e - ref_e
        dft_target = target_by_pair[(system_id, config_id)]
        dft_best = dft_best_by_system[system_id]
        pc = per_config[
            (per_config["system_id"].astype(str) == system_id)
            & (per_config["config_id"].astype(str) == config_id)
            & (per_config["sid"].astype(int) == sid)
        ].iloc[0]

        rows.append(
            {
                "system_id": system_id,
                "config_id": config_id,
                "sid": sid,
                "n_dft_frames": len(dft_frames),
                "natoms": len(dft_final),
                "start_species_match": initial_atoms.get_chemical_symbols()
                == dft_start.get_chemical_symbols(),
                "end_species_match": ml_final.get_chemical_symbols()
                == dft_final.get_chemical_symbols(),
                "start_all_atom_rmsd_A": _rmsd(start_pos, dft_start_pos),
                "start_active_atom_rmsd_A": _rmsd(
                    start_pos[active], dft_start_pos[active]
                ),
                "start_adsorbate_rmsd_A": _rmsd(
                    start_pos[adsorbate], dft_start_pos[adsorbate]
                )
                if adsorbate.any()
                else float("nan"),
                "raw_all_atom_rmsd_A": _rmsd(ml_pos, dft_pos),
                "raw_active_atom_rmsd_A": _rmsd(ml_pos[active], dft_pos[active]),
                "raw_adsorbate_rmsd_A": _rmsd(ml_pos[adsorbate], dft_pos[adsorbate])
                if adsorbate.any()
                else float("nan"),
                "mic_all_atom_rmsd_A": _mic_rmsd(
                    ml_pos,
                    dft_pos,
                    cell=cell,
                    pbc=pbc,
                ),
                "mic_active_atom_rmsd_A": _mic_rmsd(
                    ml_pos[active],
                    dft_pos[active],
                    cell=cell,
                    pbc=pbc,
                ),
                "mic_adsorbate_rmsd_A": _mic_rmsd(
                    ml_pos[adsorbate],
                    dft_pos[adsorbate],
                    cell=cell,
                    pbc=pbc,
                )
                if adsorbate.any()
                else float("nan"),
                "dft_raw_final_energy_eV": dft_raw_e,
                "dft_reference_energy_eV": ref_e,
                "dft_adsorption_energy_from_traj_eV": dft_ads_from_traj,
                "dft_adsorption_energy_target_eV": dft_target,
                "dft_traj_minus_target_eV": None
                if dft_ads_from_traj is None
                else dft_ads_from_traj - dft_target,
                "dft_gap_to_system_best_eV": dft_target - dft_best,
                "ml_relaxed_total_energy_eV": float(pc.ml_total_energy_eV),
                "ml_initial_sp_total_energy_eV": float(pc.ml_initial_sp_total_energy_eV),
                "ml_relaxed_rank": int(pc.ml_relaxed_rank),
                "ml_initial_sp_rank": int(pc.ml_initial_sp_rank),
                "dft_rank": int(pc.dft_rank),
                "adsorbate": str(pc.adsorbate),
                "adsorbate_reference_species": str(
                    getattr(pc, "adsorbate_reference_species", "unpinned")
                ),
                "mace_rank_basis": MACE_RANK_BASIS,
                "mace_eads_reference_status": MACE_EADS_REFERENCE_STATUS,
                "dft_trajectory_path": str(item.extracted_path),
                "dft_trajectory_extxyz_path": str(dft_extxyz_path),
                "dft_trajectory_log_path": str(dft_log_path),
                "initial_structure_path": str(initial_path),
                "ml_relaxed_path": str(ml_path),
            }
        )

    result = pd.DataFrame(rows).sort_values(["system_id", "dft_rank", "config_id"])
    result.to_csv(args.outdir / "dft_reference_comparison.csv", index=False)
    _write_report(args, result)
    return result


def run_oc20dense_dft_reference_checks(args: argparse.Namespace) -> int:
    require_precomputed_write_allowed(args.outdir)
    keys = _load_requested_keys(args)
    if args.mode == "list":
        rows = []
        for key, member in _iter_matching_members(
            args.archive, keys, max_members=args.max_members
        ):
            rows.append(
                {
                    "system_id": key.system_id,
                    "config_id": key.config_id,
                    "sid": key.sid,
                    "archive_member": member.name,
                    "size": int(member.size),
                }
            )
            print(member.name, flush=True)
        args.outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.outdir / "matching_members.csv", index=False)
        print(f"Wrote {args.outdir / 'matching_members.csv'}")
        return 0

    if args.mode == "extract":
        rows = _extract_matches(args, keys)
        print(f"Extracted {len(rows)} matching trajectories")
        return 0

    result = _compare(args, keys)
    if len(result) <= 10:
        print(result.to_string(index=False))
    else:
        print(
            "Compared "
            f"{len(result)} exact DFT trajectories across "
            f"{result['system_id'].nunique()} systems; "
            f"max target delta = {result['dft_traj_minus_target_eV'].abs().max():.3e} eV; "
            f"max start active RMSD = {result['start_active_atom_rmsd_A'].max():.3e} A; "
            f"max final active RMSD = {result['mic_active_atom_rmsd_A'].max():.3e} A",
            flush=True,
        )
    print(f"Wrote {args.outdir / 'dft_reference_comparison.csv'}")
    return 0


def main() -> int:
    return run_oc20dense_dft_reference_checks(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
