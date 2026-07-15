"""Helper functions for the SEI Pareto challenge solution notebook.

The notebook intentionally mirrors the Part 1 adsorption tutorial:

1. Build or load relaxed clean hosts.
2. Enumerate a small site x orientation x rotation x height grid.
3. Relax every start in Toolkit batches.
4. Select the lowest-energy valid relaxed start for each pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from ase.data import covalent_radii, vdw_radii
from ase.geometry import find_mic
from ase.io import write as ase_write


@dataclass(frozen=True)
class SolutionSettings:
    min_adsorption_clearance_A: float = 1.6
    vdw_height_scale: float = 0.66
    surface_height_tolerance_A: float = 1.2
    gas_box_A: float = 20.0
    adsorption_site_limit: int = 3
    max_surface_displacement_A: float = 1.5
    frozen_surface_fraction: float = 0.5
    li_bcc_a: float = 3.51
    lif_rocksalt_a: float = 4.02


ADSORPTION_ANCHOR_ELEMENTS = {"O", "N", "F", "P", "S"}

LI2CO3_COD_9008283_CIF = """
data_9008283
_chemical_formula_sum 'C Li2 O3'
_chemical_name_mineral Zabuyelite
_space_group_IT_number 15
_symmetry_space_group_name_Hall '-C 2yc'
_symmetry_space_group_name_H-M 'C 1 2/c 1'
_cell_angle_alpha 90
_cell_angle_beta 114.83
_cell_angle_gamma 90
_cell_length_a 8.3593
_cell_length_b 4.9725
_cell_length_c 6.1975
_cell_formula_units_Z 4
loop_
_symmetry_equiv_pos_as_xyz
x,y,z
1/2+x,1/2+y,z
x,-y,1/2+z
1/2+x,1/2-y,1/2+z
-x,y,1/2-z
1/2-x,1/2+y,1/2-z
-x,-y,-z
1/2-x,1/2-y,-z
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li Li 0.19650 0.44840 0.83440
C C 0.00000 0.06570 0.25000
O1 O 0.00000 0.32130 0.25000
O2 O 0.14590 -0.06350 0.31270
"""


def gas_box(atoms, *, settings: SolutionSettings):
    gas = atoms.copy()
    gas.set_cell([settings.gas_box_A] * 3)
    gas.set_pbc([True, True, True])
    gas.center()
    return gas


def prepare_challenge_tables(
    molecules_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    example_systems: Iterable[tuple[str, str]] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    challenge_df = molecules_df.merge(
        lookup_df[["molecule_class", "passivating_surface_id"]],
        on="molecule_class",
        how="left",
        validate="many_to_one",
    )
    if challenge_df["passivating_surface_id"].isna().any():
        missing = challenge_df.loc[
            challenge_df["passivating_surface_id"].isna(),
            ["candidate_id", "molecule_class"],
        ]
        raise RuntimeError(f"Missing passivating-surface lookup rows:\n{missing}")

    run_system_rows: list[dict] = []
    examples = tuple(example_systems)
    if examples:
        for candidate_id, interaction in examples:
            if interaction not in {"li_metal", "passivating"}:
                raise ValueError(f"Unknown interaction in EXAMPLE_SYSTEMS: {interaction!r}")
            matches = challenge_df[challenge_df["candidate_id"].eq(candidate_id)]
            if matches.empty:
                raise RuntimeError(f"EXAMPLE_SYSTEMS requested unknown candidate_id: {candidate_id}")
            source = matches.iloc[0].to_dict()
            source["interaction"] = interaction
            source["surface_id"] = (
                "Li_metal" if interaction == "li_metal" else source["passivating_surface_id"]
            )
            run_system_rows.append(source)
    else:
        for row in challenge_df.itertuples(index=False):
            for interaction, surface_id in (
                ("li_metal", "Li_metal"),
                ("passivating", row.passivating_surface_id),
            ):
                source = row._asdict()
                source["interaction"] = interaction
                source["surface_id"] = surface_id
                run_system_rows.append(source)

    run_systems_df = pd.DataFrame(run_system_rows)
    run_challenge_df = challenge_df[
        challenge_df["candidate_id"].isin(run_systems_df["candidate_id"].unique())
    ].copy()
    return challenge_df, run_systems_df, run_challenge_df


def build_li_metal_bulk(settings: SolutionSettings):
    from pymatgen.core import Lattice, Structure

    return Structure.from_spacegroup(
        "Im-3m",
        Lattice.cubic(settings.li_bcc_a),
        ["Li"],
        [[0.0, 0.0, 0.0]],
    )


def build_lif_bulk(settings: SolutionSettings):
    from pymatgen.core import Lattice, Structure

    return Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(settings.lif_rocksalt_a),
        ["Li", "F"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def build_li2co3_bulk(settings: SolutionSettings):
    """Build ambient monoclinic Li2CO3 from the COD 9008283 CIF.

    The CIF is the zabuyelite C2/c refinement from COD 9008283:
    Effenberger and Zemann, Zeitschrift fur Kristallographie 150, 133-138 (1979).
    """
    del settings
    from pymatgen.core import Structure

    return Structure.from_str(LI2CO3_COD_9008283_CIF, fmt="cif")


def tag_adsorption_slab(atoms, *, surface_id: str, provenance: str):
    slab = atoms.copy()
    slab.set_pbc([True, True, True])
    slab.info["surface_id"] = surface_id
    slab.info["surface_provenance"] = provenance
    return slab


def build_li_metal_slab(settings: SolutionSettings):
    from helpers import build_slab

    slab = build_slab(
        build_li_metal_bulk(settings),
        miller_index=(1, 0, 0),
        min_slab_size=8.0,
        min_vacuum_size=20.0,
        supercell=(5, 3, 1),
    )
    return tag_adsorption_slab(slab, surface_id="Li_metal", provenance="bcc-Li(100)-physical")


def build_lif_slab(settings: SolutionSettings):
    from helpers import build_slab

    slab = build_slab(
        build_lif_bulk(settings),
        miller_index=(1, 0, 0),
        min_slab_size=8.0,
        min_vacuum_size=20.0,
        supercell=(6, 3, 1),
    )
    return tag_adsorption_slab(slab, surface_id="LiF", provenance="rocksalt-LiF(100)-physical")


def build_li2co3_slab(settings: SolutionSettings):
    from helpers import build_slab

    slab = build_slab(
        build_li2co3_bulk(settings),
        miller_index=(0, 0, 1),
        min_slab_size=8.0,
        min_vacuum_size=20.0,
        supercell=(3, 2, 1),
    )
    return tag_adsorption_slab(
        slab,
        surface_id="Li2CO3",
        provenance="zabuyelite-Li2CO3(001)-COD9008283",
    )


def prepare_adsorption_surface(surface_id: str, *, settings: SolutionSettings):
    builders = {
        "Li_metal": build_li_metal_slab,
        "LiF": build_lif_slab,
        "Li2CO3": build_li2co3_slab,
    }
    if surface_id not in builders:
        raise RuntimeError(
            f"No physical slab builder is defined for {surface_id}. "
            "Keep EXAMPLE_SYSTEMS on supported surfaces, or add a bulk-derived builder."
        )
    return builders[surface_id](settings)


def surface_active_mask(surface, *, settings: SolutionSettings):
    from helpers import make_active_mask

    return make_active_mask(surface, bottom_fraction=settings.frozen_surface_fraction)


def build_adsorption_surfaces(surface_ids: Iterable[str], *, settings: SolutionSettings):
    return {
        surface_id: prepare_adsorption_surface(surface_id, settings=settings)
        for surface_id in sorted(surface_ids)
    }


def surface_summary(surfaces: dict[str, object], *, settings: SolutionSettings) -> pd.DataFrame:
    rows = []
    for surface_id, atoms in surfaces.items():
        cell = atoms.cell.lengths()
        active = np.asarray(surface_active_mask(atoms, settings=settings), dtype=bool)
        rows.append(
            {
                "surface_id": surface_id,
                "cell_A": f"{cell[0]:.1f} x {cell[1]:.1f} x {cell[2]:.1f}",
                "atoms": len(atoms),
                "mobile_surface_atoms": int(active.sum()),
                "provenance": atoms.info.get("surface_provenance", ""),
            }
        )
    return pd.DataFrame(rows)


def surface_center_xy(surface) -> np.ndarray:
    cell = np.asarray(surface.cell.array, dtype=float)
    return (0.5 * (cell[0] + cell[1]))[:2]


def lateral_distance_A(surface, xy_a, xy_b) -> float:
    lengths = np.asarray(surface.cell.lengths()[:2], dtype=float)
    delta = np.abs(np.asarray(xy_a, dtype=float) - np.asarray(xy_b, dtype=float))
    periodic_delta = np.minimum(delta, np.maximum(lengths - delta, 0.0))
    return float(np.linalg.norm(periodic_delta))


def adsorption_site_candidates(surface, *, settings: SolutionSettings, top_layer_tolerance_A=0.6):
    center = surface_center_xy(surface)
    top_z = float(np.max(surface.positions[:, 2]))
    top_indices = [
        index for index, atom in enumerate(surface)
        if float(atom.position[2]) >= top_z - top_layer_tolerance_A
    ]
    top_indices.sort(key=lambda index: lateral_distance_A(surface, surface.positions[index, :2], center))

    pool = [{"site_label": "center", "xy": np.asarray(center, dtype=float), "z": top_z}]
    for index in top_indices[:12]:
        atom = surface[index]
        pool.append({
            "site_label": f"top_{atom.symbol}{index}",
            "xy": np.asarray(atom.position[:2], dtype=float),
            "z": float(atom.position[2]),
        })

    bridge_pool = []
    for left, i in enumerate(top_indices[:8]):
        for j in top_indices[left + 1:8]:
            distance = lateral_distance_A(surface, surface.positions[i, :2], surface.positions[j, :2])
            if distance <= 4.5:
                xy = 0.5 * (surface.positions[i, :2] + surface.positions[j, :2])
                bridge_pool.append((lateral_distance_A(surface, xy, center), i, j, xy))
    for _, i, j, xy in sorted(bridge_pool)[:8]:
        pool.append({"site_label": f"bridge_{i}_{j}", "xy": np.asarray(xy, dtype=float), "z": top_z})

    selected = []
    for site in pool:
        separated = all(lateral_distance_A(surface, site["xy"], prev["xy"]) >= 0.75 for prev in selected)
        if separated:
            site["site_id"] = f"s{len(selected)}"
            site["position"] = np.array([site["xy"][0], site["xy"][1], site["z"]], dtype=float)
            selected.append(site)
        if len(selected) >= settings.adsorption_site_limit:
            break
    return selected


def molecule_anchor_indices(molecule, *, anchor_element: str | None = None) -> list[int]:
    indices = [
        index for index, atom in enumerate(molecule)
        if atom.symbol in ADSORPTION_ANCHOR_ELEMENTS
        and (anchor_element is None or atom.symbol == anchor_element)
    ]
    if indices:
        return indices
    if anchor_element is not None:
        raise ValueError(f"Requested {anchor_element}-down orientation, but molecule has no {anchor_element} atom.")
    return list(range(len(molecule)))


def _maximin_anchor_direction(atoms, anchor_idx: int) -> tuple[np.ndarray, float]:
    rel = atoms.positions - atoms.positions[anchor_idx]
    rel = np.delete(rel, anchor_idx, axis=0)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    best_score = -np.inf
    best = np.array([0.0, 0.0, 1.0])
    for i in range(512):
        z = 1.0 - 2.0 * (i + 0.5) / 512
        radius = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * i
        direction = np.array([np.cos(theta) * radius, np.sin(theta) * radius, z])
        score = float(np.min(rel @ direction))
        if score > best_score:
            best_score = score
            best = direction
    if best_score <= 1e-6:
        raise ValueError("Requested anchor atom is not exposed for this molecule geometry.")
    return best, best_score


def _orient_anchor_lowest(atoms, anchor_idx: int):
    oriented = atoms.copy()
    oriented.translate(-oriented.positions[anchor_idx])
    direction, _ = _maximin_anchor_direction(oriented, anchor_idx)
    oriented.rotate(direction, (0.0, 0.0, 1.0), center=(0.0, 0.0, 0.0))
    oriented.positions[anchor_idx] = [0.0, 0.0, 0.0]
    return oriented


def orientation_anchor_element(orientation: str) -> str | None:
    if orientation == "anchor-down":
        return None
    if orientation.endswith("-down"):
        return orientation.split("-", 1)[0]
    raise ValueError(f"Unsupported orientation label: {orientation!r}")


def build_oriented_adsorbate(molecule, orientation: str):
    anchor_element = orientation_anchor_element(orientation)
    scored = []
    for index in molecule_anchor_indices(molecule, anchor_element=anchor_element):
        trial = molecule.copy()
        trial.translate(-trial.positions[index])
        try:
            _, score = _maximin_anchor_direction(trial, index)
        except ValueError:
            continue
        scored.append((score, index))
    if not scored:
        raise ValueError(f"Could not expose a valid {orientation} anchor for this molecule.")
    _, anchor_idx = max(scored)
    return _orient_anchor_lowest(molecule, anchor_idx)


def adsorption_orientation_names(candidate_id, interaction, surface_id, molecule) -> list[str]:
    del candidate_id, interaction, surface_id
    symbols = set(molecule.get_chemical_symbols())
    orientations = []
    if "F" in symbols:
        orientations.append("F-down")
    if "O" in symbols:
        orientations.append("O-down")
    if "N" in symbols and not orientations:
        orientations.append("N-down")
    if "S" in symbols and not orientations:
        orientations.append("S-down")
    if not orientations:
        orientations.append("anchor-down")
    return orientations[:2]


def atomic_vdw_radius_A(atomic_number: int) -> float:
    radius = float(vdw_radii[int(atomic_number)])
    if np.isfinite(radius) and radius > 0:
        return radius
    return float(covalent_radii[int(atomic_number)] + 0.8)


def anchor_index_for_oriented_adsorbate(adsorbate_atoms) -> int:
    distances = np.linalg.norm(np.asarray(adsorbate_atoms.positions, dtype=float), axis=1)
    return int(np.argmin(distances))


def adsorption_height_for_start(surface, site, adsorbate_atoms, *, settings: SolutionSettings) -> float:
    anchor = adsorbate_atoms[anchor_index_for_oriented_adsorbate(adsorbate_atoms)]
    anchor_radius = atomic_vdw_radius_A(anchor.number)
    top_z = float(np.max(surface.positions[:, 2]))
    site_z = float(site["position"][2])
    required_height = float(settings.min_adsorption_clearance_A)

    for atom in surface:
        if float(atom.position[2]) < top_z - settings.surface_height_tolerance_A:
            continue
        lateral = lateral_distance_A(surface, site["position"][:2], atom.position[:2])
        contact_distance = settings.vdw_height_scale * (anchor_radius + atomic_vdw_radius_A(atom.number))
        if lateral >= contact_distance:
            continue
        vertical_clearance = np.sqrt(max(contact_distance**2 - lateral**2, 0.0))
        required_height = max(required_height, float(atom.position[2]) + vertical_clearance - site_z)

    return required_height


def surface_normal_from_cell(slab) -> np.ndarray:
    normal = np.cross(slab.cell[0], slab.cell[1])
    return normal / np.linalg.norm(normal)


def place_adsorbate_for_start(slab, adsorbate_atoms, site_position, *, height_A: float):
    ads = adsorbate_atoms.copy()
    ads.translate(np.asarray(site_position, dtype=float) + height_A * surface_normal_from_cell(slab))
    combined = slab.copy() + ads
    combined.set_cell(slab.cell)
    combined.set_pbc(slab.pbc)
    return combined


def build_sei_config_grid(candidate_id, interaction, surface_id, surface, molecule, *, settings: SolutionSettings):
    from helpers.config_search import Configuration

    base_mask = surface_active_mask(surface, settings=settings)
    configs = []
    for site in adsorption_site_candidates(surface, settings=settings):
        for orientation in adsorption_orientation_names(candidate_id, interaction, surface_id, molecule):
            adsorbate_atoms = build_oriented_adsorbate(molecule, orientation)
            height_A = adsorption_height_for_start(
                surface,
                site,
                adsorbate_atoms,
                settings=settings,
            )
            atoms = place_adsorbate_for_start(
                surface,
                adsorbate_atoms,
                site["position"],
                height_A=height_A,
            )
            label = (
                f"{candidate_id}_{interaction}_{surface_id}_"
                f"{site['site_label']}_{orientation}_h{height_A:.1f}"
            )
            configs.append(Configuration(
                label=label,
                host=surface_id,
                adsorbate=candidate_id,
                site=site["site_label"],
                orientation=orientation,
                rot_deg=0.0,
                height=float(height_A),
                atoms=atoms,
                active_mask=base_mask + [True] * len(adsorbate_atoms),
            ))
    return configs


def make_gas_jobs(run_challenge_df: pd.DataFrame, molecule_atoms: dict, *, settings: SolutionSettings):
    jobs = []
    for row in run_challenge_df.itertuples(index=False):
        atoms = gas_box(molecule_atoms[row.candidate_id], settings=settings)
        atoms.info["charge"] = int(row.charge)
        atoms.info["mult"] = int(row.multiplicity)
        jobs.append({
            "job_id": f"gas_{row.candidate_id}",
            "candidate_id": row.candidate_id,
            "interaction": "gas",
            "surface_id": "",
            "atoms": atoms,
            "reference_molecule": molecule_atoms[row.candidate_id],
        })
    return jobs


def make_clean_surface_jobs(surface_ids, adsorption_surfaces, surface_meta, *, settings: SolutionSettings):
    jobs = []
    for surface_id in sorted(surface_ids):
        atoms = adsorption_surfaces[surface_id].copy()
        atoms.info["charge"] = int(surface_meta.loc[surface_id, "charge"])
        atoms.info["mult"] = int(surface_meta.loc[surface_id, "multiplicity"])
        jobs.append({
            "job_id": f"clean_{surface_id}",
            "surface_id": surface_id,
            "interaction": "clean_surface",
            "atoms": atoms,
            "active_mask": surface_active_mask(atoms, settings=settings),
        })
    return jobs


def make_combined_jobs(run_systems_df, molecule_atoms, surface_meta, clean_surface_results, *, settings: SolutionSettings):
    relaxed_surface_atoms = {
        row["surface_id"]: row["relaxed_atoms"].copy()
        for row in clean_surface_results
    }
    jobs = []
    for row in run_systems_df.itertuples(index=False):
        surface = relaxed_surface_atoms[row.surface_id].copy()
        molecule = molecule_atoms[row.candidate_id].copy()
        configs = build_sei_config_grid(
            row.candidate_id,
            row.interaction,
            row.surface_id,
            surface,
            molecule,
            settings=settings,
        )
        for config in configs:
            combined = config.atoms.copy()
            combined.info["charge"] = int(row.charge) + int(surface_meta.loc[row.surface_id, "charge"])
            combined.info["mult"] = 1
            jobs.append({
                "job_id": f"combined_{config.label}",
                "candidate_id": row.candidate_id,
                "interaction": row.interaction,
                "surface_id": row.surface_id,
                "site_label": config.site,
                "start_orientation": config.orientation,
                "initial_adsorption_height_A": config.height,
                "atoms": combined,
                "active_mask": config.active_mask,
                "reference_molecule": molecule,
                "reference_surface": surface.copy(),
                "allow_molecule_geometry_failure": True,
            })
    return jobs


def max_force_from_result(result) -> float:
    forces = np.asarray(result.forces, dtype=float).reshape(-1, 3)
    return 0.0 if len(forces) == 0 else float(np.linalg.norm(forces, axis=1).max())


def verify_frozen_atoms_unchanged(job, relaxed_atoms, *, tolerance_A=5e-4) -> float:
    active_mask = job.get("active_mask")
    if active_mask is None:
        return 0.0
    frozen = ~np.asarray(active_mask, dtype=bool)
    if not frozen.any():
        return 0.0
    displacement = np.linalg.norm(
        np.asarray(relaxed_atoms.positions)[frozen] - np.asarray(job["atoms"].positions)[frozen],
        axis=1,
    )
    max_displacement = float(displacement.max()) if len(displacement) else 0.0
    if max_displacement > tolerance_A:
        raise RuntimeError(f"{job['job_id']}: frozen atoms moved by {max_displacement:.4f} A.")
    return max_displacement


def reference_bond_pairs(atoms, *, scale=1.25):
    pairs = []
    h_indices = [index for index, atom in enumerate(atoms) if atom.symbol == "H"]
    heavy_indices = [index for index, atom in enumerate(atoms) if atom.symbol != "H"]
    for h_index in h_indices:
        candidates = []
        for heavy_index in heavy_indices:
            cutoff = scale * (covalent_radii[atoms[h_index].number] + covalent_radii[atoms[heavy_index].number])
            distance = float(np.linalg.norm(atoms.positions[h_index] - atoms.positions[heavy_index]))
            if distance <= cutoff:
                candidates.append((distance, h_index, heavy_index))
        if candidates:
            distance, i, j = min(candidates)
            pairs.append((min(i, j), max(i, j), distance))
    for i, j in zip(heavy_indices, heavy_indices[1:]):
        cutoff = scale * (covalent_radii[atoms[i].number] + covalent_radii[atoms[j].number])
        distance = float(np.linalg.norm(atoms.positions[i] - atoms.positions[j]))
        if distance <= cutoff:
            pairs.append((i, j, distance))
    return sorted(set(pairs))


def verify_molecule_integrity(job, relaxed_atoms, *, stretch_ratio=1.8, stretch_A=0.8, compress_ratio=0.6):
    reference = job.get("reference_molecule")
    if reference is None:
        return []
    molecule = relaxed_atoms[-len(reference):]
    flags = []
    for i, j, start_distance in reference_bond_pairs(reference):
        delta, _ = find_mic(
            molecule.positions[i] - molecule.positions[j],
            cell=relaxed_atoms.cell,
            pbc=relaxed_atoms.pbc,
        )
        final_distance = float(np.linalg.norm(delta))
        if final_distance > max(stretch_ratio * start_distance, start_distance + stretch_A):
            flags.append(f"{reference[i].symbol}{i}-{reference[j].symbol}{j} stretched {start_distance:.2f}->{final_distance:.2f} A")
        elif final_distance < compress_ratio * start_distance:
            flags.append(f"{reference[i].symbol}{i}-{reference[j].symbol}{j} compressed {start_distance:.2f}->{final_distance:.2f} A")
    if flags and not job.get("allow_molecule_geometry_failure", False):
        raise RuntimeError(f"{job['job_id']}: relaxed molecule geometry is not usable: " + "; ".join(flags[:4]))
    return flags


def verify_surface_integrity(job, relaxed_atoms, *, settings: SolutionSettings):
    reference = job.get("reference_surface")
    if reference is None:
        return [], 0.0
    n_surface = len(reference)
    relaxed_surface = relaxed_atoms[:n_surface]
    delta, _ = find_mic(
        relaxed_surface.positions - reference.positions,
        cell=relaxed_atoms.cell,
        pbc=relaxed_atoms.pbc,
    )
    displacements = np.linalg.norm(delta, axis=1)
    max_displacement = float(np.max(displacements)) if len(displacements) else 0.0
    moved = np.where(displacements > settings.max_surface_displacement_A)[0]
    if not len(moved):
        return [], max_displacement
    worst = moved[np.argsort(displacements[moved])[-4:]][::-1]
    flags = [f"{reference[index].symbol}{int(index)} moved {displacements[index]:.2f} A" for index in worst]
    return flags, max_displacement


def relax_structures(jobs, relaxation_engine, *, settings: SolutionSettings, batch_size: int, label_prefix: str):
    from helpers import ase_to_atomic_data, atomic_data_to_ase

    rows = []
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        payloads = [
            ase_to_atomic_data(job["atoms"], structure_id=job["job_id"], active_mask=job.get("active_mask"))
            for job in chunk
        ]
        reply = relaxation_engine.relax(payloads, label=f"{label_prefix}_{start // batch_size + 1:03d}")
        for job, result in zip(chunk, reply.atoms):
            relaxed_atoms = atomic_data_to_ase(result)
            surface_flags, surface_max_displacement_A = verify_surface_integrity(job, relaxed_atoms, settings=settings)
            rows.append({
                **{key: value for key, value in job.items() if key not in {"atoms", "active_mask", "reference_molecule", "reference_surface"}},
                "energy_eV": float(result.energy),
                "converged": bool(result.converged),
                "optimizer_nsteps": int(result.num_optimization_steps),
                "fmax_eV_A": max_force_from_result(result),
                "frozen_max_displacement_A": verify_frozen_atoms_unchanged(job, relaxed_atoms),
                "molecule_geometry_flags": verify_molecule_integrity(job, relaxed_atoms),
                "surface_geometry_flags": surface_flags,
                "surface_max_displacement_A": surface_max_displacement_A,
                "relaxed_atoms": relaxed_atoms,
            })
    return rows


def require_all_converged(results, *, label: str):
    bad_rows = [
        {
            "job_id": row.get("job_id", ""),
            "candidate_id": row.get("candidate_id", ""),
            "interaction": row.get("interaction", ""),
            "surface_id": row.get("surface_id", ""),
            "fmax_eV_A": row.get("fmax_eV_A", np.nan),
            "optimizer_nsteps": row.get("optimizer_nsteps", np.nan),
        }
        for row in results
        if not bool(row.get("converged", False))
    ]
    if bad_rows:
        bad_df = pd.DataFrame(bad_rows).sort_values("fmax_eV_A", ascending=False)
        raise RuntimeError(f"{label} has unconverged relaxation(s):\n{bad_df.to_string(index=False)}")


def select_lowest_energy_site_results(results):
    df = pd.DataFrame(results)
    if df.empty:
        raise RuntimeError("No adsorption site-search results were produced.")
    df["molecule_geometry_ok"] = df["molecule_geometry_flags"].apply(lambda flags: len(flags) == 0)
    df["surface_geometry_ok"] = df["surface_geometry_flags"].apply(lambda flags: len(flags) == 0)
    df["reliable_for_minimum"] = (
        df["converged"].astype(bool)
        & df["molecule_geometry_ok"].astype(bool)
        & df["surface_geometry_ok"].astype(bool)
    )

    selected_rows = []
    failure_rows = []
    for key, group in df.groupby(["candidate_id", "interaction"], sort=False):
        reliable = group[group["reliable_for_minimum"]]
        if reliable.empty:
            best_attempt = group.sort_values("energy_eV").iloc[0]
            failure_rows.append({
                "candidate_id": key[0],
                "interaction": key[1],
                "n_starts": int(len(group)),
                "n_converged": int(group["converged"].sum()),
                "best_attempt_job_id": best_attempt.get("job_id", ""),
                "best_attempt_flags": "; ".join(
                    best_attempt.get("molecule_geometry_flags", [])
                    + best_attempt.get("surface_geometry_flags", [])
                ),
            })
            continue
        selected_rows.append(reliable.loc[reliable["energy_eV"].idxmin()].to_dict())

    if failure_rows:
        failure_df = pd.DataFrame(failure_rows)
        raise RuntimeError(f"At least one pair has no valid adsorption start:\n{failure_df.to_string(index=False)}")
    return pd.DataFrame(selected_rows).sort_values(["candidate_id", "interaction"]).reset_index(drop=True).to_dict("records")


def selected_site_summary(selected_results) -> pd.DataFrame:
    columns = [
        "candidate_id", "interaction", "surface_id", "site_label", "start_orientation",
        "energy_eV", "fmax_eV_A", "surface_max_displacement_A",
    ]
    return pd.DataFrame(selected_results)[columns].copy()


def component_energy_table(run_systems_df, gas_results, clean_surface_results, combined_results) -> pd.DataFrame:
    gas_energy = {row["candidate_id"]: float(row["energy_eV"]) for row in gas_results}
    surface_energy = {row["surface_id"]: float(row["energy_eV"]) for row in clean_surface_results}
    combined_by_key = {(row["candidate_id"], row["interaction"]): row for row in combined_results}

    rows = []
    for row in run_systems_df.itertuples(index=False):
        selected = combined_by_key[(row.candidate_id, row.interaction)]
        if selected["surface_id"] != row.surface_id:
            raise RuntimeError(
                f"Unexpected surface for {row.candidate_id}/{row.interaction}: "
                f"{selected['surface_id']} != {row.surface_id}"
            )
        rows.append({
            "candidate_id": row.candidate_id,
            "interaction": row.interaction,
            "surface_id": row.surface_id,
            "E_surface_species_eV": float(selected["energy_eV"]),
            "E_surface_eV": surface_energy[row.surface_id],
            "E_species_eV": gas_energy[row.candidate_id],
            "selected_site_label": selected.get("site_label", ""),
            "selected_start_orientation": selected.get("start_orientation", ""),
        })
    return pd.DataFrame(rows)


def binding_energy_table(run_challenge_df, raw_component_energies_df) -> pd.DataFrame:
    raw_by_key = raw_component_energies_df.set_index(["candidate_id", "interaction"])
    rows = []
    for row in run_challenge_df.itertuples(index=False):
        li = raw_by_key.loc[(row.candidate_id, "li_metal")]
        pas = raw_by_key.loc[(row.candidate_id, "passivating")]
        rows.append({
            "candidate_id": row.candidate_id,
            "role": row.role,
            "molecule_class": row.molecule_class,
            "passivating_surface_id": row.passivating_surface_id,
            "E_bind_Li_eV": float(li["E_surface_species_eV"] - li["E_surface_eV"] - li["E_species_eV"]),
            "E_bind_passivating_eV": float(pas["E_surface_species_eV"] - pas["E_surface_eV"] - pas["E_species_eV"]),
        })
    return pd.DataFrame(rows)


def safe_structure_name(*parts) -> str:
    text = "_".join(str(part) for part in parts if str(part))
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def write_ovito_inspection_structures(results, *, output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        if result.get("interaction") not in {"li_metal", "passivating"}:
            continue
        atoms = result["relaxed_atoms"].copy()
        atoms.info["candidate_id"] = result["candidate_id"]
        atoms.info["interaction"] = result["interaction"]
        atoms.info["surface_id"] = result["surface_id"]
        atoms.info["energy_eV"] = float(result["energy_eV"])
        path = output_dir / f"{safe_structure_name(result['candidate_id'], result['interaction'], result['surface_id'])}.extxyz"
        ase_write(path, atoms, format="extxyz")
        rows.append({
            "candidate_id": result["candidate_id"],
            "interaction": result["interaction"],
            "surface_id": result["surface_id"],
            "energy_eV": float(result["energy_eV"]),
            "converged": bool(result["converged"]),
            "structure_path": path.as_posix(),
        })
    if not rows:
        raise RuntimeError("No relaxed combined structures were found.")
    return pd.DataFrame(rows).sort_values(["candidate_id", "interaction"]).reset_index(drop=True)


def choose_inspection_candidates(binding_table: pd.DataFrame, *, max_candidates=6) -> list[str]:
    baseline_ids = binding_table.loc[binding_table["role"].eq("baseline"), "candidate_id"].tolist()
    strongest_li_ids = binding_table.sort_values("E_bind_Li_eV").head(2)["candidate_id"].tolist()
    weakest_passivating_ids = binding_table.sort_values("E_bind_passivating_eV", ascending=False).head(2)["candidate_id"].tolist()
    return list(dict.fromkeys([*baseline_ids, *strongest_li_ids, *weakest_passivating_ids]))[:max_candidates]


def inspection_widget_rows(inspection_df: pd.DataFrame, candidate_ids: Iterable[str]):
    widget_rows = []
    for candidate_id in candidate_ids:
        row_widgets = []
        for interaction in ("li_metal", "passivating"):
            matches = inspection_df[
                inspection_df["candidate_id"].eq(candidate_id)
                & inspection_df["interaction"].eq(interaction)
            ]
            if matches.empty:
                continue
            row = matches.iloc[0]
            row_widgets.append((f"{candidate_id} | {row.interaction} | {row.surface_id}", row.structure_path))
        if row_widgets:
            widget_rows.append(row_widgets)
    return widget_rows
