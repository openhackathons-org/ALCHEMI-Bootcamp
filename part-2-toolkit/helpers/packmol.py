"""Packmol-based supercell construction for warmup.py.

Alternative to the default CIF-based crystal replication
(``unit_cell * args.supercell``). Packmol packs N copies of a single
molecule into a cubic periodic box at a target initial density,
producing a disordered / liquid-like starting configuration that the
downstream FIRE -> NVT -> NPT pipeline then equilibrates.

Crystal vs. packmol starting points are physically different:

- CIF supercell: ordered crystal at experimental geometry, monoclinic
  cell preserved.
- Packmol box: amorphous arrangement at user-specified density, cubic
  cell.

Use ``--packmol`` when you want a glassy / liquid initial condition;
for solid-phase warmup feeding the canonical Part-2 SLC pipeline, the
default CIF-supercell path is what you want.

The single-molecule template is auto-extracted from
``data/<material>.cif`` via covalent-radius connectivity (largest
connected component, PBC re-stitched via the minimum image convention
so molecules straddling the unit-cell boundary aren't torn in half).

Adapted from ``alchemi-toolkit-demo/example/utils.py::pack_liquid_box``.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.neighborlist import build_neighbor_list

N_AVOGADRO = 6.02214076e23


def extract_single_molecule(unit_cell: Atoms) -> Atoms:
    """Return one molecule from a unit-cell Atoms via covalent connectivity.

    Builds a neighbour list with default ASE covalent radii, walks connected
    components, and returns the largest as a fresh PBC-free Atoms. Bonds
    that wrap across the unit cell are stitched back together using the
    minimum-image convention so the returned molecule is contiguous and
    safe to write to PDB for packmol.
    """
    nl = build_neighbor_list(unit_cell, self_interaction=False, bothways=True)
    adjacency = nl.get_connectivity_matrix(sparse=False)

    n = len(unit_cell)
    visited = np.zeros(n, dtype=bool)
    components: list[list[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        comp: list[int] = []
        while stack:
            i = stack.pop()
            if visited[i]:
                continue
            visited[i] = True
            comp.append(i)
            stack.extend(int(j) for j in np.where(adjacency[i])[0] if not visited[j])
        components.append(comp)

    if not components:
        raise RuntimeError("No connected components found in unit cell")

    largest = max(components, key=len)
    positions = unit_cell.get_positions()
    cell = np.asarray(unit_cell.cell)
    inv_cell = np.linalg.inv(cell)

    # BFS from component root, unwrapping each bond via MIC so the molecule
    # is contiguous regardless of how the CIF placed it inside the cell.
    root = largest[0]
    placed = {root: positions[root].copy()}
    frontier = [root]
    while frontier:
        i = frontier.pop()
        for j in np.where(adjacency[i])[0]:
            j = int(j)
            if j in placed:
                continue
            delta = positions[j] - placed[i]
            frac = delta @ inv_cell
            frac -= np.round(frac)
            placed[j] = placed[i] + frac @ cell
            frontier.append(j)

    mol_indices = sorted(placed.keys())
    mol = unit_cell[mol_indices].copy()
    mol.set_positions(np.stack([placed[i] for i in mol_indices]))
    mol.set_pbc(False)
    mol.set_cell([0.0, 0.0, 0.0])
    mol.center()
    return mol


def pack_liquid_box(
    molecule: Atoms,
    n_molecules: int,
    target_density: float,
    model_cutoff: float,
    tolerance: float = 2.0,
    nloop: int = 50,
) -> tuple[Atoms, int]:
    """Pack N copies of ``molecule`` into a cubic periodic box.

    Box side is sized to hit ``target_density`` and then expanded -- with
    proportional increase in molecule count -- to satisfy the MIC
    constraint ``box_side >= 2 * model_cutoff``. Returns
    ``(box_atoms, n_molecules)``; the returned ``n_molecules`` may exceed
    the requested value if MIC forced an enlargement.

    ``nloop`` is forwarded to packmol's GENCAN iteration cap (packmol
    default = 50). At densities close to experimental crystal packing,
    packmol may exit non-zero with "ENDED WITHOUT PERFECT PACKING" yet
    still write a usable file -- that output is accepted with a warning;
    bumping ``nloop`` (e.g. 200) gives packmol more attempts to satisfy
    the ``tolerance`` constraint cleanly.
    """
    mw = float(sum(molecule.get_masses()))
    total_mass_g = n_molecules * mw / N_AVOGADRO
    volume_cm3 = total_mass_g / target_density
    volume_A3 = volume_cm3 * 1e24
    box_side = volume_A3 ** (1.0 / 3.0)

    min_box = 2.0 * model_cutoff
    if box_side < min_box:
        new_volume_A3 = min_box**3
        new_volume_cm3 = new_volume_A3 * 1e-24
        new_total_mass_g = target_density * new_volume_cm3
        n_molecules_new = max(round(new_total_mass_g * N_AVOGADRO / mw), n_molecules)
        print(
            f"  WARNING: box side {box_side:.2f} A < 2*cutoff ({min_box:.1f} A). "
            f"Enlarging to {min_box:.1f} A and increasing molecules "
            f"{n_molecules} -> {n_molecules_new} to preserve "
            f"density={target_density} g/cm^3."
        )
        box_side = min_box
        n_molecules = n_molecules_new

    # Lazy import: keeps helpers package importable in host-side envs that
    # don't ship the packmol Python wrapper.
    from packmol.cli import get_binary_path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        mol_pdb = tmp / "molecule.pdb"
        out_pdb = tmp / "packed.pdb"
        inp_file = tmp / "pack.inp"

        ase_write(str(mol_pdb), molecule, format="proteindatabank")

        margin = tolerance / 2.0
        inp_file.write_text(
            f"tolerance {tolerance}\n"
            f"filetype pdb\n"
            f"output {out_pdb}\n"
            f"nloop {nloop}\n"
            f"\n"
            f"structure {mol_pdb}\n"
            f"  number {n_molecules}\n"
            f"  inside box {margin} {margin} {margin} "
            f"{box_side - margin} {box_side - margin} {box_side - margin}\n"
            f"end structure\n"
        )

        with open(inp_file) as inp_stream:
            result = subprocess.run(
                [str(get_binary_path())],
                stdin=inp_stream,
                capture_output=True,
                text=True,
                timeout=600,
            )
        if not out_pdb.exists():
            raise RuntimeError(
                f"Packmol failed without producing output (exit "
                f"{result.returncode}):\n{result.stdout[-2000:]}\n"
                f"{result.stderr}"
            )
        if result.returncode != 0:
            # Packmol convention: non-zero exit with output file = "ENDED
            # WITHOUT PERFECT PACKING". The file is a best-effort solution
            # and is the recommended starting configuration; downstream
            # FIRE2 relaxation absorbs the remaining overlap.
            print(
                f"  WARNING: packmol exited {result.returncode} "
                f"(no perfect packing at rho={target_density} g/cm^3, "
                f"tol={tolerance} A, nloop={nloop}). Using best-effort "
                f"output; increase nloop or lower density for a cleaner pack."
            )

        box_atoms = ase_read(str(out_pdb), format="proteindatabank")

    box_atoms.set_cell([box_side, box_side, box_side])
    box_atoms.set_pbc(True)
    return box_atoms, n_molecules


def pack_with_fixed_obstacle(
    monomer: Atoms,
    n_molecules: int,
    fixed_atoms: Atoms,
    pbc_box: np.ndarray,
    plane_constraints: list[tuple[str, np.ndarray, float]],
    *,
    tolerance: float = 2.0,
    nloop: int = 20,
    seed: int | None = None,
) -> Atoms:
    """Pack ``n_molecules`` copies of ``monomer`` around an immovable
    ``fixed_atoms`` structure in a single Packmol invocation.

    Packmol's ``tolerance`` enforces minimum inter-atom separation against
    ALL atoms -- the fixed obstacle, the already-placed monomers, AND their
    PBC images under ``pbc Lx Ly Lz``. Place the fixed obstacle in the
    Packmol coordinate frame (anchored at the world origin); caller is
    responsible for any frame shifts.

    ``plane_constraints`` is a list of ``(direction, normal, d)`` tuples
    applied to the monomer structure, where ``direction`` is ``"above"`` or
    ``"below"`` and ``normal`` is a length-3 array (unit-normalised before
    emission so ``d`` is a Cartesian signed distance from the origin).

    Returns a single Atoms with the fixed atoms first (unchanged) and the
    packed monomers after, matching Packmol's input-order output convention.
    No cell / PBC is set on the result -- the caller assembles the final
    Batch with the desired cell.
    """
    from packmol.cli import get_binary_path

    # Packmol's `fixed x y z α β γ` directive **translates** the input
    # structure by (x, y, z) and rotates by the Euler angles -- it does NOT
    # interpret (x, y, z) as the target centroid. Since `fixed_atoms` is
    # already in the desired Packmol-frame coordinates (the caller shifted
    # them so they sit inside the bbox at world origin), pass an identity
    # transform.

    plane_lines: list[str] = []
    for direction, normal, d in plane_constraints:
        n = np.asarray(normal, dtype=float)
        n_hat = n / np.linalg.norm(n)
        plane_lines.append(
            f"  {direction} plane {n_hat[0]:.8f} {n_hat[1]:.8f} {n_hat[2]:.8f} {float(d):.8f}"
        )
    plane_block = "\n".join(plane_lines)

    pbc_box = np.asarray(pbc_box, dtype=float)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        fixed_xyz = tmp / "fixed.xyz"
        monomer_xyz = tmp / "monomer.xyz"
        out_xyz = tmp / "combined.xyz"
        inp_file = tmp / "pack.inp"

        ase_write(str(fixed_xyz), fixed_atoms, format="xyz")
        ase_write(str(monomer_xyz), monomer, format="xyz")

        seed_line = f"seed {seed}\n" if seed is not None else ""
        inp_file.write_text(
            f"tolerance {tolerance}\n"
            f"filetype xyz\n"
            f"output {out_xyz}\n"
            f"nloop {nloop}\n"
            f"{seed_line}"
            f"pbc {pbc_box[0]:.8f} {pbc_box[1]:.8f} {pbc_box[2]:.8f}\n"
            f"\n"
            f"structure {fixed_xyz}\n"
            f"  number 1\n"
            f"  fixed 0. 0. 0. 0. 0. 0.\n"
            f"end structure\n"
            f"\n"
            f"structure {monomer_xyz}\n"
            f"  number {n_molecules}\n"
            f"{plane_block}\n"
            f"end structure\n"
        )

        with open(inp_file) as inp_stream:
            result = subprocess.run(
                [str(get_binary_path())],
                stdin=inp_stream,
                capture_output=True,
                text=True,
                timeout=600,
            )
        if not out_xyz.exists():
            raise RuntimeError(
                f"Packmol failed without producing output (exit "
                f"{result.returncode}):\n{result.stdout[-2000:]}\n"
                f"{result.stderr}"
            )
        if result.returncode != 0:
            print(
                f"  WARNING: packmol exited {result.returncode} "
                f"(no perfect packing with fixed obstacle, tol={tolerance} A, "
                f"nloop={nloop}). Using best-effort output; increase nloop "
                f"or tolerance for a cleaner pack."
            )

        combined = ase_read(str(out_xyz), format="xyz")

    combined.set_pbc(False)
    combined.set_cell([0.0, 0.0, 0.0])
    return combined
