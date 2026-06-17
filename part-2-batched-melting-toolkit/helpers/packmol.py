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
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.neighborlist import build_neighbor_list

if TYPE_CHECKING:
    # Toolkit-only annotations. Guarded so the module stays importable in the
    # host conda env (numpy/ase only) -- the names are needed solely for type
    # hints, which `from __future__ import annotations` keeps as strings.
    import torch
    from nvalchemi.data import Batch

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


# Naphthalene atom layout in an nvalchemi Batch built from the (3,6,3) P2_1/a
# supercell is not a simple stride-N grouping: atoms block per unit cell as
# [C x 20, H x 16], but the 20 C atoms within a block are NOT partitioned
# into two contiguous 10-atom molecules -- C-C bonds span across [0:10] and
# [10:20]. Trying to guess the per-molecule partition analytically is
# fragile, so `_detect_naphthalene_molecules` instead runs a neighbor-list
# connected-components search once per half to discover molecules from
# bonding topology (cutoff 1.8 A catches C-C and C-H but not H-H or
# close-contact van-der-Waals). Topology is identity-preserving through
# dynamics, so the mapping computed on the unwrapped checkpoint is valid
# for the whole simulation.


def _detect_naphthalene_molecules(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell: torch.Tensor,
    c_c_cutoff: float = 1.6,
    c_h_cutoff: float = 1.4,
) -> list[torch.Tensor]:
    """Discover per-molecule atom indices via PBC-aware neighbor-list
    connected components with element-pair-aware bond cutoffs:

      - C-C bond: d < 1.6 A  (typical 1.40, max ~1.55)
      - C-H bond: d < 1.4 A  (typical 1.09, stretched up to ~1.35 at 500 K)
      - H-H: never bond (no H-H bonds in naphthalene)

    A single isotropic cutoff cannot separate the crystal (tight C-H)
    from the melt (stretched C-H): 1.5 A fragments melt C-H bonds, 1.7 A
    merges adjacent molecules via intermolecular H-H contacts. Element-
    pair filtering sidesteps both failure modes -- verified to partition
    both `nvt_200k` and `meltgen_nvt_500k` into exactly 108 C10H8 groups.
    """
    import torch
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    atoms = Atoms(
        numbers=atomic_numbers.cpu().numpy(),
        positions=positions.detach().cpu().numpy(),
        cell=cell.detach().cpu().numpy(),
        pbc=True,
    )
    i_list, j_list, d_list = neighbor_list(
        "ijd", atoms, cutoff=max(c_c_cutoff, c_h_cutoff)
    )
    n = positions.shape[0]
    Z = atomic_numbers.cpu().numpy()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b, d in zip(i_list.tolist(), j_list.tolist(), d_list.tolist()):
        za, zb = int(Z[a]), int(Z[b])
        if za == 6 and zb == 6 and d < c_c_cutoff:
            union(a, b)
        elif {za, zb} == {1, 6} and d < c_h_cutoff:
            union(a, b)
        # H-H pairs never bond

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out: list[torch.Tensor] = []
    for gid, members in groups.items():
        assert len(members) == 18, (
            f"molecule {gid} has {len(members)} atoms (expected 18)"
        )
        Zs = atomic_numbers[members].tolist()
        assert sorted(Zs) == [1] * 8 + [6] * 10, (
            f"molecule {gid} stoichiometry {sorted(Zs)} != C10H8"
        )
        out.append(torch.tensor(sorted(members), dtype=torch.long))
    return out


def _unwrap_molecules_inplace(
    positions: torch.Tensor,
    cell: torch.Tensor,
    mol_indices: list[torch.Tensor],
) -> None:
    """Pull every atom of each molecule to the minimum-image position relative
    to that molecule's first atom. Must be called on each SLC half *before*
    concatenation, using that half's *own* cell. Caller supplies the
    molecule partition (from ``_detect_naphthalene_molecules``) so this
    routine does not need to rediscover topology.
    """
    import torch

    cell_inv = torch.linalg.inv(cell)
    for idx in mol_indices:
        idx = idx.to(positions.device)
        ref = positions[idx[0]].clone()
        dr = positions[idx] - ref
        frac = dr @ cell_inv
        frac -= frac.round()
        positions[idx] = ref + frac @ cell


def _wrap_molecules_by_com_inplace(
    positions: torch.Tensor,
    cell: torch.Tensor,
    mol_indices: list[torch.Tensor],
) -> None:
    """After stacking, fold whole molecules back into the stacked cell by
    their COM (integer lattice shifts only), so no molecule is split across
    the PBC. Uses uniform-mass COM for speed -- the shift is an integer
    number of lattice vectors so atomic-mass weighting cannot change it for
    compact molecules. Apply AFTER concatenation, using the STACKED cell.
    """
    import torch

    cell_inv = torch.linalg.inv(cell)
    for idx in mol_indices:
        idx = idx.to(positions.device)
        com = positions[idx].mean(dim=0)
        com_frac = com @ cell_inv
        shift_frac = com_frac.floor()
        if shift_frac.abs().sum() > 0:
            positions[idx] = positions[idx] - shift_frac @ cell


def build_packmol_slc_stack(
    crystal_batch: Batch,
    monomer: Atoms,
    n_melt_molecules: int,
    *,
    device: str,
    tolerance: float = 2.0,
    nloop: int = 20,
    seed: int | None = None,
) -> Batch:
    """Construct the full SLC stack in a single Packmol invocation, with
    the crystal half held fixed as an obstacle.

    Packmol's ``tolerance`` enforces minimum inter-atom separation against
    the crystal atoms (including PBC-wrap images), so interface clashes are
    prevented at construction rather than relaxed away afterwards. No
    vacuum gap: the stacked cell is exactly (a, 2*b, c) of the crystal
    cell.

    The melt half is confined to the upper-b parallelepiped via six
    ``above plane`` / ``below plane`` constraints. Packmol's ``pbc Lx Ly Lz``
    operates over the orthorhombic bounding box of the stacked cell; we
    shift all coordinates so the bbox sits at the world origin (see the
    plan file's "Why the shift" section for the rationale), then shift
    back into the crystal's original frame on output.

    Returns a single-graph Batch with the crystal atoms first (positions
    unchanged from the warmup endpoint after unwrap + COM-wrap) and the
    packmol-placed melt atoms after.
    """
    import torch
    from nvalchemi.data import AtomicData, Batch

    # 1. Unwrap + COM-wrap crystal molecules in the crystal cell so any
    # WrapPeriodicHook splits from the warmup NPT endpoint are stitched
    # back together before we hand atoms to Packmol.
    crystal_pos = crystal_batch.positions.clone()
    crystal_Z = crystal_batch.atomic_numbers
    crystal_cell = crystal_batch.cell.squeeze().clone()
    crystal_mols = _detect_naphthalene_molecules(
        crystal_pos, crystal_Z, crystal_cell,
    )
    _unwrap_molecules_inplace(crystal_pos, crystal_cell, crystal_mols)
    _wrap_molecules_by_com_inplace(crystal_pos, crystal_cell, crystal_mols)

    # 2. Build the stacked cell + Packmol-frame shift.
    cell_np = crystal_cell.detach().cpu().numpy()
    a, b, c = cell_np[0], cell_np[1], cell_np[2]
    stacked_cell_np = np.stack([a, 2.0 * b, c])
    vertices = np.stack([
        np.zeros(3), a, 2.0 * b, c,
        a + 2.0 * b, a + c, 2.0 * b + c, a + 2.0 * b + c,
    ])
    bbox_lo = vertices.min(axis=0)
    bbox_hi = vertices.max(axis=0)
    shift = -bbox_lo
    pbc_box = bbox_hi - bbox_lo

    # 3. Six plane constraints carving the upper-b parallelepiped (the melt
    # half) out of the stacked-cell bbox. Origin = shift + b_vec so the
    # lower-b face sits at the crystal-half top.
    n_a = np.cross(b, c); n_a /= np.linalg.norm(n_a)
    n_b = np.cross(c, a); n_b /= np.linalg.norm(n_b)
    n_c = np.cross(a, b); n_c /= np.linalg.norm(n_c)
    margin = max(tolerance / 2.0, 0.5)
    melt_origin = shift + b
    plane_constraints = [
        ("above", n_a, float(n_a @ melt_origin) + margin),
        ("below", n_a, float(n_a @ (melt_origin + a)) - margin),
        ("above", n_b, float(n_b @ melt_origin) + margin),
        ("below", n_b, float(n_b @ (melt_origin + b)) - margin),
        ("above", n_c, float(n_c @ melt_origin) + margin),
        ("below", n_c, float(n_c @ (melt_origin + c)) - margin),
    ]

    # 4. Per-atom wrap into the STACKED cell so any unwrap-protrusion that
    # sat at e.g. y = -0.1 (just below the crystal cell's y=0 boundary)
    # lands at y = 2|b| - 0.1 (top of the stacked cell, adjacent to the
    # melt-top across the stacked-cell PBC). This puts every fixed atom
    # inside the Packmol bbox and lets Packmol's tolerance check work
    # correctly at the wrap-side interface. Molecules torn by this wrap
    # are re-stitched after Packmol returns (step 7).
    crystal_pos_np = crystal_pos.detach().cpu().numpy()
    crystal_Z_np = crystal_Z.detach().cpu().numpy()
    stacked_cell_inv = np.linalg.inv(stacked_cell_np)
    crystal_frac = crystal_pos_np @ stacked_cell_inv
    crystal_frac -= np.floor(crystal_frac)
    crystal_pos_np = crystal_frac @ stacked_cell_np
    fixed_atoms = Atoms(
        numbers=crystal_Z_np, positions=crystal_pos_np + shift,
    )

    # 5. Single Packmol run -- tolerance enforces interface separation.
    combined = pack_with_fixed_obstacle(
        monomer, n_melt_molecules, fixed_atoms,
        pbc_box=pbc_box,
        plane_constraints=plane_constraints,
        tolerance=tolerance,
        nloop=nloop,
        seed=seed,
    )

    # 6. Undo the bbox shift; combined now sits in the crystal's frame.
    combined_pos_np = combined.get_positions() - shift
    combined_Z_np = combined.get_atomic_numbers()
    n_total = len(combined_pos_np)

    # 7. Re-stitch molecules in the stacked cell. The per-atom wrap in step 4
    # may have split some crystal molecules across the stacked-cell PBC
    # (bottom-protruders wrapped to the top); molecule detection here uses
    # the stacked-cell PBC neighbour list so the split atoms are still
    # recognised as one molecule. Unwrap pulls them back together, then
    # COM-wrap folds whole molecules into the stacked cell.
    combined_pos = torch.tensor(
        combined_pos_np, dtype=torch.float32, device=device,
    )
    combined_Z = torch.tensor(combined_Z_np, dtype=torch.long, device=device)
    stacked_cell = torch.tensor(
        stacked_cell_np, dtype=torch.float32, device=device,
    )
    stacked_mols = _detect_naphthalene_molecules(
        combined_pos, combined_Z, stacked_cell,
    )
    _unwrap_molecules_inplace(combined_pos, stacked_cell, stacked_mols)
    _wrap_molecules_by_com_inplace(combined_pos, stacked_cell, stacked_mols)

    # 8. Wrap into a single-graph SLC Batch. Velocities are zero -- Stage B
    # reinitialises at per-T Maxwell-Boltzmann after multi-graph expansion.
    slc_data = AtomicData(
        positions=combined_pos,
        atomic_numbers=combined_Z,
        velocities=torch.zeros(n_total, 3, device=device),
        forces=torch.zeros(n_total, 3, device=device),
        energy=torch.zeros(1, 1, device=device),
        stress=torch.zeros(1, 3, 3, device=device),
        cell=stacked_cell.unsqueeze(0),
        pbc=torch.tensor([[True, True, True]], device=device),
    )
    slc_data.charge = torch.zeros(1, 1, device=device)
    return Batch.from_data_list([slc_data], device=device)
