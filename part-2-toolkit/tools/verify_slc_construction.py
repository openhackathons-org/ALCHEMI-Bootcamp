"""Construct + sanity-check the SLC stack (debug, no dynamics).

Loads warmup + melt checkpoints, runs the production
``construct_slc_stack`` from ``slc``, verifies geometric
sanity, and exports the stacked geometry as a single-frame extxyz so
``visualize_warmup_trajectory.py`` can open it in ASE GUI for visual
inspection of the constructed stack BEFORE any dynamics run.

Sanity checks reported:

  * total atom count = 2 x crystal half (asserted inside
    ``construct_slc_stack``, mirrored here for visibility);
  * molecule re-partition on the stacked geometry yields exactly
    2 x N_mol C10H8 groups -- the assertions inside
    ``_detect_naphthalene_molecules`` will fail loudly if any molecule
    was torn (unexpected atom count or stoichiometry per group);
  * minimum PBC pairwise distance is above the ~0.5 A clash threshold;
  * no H atom sits more than 1.3 A from its nearest C (the H-C
    dissociation diagnostic from CLAUDE.md).

This is the canonical pre-flight check for a new (run-name, supercell)
combination -- run it once after a fresh warmup + melt to confirm the
stacking algorithm preserves molecular integrity for that geometry,
then launch the full SLC pipeline only if all checks pass.

Usage::

    python verify_slc_construction.py [--run-name NAME]
                                      [--source {npt,nvt,auto}]
                                      [--t-warmup 200] [--dt 0.5]
                                      [--device cuda]

Output goes to ``logs/<run-name>/slc_init_from_<src>_<T_TAG>_<DT_TAG>.extxyz``.
The script prints the visualizer command on exit.
"""

import argparse
import sys
from pathlib import Path

import torch
from ase import Atoms
from ase.io import write as ase_write

import _path  # noqa: F401  # parent dir on sys.path for `helpers` and `slc` imports
from ase.io import read as ase_read
from helpers import (
    checkpoint_exists,
    extract_single_molecule,
    load_checkpoint,
    min_pbc_distance,
)
from slc import (
    SLC_INTERFACE_GAP,
    _detect_naphthalene_molecules,
    build_packmol_slc_stack,
    construct_slc_stack,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-name", default="naphthalene_long",
        help="Run name; resolves to logs/<run-name>/checkpoints/ for the seed pair.",
    )
    p.add_argument(
        "--source", choices=["npt", "nvt", "auto"], default="auto",
        help="Which warmup branch feeds both halves (matches "
             "slc.py --source). 'auto' tries npt first.",
    )
    p.add_argument(
        "--t-warmup", type=float, default=200.0,
        help="Warmup target temperature in K; tagged into checkpoint lookup "
             "as f'{int(T)}k'. Must match the warmup driver's value.",
    )
    p.add_argument(
        "--dt", type=float, default=0.5,
        help="MD timestep in fs (default 0.5). Used to tag the seed checkpoint "
             "lookup -- must match the warmup driver.",
    )
    p.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="PyTorch device for the construction (CPU is fine -- no compute, "
             "just tensor ops + neighbor list).",
    )
    p.add_argument(
        "--gap", type=float, default=SLC_INTERFACE_GAP,
        help=f"Interface vacuum gap in A for the meltgen-melt path "
             f"(default {SLC_INTERFACE_GAP}). Ignored when --packmol is set "
             f"(the single-Packmol-run path has no gap concept). Sweep smaller "
             "values for meltgen-melt to find the minimum gap that still "
             "passes the post-construction clash check. The output extxyz "
             "filename includes the gap so multiple values coexist.",
    )
    p.add_argument(
        "--no-write", action="store_true",
        help="Skip the extxyz write (useful for fast sweeps). Sanity checks "
             "still run and report.",
    )
    p.add_argument(
        "--packmol", action="store_true",
        help="Build the melt half via Packmol instead of loading a melt NVT "
             "checkpoint. Mirrors `slc.py --packmol`. Skip the melt-checkpoint "
             "existence requirement; pack N_mol_crystal copies of the molecule "
             "extracted from data/<material>.cif into the crystal-half "
             "parallelepiped. Output extxyz is tagged `slc_init_packmol_melt_"
             "from_{src}_...` so packmol-melt and melt-NVT init geometries "
             "coexist on disk.",
    )
    p.add_argument(
        "--material", default="naphthalene",
        help="Material name; resolves to data/<material>.cif for --packmol "
             "monomer extraction. Default 'naphthalene'. Ignored without "
             "--packmol.",
    )
    p.add_argument(
        "--packmol-tolerance", type=float, default=2.0,
        help="Packmol minimum inter-atom distance in A (default 2.0).",
    )
    p.add_argument(
        "--packmol-nloop", type=int, default=20,
        help="Packmol GENCAN iteration cap (default 20, Packmol stock default). "
             "Bump (e.g. 200) if Packmol exits 'ENDED WITHOUT PERFECT PACKING'.",
    )
    return p.parse_args()


def resolve_source(
    source_arg: str,
    log_dir: Path,
    dt_tag: str,
    t_warmup_tag: str,
    packmol: bool = False,
) -> tuple[str, str, str | None]:
    """Mirror the source-resolution loop in slc.main. When ``packmol`` is
    True, only the warmup checkpoint is required and the returned melt key
    is None (melt half is generated, not loaded).
    """
    if source_arg == "auto":
        order = ("npt", "nvt")
    else:
        order = (source_arg,)
    misses: list[tuple[str, list[str]]] = []
    for src in order:
        warmup = f"{src}_{t_warmup_tag}_{dt_tag}"
        has_w = checkpoint_exists(warmup, log_dir)
        if packmol:
            if has_w:
                return src, warmup, None
            misses.append((src, [f"after_{warmup}.zarr"]))
            continue
        melt = f"meltgen_nvt_500k_from_{src}_{t_warmup_tag}_{dt_tag}"
        has_m = checkpoint_exists(melt, log_dir)
        if has_w and has_m:
            return src, warmup, melt
        miss = []
        if not has_w:
            miss.append(f"after_{warmup}.zarr")
        if not has_m:
            miss.append(f"after_{melt}.zarr")
        misses.append((src, miss))
    if packmol:
        sys.exit(
            f"No warmup checkpoint under {log_dir / 'checkpoints'} for "
            f"T_WARMUP={t_warmup_tag}, DT={dt_tag}. Missing per source: "
            f"{dict(misses)}"
        )
    sys.exit(
        f"No matching (warmup, melt) pair under {log_dir / 'checkpoints'} for "
        f"T_WARMUP={t_warmup_tag}, DT={dt_tag}. Missing per source: {dict(misses)}"
    )


def hc_diagnostic(
    positions: torch.Tensor, atomic_numbers: torch.Tensor, cell: torch.Tensor,
) -> tuple[int, float, float]:
    """Per-H minimum distance to nearest C atom (PBC, MIC).

    Returns ``(n_dissociated, max_min_d, median_min_d)`` where
    ``n_dissociated`` counts H atoms whose nearest-C distance exceeds
    1.3 A (typical aromatic C-H is 1.09 A; 1.35 A is already a stretched
    bond at 500 K, so >1.3 A is a strong dissociation signal).

    Vectorised pairwise computation: an (N_H, N_C, 3) tensor of MIC
    displacement vectors. For a (5,5,4) supercell stacked half-on-half
    that's about (1600, 2000, 3) ~ 38 MB of float32 -- fits comfortably.
    """
    H_idx = (atomic_numbers == 1).nonzero(as_tuple=True)[0]
    C_idx = (atomic_numbers == 6).nonzero(as_tuple=True)[0]
    H_pos = positions[H_idx]
    C_pos = positions[C_idx]
    dr = H_pos[:, None, :] - C_pos[None, :, :]  # (N_H, N_C, 3)
    cell_inv = torch.linalg.inv(cell)
    dr_frac = dr @ cell_inv
    dr_frac -= dr_frac.round()
    dr_mic = dr_frac @ cell
    d = dr_mic.norm(dim=-1)
    d_min = d.min(dim=-1).values  # (N_H,)
    return (
        int((d_min > 1.3).sum().item()),
        float(d_min.max().item()),
        float(d_min.median().item()),
    )


def main() -> None:
    args = parse_args()
    dt_tag = f"dt{str(args.dt).replace('.', 'p')}fs"
    t_warmup_tag = f"{int(args.t_warmup)}k"
    log_dir = Path("logs") / args.run_name

    src, warmup_ck, melt_ck = resolve_source(
        args.source, log_dir, dt_tag, t_warmup_tag, packmol=args.packmol,
    )
    if args.packmol:
        print(
            f"Source: {src} -- crystal '{warmup_ck}', melt generated via Packmol "
            f"(material={args.material}, tol={args.packmol_tolerance} A, "
            f"nloop={args.packmol_nloop})"
        )
    else:
        print(f"Source: {src} -- crystal '{warmup_ck}', melt '{melt_ck}'")

    crystal_batch = load_checkpoint(warmup_ck, log_dir, args.device)
    cryst_cl = crystal_batch.cell.squeeze().norm(dim=-1)
    print(f"Crystal: {crystal_batch.num_nodes} atoms, "
          f"cell {[f'{l:.2f}' for l in cryst_cl.tolist()]} A")

    if args.packmol:
        cif_path = Path("data") / f"{args.material}.cif"
        if not cif_path.exists():
            sys.exit(
                f"{cif_path} not found. --packmol needs the CIF to extract "
                f"the single-molecule template. Available: "
                f"{sorted(p.stem for p in Path('data').glob('*.cif'))}"
            )
        unit_cell = ase_read(str(cif_path))
        monomer = extract_single_molecule(unit_cell)
        atoms_per_mol = len(monomer)
        assert crystal_batch.num_nodes % atoms_per_mol == 0, (
            f"crystal half {crystal_batch.num_nodes} atoms not divisible by "
            f"monomer {atoms_per_mol} ({monomer.get_chemical_formula()})"
        )
        n_mol_crystal = crystal_batch.num_nodes // atoms_per_mol
        print(
            f"Monomer: {monomer.get_chemical_formula()} ({atoms_per_mol} atoms); "
            f"packing {n_mol_crystal} copies around fixed crystal (single Packmol run)"
        )
        slc_batch = build_packmol_slc_stack(
            crystal_batch,
            monomer,
            n_mol_crystal,
            device=args.device,
            tolerance=args.packmol_tolerance,
            nloop=args.packmol_nloop,
        )
    else:
        melt_batch = load_checkpoint(melt_ck, log_dir, args.device)
        melt_cl = melt_batch.cell.squeeze().norm(dim=-1)
        print(f"Melt:    {melt_batch.num_nodes} atoms, "
              f"cell {[f'{l:.2f}' for l in melt_cl.tolist()]} A")
        print(f"\nGap: {args.gap} A")
        try:
            slc_batch, _stacked_mols, _n_crystal = construct_slc_stack(
                crystal_batch, melt_batch, device=args.device, gap=args.gap,
            )
        except AssertionError as e:
            # `_detect_naphthalene_molecules` asserts when stacking tears a
            # molecule (atom count or stoichiometry per group). For the sweep
            # path we want this to print FAIL instead of crashing.
            print(f"  FAIL  construction asserted: {e}")
            sys.exit(2)

    slc_cell = slc_batch.cell.squeeze()
    slc_pos = slc_batch.positions
    slc_Z = slc_batch.atomic_numbers
    slc_cl = slc_cell.norm(dim=-1)
    print(f"Stacked: {slc_batch.num_nodes} atoms, "
          f"cell {[f'{l:.2f}' for l in slc_cl.tolist()]} A "
          f"(b axis: {cryst_cl[1].item():.2f} -> {slc_cl[1].item():.2f})")

    # --- Sanity checks -----------------------------------------------------
    print("\n=== Sanity checks ===")

    # 1. Re-detect molecules on stacked geometry. The internal asserts in
    #    `_detect_naphthalene_molecules` will raise if any molecule has
    #    !=18 atoms or !=C10H8 stoichiometry, i.e. if stacking tore one.
    expected = slc_batch.num_nodes // 18  # naphthalene-specific assumption
    try:
        re_mols = _detect_naphthalene_molecules(slc_pos, slc_Z, slc_cell)
        actual = len(re_mols)
        label = "OK   " if actual == expected else "FAIL "
        print(f"  {label} molecule count after stacking: {actual} "
              f"(expected {expected}, = 2 x {expected // 2})")
    except AssertionError as e:
        print(f"  FAIL  re-detect torn a molecule: {e}")

    # 2. Min pairwise PBC distance -- the FIRE clash-guard threshold (post-FIRE
    # assertion in slc.py is `min_dist > 0.5`). Both source paths produce
    # geometries that should satisfy this at construction time: the
    # meltgen-melt path via the vacuum gap, the --packmol path via Packmol's
    # tolerance enforcement against the fixed crystal.
    min_d = min_pbc_distance(slc_pos, slc_cell)
    label = "OK   " if min_d > 0.5 else "FAIL "
    print(f"  {label} min pairwise distance (PBC, all atoms): "
          f"{min_d:.3f} A (threshold 0.5)")

    # 3. H-C dissociation diagnostic (CLAUDE.md gotcha).
    n_dissoc, max_d, med_d = hc_diagnostic(slc_pos, slc_Z, slc_cell)
    label = "OK   " if n_dissoc == 0 else "WARN "
    print(f"  {label} H-C dissociation: {n_dissoc} H atoms with nearest C "
          f"> 1.3 A; median nearest H-C = {med_d:.3f} A, max = {max_d:.3f} A "
          "(typical aromatic C-H = 1.09 A)")

    # --- Export to extxyz --------------------------------------------------
    if args.no_write:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    # Gap tag only meaningful for the meltgen-melt path; the --packmol path
    # has no gap concept (interfaces are flush, tolerance-enforced).
    gap_tag = (
        f"_gap{str(args.gap).replace('.', 'p')}A"
        if not args.packmol and args.gap != SLC_INTERFACE_GAP
        else ""
    )
    src_tag = f"packmol_melt_from_{src}" if args.packmol else f"from_{src}"
    out_name = f"slc_init_{src_tag}_{t_warmup_tag}_{dt_tag}{gap_tag}.extxyz"
    out_path = log_dir / out_name
    atoms = Atoms(
        numbers=slc_Z.cpu().numpy(),
        positions=slc_pos.detach().cpu().numpy(),
        cell=slc_cell.detach().cpu().numpy(),
        pbc=True,
    )
    atoms.info["step"] = 0
    atoms.info["source"] = src
    atoms.info["t_warmup_K"] = float(args.t_warmup)
    atoms.info["packmol_melt"] = bool(args.packmol)
    if not args.packmol:
        atoms.info["slc_gap_A"] = float(args.gap)
    ase_write(str(out_path), atoms, format="extxyz")
    size_kb = out_path.stat().st_size / 1024
    print(f"\nWrote {out_path} ({size_kb:.0f} KB, single frame)")
    print(f"\nView with:\n"
          f"  python visualize_warmup_trajectory.py {out_path}")


if __name__ == "__main__":
    main()
