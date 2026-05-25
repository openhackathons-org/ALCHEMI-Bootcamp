"""Pluck the lowest-PE NPT-trajectory frame and write it as the canonical
``after_npt_<TWARM>_<DT>.zarr`` checkpoint, so that the melt + slc drivers
seed from the most-relaxed configuration in the run rather than the
arbitrary endpoint.

Reads ``logs/<run-name>/warmup_npt_<TWARM>_<DT>.csv`` to find the row with
minimum potential energy, then pulls the matching frame from
``warmup_npt_<TWARM>_<DT>.zarr`` (per-frame Batch sampled every
``SNAPSHOT_EVERY=100`` MD steps), wraps it via ``Batch.from_data_list``,
and writes via the helpers used by the warmup driver itself --
``save_checkpoint`` + ``save_stage_meta`` -- so ``melt.py`` and ``slc.py``
consume it unchanged.

Usage::

    python extract_min_pe_checkpoint.py \\
        --run-name naphthalene_long_2025 --t-warmup 100 --dt 0.5

The existing endpoint checkpoint should be backed up to ``.bak`` before
running this script (it overwrites the canonical key in place).
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader

import _path  # noqa: F401  # parent dir on sys.path for `helpers` import
from helpers import save_checkpoint, save_stage_meta

SNAPSHOT_EVERY = 100  # matches warmup driver constant


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-name", default="naphthalene_long")
    p.add_argument("--t-warmup", type=float, default=200.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dt_tag = f"dt{str(args.dt).replace('.', 'p')}fs"
    t_warmup_tag = f"{int(args.t_warmup)}k"
    log_dir = Path("logs") / args.run_name
    basename = f"warmup_npt_{t_warmup_tag}_{dt_tag}"
    csv_path = log_dir / f"{basename}.csv"
    zarr_path = log_dir / f"{basename}.zarr"

    df = pd.read_csv(csv_path)
    i_min = int(df["energy"].idxmin())
    step = int(df.loc[i_min, "step"])
    frame_idx = step // SNAPSHOT_EVERY
    print(
        f"min-PE frame: step={step} ({step * args.dt / 1000:.2f} ps), "
        f"frame_idx={frame_idx}"
    )
    print(
        f"  E={df.loc[i_min, 'energy']:.3f} eV  "
        f"T={df.loc[i_min, 'temperature']:.2f} K  "
        f"ρ={df.loc[i_min, 'density_g_cm3']:.4f} g/cm³  "
        f"fmax={df.loc[i_min, 'fmax']:.3f} eV/Å"
    )

    reader = AtomicDataZarrReader(store=str(zarr_path))
    data_dict, _ = reader[frame_idx]
    kw = {k: v.to(args.device) for k, v in data_dict.items()}
    atomic_data = AtomicData(**kw)
    batch = Batch.from_data_list([atomic_data], device=args.device)
    print(
        f"loaded frame: {batch.num_nodes} atoms, "
        f"cell {[f'{x:.2f}' for x in batch.cell.squeeze().norm(dim=-1).tolist()]} Å"
    )

    npt_ck = f"npt_{t_warmup_tag}_{dt_tag}"
    save_checkpoint(batch, npt_ck, log_dir)
    save_stage_meta(npt_ck, log_dir, step + SNAPSHOT_EVERY)
    print(
        f"wrote {log_dir}/checkpoints/after_{npt_ck}.zarr "
        f"(steps_completed={step + SNAPSHOT_EVERY})"
    )
    print(
        "note: integrator state .pt is NOT regenerated; melt + slc "
        "re-init velocities anyway, so this is fine for downstream stages."
    )


if __name__ == "__main__":
    main()
