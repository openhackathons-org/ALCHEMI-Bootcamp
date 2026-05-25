"""Host-side zarr -> extended-XYZ converter (no nvalchemi-toolkit dependency).

Reads the toolkit's per-snapshot trajectory zarr (CSR layout: atom-level
arrays flattened across all samples + ``meta/atoms_ptr`` offsets, system-
level arrays (cell, energy, pbc, stress) shape ``(num_samples, ...)`` per
sample) using plain ``zarr``. Writes extxyz frames with a streamed Python
formatter — same per-atom-line format as ``ase.io.write(format='extxyz')``
but no batch_to_ase round-trip and no full-list staging in memory, so it
starts emitting bytes as soon as the first frame is read.

Faster than ``export_zarr_to_extxyz.py`` (the in-container path) on this
hardware because:

  * runs on the host rather than the contended container CPU
  * skips the toolkit's ``load_zarr_trajectory`` (which materialises every
    sample as a ``Batch`` object) and ``batch_to_ase`` per-frame conversion
  * streams directly to disk so the local file grows progressively (handy
    for ETA monitoring)

Read with ``visualize_warmup_trajectory.py`` / ``analyze_s0.py`` exactly
as if the file had been produced by ``export_zarr_to_extxyz.py``.

Usage::

    python zarr_to_extxyz.py <zarr-dir> <out-path> [--dt-fs 0.5] [--snapshot-every 100]

If ``out-path`` ends in ``.gz`` the file is gzip-compressed via
``gzip.open``. ``atoms.info["step"]`` is stamped as ``i*--snapshot-every``;
``--dt-fs`` additionally stamps ``info["time_fs"]``. ``pbc`` flags are
written verbatim from the zarr.
"""

import argparse
import gzip
import time
from pathlib import Path

import zarr

# Naphthalene-only; extend if other elements appear. ase.data.chemical_symbols
# would work but adds import weight for one lookup.
Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("zarr_path", type=Path, help="Path to the local zarr directory.")
    p.add_argument(
        "out_path",
        type=Path,
        help="Output path; .gz suffix triggers gzip compression.",
    )
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=100,
        help="MD steps between snapshots (matches SnapshotHook frequency).",
    )
    p.add_argument(
        "--dt-fs",
        type=float,
        default=None,
        help="MD timestep in fs; if set, stamps info['time_fs'] per frame.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.zarr_path.exists():
        raise SystemExit(f"Not found: {args.zarr_path}")

    t0 = time.monotonic()
    root = zarr.open(str(args.zarr_path), mode="r")
    atoms_ptr = root["meta"]["atoms_ptr"][:]
    pos_arr = root["core"]["positions"]
    cell_arr = root["core"]["cell"]
    Z_arr = root["core"]["atomic_numbers"]
    pbc_arr = root["core"]["pbc"]
    forces_arr = root["core"]["forces"] if "forces" in root["core"] else None

    # When the source zarr is being live-written (e.g. tar'd off a running
    # warmup driver), two failure modes are possible:
    #   * attrs claims N samples but chunked storage only has M < N flushed.
    #   * attrs missing 'num_samples' entirely because zarr.json was tar'd
    #     before its update completed.
    # Treat the sample-level arrays as the ground truth and either clamp
    # the metadata claim or infer the count directly from them.
    chunk_capped = min(
        len(atoms_ptr) - 1,  # CSR pointer has N+1 entries for N samples
        cell_arr.shape[0],
        pbc_arr.shape[0],
    )
    if "num_samples" in root.attrs:
        n_samples = int(root.attrs["num_samples"])
        if chunk_capped < n_samples:
            print(
                f"  WARNING: zarr metadata claims {n_samples} samples but disk "
                f"chunks only cover {chunk_capped}; clamping to {chunk_capped} "
                f"(likely a mid-write tar of a running simulation)."
            )
            n_samples = chunk_capped
    else:
        n_samples = chunk_capped
        print(
            f"  WARNING: zarr.json missing 'num_samples' attr (mid-write tar?); "
            f"inferring {n_samples} from sample-level arrays."
        )

    print(
        f"Opened {args.zarr_path.name}: {n_samples} samples, "
        f"{atoms_ptr[n_samples]} total atom entries"
        f"{' [+ forces]' if forces_arr is not None else ''} "
        f"({time.monotonic() - t0:.1f}s)"
    )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.out_path.suffix == ".gz" else open
    t1 = time.monotonic()
    n_written = 0
    with opener(args.out_path, "wt") as out:
        for i in range(n_samples):
            s, e = int(atoms_ptr[i]), int(atoms_ptr[i + 1])
            n_atoms = e - s
            Z = Z_arr[s:e]
            pos = pos_arr[s:e]
            c = cell_arr[i].reshape(-1)  # 9 floats, row-major
            pbc = pbc_arr[i]

            cell_str = " ".join(f"{x:.10f}" for x in c)
            pbc_str = "T T T" if pbc.all() else " ".join("T" if b else "F" for b in pbc)
            step = i * args.snapshot_every
            time_field = f' time_fs="{step * args.dt_fs}"' if args.dt_fs is not None else ""
            properties = "species:S:1:pos:R:3:Z:I:1"
            if forces_arr is not None:
                properties += ":forces:R:3"
            header = (
                f'Lattice="{cell_str}" Properties={properties} '
                f'step="{step}"{time_field} pbc="{pbc_str}"'
            )
            # Symbols once per frame
            symbols = [Z_TO_SYMBOL[int(z)] for z in Z]

            # Frame body: atom count + header + per-atom lines, joined once.
            body = [str(n_atoms), header]
            if forces_arr is not None:
                forces = forces_arr[s:e]
                body.extend(
                    f"{sym:<2s} {p[0]:18.10f} {p[1]:18.10f} {p[2]:18.10f} {z:>3d}"
                    f" {f[0]:14.6e} {f[1]:14.6e} {f[2]:14.6e}"
                    for sym, p, z, f in zip(symbols, pos, Z, forces)
                )
            else:
                body.extend(
                    f"{sym:<2s} {p[0]:18.10f} {p[1]:18.10f} {p[2]:18.10f} {z:>3d}"
                    for sym, p, z in zip(symbols, pos, Z)
                )
            out.write("\n".join(body))
            out.write("\n")
            n_written += 1

            if n_written % 100 == 0 or n_written == n_samples:
                elapsed = time.monotonic() - t1
                rate = n_written / elapsed
                eta = (n_samples - n_written) / rate if rate > 0 else 0
                print(
                    f"  {n_written}/{n_samples} frames "
                    f"({elapsed:.1f}s, {rate:.0f} fps, ETA {eta:.0f}s)"
                )

    size_mb = args.out_path.stat().st_size / 1e6
    print(
        f"Wrote {n_written} frames to {args.out_path} "
        f"({size_mb:.1f} MB, {time.monotonic() - t1:.1f}s)"
    )


if __name__ == "__main__":
    main()
