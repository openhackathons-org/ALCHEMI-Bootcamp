"""BGR NIM batch-throughput scaling utilities.

Drives the hello-world scaling demonstration in the tutorial: send N
copies of a small gas-phase structure (usually a single H2O) in a single
BGR ``run_bgr`` call, measure wall-clock time, and report structures/sec
and atoms/sec throughput. Sweep N across a doubling series until the
NIM refuses the batch or throughput plateaus, then plot the curve.

Results are pickled to ``cached_responses/water-sorbents/throughput_sweep.json``
so ``FAST_DEMO`` replays reproduce the figure without calling the NIM.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .api_client import run_bgr
from .models import ase_to_atomic_data

if TYPE_CHECKING:
    import ase
    import matplotlib.axes


# ---------------------------------------------------------------------------
# Single-batch timing
# ---------------------------------------------------------------------------


def measure_batch_throughput(
    base_atoms: ase.Atoms,
    batch_size: int,
    server_url: str,
    opttol: float | None = None,
    timeout: int = 1800,
) -> dict:
    """Send ``batch_size`` copies of ``base_atoms`` to BGR, time the call.

    The BGR NIM treats each entry in the list as an independent structure;
    sending N copies exercises batch parallelism without changing the
    single-structure compute cost.

    Returns
    -------
    dict with keys: batch_size, wall_time_s, n_atoms_total, struct_per_s,
    atoms_per_s, success, error.
    """
    atoms_list = [
        ase_to_atomic_data(base_atoms, structure_id=f"hello_{i:06d}")
        for i in range(batch_size)
    ]
    n_atoms_total = len(base_atoms) * batch_size
    t0 = time.perf_counter()
    try:
        run_bgr(atoms_list, server_url=server_url, opttol=opttol, timeout=timeout)
        dt = time.perf_counter() - t0
        return {
            "batch_size": batch_size,
            "wall_time_s": dt,
            "n_atoms_total": n_atoms_total,
            "struct_per_s": batch_size / dt,
            "atoms_per_s": n_atoms_total / dt,
            "success": True,
            "error": None,
        }
    except Exception as exc:  # broad: OOM, timeout, HTTP 5xx, etc.
        dt = time.perf_counter() - t0
        return {
            "batch_size": batch_size,
            "wall_time_s": dt,
            "n_atoms_total": n_atoms_total,
            "struct_per_s": math.nan,
            "atoms_per_s": math.nan,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Sweep driver with caching
# ---------------------------------------------------------------------------


def sweep_batch_throughput(
    base_atoms: ase.Atoms,
    sizes: list[int],
    server_url: str,
    cache_path: str | Path,
    endpoint_live: bool,
    stop_on_failure: bool = True,
    opttol: float | None = None,
    timeout: int = 1800,
) -> list[dict]:
    """Run ``measure_batch_throughput`` for each size in ``sizes``.

    If the cache file at ``cache_path`` exists it is loaded and returned
    unchanged (FAST_DEMO mode). Otherwise, if ``endpoint_live`` is True,
    the sweep runs live and results are persisted to ``cache_path``.

    Parameters
    ----------
    base_atoms : ase.Atoms
        Structure to replicate (e.g. a single H2O in a 15 A vacuum cube).
    sizes : list[int]
        Batch sizes to measure (e.g. [1, 2, 4, 8, ...]).
    server_url : str
        BGR endpoint URL.
    cache_path : str or Path
        JSON file holding the sweep results. Loaded if present.
    endpoint_live : bool
        If False and cache is missing, raise rather than call the NIM.
    stop_on_failure : bool
        If True, stop the sweep at the first failure (typical for
        finding an OOM ceiling). If False, continue through all sizes.
    """
    cache_path = Path(cache_path)
    if cache_path.is_file():
        print(f"  Loading cached throughput sweep: {cache_path}")
        return json.loads(cache_path.read_text())

    if not endpoint_live:
        raise RuntimeError(
            f"Throughput cache missing at {cache_path} and BGR endpoint "
            "is not live. Start the NIM or provide the cached sweep."
        )

    results: list[dict] = []
    for n in sizes:
        print(f"  N = {n:>6d} ... ", end="", flush=True)
        r = measure_batch_throughput(
            base_atoms, n, server_url=server_url, opttol=opttol, timeout=timeout
        )
        if r["success"]:
            print(
                f"{r['wall_time_s']:6.2f} s, "
                f"{r['struct_per_s']:>8.1f} struct/s, "
                f"{r['atoms_per_s']:>9.1f} atoms/s"
            )
        else:
            print(f"FAILED after {r['wall_time_s']:.2f} s: {r['error']}")
        results.append(r)
        if stop_on_failure and not r["success"]:
            break

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(results, indent=2))
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_throughput(
    results: list[dict],
    output_path: str | Path,
    title: str = "BGR NIM batch-throughput scaling (single H2O per structure)",
) -> str:
    """Render a two-panel log-log throughput plot and save as PNG.

    Panel 1: structures per second vs batch size.
    Panel 2: atoms per second vs batch size (same curve scaled by
    len(base_atoms)).
    """
    import matplotlib.pyplot as plt

    ok = [r for r in results if r["success"]]
    if not ok:
        raise RuntimeError("No successful batches in sweep; nothing to plot.")

    ns = [r["batch_size"] for r in ok]
    sps = [r["struct_per_s"] for r in ok]
    aps = [r["atoms_per_s"] for r in ok]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, y, ylabel, color in [
        (axes[0], sps, "structures / second", "#2563eb"),
        (axes[1], aps, "atoms / second", "#16a34a"),
    ]:
        ax.loglog(ns, y, "o-", color=color, lw=2, markersize=7)
        ax.set_xlabel("batch size  N  (structures per BGR call)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", ls="--", alpha=0.4)

    # Annotate failure at end of sweep if any
    failed = [r for r in results if not r["success"]]
    if failed:
        r = failed[0]
        for ax in axes:
            ax.axvline(r["batch_size"], color="#dc2626", ls=":", alpha=0.7)
            ax.text(
                r["batch_size"], ax.get_ylim()[0] * 1.5,
                f"ceiling: N={r['batch_size']}",
                color="#dc2626", fontsize=9, ha="left", va="bottom",
            )

    fig.suptitle(title)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)
