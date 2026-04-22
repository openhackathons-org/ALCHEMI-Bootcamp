"""Trajectory persistence (Zarr), CSV log concatenation, JSON (de)serialization,
and per-stage checkpoint management for the fused warmup pipeline.

All checkpoint and log-concatenation helpers take ``log_dir`` as an argument;
the notebook owns the value of ``LOG_DIR`` / ``CKPT_DIR`` and passes it in.
"""

import csv
import shutil
from pathlib import Path

import numpy as np

from nvalchemi.data import Batch
from nvalchemi.data.datapipes import AtomicDataZarrReader, DataLoader, Dataset
from nvalchemi.dynamics import ZarrData

from .constants import STATUS_BY_STAGE, WARMUP_STAGE_NAMES, _ARRAY_KEYS


def load_zarr_trajectory(zarr_path, device="cpu"):
    """Load a Zarr trajectory into a list of Batch objects."""
    reader = AtomicDataZarrReader(str(zarr_path))
    ds = Dataset(reader, device=device, num_workers=1)
    try:
        loader = DataLoader(ds, batch_size=1)
        return list(loader)
    finally:
        ds.close()


def load_zarr_frames(zarr_path, indices, device="cpu"):
    """Load specific frames (by index) from a Zarr trajectory.

    Cheaper than :func:`load_zarr_trajectory` when the caller only needs a
    handful of frames -- e.g. :func:`plot_trajectory_frames` picks 4 evenly
    spaced frames out of hundreds, and paying for the full decode is
    wasteful. ``indices`` is any iterable of ints; each is bounds-checked
    against the dataset length.
    """
    reader = AtomicDataZarrReader(str(zarr_path))
    ds = Dataset(reader, device=device, num_workers=1)
    try:
        n = len(ds)
        frames = []
        for idx in indices:
            i = int(idx)
            if i < 0 or i >= n:
                raise IndexError(f"frame {i} out of range for trajectory of length {n}")
            data, _meta = ds[i]
            frames.append(Batch.from_data_list([data], device=device))
        return frames
    finally:
        ds.close()


def zarr_trajectory_length(zarr_path):
    """Return the number of frames in a Zarr trajectory without decoding any."""
    with AtomicDataZarrReader(str(zarr_path)) as reader:
        return len(reader)


def fresh_zarr_sink(path, capacity):
    """Return a ZarrData ready for a fresh write.

    AtomicDataZarrWriter.write() refuses to overwrite an existing store
    (FileExistsError). A stage that crashed mid-run leaves its log zarr on
    disk; the next attempt would crash on sink startup. Wipe the store first
    so every run-branch sink sees a clean slate.
    """
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    return ZarrData(store=str(path), capacity=capacity)


def extract_per_graph_trajectory(batches, graph_idx, num_graphs):
    """Slice frames for a single graph out of a multi-graph trajectory.

    ``SnapshotHook`` + ``ZarrData`` writes each graph of a multi-graph Batch
    as a separate Zarr sample, and ``DataLoader(batch_size=1)`` yields one
    single-graph Batch per iteration. So frames for ``num_graphs`` systems
    are interleaved: ``[g0_s0, g1_s0, ..., g{M-1}_s0, g0_s1, ...]``. Recover
    one system's trajectory by striding with ``step=num_graphs``.
    """
    return batches[graph_idx::num_graphs]


def read_csv_log(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def checkpoint_exists(stage_name, log_dir):
    return (Path(log_dir) / "checkpoints" / f"after_{stage_name}.zarr").exists()


def save_checkpoint(batch, stage_name, log_dir):
    """Persist ``batch`` (single- or multi-graph) to
    ``checkpoints/after_{stage_name}.zarr``.
    """
    n = max(1, batch.num_graphs or 1)
    sink = fresh_zarr_sink(
        Path(log_dir) / "checkpoints" / f"after_{stage_name}.zarr",
        capacity=n,
    )
    sink.write(batch)


def load_checkpoint(stage_name, log_dir, device):
    """Load an end-of-stage Batch (single- or multi-graph) onto ``device``.

    ZarrData is lazy; count samples via AtomicDataZarrReader, then poke
    ``_count`` / ``_written_once`` so ``.read()`` skips its empty-store guard
    and hands back a Batch with the correct num_graphs.
    """
    path = Path(log_dir) / "checkpoints" / f"after_{stage_name}.zarr"
    with AtomicDataZarrReader(str(path)) as reader:
        n = len(reader)
    sink = ZarrData(store=str(path), capacity=max(1, n))
    sink._count = n
    sink._written_once = True
    return sink.read().to(device)


def load_warmup_trajectory(log_dir, device="cpu"):
    """Concatenate per-stage warmup Zarr trajectories in FIRE -> NVT -> NPT order.

    Returns ``(frames, stage_labels)`` where ``stage_labels[i]`` is the stage
    key (``"fire"`` / ``"nvt_200k"`` / ``"npt_200k"``) for frame i. Missing
    stages contribute no frames.
    """
    frames, labels = [], []
    for stage in WARMUP_STAGE_NAMES:
        p = Path(log_dir) / f"warmup_{stage}.zarr"
        if not p.exists():
            continue
        stage_frames = load_zarr_trajectory(p, device=device)
        frames.extend(stage_frames)
        labels.extend([stage] * len(stage_frames))
    return frames, labels


def load_warmup_csv(log_dir, log_every=100):
    """Concatenate per-stage warmup CSVs with cumulative step counts.

    Each per-stage CSV emits ``step`` starting at 0. To plot a single time
    axis we shift each stage's steps by the total steps its predecessors
    ran, plus ``log_every`` to account for the one-step gap between the last
    logged step of a stage and the first of the next. The ``status`` column
    is rewritten from the file name so downstream code sees FIRE=0, NVT=1,
    NPT=2 regardless of what the hook emitted.
    """
    rows, offset = [], 0
    for stage in WARMUP_STAGE_NAMES:
        p = Path(log_dir) / f"warmup_{stage}.csv"
        if not p.exists():
            continue
        stage_rows = read_csv_log(p)
        last_step = 0
        for r in stage_rows:
            r = dict(r)
            s = int(float(r["step"]))
            r["step"] = str(s + offset)
            r["status"] = str(STATUS_BY_STAGE[stage])
            rows.append(r)
            last_step = s
        offset += last_step + log_every
    return rows


def _to_jsonable(obj):
    """Recursively convert numpy/tuple containers to json-native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def _restore_arrays(analysis_d):
    """Rehydrate the time-series lists back into ndarrays in-place."""
    for a in analysis_d.values():
        for k in _ARRAY_KEYS:
            if k in a:
                a[k] = np.asarray(a[k])
    return analysis_d
