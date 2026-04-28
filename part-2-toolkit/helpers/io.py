"""Trajectory persistence (Zarr), CSV log concatenation, JSON (de)serialization,
and per-stage checkpoint management for the fused warmup pipeline.

All checkpoint and log-concatenation helpers take ``log_dir`` as an argument;
the notebook owns the value of ``LOG_DIR`` / ``CKPT_DIR`` and passes it in.
"""

import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch

from nvalchemi.data import Batch
from nvalchemi.data.datapipes import AtomicDataZarrReader, DataLoader, Dataset
from nvalchemi.dynamics import ZarrData

from .constants import _ARRAY_KEYS, status_by_stage, warmup_stage_names

_PART_RE = re.compile(r"\.part(\d+)$")

# Full set of per-system tensors the NPT integrator stores on its internal
# _state Batch (nvalchemi/dynamics/integrators/npt.py:185-239). Used to
# serialize the _state fields we need to restore on resume. Missing keys are
# silently skipped (handled by hasattr below), so using this tuple also with
# future integrators that add fields will degrade gracefully.
_NPT_STATE_KEYS = (
    "dt",
    "temperature",
    "pressure",
    "barostat_time",
    "thermostat_time",
    "W",
    "cell_velocity",
    "num_atoms_per_system",
    "nhc_eta",
    "nhc_eta_dot",
    "nhc_Q",
    "nhc_b_eta",
    "nhc_b_eta_dot",
    "nhc_b_Q",
    "kinetic_tensors",
    "pressure_tensors",
    "volumes",
)


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


def integrator_state_exists(stage_name, log_dir):
    return (
        Path(log_dir) / "checkpoints" / f"after_{stage_name}_integrator.pt"
    ).exists()


def save_integrator_state(state_batch, stage_name, log_dir):
    """Persist an integrator's internal ``_state`` Batch so a later run can
    restore thermostat-chain and barostat-cell momenta and continue without
    the ~several-tau_P transient that a zero-init incurs.

    Saves as a plain dict of CPU tensors via :func:`torch.save`. The
    system-only Batch produced by :func:`_make_state_batch` does **not**
    round-trip through :class:`~nvalchemi.dynamics.ZarrData`: the reader
    assumes atom-level pointer arrays of length ``num_graphs + 1`` but a
    system-only Batch writes only length ``num_graphs``, so subsequent
    reads raise ``IndexError``. ``torch.save`` of a tensor dict is the
    right granularity for this data.

    Uses a tmp-then-rename pattern so a crash during the write leaves
    either the old file intact or nothing at all — never a torn blob.
    """
    state_dict = {
        k: getattr(state_batch, k).detach().cpu()
        for k in _NPT_STATE_KEYS
        if hasattr(state_batch, k)
    }
    path = Path(log_dir) / "checkpoints" / f"after_{stage_name}_integrator.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state_dict, tmp)
    tmp.replace(path)


def load_integrator_state(stage_name, log_dir, device):
    """Reload the integrator ``_state`` Batch saved by
    :func:`save_integrator_state`. Assign the return value to
    ``integrator._state`` *before* the first ``run()`` call so
    ``BaseDynamics._ensure_state_initialized`` becomes a no-op and the
    preloaded NHC / barostat variables are used as-is.
    """
    # Private toolkit import: the Batch reconstruction helper lives in
    # nvalchemi.dynamics._ops._bridge. Leading-underscore module implies
    # "internal", but there is no public equivalent today, and npt.py
    # itself imports from there.
    from nvalchemi.dynamics._ops._bridge import _make_state_batch

    path = Path(log_dir) / "checkpoints" / f"after_{stage_name}_integrator.pt"
    state_dict = torch.load(path, map_location=device, weights_only=True)
    dev = torch.device(device) if isinstance(device, str) else device
    state_dict = {k: v.to(device=dev) for k, v in state_dict.items()}
    return _make_state_batch(state_dict, dev)


def save_stage_meta(stage_name, log_dir, steps_completed):
    """Write ``checkpoints/after_{stage}.json`` recording how many steps
    the stage has covered, so a re-entry can decide skip vs extend by
    comparing against the target ``n_steps`` in the notebook.

    Tmp-then-rename so a crash during the write never leaves a truncated
    json that a later parse would choke on.
    """
    path = Path(log_dir) / "checkpoints" / f"after_{stage_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"steps_completed": int(steps_completed)}))
    tmp.replace(path)


def load_stage_meta(stage_name, log_dir):
    path = Path(log_dir) / "checkpoints" / f"after_{stage_name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def next_part_index(log_dir, basename):
    """Return the next ``.partN`` index to use for an extension write.

    Convention: the first run writes ``{basename}.csv`` / ``{basename}.zarr``
    (no suffix). Subsequent runs write ``.part2``, ``.part3``, ... Chose the
    next index by globbing either extension so callers only have to supply
    the base stem (e.g. ``"warmup_npt_200k_dt0p5fs"``).
    """
    base_dir = Path(log_dir)
    csv_parts = list(base_dir.glob(f"{basename}.part*.csv"))
    zarr_parts = list(base_dir.glob(f"{basename}.part*.zarr"))
    existing = csv_parts + zarr_parts
    max_seen = 1  # base file (no suffix) is implicitly part 1
    for p in existing:
        m = _PART_RE.search(p.stem)
        if m:
            max_seen = max(max_seen, int(m.group(1)))
    return max_seen + 1


def part_paths(log_dir, basename, part):
    """Return ``(csv_path, zarr_path)`` for a given ``basename`` and 1-based
    ``part`` index. Part 1 uses no suffix (matches the fresh-run convention).
    """
    suffix = "" if part <= 1 else f".part{part}"
    base_dir = Path(log_dir)
    return (
        base_dir / f"{basename}{suffix}.csv",
        base_dir / f"{basename}{suffix}.zarr",
    )


def _enumerate_parts(log_dir, basename, ext):
    """List ``{basename}.{ext}`` and any ``.partN`` sibling parts in index
    order. Missing files are skipped silently.
    """
    base_dir = Path(log_dir)
    base = base_dir / f"{basename}{ext}"
    parts = []
    if base.exists():
        parts.append((1, base))
    for p in base_dir.glob(f"{basename}.part*{ext}"):
        m = _PART_RE.search(p.stem)
        if m:
            parts.append((int(m.group(1)), p))
    parts.sort(key=lambda t: t[0])
    return [p for _, p in parts]


def load_warmup_trajectory(log_dir, device="cpu", stage_suffix="", t_warmup=200.0):
    """Concatenate per-stage warmup Zarr trajectories in FIRE -> NVT -> NPT order.

    Each stage may have been extended across multiple parts
    (``warmup_{stage}{stage_suffix}.zarr``, then
    ``warmup_{stage}{stage_suffix}.part2.zarr``, ...); all parts are
    enumerated and concatenated in index order.

    ``stage_suffix`` is appended to the on-disk stage name (not the returned
    label) so callers can find DT-tagged assets while keeping the base label
    for status lookups -- e.g. ``stage_suffix="_dt0p5fs"`` finds
    ``warmup_nvt_200k_dt0p5fs.zarr`` but still labels frames as ``"nvt_200k"``.

    ``t_warmup`` selects which warmup-temperature stage tuple to enumerate
    (default 200 K matches the canonical baseline). Pass 100.0 to load the
    100 K warmup artefacts (``warmup_nvt_100k_*`` / ``warmup_npt_100k_*``).

    Returns ``(frames, stage_labels)`` where ``stage_labels[i]`` is the base
    stage key (e.g. ``"fire"`` / ``"nvt_200k"`` / ``"npt_200k"``) for frame i.
    Missing stages contribute no frames.
    """
    frames, labels = [], []
    for stage in warmup_stage_names(t_warmup):
        physical = f"{stage}{stage_suffix}"
        for p in _enumerate_parts(log_dir, f"warmup_{physical}", ".zarr"):
            stage_frames = load_zarr_trajectory(p, device=device)
            frames.extend(stage_frames)
            labels.extend([stage] * len(stage_frames))
    return frames, labels


def load_warmup_csv(log_dir, log_every=100, stage_suffix="", t_warmup=200.0):
    """Concatenate per-stage warmup CSVs with cumulative step counts.

    Each per-stage part CSV emits ``step`` starting at 0. The global time
    axis is built by shifting every row by the cumulative steps that ran
    in all earlier parts (within this stage) and all earlier stages, plus
    ``log_every`` per transition to account for the one-step gap between
    the last logged step of a run and the first of the next. The
    ``status`` column is rewritten from the file name so downstream code
    sees FIRE=0, NVT=1, NPT=2 regardless of what the hook emitted.

    ``stage_suffix`` is appended to the on-disk stage name (mirrors
    :func:`load_warmup_trajectory`): e.g. ``stage_suffix="_dt0p5fs"`` finds
    ``warmup_nvt_200k_dt0p5fs.csv`` but still tags rows with the base status.
    ``t_warmup`` selects the warmup-temperature stage tuple (default 200 K).
    """
    status_map = status_by_stage(t_warmup)
    rows, offset = [], 0
    for stage in warmup_stage_names(t_warmup):
        physical = f"{stage}{stage_suffix}"
        for p in _enumerate_parts(log_dir, f"warmup_{physical}", ".csv"):
            stage_rows = read_csv_log(p)
            last_step = 0
            for r in stage_rows:
                r = dict(r)
                s = int(float(r["step"]))
                r["step"] = str(s + offset)
                r["status"] = str(status_map[stage])
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
