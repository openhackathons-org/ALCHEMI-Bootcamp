"""Multi-temperature SLC diagnostic for the SLC pre-equilibration / production sweep.

Reads ``logs/<run-name>/slc_<stem>_from_{src}_{T_WARMUP_TAG}_{DT_TAG}.csv``
(plus any ``.partN`` siblings from extend runs), or — when launched via
``slc_multi_gpu.sh`` — the per-rank shards ``..._t{subset}.csv`` with their
local ``graph_idx`` rebased onto the global ``TEMPS`` order. Splits the
merged frame by ``graph_idx`` into five temperature series and produces a
stacked-panel figure with raw (alpha=0.22) + rolling-mean (w=20) overlay,
one viridis colour per T.

NVT panels (4): temperature / energy / fmax / pressure. Cell is fixed.

NPT panels (5): temperature / density / pressure / energy / fmax.

``stem`` is ``slc_nvt`` for the pre-equilibration stage and ``slc_all`` for
the production NPT (matches ``slc.py`` artefact naming).
Diverged runs (T outside [50, 1000] K, fmax > 20 eV/Å, |P| > 0.1 eV/Å³)
are filtered out of the ylim Tukey fence so a single blown-up graph
doesn't crush the healthy ones.

Pull the CSVs off the remote container first, e.g.::

    source /tmp/alchemi-playbook-part2-deploy.env
    RUN=naphthalene_long_2025; STAGE=npt; SRC=nvt; T=200k; TAG=dt0p5fs
    STEM=slc_nvt; [ "$STAGE" = npt ] && STEM=slc_all
    ssh -J "$LOGIN_HOST" "$COMPUTE_NODE" \\
      "docker exec alchemi-playbook-part2 tar czf - -C /workspace/logs/$RUN \\
         ${STEM}_from_${SRC}_${T}_${TAG}_t250_300.csv \\
         ${STEM}_from_${SRC}_${T}_${TAG}_t350.csv \\
         ${STEM}_from_${SRC}_${T}_${TAG}_t400.csv \\
         ${STEM}_from_${SRC}_${T}_${TAG}_t450.csv" \\
      | tar xzf - -C dev/part-2-toolkit/logs/$RUN

``tar`` silently drops missing members, so over-specifying shard names
is safe. Run locally from ``dev/part-2-toolkit/``::

    python plot_slc.py --stage {nvt,npt} [--src {npt,nvt,auto}] [--dt 0.5]
                       [--t-warmup 200] [--total-ps N]
                       [--run-name naphthalene_long]

``--total-ps`` defaults to a stage-aware value (10 for NVT, 50 for NPT).
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent  # part-2-toolkit/ root

TEMPS = [200, 250, 300, 350, 400, 450, 500]
LOG_EVERY = 100
WINDOW = 20
P_1ATM = 101325.0 / 1.602176634e11  # eV/A^3


# Physical bands each column should stay in when the system is behaving.
# A graph whose rolling-median falls outside its column's band is treated
# as diverged and excluded from the ylim fence. Bands are wide enough to
# absorb transient spikes but tight enough to reject the fmax-clamped
# (~50 eV/Å), T~1e5 K, P~1 eV/Å³ runaway regime we see from bare AIMNet2
# in pathological regions. Energy and density are system-size dependent
# (no fixed band) — they fall through to a plain Tukey on all graphs.
HEALTHY_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature": (50.0, 1000.0),  # K    (TEMPS spans 250..450, pad ~2x)
    "fmax": (0.0, 20.0),  # eV/Å (clamp is 50; healthy is < ~15)
    "pressure_eV_A3": (-0.1, 0.1),  # eV/Å³ (condensed phase at 1 atm ~ 0)
}


# Panel registry: (csv_col, ylabel, ref, ref_label_template).
# `ref == "TEMPS"` -> per-T viridis axhlines (temperature panel).
# `ref` numeric  -> single black axhline at that value.
# `ref` None     -> no reference line.
_PANEL_SPECS = {
    "temperature": (
        "temperature",
        "Temperature (K)",
        "TEMPS",
        None,
    ),
    "density": (
        "density_g_cm3",
        "Density (g/cm$^3$)",
        None,
        None,
    ),
    "pressure": (
        "pressure_eV_A3",
        "Pressure (eV/Å$^3$)",
        P_1ATM,
        "1 atm = {ref:.2e} eV/Å$^3$",
    ),
    "energy": (
        "energy",
        "Potential energy (eV)",
        None,
        None,
    ),
    "fmax": (
        "fmax",
        "f$_{\\mathrm{max}}$ (eV/Å)",
        None,
        None,
    ),
}

STAGE_CONFIG = {
    "nvt": {
        "default_total_ps": 10.0,
        "input_stem": "slc_nvt",
        "output_stem": "slc_nvt",
        "title_phrase": "pre-equilibration",
        "panels": ("temperature", "energy", "fmax", "pressure"),
    },
    "npt": {
        "default_total_ps": 50.0,
        "input_stem": "slc_all",
        "output_stem": "slc_npt",
        "title_phrase": "sweep",
        "panels": ("temperature", "density", "pressure", "energy", "fmax"),
    },
}


def dt_tag(dt: float) -> str:
    return f"dt{str(dt).replace('.', 'p')}fs"


def _median_in_band(rolling: np.ndarray, band: tuple[float, float]) -> bool:
    vals = np.asarray(rolling, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return False
    m = float(np.median(vals))
    return band[0] <= m <= band[1]


def _clip_ylim(graph_series, col, refs=None, k=1.5, pad=0.08):
    """Physical-band-filtered Tukey ylim so diverged trajectories don't crush
    the healthy ones. Per-graph filter on ``HEALTHY_BOUNDS[col]`` (when
    defined) followed by Tukey-fence on the surviving pooled samples.
    Falls back to all graphs if the band rejects everything.
    """
    band = HEALTHY_BOUNDS.get(col)
    if band is not None:
        healthy = [g for g in graph_series if _median_in_band(g, band)]
        if not healthy:
            healthy = graph_series
    else:
        healthy = graph_series
    pooled = np.concatenate([np.asarray(g, dtype=float) for g in healthy])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return None, None
    q25, q75 = np.nanpercentile(pooled, [25, 75])
    iqr = q75 - q25
    data_lo, data_hi = float(pooled.min()), float(pooled.max())
    if iqr > 0:
        lo = max(data_lo, q25 - k * iqr)
        hi = min(data_hi, q75 + k * iqr)
    else:
        lo, hi = data_lo, data_hi
    if refs:
        lo = min(lo, min(refs))
        hi = max(hi, max(refs))
    span = hi - lo
    if span == 0:
        span = abs(hi) * 0.1 if hi != 0 else 1.0
    return lo - pad * span, hi + pad * span


def _has_data(log_dir: Path, basename: str) -> bool:
    if (log_dir / f"{basename}.csv").exists():
        return True
    if list(log_dir.glob(f"{basename}.part*.csv")):
        return True
    if list(log_dir.glob(f"{basename}_t*.csv")):
        return True
    return False


def resolve_src(
    log_dir: Path, stem: str, t_warmup_tag: str, dt_suffix: str, src_arg: str
) -> str:
    def base(s: str) -> str:
        return f"{stem}_from_{s}_{t_warmup_tag}_{dt_suffix}"

    if src_arg in ("npt", "nvt"):
        if not _has_data(log_dir, base(src_arg)):
            sys.exit(f"no {base(src_arg)}.csv* in {log_dir}")
        return src_arg
    # auto: prefer npt (matches the driver's auto fallback order).
    for s in ("npt", "nvt"):
        if _has_data(log_dir, base(s)):
            return s
    sys.exit(
        f"no {stem}_from_{{npt,nvt}}_{t_warmup_tag}_{dt_suffix}.csv* under "
        f"{log_dir} -- pull the remote logs first (see module docstring)"
    )


def _load_part_chain(log_dir: Path, basename: str) -> pd.DataFrame | None:
    """Concatenate ``{basename}.csv`` and its ``.partN`` siblings in order,
    shifting each part's ``step`` by the cumulative step count of earlier
    parts so the combined ``global_step`` axis is monotonic. Returns
    ``None`` if no matching files exist (caller falls back to multi-GPU
    shard discovery).
    """
    bare = log_dir / f"{basename}.csv"
    extras = []
    for p in log_dir.glob(f"{basename}.part*.csv"):
        m = re.search(r"\.part(\d+)$", p.stem)
        if m:
            extras.append((int(m.group(1)), p))
    extras.sort(key=lambda t: t[0])
    parts = ([bare] if bare.exists() else []) + [p for _, p in extras]
    if not parts:
        return None

    frames = []
    offset = 0
    for p in parts:
        df = pd.read_csv(p)
        df["global_step"] = df["step"] + offset
        frames.append(df)
        last_step = int(df["step"].max())
        offset += last_step + LOG_EVERY
    return pd.concat(frames, ignore_index=True)


def _load_multigpu_shards(log_dir: Path, basename: str) -> pd.DataFrame:
    """Merge ``{basename}_t<subset>[.partN].csv`` shards (one set per rank).

    Each shard's local ``graph_idx`` is 0..n-1 over its own subset; we
    rebase to the global ``TEMPS`` order so a single ``graph_idx == i``
    filter downstream selects the right T regardless of which rank wrote
    the row. ``.partN`` extension is honoured per-shard.
    """
    pat = re.compile(rf"^{re.escape(basename)}_t([0-9_]+)(?:\.part(\d+))?$")
    by_subset: dict[tuple[int, ...], list[tuple[int, Path]]] = {}
    for p in log_dir.glob(f"{basename}_t*.csv"):
        m = pat.match(p.stem)
        if not m:
            continue
        temps = tuple(int(t) for t in m.group(1).split("_"))
        part = int(m.group(2)) if m.group(2) else 0
        by_subset.setdefault(temps, []).append((part, p))

    if not by_subset:
        sys.exit(f"no {basename}_t*.csv shards in {log_dir}")

    frames = []
    for temps, paths in by_subset.items():
        paths.sort(key=lambda t: t[0])  # bare (part=0) first, then partN
        offset = 0
        for _, p in paths:
            df = pd.read_csv(p)
            df["global_step"] = df["step"] + offset
            df["graph_idx"] = (
                df["graph_idx"].astype(int).map(lambda i, ts=temps: TEMPS.index(ts[i]))
            )
            frames.append(df)
            last_step = int(df["step"].max())
            offset += last_step + LOG_EVERY
    return pd.concat(frames, ignore_index=True)


def load_multipart_csv(log_dir: Path, basename: str) -> pd.DataFrame:
    """Prefer single CSV (+ .partN chain); fall back to multi-GPU
    ``_t<subset>`` shards when no single CSV is present (slc_multi_gpu.sh
    layout).
    """
    df = _load_part_chain(log_dir, basename)
    if df is not None:
        return df
    return _load_multigpu_shards(log_dir, basename)


def _plot_panel(ax, df: pd.DataFrame, panel_key: str, colors) -> None:
    col, ylabel, ref, ref_label = _PANEL_SPECS[panel_key]
    per_graph_raw = []
    per_graph_rolling = []
    for i, T in enumerate(TEMPS):
        sub = df[df["graph_idx"] == i].sort_values("global_step")
        if sub.empty:
            continue
        raw = sub[col].to_numpy()
        rolling = pd.Series(raw).rolling(WINDOW, center=True).mean().to_numpy()
        per_graph_raw.append(raw)
        per_graph_rolling.append(rolling)
        ax.plot(sub["global_step"], raw, color=colors[i], alpha=0.22, lw=0.7)
        # Label T only on the temperature panel; other panels reuse the legend.
        label = f"{T} K" if panel_key == "temperature" else None
        ax.plot(sub["global_step"], rolling, color=colors[i], lw=1.6, label=label)

    refs_for_ylim = None
    if ref == "TEMPS":
        for T, c in zip(TEMPS, colors):
            ax.axhline(T, color=c, ls=":", lw=0.7, alpha=0.55)
        refs_for_ylim = list(TEMPS)
    elif isinstance(ref, (int, float)):
        line = ax.axhline(ref, color="black", ls=":", lw=0.9)
        ax.legend(
            [line], [ref_label.format(ref=ref)], loc="best", fontsize=9, framealpha=0.9
        )
        refs_for_ylim = [ref]

    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if per_graph_rolling:
        # Pressure is noisy relative to its rolling mean (raw cloud ~10^-3,
        # rolling ~10^-4) — using the rolling-mean Tukey fence clips the raw
        # cloud off the panel. Use raw values + a wider k=3 fence so the
        # cloud is visible. Other panels use the rolling mean to stay tight.
        if col == "pressure_eV_A3":
            ax.set_ylim(*_clip_ylim(per_graph_raw, col, refs=refs_for_ylim, k=3.0))
        else:
            ax.set_ylim(*_clip_ylim(per_graph_rolling, col, refs=refs_for_ylim))


def _print_tail_summary(df: pd.DataFrame, stage: str, basename: str) -> None:
    print(f"\nlast {WINDOW} samples per T  (basename={basename}):")
    if stage == "npt":
        print(
            f"{'T':>6}  {'Tact':>14}  {'ρ':>18}  {'P':>14}  {'E_pot':>14}  {'fmax':>10}"
        )
    else:  # nvt
        print(f"{'T':>6}  {'Tact':>14}  {'E_pot':>14}  {'fmax':>10}  {'P':>14}")
    for i, T in enumerate(TEMPS):
        sub = df[df["graph_idx"] == i].tail(WINDOW)
        if sub.empty:
            print(f"{T:>6}  (no rows)")
            continue
        ts = f"{sub['temperature'].mean():6.1f} ± {sub['temperature'].std():4.1f}"
        es = f"{sub['energy'].mean():+.3e}"
        fm = f"{sub['fmax'].mean():6.2f}"
        ps = f"{sub['pressure_eV_A3'].mean():+.2e}"
        if stage == "npt":
            rs = f"{sub['density_g_cm3'].mean():.3f} ± {sub['density_g_cm3'].std():.3f}"
            print(f"{T:>6}  {ts:>14}  {rs:>18}  {ps:>14}  {es:>14}  {fm:>10}")
        else:
            print(f"{T:>6}  {ts:>14}  {es:>14}  {fm:>10}  {ps:>14}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stage",
        required=True,
        choices=("nvt", "npt"),
        help="SLC sub-stage. nvt = pre-equilibration (fixed cell); "
        "npt = production sweep (variable cell).",
    )
    ap.add_argument(
        "--src",
        choices=["npt", "nvt", "auto"],
        default="auto",
        help="Melt-endpoint tag (matches driver --source). 'auto' prefers "
        "npt over nvt when both present.",
    )
    ap.add_argument(
        "--dt",
        type=float,
        default=0.5,
        help="MD timestep in fs (matches driver DT). Default 0.5.",
    )
    ap.add_argument(
        "--t-warmup",
        type=float,
        default=200.0,
        help="Warmup target temperature in K (matches the warmup driver "
        "--t-warmup; default 200). Used to build the CSV basename.",
    )
    ap.add_argument(
        "--total-ps",
        type=float,
        default=None,
        help="Expected stage duration in ps. Default depends on --stage "
        "(10 for nvt, 50 for npt). Drives x-axis extent and title totals.",
    )
    ap.add_argument(
        "--run-name",
        type=str,
        default="naphthalene_long",
        help="Run name; drives default --log-dir and --out paths.",
    )
    ap.add_argument(
        "--packmol",
        action="store_true",
        help="The SLC stack was generated via `slc.py --packmol` (single "
        "Packmol run with the warmup crystal as fixed obstacle). Adds the "
        "`packmol_melt_` tag to the CSV basename to match the slc.py "
        "artefact naming.",
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Default: logs/<run-name>/",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG. Default assets/<run-name>/figs/"
        "<output_stem>_from_{src}_{T_WARMUP_TAG}_{DT_TAG}.png.",
    )
    args = ap.parse_args()
    cfg = STAGE_CONFIG[args.stage]
    if args.log_dir is None:
        args.log_dir = HERE / "logs" / args.run_name
    if args.total_ps is None:
        args.total_ps = cfg["default_total_ps"]

    tag = dt_tag(args.dt)
    t_warmup_tag = f"{int(args.t_warmup)}k"
    src_tag = f"packmol_melt_from_{{}}" if args.packmol else "from_{}"

    def _stem_with_src(s: str) -> str:
        return f"{cfg['input_stem']}_{src_tag.format(s)}_{t_warmup_tag}_{tag}"

    # Tweak resolve_src to use the packmol-prefixed stem.
    if args.src in ("npt", "nvt"):
        candidates = (args.src,)
    else:
        candidates = ("npt", "nvt")
    src = None
    for s in candidates:
        if _has_data(args.log_dir, _stem_with_src(s)):
            src = s
            break
    if src is None:
        sys.exit(
            f"no {_stem_with_src('{npt,nvt}')}.csv* under {args.log_dir} -- "
            f"pull the remote logs first (see module docstring)"
        )
    basename = _stem_with_src(src)
    df = load_multipart_csv(args.log_dir, basename)

    n_total = int(args.total_ps * 1000 / args.dt)
    done = int(df["global_step"].max())
    done_ps = done * args.dt / 1000

    n_panels = len(cfg["panels"])
    fig_height = 2 * n_panels + 3  # 11 (4 panels), 13 (5 panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, fig_height), sharex=True)
    axes = list(axes)
    colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(TEMPS)))

    for ax, panel_key in zip(axes, cfg["panels"]):
        _plot_panel(ax, df, panel_key, colors)

    axes[0].legend(
        loc="center right", fontsize=9, framealpha=0.9, title="Target T", ncol=1
    )

    # Part-boundary guides at each cumulative offset (visible when extend
    # runs append .partN files).
    offsets = sorted(set(df["global_step"] - df["step"]))
    for off in offsets:
        if off == 0:
            continue
        for ax in axes:
            ax.axvline(off, color="gray", ls="--", lw=0.7, alpha=0.45)

    axes[0].set_title(
        f"SLC {args.stage.upper()} {cfg['title_phrase']} from {src.upper()} "
        f"endpoint (DT = {args.dt} fs) — live "
        f"({done} / {n_total} steps = {done_ps:.1f} / {args.total_ps:.0f} ps)"
    )
    axes[-1].set_xlabel("Global step")
    axes[-1].set_xlim(0, n_total)

    _print_tail_summary(df, args.stage, basename)

    src_out_tag = f"packmol_melt_from_{src}" if args.packmol else f"from_{src}"
    out = args.out or (
        HERE
        / "assets"
        / args.run_name
        / "figs"
        / f"{cfg['output_stem']}_{src_out_tag}_{t_warmup_tag}_{tag}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
