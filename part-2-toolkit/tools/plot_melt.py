"""NVT melt thermalization diagnostic for the naphthalene_long pipeline.

Reads ``logs/naphthalene_long/melt_nvt_500k_from_{src}_{DT_TAG}.csv``
(plus any ``.partN`` siblings) and produces a 3-panel figure analogous
to ``assets/figs/melt_nvt_thermalization_from_*.png`` -- temperature,
fmax, pressure vs global step, raw at ``alpha=0.25`` + rolling-mean
(w=20) overlay in bold. ``--src`` selects which warmup endpoint seeded
the melt (``npt`` = melt from NPT-equilibrated solid, ``nvt`` = melt-in-
box from unexpanded NVT crystal, ``auto`` = prefer npt). Promoted from
the throwaway ``/tmp/claude/naph_nvt/plot_melt.py`` prototype.

Pull the CSVs off the remote container first, e.g.::

    source /tmp/alchemi-playbook-part2-deploy.env
    SRC=npt; TAG=dt0p5fs
    ssh -J "$LOGIN_HOST" "$COMPUTE_NODE" \\
      "docker exec alchemi-playbook-part2 tar czf - -C /workspace/logs/naphthalene_long \\
         melt_nvt_500k_from_${SRC}_${TAG}.csv \\
         $(for p in 2 3; do printf 'melt_nvt_500k_from_%s_%s.part%d.csv ' $SRC $TAG $p; done)" \\
      | tar xzf - -C dev/part-2-toolkit/logs/naphthalene_long

Usage::

    python plot_melt.py [--src {npt,nvt,auto}] [--dt 0.5]
                        [--log-dir logs/naphthalene_long]
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent

LOG_EVERY = 100
WINDOW = 20
T_TARGET = 500.0
P_1ATM = 101325.0 / 1.602176634e11


def dt_tag(dt: float) -> str:
    return f"dt{str(dt).replace('.', 'p')}fs"


def resolve_src(log_dir: Path, dt_suffix: str, t_warmup_tag: str, src_arg: str) -> str:
    def present(s: str) -> bool:
        stem = f"melt_nvt_500k_from_{s}_{t_warmup_tag}_{dt_suffix}"
        return (log_dir / f"{stem}.csv").exists() \
            or bool(list(log_dir.glob(f"{stem}.part*.csv")))

    if src_arg in ("npt", "nvt"):
        if not present(src_arg):
            sys.exit(
                f"no melt_nvt_500k_from_{src_arg}_{t_warmup_tag}_{dt_suffix}.csv* in {log_dir}"
            )
        return src_arg
    for s in ("npt", "nvt"):
        if present(s):
            return s
    sys.exit(
        f"no melt_nvt_500k_from_{{npt,nvt}}_{t_warmup_tag}_{dt_suffix}.csv* under {log_dir} -- "
        "pull the remote logs first (see module docstring)"
    )


def load_multipart_csv(log_dir: Path, basename: str) -> pd.DataFrame:
    bare = log_dir / f"{basename}.csv"
    extras = []
    for p in log_dir.glob(f"{basename}.part*.csv"):
        m = re.search(r"\.part(\d+)$", p.stem)
        if m:
            extras.append((int(m.group(1)), p))
    extras.sort(key=lambda t: t[0])
    parts = ([bare] if bare.exists() else []) + [p for _, p in extras]
    if not parts:
        sys.exit(f"no {basename}.csv* in {log_dir}")

    frames = []
    offset = 0
    for p in parts:
        df = pd.read_csv(p)
        df["global_step"] = df["step"] + offset
        frames.append(df)
        last_step = int(df["step"].max())
        offset += last_step + LOG_EVERY
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", choices=["npt", "nvt", "auto"], default="auto")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--t-warmup", type=float, default=200.0,
                    help="Warmup target temperature in K (default 200). Tags the "
                         "input CSV stem as 'melt_nvt_500k_from_{src}_{int(T)}k_{DT_TAG}'; "
                         "must match the value passed to melt.py.")
    ap.add_argument("--total-ps", type=float, default=15.0,
                    help="Expected melt NVT total duration in ps (default 15, matching "
                         "melt.py --melt-ps default). Sets the x-axis "
                         "upper bound and the title's denominator.")
    ap.add_argument("--run-name", type=str, default="naphthalene_long",
                    help="Run name; drives default --log-dir and --out paths.")
    ap.add_argument("--log-dir", type=Path, default=None,
                    help="Default: logs/<run-name>/")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.log_dir is None:
        args.log_dir = HERE / "logs" / args.run_name

    tag = dt_tag(args.dt)
    t_warmup_tag = f"{int(args.t_warmup)}k"
    src = resolve_src(args.log_dir, tag, t_warmup_tag, args.src)
    basename = f"melt_nvt_500k_from_{src}_{t_warmup_tag}_{tag}"
    df = load_multipart_csv(args.log_dir, basename)

    n_total = int(args.total_ps * 1000 / args.dt)
    done = int(df["global_step"].max())
    done_ps = done * args.dt / 1000

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    panels = [
        (axes[0], "temperature",    "Temperature (K)",     "#ef6c00",
         T_TARGET, f"target = {T_TARGET:.0f} K"),
        (axes[1], "fmax",           "f$_{\\mathrm{max}}$ (eV/Å)", "#6a1b9a",
         None,     None),
        (axes[2], "pressure_eV_A3", "Pressure (eV/Å$^3$)", "#1565c0",
         P_1ATM,   f"1 atm = {P_1ATM:.2e} eV/Å$^3$"),
    ]
    for ax, col, ylabel, color, ref, ref_label in panels:
        ax.plot(df["global_step"], df[col],
                color=color, alpha=0.25, lw=0.7, label="raw (100-step stride)")
        rolling = df[col].rolling(WINDOW, center=True).mean()
        ax.plot(df["global_step"], rolling,
                color=color, lw=1.8, label=f"rolling mean (w={WINDOW})")
        if ref is not None:
            ax.axhline(ref, color="black", ls=":", lw=0.9, label=ref_label)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

    offsets = sorted(set(df["global_step"] - df["step"]))
    for off in offsets:
        if off == 0:
            continue
        for ax in axes:
            ax.axvline(off, color="gray", ls="--", lw=0.7, alpha=0.45)

    axes[0].set_title(
        f"NVT {T_TARGET:.0f} K melt thermalization from {src.upper()} endpoint "
        f"(DT = {args.dt} fs) — live "
        f"({done} / {n_total} steps = {done_ps:.1f} / {args.total_ps:.0f} ps)"
    )
    axes[-1].set_xlabel("Global step")
    axes[-1].set_xlim(0, n_total)

    last = df.tail(WINDOW)
    print(
        f"\nlast {min(WINDOW, len(df))} samples "
        f"(step {int(last['global_step'].min())} – {int(last['global_step'].max())}):\n"
        f"  T    = {last['temperature'].mean():.1f} ± {last['temperature'].std():.1f}  K  "
        f"(target {T_TARGET:.0f})\n"
        f"  fmax = {last['fmax'].mean():.2f} ± {last['fmax'].std():.2f}  eV/Å\n"
        f"  P    = {last['pressure_eV_A3'].mean():+.3e} ± {last['pressure_eV_A3'].std():.2e}  eV/Å³\n"
        f"  ρ    = {last['density_g_cm3'].mean():.4f}  g/cm³  (NVT: fixed)\n"
        f"  V    = {last['volume_A3'].mean():.1f}  Å³    (NVT: fixed)"
    )

    out = args.out or (HERE / "assets" / args.run_name / "figs" / f"melt_nvt_thermalization_from_{src}_{t_warmup_tag}_{tag}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
