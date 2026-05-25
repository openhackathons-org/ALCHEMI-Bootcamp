"""Warmup MD diagnostic for the naphthalene_long warmup driver.

Reads ``logs/<run-name>/warmup_<stage>_<T_WARMUP_TAG>_<DT_TAG>.csv`` (plus
any ``.partN`` siblings from extend runs) for ``--stage ∈ {nvt, npt}`` and
renders a stacked-panel figure for the selected stage. Both stages share
the raw + rolling-mean (w=20) overlay convention; every panel plots the
raw trace at ``alpha=0.25`` with a centred rolling mean in bold.

NVT panels (4): temperature / potential energy / fmax / pressure. The
cell is fixed in NVT, so density and lattice are not plotted.

NPT panels (4): temperature / density+pressure (twin y-axes) / potential
energy / fmax. ρ and P are conjugate variables in NPT — the barostat
fluctuates V to drive ⟨P⟩ → P_target and ρ = m/V tracks V inversely — so
co-plotting them on a single panel exposes the response loop and frees a
row for the energy trace. If a per-frame extxyz has been pulled to
``assets/<run-name>/traj/warmup_npt_<TWARM>_<DT_TAG>.extxyz[.gz]`` (via
``pull_trajectory.py --stage npt``), an additional 5th panel plotting the
three cell-vector norms |a|, |b|, |c| vs step is appended. The trajectory
is parsed by streaming regex over the ``Lattice="..."`` header lines (no
ASE full-atom load), so the extra panel adds a few seconds even for
~400-frame gzipped trajectories.

Pull the CSVs off the remote container first, e.g.::

    source /tmp/alchemi-playbook-part2-deploy.env
    STAGE=npt        # or nvt
    TAG=dt0p5fs
    TWARM=200k
    ssh -J "$LOGIN_HOST" "$COMPUTE_NODE" \\
      "docker exec alchemi-playbook-part2 tar czf - -C /workspace/logs/naphthalene_long \\
         warmup_${STAGE}_${TWARM}_${TAG}.csv \\
         $(for p in 2 3 4 5; do printf 'warmup_%s_%s_%s.part%d.csv ' $STAGE $TWARM $TAG $p; done)" \\
      | tar xzf - -C dev/part-2-toolkit/logs/naphthalene_long

``tar`` silently drops missing members, so over-specifying ``.partN``
is safe. Run locally from ``dev/part-2-toolkit/``::

    python plot_warmup.py --stage {nvt,npt} [--dt 0.5] [--total-ps N]
                          [--t-warmup 200] [--log-dir logs/naphthalene_long]

``--total-ps`` defaults to a stage-aware value (30 for NVT, 75 for NPT).
"""

import argparse
import gzip
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent

LOG_EVERY = 100
WINDOW = 20
P_1ATM = 101325.0 / 1.602176634e11
# Default experimental density for naphthalene (Brock & Dunitz 1982, 295 K).
# Override via --rho-exp for other materials (e.g. 0.703 for liquid n-octane @ 298 K).
RHO_EXP_DEFAULT = 1.18
# Distinguishable colours for the three cell-vector norms (a, b, c) on the
# lattice panel. Distinct from the per-scalar palette below.
LATTICE_COLORS = ("#ef6c00", "#00838f", "#ad1457")

# Panel registry: (csv_col, ylabel, color, ref_value, ref_label_template).
# `ref_value == "T_TARGET"` is a sentinel resolved at runtime to the warmup
# target temperature; templates are formatted with T / RHO_EXP / P_1ATM.
_PANEL_SPECS = {
    "temperature": (
        "temperature",
        "Temperature (K)",
        "#c62828",
        "T_TARGET",
        "target = {T:.0f} K",
    ),
    "density": (
        "density_g_cm3",
        "Density (g/cm$^3$)",
        "#2e7d32",
        "RHO_EXP",
        "experimental = {RHO_EXP} g/cm$^3$",
    ),
    "pressure": (
        "pressure_eV_A3",
        "Pressure (eV/Å$^3$)",
        "#1565c0",
        P_1ATM,
        "1 atm = {P_1ATM:.2e} eV/Å$^3$",
    ),
    "energy": (
        "energy",
        "Potential energy (eV)",
        "#455a64",
        None,
        None,
    ),
    "fmax": (
        "fmax",
        "f$_{\\mathrm{max}}$ (eV/Å)",
        "#6a1b9a",
        None,
        None,
    ),
}

STAGE_CONFIG = {
    "nvt": {
        "default_total_ps": 30.0,
        "input_stem": "warmup_nvt",
        "output_stem": "nvt_thermalization",
        "title_phrase": "thermalization",
        "title_target": "{T:.0f} K",
        "panels": ("temperature", "energy", "fmax", "pressure"),
        "include_lattice": False,
    },
    "npt": {
        "default_total_ps": 75.0,
        "input_stem": "warmup_npt",
        "output_stem": "npt_equilibration",
        "title_phrase": "equilibration",
        "title_target": "{T:.0f} K / 1 atm",
        "panels": ("temperature", "density", "pressure", "energy", "fmax"),
        "include_lattice": True,
    },
}


def dt_tag(dt: float) -> str:
    return f"dt{str(dt).replace('.', 'p')}fs"


_LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
_STEP_RE = re.compile(r"\bstep=(\d+)")


def read_lattice_from_extxyz(path: Path) -> pd.DataFrame:
    """Parse per-frame cell-vector norms from the extxyz ``Lattice="..."`` header.

    Streams through the file line-by-line, capturing the 9-float lattice per
    frame along with the ``step=N`` info stamp (written by
    ``export_zarr_to_extxyz.py`` as ``i*SNAPSHOT_EVERY``). Falls back to
    ``frame_idx * LOG_EVERY`` if the info stamp is missing. Avoids ASE's
    full-atom parsing: ~seconds instead of ~tens of seconds for a 400-frame
    ~75 MB trajectory.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    frame_idx = 0
    with opener(path, "rt") as f:
        for line in f:
            m = _LATTICE_RE.search(line)
            if m is None:
                continue
            # ASE writes the cell as row-major 9 floats = [ax, ay, az, bx, by, bz, cx, cy, cz].
            cell = np.fromstring(m.group(1), sep=" ").reshape(3, 3)
            a, b, c = np.linalg.norm(cell, axis=1)
            step_match = _STEP_RE.search(line)
            step = int(step_match.group(1)) if step_match else frame_idx * LOG_EVERY
            rows.append({"step": step, "a": a, "b": b, "c": c})
            frame_idx += 1
    if not rows:
        return pd.DataFrame(columns=["step", "a", "b", "c"])
    return pd.DataFrame(rows)


def find_extxyz(run_name: str, basename: str) -> Path | None:
    """Locate a local extxyz trajectory for this run, if pulled via
    ``pull_trajectory.py``. Prefer the uncompressed file (faster read)."""
    traj_dir = HERE / "assets" / run_name / "traj"
    for ext in (".extxyz", ".extxyz.gz"):
        p = traj_dir / f"{basename}{ext}"
        if p.exists():
            return p
    return None


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


def _plot_panel(ax, df: pd.DataFrame, spec: tuple, t_target: float, rho_exp: float) -> None:
    col, ylabel, color, ref, ref_label = spec
    ax.plot(df["global_step"], df[col], color=color, alpha=0.25, lw=0.7)
    rolling = df[col].rolling(WINDOW, center=True).mean()
    ax.plot(df["global_step"], rolling, color=color, lw=1.8)
    if ref is not None:
        ref_value = (
            t_target if ref == "T_TARGET"
            else rho_exp if ref == "RHO_EXP"
            else ref
        )
        label = ref_label.format(T=t_target, RHO_EXP=rho_exp, P_1ATM=P_1ATM)
        ax.axhline(ref_value, color="black", ls=":", lw=0.9, label=label)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def _print_tail_summary(
    df: pd.DataFrame,
    stage: str,
    t_target: float,
    rho_exp: float,
    lattice_df: pd.DataFrame | None,
) -> None:
    last = df.tail(WINDOW)
    lines = [
        f"\nlast {min(WINDOW, len(df))} samples "
        f"(step {int(last['global_step'].min())} – {int(last['global_step'].max())}):",
        f"  T     = {last['temperature'].mean():.1f} ± {last['temperature'].std():.1f}  K  "
        f"(target {t_target:.0f})",
        f"  E_pot = {last['energy'].mean():.4e} ± {last['energy'].std():.2e}  eV",
        f"  fmax  = {last['fmax'].mean():.2f} ± {last['fmax'].std():.2f}  eV/Å",
        f"  P     = {last['pressure_eV_A3'].mean():+.2e} ± {last['pressure_eV_A3'].std():.2e}  eV/Å³  "
        f"(target {P_1ATM:.2e})",
    ]
    if stage == "npt":
        lines.append(
            f"  ρ     = {last['density_g_cm3'].mean():.3f} ± {last['density_g_cm3'].std():.3f}  g/cm³  "
            f"(target {rho_exp})"
        )
        lines.append(f"  V     = {last['volume_A3'].mean():.1f}  Å³")
    else:  # nvt
        lines.append(
            f"  ρ     = {last['density_g_cm3'].mean():.4f}  g/cm³  (NVT: fixed)"
        )
        lines.append(f"  V     = {last['volume_A3'].mean():.1f}  Å³    (NVT: fixed)")
    if lattice_df is not None:
        last_lat = lattice_df.tail(WINDOW)
        lines.append(
            f"  |a|   = {last_lat['a'].mean():.2f} ± {last_lat['a'].std():.2f}  Å"
        )
        lines.append(
            f"  |b|   = {last_lat['b'].mean():.2f} ± {last_lat['b'].std():.2f}  Å"
        )
        lines.append(
            f"  |c|   = {last_lat['c'].mean():.2f} ± {last_lat['c'].std():.2f}  Å"
        )
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stage",
        required=True,
        choices=("nvt", "npt"),
        help="Warmup sub-stage to plot. nvt = thermalization (fixed cell); "
        "npt = equilibration (variable cell, ρ+P twin-axis panel).",
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
        help="Warmup target temperature in K (default 200). Must match the "
        "value passed to warmup.py so this script finds the right CSV / "
        "trajectory.",
    )
    ap.add_argument(
        "--total-ps",
        type=float,
        default=None,
        help="Expected stage duration in ps. Default depends on --stage "
        "(30 for nvt, 75 for npt). Drives x-axis extent and title totals; "
        "pass smaller values for the shortened Ewald/2025 driver budgets.",
    )
    ap.add_argument(
        "--run-name",
        type=str,
        default="naphthalene_long",
        help="Run name; drives default --log-dir and --out paths.",
    )
    ap.add_argument(
        "--log-dir", type=Path, default=None, help="Default: logs/<run-name>/"
    )
    ap.add_argument(
        "--rho-exp",
        type=float,
        default=RHO_EXP_DEFAULT,
        help=f"Experimental density (g/cm³) for the reference line on the "
        f"density panel. Default {RHO_EXP_DEFAULT} (naphthalene, Brock & Dunitz "
        f"1982, 295 K). For other materials pass the literature value, e.g. "
        f"0.703 for liquid n-octane @ 298 K.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG. Default assets/<run-name>/figs/<output_stem>_<TWARM>_<DT_TAG>.png.",
    )
    args = ap.parse_args()
    cfg = STAGE_CONFIG[args.stage]
    if args.log_dir is None:
        args.log_dir = HERE / "logs" / args.run_name
    if args.total_ps is None:
        args.total_ps = cfg["default_total_ps"]

    tag = dt_tag(args.dt)
    t_warmup_tag = f"{int(args.t_warmup)}k"
    T_TARGET = args.t_warmup
    basename = f"{cfg['input_stem']}_{t_warmup_tag}_{tag}"
    df = load_multipart_csv(args.log_dir, basename)

    n_total = int(args.total_ps * 1000 / args.dt)

    # NPT only: try to load per-frame lattice from a pulled extxyz.
    # Silent fallback to no-lattice layout if no trajectory is locally available.
    lattice_df: pd.DataFrame | None = None
    if cfg["include_lattice"]:
        extxyz_path = find_extxyz(args.run_name, basename)
        if extxyz_path is not None:
            print(f"Reading lattice dimensions from {extxyz_path} ...")
            lattice_df = read_lattice_from_extxyz(extxyz_path)
            if lattice_df.empty:
                print("  (parsed 0 frames -- skipping lattice panel)")
                lattice_df = None
            else:
                print(
                    f"  {len(lattice_df)} frames, step range "
                    f"{int(lattice_df['step'].min())}..{int(lattice_df['step'].max())}"
                )

    panel_keys = list(cfg["panels"])
    if lattice_df is not None:
        panel_keys.append("lattice")
    n_panels = len(panel_keys)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes_grid = plt.subplots(
        n_rows, n_cols, figsize=(14, 3.2 * n_rows + 1.5), sharex=True
    )
    axes = axes_grid.flatten().tolist()

    for ax, panel_key in zip(axes, panel_keys):
        if panel_key == "lattice":
            for col, color in zip(("a", "b", "c"), LATTICE_COLORS):
                ax.plot(
                    lattice_df["step"], lattice_df[col],
                    color=color, alpha=0.25, lw=0.7,
                )
                rolling = lattice_df[col].rolling(WINDOW, center=True).mean()
                ax.plot(
                    lattice_df["step"], rolling,
                    color=color, lw=1.8, label=f"|{col}|",
                )
            ax.set_ylabel("Cell-vector norm (Å)")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=9, framealpha=0.9)
        else:
            _plot_panel(ax, df, _PANEL_SPECS[panel_key], T_TARGET, args.rho_exp)

    # Hide leftover grid cells when n_panels is odd.
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    offsets = sorted(set(df["global_step"] - df["step"]))
    for off in offsets:
        if off == 0:
            continue
        for ax in axes[:n_panels]:
            ax.axvline(off, color="gray", ls="--", lw=0.7, alpha=0.45)

    title_target = cfg["title_target"].format(T=T_TARGET)
    fig.suptitle(
        f"{args.stage.upper()} {title_target} {cfg['title_phrase']} "
        f"(DT = {args.dt} fs)",
        fontsize=13,
    )
    # sharex=True suppresses x labels on non-bottom rows; bottom-row cells
    # need the label + xlim. With odd n_panels the last cell is hidden, so
    # the cell directly above it also becomes a column's bottom; add the
    # label there too.
    bottom_row_indices = set()
    for col in range(n_cols):
        for row in range(n_rows - 1, -1, -1):
            idx = row * n_cols + col
            if idx < n_panels:
                bottom_row_indices.add(idx)
                break
    for idx in bottom_row_indices:
        axes[idx].set_xlabel("Global step")
        axes[idx].tick_params(axis="x", labelbottom=True)
    for ax in axes[:n_panels]:
        ax.set_xlim(0, n_total)

    _print_tail_summary(df, args.stage, T_TARGET, args.rho_exp, lattice_df)

    out = args.out or (
        HERE
        / "assets"
        / args.run_name
        / "figs"
        / f"{cfg['output_stem']}_{t_warmup_tag}_{tag}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
