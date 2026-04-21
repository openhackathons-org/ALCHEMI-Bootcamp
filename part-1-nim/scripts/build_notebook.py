#!/usr/bin/env python3
"""Build the AWH water-sorbent tutorial notebook from source cells.

Regenerates ``part-1-nim/alchemi-mace-water-sorbents.ipynb``. Run from the
part-1-nim/ directory:

    python3 scripts/build_notebook.py

The notebook is committed, so this script only needs to be re-run when
cell content changes. Keeping the source in a single Python file keeps
diffs reviewable (ipynb JSON is noisy).
"""

from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


cells: list[dict] = []


# -- Section 1 --------------------------------------------------------------
cells.append(md(
    """# Atmospheric Water Harvesting: MACE-MP-0 Screening with NVIDIA ALCHEMI

*Interactive workshop - approximately 60 min on a single NVIDIA A100.*

This notebook teaches two things at once:

1. **A discovery workflow** for the atmospheric water harvesting (AWH) problem. We relax H2O adsorbed on six inorganic sorbent frameworks and rank them by binding energy, validating against published DFT and CCSD(T)/CBS reference data wherever it exists and stating the uncertainty explicitly where it does not.
2. **The NVIDIA ALCHEMI Batch Geometry Relaxation (BGR) NIM** as a working tool. Every relaxation below is a single HTTP call to a local NIM container running the MACE-MPA-0 foundation model with DFT-D3(BJ) dispersion. The notebook also doubles as a hands-on introduction to the NIM wire protocol, batch parallelism, active-mask constraints, and Prometheus/Grafana observability.

We write in scientifically honest prose: "agrees within the published sub-category MAD" rather than "agrees with DFT". MACE-MP-0's limitations (12 A receptive-field cutoff, no spin, MPtrj gaps for gas-phase molecules and surface slabs) are stated at the points where they bite.
"""
))

# -- Section 2 --------------------------------------------------------------
cells.append(md(
    """## Why AWH, and why inorganic sorbents

Atmospheric water harvesting addresses a real problem (water scarcity in arid regions) with a real commercial deployment pipeline (MOF-303, MOF-801, AQSOA sorbents in field devices). The MOF champions, however, fall outside MACE-MP-0's training distribution: the MPtrj dataset that MACE was trained on contains no metal-organic frameworks and no gas-phase molecules. Running a zero-shot foundation model outside its training distribution is the fastest way to get an overconfident wrong answer, so we deliberately **exclude MOFs** and focus on the **inorganic sorbent tier**, which is in-distribution and experimentally well-characterised:

- Zeolite frameworks (chabazite, MFI/silicalite, SAPO-34)
- Ionic-class oxides (alpha-Al2O3, rutile TiO2, monoclinic ZrO2)

All six hosts are **closed-shell singlets**, with no magnetic 3d metal oxides, no reducible cations (no Ti3+ / Ce3+ chemistry), no lanthanides or actinides. This is the scope where MACE-MP-0 was validated in Batatia 2024 (paper's S24 and A.31 benchmark panels) and where we can responsibly run it.

## The six-host panel

| # | Host | S24 class | Validation tier | Expected E_ads (kJ/mol) |
|---|------|-----------|-----------------|-------------------------|
| 1 | H-SAPO-34 (CHA topology) | Zeolite-analogue | Tier 3 (Fischer 2015 CP2K) + commercial (AQSOA FAM-Z02) | -55 to -75 |
| 2 | H-CHA (Al-substituted chabazite) | Zeolite | Tier 1 (S24 MAD 229 meV) | -50 to -65 |
| 3 | H-MFI (silicalite) | Zeolite (not in S24) | Tier 2 (Plessow 2024 CCSD(T)/CBS) | -55 to -65 |
| 4 | alpha-Al2O3(0001) | Ionic | Tier 1 (S24 MAD 361 meV) | -70 to -90 |
| 5 | TiO2(110) rutile | Ionic | Tier 1 (S24 MAD 361 meV) | -70 to -90 |
| 6 | ZrO2(-1,1,1) monoclinic | Ionic (no checkpoint) | Tier 4 (MACE only - candidate for the lab) | open |

The validation spine is the S24 benchmark in Batatia et al. 2024 (arXiv:2401.00096), supplemented by Plessow 2024 CCSD(T)/CBS and Fischer 2015 CP2K PBE-D3 for the zeolite hosts that are not in S24.
"""
))

# -- Section 3 --------------------------------------------------------------
cells.append(md(
    """## What is ALCHEMI, and what does the BGR NIM do

The **ALCHEMI Batch Geometry Relaxation (BGR) NIM** is a containerised inference service that wraps a geometry-optimisation loop around a machine-learning interatomic potential (MLIP) and exposes it over HTTP. The Docker Compose stack shipped with this tutorial pulls `nvcr.io/nim/nvidia/alchemi-bgr:1.0.0`, which by default runs the **MACE-MPA-0** foundation model (MACE trained on MPtrj plus sAlex) with **DFT-D3(BJ)** dispersion corrections enabled. Every relaxation in this notebook is a single POST to `localhost:8000/infer` carrying one or more `BGRAtomicData` structures; the NIM returns optimised coordinates, energies, forces, and stresses.

The key practical capabilities exercised below:

- **Batch parallelism** - a single HTTP request carries a list of N independent structures, and the NIM runs them through the model in one batched forward pass. This is what the "hello-world" cell a few rows down demonstrates: we send 1, then 128, then thousands of H2O molecules in a single call and measure how throughput scales.
- **Active-mask constraints** - each structure can specify per-atom boolean flags to freeze selected atoms during relaxation. We use this to freeze the bottom half of each oxide slab (a standard idiom for preventing spurious bulk rearrangement).
- **Periodic boundary conditions** - the NIM is configured with PBC on; zeolite bulks and oxide slabs both rely on it.
- **Dispersion on the server** - when `ALCHEMI_NIM_DFT3_ENABLED=true`, the NIM adds a post-hoc D3(BJ) correction to the MLIP energy and forces (matching the MACE-MP-0 training protocol).
- **Observability** - Prometheus scrapes `/v1/metrics`; Grafana at `localhost:3000` shows live GPU, request, and queue metrics. Useful for the batch-scaling study below.

FAST_DEMO mode replaces every live BGR call with a cached JSON reply so the notebook replays fully offline.
"""
))

# -- Section 4 --------------------------------------------------------------
cells.append(md(
    """## Scope caveats (read these first)

- **MACE-MP-0 receptive field is 12 A.** Long-range dispersion beyond that cutoff is not captured; H2O clustering beyond the first adsorbate shell is out of scope.
- **No explicit spin.** Every host here is a closed-shell singlet. Magnetic 3d metal oxides, reducible cations, and f-electron systems are explicitly excluded.
- **MPtrj training gaps.** MPtrj contains no gas-phase molecules, no surface slabs, and no MOFs. We expect some systematic offset in gas-phase H2O reference energies, and we calibrate that offset by computing E_ads as a difference (cancels the gas-phase error to leading order).
- **No free energy / entropy.** E_ads here is electronic-energy-only. Thermal and configurational corrections require MD + thermodynamic integration; out of tutorial scope.
- **No VASP in this notebook.** We compare against published DFT/CC numbers rather than running DFT ourselves. The only compute engine in the notebook is MACE-MPA-0 via the BGR NIM.

Whenever MACE and a reference disagree, we report the delta in meV **and** as a fraction of the published S24 sub-category MAD (Zeolite 229 meV, Ionic 361 meV), never as "MACE agrees with DFT".
"""
))

# -- Section 5 --------------------------------------------------------------
cells.append(md(
    """---

## Control panel

Edit once; all downstream cells read from here."""
))

cells.append(code(
    """import os

# FAST_DEMO: when True, every BGR call is replaced by a cached JSON response.
# Useful for offline replay and for workshops where no GPU is available.
FAST_DEMO = False

# Local paths
OUTPUT_DIR = "outputs"
CACHE_DIR = os.path.join("cached_responses", "water-sorbents")
ASSETS_DIR = "assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# BGR NIM endpoint. The Docker Compose stack binds the NIM to localhost:8000.
BGR_SERVER = os.environ.get("BGR_SERVER", "http://localhost:8000")

# Geometry-optimisation convergence tolerance (eV/A on maximum force).
# The NIM's 'materials' preset defaults to 0.05; we'll pass opttol=None
# to honour that default unless a cell needs something tighter.
OPTTOL = None

print(f"FAST_DEMO  : {FAST_DEMO}")
print(f"BGR_SERVER : {BGR_SERVER}")
print(f"CACHE_DIR  : {CACHE_DIR}")
"""
))

# -- Section 6 --------------------------------------------------------------
cells.append(md(
    """## Package versions and imports"""
))

cells.append(code(
    """import sys
import ase
import numpy as np
import pandas as pd
import matplotlib
import pymatgen

print(f"Python     : {sys.version.split()[0]}")
print(f"ase        : {ase.__version__}")
print(f"numpy      : {np.__version__}")
print(f"pandas     : {pd.__version__}")
print(f"matplotlib : {matplotlib.__version__}")
print(f"pymatgen   : {pymatgen.__version__}")
"""
))

cells.append(code(
    """from helpers import (
    # BGR client + cache
    check_endpoint,
    run_bgr_or_load_cache,
    async_run_bgr_or_load_cache,
    # Data models
    BGRAtomicData, BGRReply, OptimizationResult,
    ase_to_atomic_data, atomic_data_to_ase,
    # Host builders
    build_siliceous_cha, build_siliceous_mfi,
    build_h_cha, build_h_mfi, build_h_sapo34,
    build_alpha_alumina_0001_slab, build_tio2_110_slab, build_zro2_m111_slab,
    # Adsorbate + slab helpers
    build_adsorbate, place_adsorbate, make_active_mask,
    find_central_site,
    # Energy + displacement analysis
    compute_adsorption_energy, compute_surface_displacement,
    classify_relaxation,
    # Visualisation
    render_structure_ovito, create_interactive_view, display_widgets_row,
    display_inline, structure_summary_table,
    # Throughput scan
    measure_batch_throughput, sweep_batch_throughput, plot_throughput,
    # Constants
    KJ_MOL_TO_EV, EV_TO_KJ_MOL,
)
print("helpers imported OK")
"""
))

# -- Section 7: endpoint + metadata -----------------------------------------
cells.append(md(
    """## Endpoint check and NIM metadata

The first NIM call: confirm the server is up and print the runtime metadata (model, version, dispersion flag). This doubles as our source of truth for *which* MACE variant is live - no hard-coded assumptions in the notebook.
"""
))

cells.append(code(
    """import requests

BGR_LIVE = check_endpoint(BGR_SERVER) if not FAST_DEMO else False
print(f"BGR endpoint live: {BGR_LIVE}")

# Try to pull runtime metadata. The NIM exposes /v1/status; if the
# container ever adds /v1/metadata we will prefer that.
if BGR_LIVE:
    for path in ("/v1/metadata", "/v1/status", "/v1/models"):
        try:
            r = requests.get(BGR_SERVER + path, timeout=5)
            if r.ok and r.headers.get("content-type", "").startswith("application/json"):
                meta = r.json()
                print(f"GET {path}:")
                for k, v in (meta.items() if isinstance(meta, dict) else []):
                    print(f"  {k}: {v}")
                break
        except requests.RequestException:
            continue
    else:
        print("No metadata endpoint responded; check docker-compose logs for the deployed model.")
else:
    print("FAST_DEMO or endpoint down - skipping live metadata query.")
"""
))

# -- Hello-world 6a ---------------------------------------------------------
cells.append(md(
    """---

## Hello-world (6a): a single gas-phase H2O

Send one water molecule through the BGR NIM to (i) confirm the wire protocol and (ii) extract the gas-phase reference energy E(H2O) that will appear in every E_ads later. The H2O sits in a 15 A vacuum cube; the NIM is in PBC mode, so a box is required even for gas-phase calculations.
"""
))

cells.append(code(
    """def gas_phase_h2o_atoms(box: float = 15.0) -> ase.Atoms:
    \"\"\"Return a single water molecule centred in a cubic vacuum box.\"\"\"
    h2o = build_adsorbate("H2O")
    h2o.set_cell(np.eye(3) * box)
    h2o.set_pbc(True)
    # centre the oxygen (which is at origin from build_adsorbate) in the box
    h2o.translate(np.array([box / 2.0, box / 2.0, box / 2.0]))
    return h2o


h2o = gas_phase_h2o_atoms()
print(f"Atoms: {len(h2o)}  |  Cell: {h2o.cell.lengths()}  |  PBC: {h2o.pbc.tolist()}")

reply_h2o = run_bgr_or_load_cache(
    [ase_to_atomic_data(h2o, structure_id="gas_h2o")],
    server_url=BGR_SERVER,
    cache_dir=CACHE_DIR,
    label="gas_h2o",
    endpoint_live=BGR_LIVE,
    opttol=OPTTOL,
)
opt = reply_h2o.atoms[0]
E_H2O_gas = float(opt.energy)
print(f"E(H2O, gas) = {E_H2O_gas:.4f} eV   |  converged={opt.converged}  |  steps={opt.num_optimization_steps}")
"""
))

# -- Hello-world 6b ---------------------------------------------------------
cells.append(md(
    """## Hello-world (6b): 128 waters in one call

Same structure, now sent as a batch of 128. The HTTP round-trip cost is paid once; the NIM runs all 128 through the model in a single batched forward pass. We expect structures/sec to be orders of magnitude higher than the N=1 case.
"""
))

cells.append(code(
    """r128 = measure_batch_throughput(
    gas_phase_h2o_atoms(),
    batch_size=128,
    server_url=BGR_SERVER,
    opttol=OPTTOL,
) if BGR_LIVE else {"batch_size": 128, "wall_time_s": float("nan"),
                    "n_atoms_total": 128 * 3,
                    "struct_per_s": float("nan"),
                    "atoms_per_s": float("nan"),
                    "success": False,
                    "error": "FAST_DEMO or endpoint down"}

print(f"N={r128['batch_size']}  t={r128['wall_time_s']:.2f}s  "
      f"throughput={r128['struct_per_s']:.1f} struct/s  "
      f"({r128['atoms_per_s']:.1f} atoms/s)")
"""
))

# -- Hello-world 6c ---------------------------------------------------------
cells.append(md(
    """## Hello-world (6c): doubling sweep to the NIM ceiling

Sweep `N` across `{1, 2, 4, ..., 2^k}` until the NIM refuses the batch (OOM, timeout, server cap) or throughput plateaus. The ceiling is an **empirical finding** for the A100 / NIM combination in this deployment - we do not hard-code it.

The sweep caches its results in `cached_responses/water-sorbents/throughput_sweep.json`; in FAST_DEMO mode the cache is read directly and the call skipped.
"""
))

cells.append(code(
    """# Doubling series; cap at 2^14 = 16384 to keep tutorial-added compute
# under a few minutes on an A100. Raise the cap if you want to chase
# the real ceiling.
SIZES = [2 ** k for k in range(0, 15)]  # 1, 2, 4, ..., 16384

throughput_cache = os.path.join(CACHE_DIR, "throughput_sweep.json")
results = sweep_batch_throughput(
    gas_phase_h2o_atoms(),
    sizes=SIZES,
    server_url=BGR_SERVER,
    cache_path=throughput_cache,
    endpoint_live=BGR_LIVE,
    stop_on_failure=True,
    opttol=OPTTOL,
)

# Pretty table
df = pd.DataFrame(results)
cols = ["batch_size", "wall_time_s", "n_atoms_total", "struct_per_s", "atoms_per_s", "success"]
df[cols]
"""
))

# -- Hello-world 6d ---------------------------------------------------------
cells.append(md(
    """## Hello-world (6d): throughput figure

Two-panel log-log plot. Small-N is overhead-dominated (HTTP round-trip + Python client); large-N is GPU-compute-dominated; the right-hand edge is where the NIM refuses the batch.
"""
))

cells.append(code(
    """fig_path = os.path.join(ASSETS_DIR, "throughput_scaling.png")
plot_throughput(results, output_path=fig_path,
                title="BGR NIM batch-throughput scaling - single H2O per structure")
display_inline(fig_path)
print(f"Saved: {os.path.abspath(fig_path)}")
"""
))

# -- Hello-world 6e ---------------------------------------------------------
cells.append(md(
    """## Hello-world (6e): what the curve tells us

- At **N=1** the wall time is HTTP overhead plus one MLIP forward-and-optimise. Throughput (structures/sec) is low because the fixed overhead dominates.
- Between N and N\' the throughput ramps up linearly on log-log axes: the NIM is amortising overhead across the batch.
- Past a crossover the throughput plateaus and then collapses. The plateau is the GPU-compute-limited regime - work per call is linear in N, so structures/sec becomes constant. The collapse is either an OOM error or a server-side batch cap.

The practical takeaway for the six-host panel that follows: **we can relax all six hosts-plus-water in a single BGR call**. Screening 100 or 1000 hosts from a larger library is still one call, not 1000. This is the mechanism that turns a multi-hour serial DFT sweep into a single minute-scale NIM call - and it is what makes MLIP-based sorbent screening tractable as a discovery tool.

The rest of the notebook drops from structures-per-second into scientific-discovery mode: build six realistic host frameworks, place H2O on each, relax them all as one batch, compare to published DFT and CCSD(T)/CBS numbers, and flag the one host without a reference as a candidate for experimental follow-up.
"""
))

# ---------------------------------------------------------------------------
# Section 7: Host construction
# ---------------------------------------------------------------------------

cells.append(md(
    """---

## Six-host panel: construction and visualisation

Build all six hosts from reproducible recipes, then inspect each as an interactive 3-D OVITO widget. The zeolites are bulk 3-D frameworks (H2O will be placed *inside* the pore network, not on a cleaved surface); the oxides are slabs with vacuum (H2O sits on the top surface).
"""
))

cells.append(code(
    """HOSTS: dict[str, ase.Atoms] = {
    # Zeolites (3-D bulk frameworks, no slab/vacuum)
    "H-CHA":       build_h_cha(),
    "H-SAPO-34":   build_h_sapo34(),
    "H-MFI":       build_h_mfi(),
    # Oxide slabs (vacuum along c-axis)
    "Al2O3(0001)": build_alpha_alumina_0001_slab(min_slab_size=8.0, min_vacuum_size=15.0),
    "TiO2(110)":   build_tio2_110_slab(min_slab_size=8.0, min_vacuum_size=15.0, supercell=(2, 2, 1)),
    "ZrO2(-1,1,1)": build_zro2_m111_slab(min_slab_size=8.0, min_vacuum_size=15.0),
}

# Tier labels (for later reporting)
TIER = {
    "H-CHA": 1, "Al2O3(0001)": 1, "TiO2(110)": 1,
    "H-MFI": 2,
    "H-SAPO-34": 3,
    "ZrO2(-1,1,1)": 4,
}

for name, atoms in HOSTS.items():
    comp = {s: atoms.get_chemical_symbols().count(s) for s in sorted(set(atoms.get_chemical_symbols()))}
    formula = " ".join(f"{s}{n}" for s, n in comp.items())
    print(f"  {name:<14}  atoms={len(atoms):>4}  cell={atoms.cell.lengths().round(2).tolist()}  {formula}")
"""
))

cells.append(md(
    """### Closed-shell / no-f / no-magnetic sanity check

MACE-MP-0 is validated for closed-shell singlets with no magnetic 3d oxides, no reducible cations (Ti3+/Ce3+), and no f-electron systems. Every host in the panel must clear these element-set constraints before we touch the NIM.
"""
))

cells.append(code(
    """F_BLOCK = set(range(57, 72)) | set(range(89, 104))  # Ln + An
MAGNETIC_3D = {"V", "Cr", "Mn", "Fe", "Co", "Ni"}
REDUCIBLE = {"Ce", "Eu", "Sm", "Tb", "Yb"}  # common reducible +3/+4 lanthanides

ok = True
for name, atoms in HOSTS.items():
    numbers = set(atoms.numbers.tolist())
    symbols = set(atoms.get_chemical_symbols())
    offending = {s for s in symbols if s in MAGNETIC_3D or s in REDUCIBLE}
    f_nums = numbers & F_BLOCK
    if offending or f_nums:
        print(f"  FAIL {name}: magnetic/reducible={offending}  f-block Z={f_nums}")
        ok = False
    else:
        print(f"  OK   {name}: {sorted(symbols)}")

assert ok, "At least one host violates the closed-shell / no-f / no-magnetic scope."
"""
))

cells.append(md(
    """### Per-host summary table"""
))

cells.append(code(
    """rows = []
for name, atoms in HOSTS.items():
    row = structure_summary_table(atoms).iloc[0].to_dict()
    row = {"Host": name, "Tier": TIER[name], **row}
    rows.append(row)
pd.DataFrame(rows)
"""
))

cells.append(md(
    """### Interactive 3-D views

OVITO widgets (drag to rotate, scroll to zoom). Oxide slabs show their vacuum gap; zeolites show their full 3-D pore networks.
"""
))

cells.append(code(
    """display_widgets_row(
    [(name, atoms) for name, atoms in HOSTS.items()],
    width="260px", height="260px",
)
"""
))


# ---------------------------------------------------------------------------
# Section 8: Bulk cell-optimisation for the oxide hosts
# ---------------------------------------------------------------------------

cells.append(md(
    """---

## Bulk cell optimisation for the three oxide hosts

Before cleaving slabs we want to know whether MACE-MPA-0 reproduces the published experimental lattice parameters for each oxide bulk. This is the Batatia 2024 Fig. A.6-style sanity check: tight agreement on bulk lattice constants is a prerequisite for trusting the surface chemistry on top.

Zeolites are skipped - the IZA CIFs were already DLS76-optimised under a pure-SiO2 composition, and re-relaxing them with MACE at this tutorial scope is needlessly expensive for the discovery story.
"""
))

cells.append(code(
    """from helpers.oxide_slabs import (
    build_alpha_alumina_bulk, build_rutile_tio2_bulk, build_monoclinic_zro2_bulk,
)
from pymatgen.io.ase import AseAtomsAdaptor

# Experimental references (same lattice parameters used in the bulk builders)
EXPT_LATTICE = {
    "Al2O3":  {"a": 4.7607, "b": 4.7607, "c": 12.9947, "ref": "Lewis 1982"},
    "TiO2":   {"a": 4.594,  "b": 4.594,  "c": 2.958,   "ref": "Bolzan 1997"},
    "ZrO2-m": {"a": 5.1454, "b": 5.2075, "c": 5.3107,  "ref": "Howard 1988"},
}

BULKS_PMG = {
    "Al2O3":  build_alpha_alumina_bulk(),
    "TiO2":   build_rutile_tio2_bulk(),
    "ZrO2-m": build_monoclinic_zro2_bulk(),
}
BULKS_ASE = {k: AseAtomsAdaptor().get_atoms(s) for k, s in BULKS_PMG.items()}
for k, a in BULKS_ASE.items():
    a.pbc = True
"""
))

cells.append(code(
    """bulk_labels = list(BULKS_ASE.keys())
bulk_atoms_data = [ase_to_atomic_data(BULKS_ASE[k], structure_id=f"bulk_{k}") for k in bulk_labels]

reply_bulks = run_bgr_or_load_cache(
    bulk_atoms_data,
    server_url=BGR_SERVER,
    cache_dir=CACHE_DIR,
    label="oxide_bulks_cellopt",
    endpoint_live=BGR_LIVE,
    cellopt=True,
    opttol=OPTTOL,
)

rows = []
for k, opt in zip(bulk_labels, reply_bulks.atoms):
    atoms_relaxed = atomic_data_to_ase(opt)
    a, b, c = atoms_relaxed.cell.lengths()
    ref = EXPT_LATTICE[k]
    rows.append({
        "Host": k,
        "a_MACE (A)": round(a, 3), "a_expt (A)": ref["a"], "Δa (A)": round(a - ref["a"], 3),
        "b_MACE (A)": round(b, 3), "b_expt (A)": ref["b"], "Δb (A)": round(b - ref["b"], 3),
        "c_MACE (A)": round(c, 3), "c_expt (A)": ref["c"], "Δc (A)": round(c - ref["c"], 3),
        "converged": opt.converged,
    })
bulk_df = pd.DataFrame(rows)
bulk_df
"""
))

cells.append(md(
    """### Parity plot: MACE vs experiment

One point per (host, lattice vector). The diagonal is y=x; deviations are MACE's systematic offset on bulk lattice constants.
"""
))

cells.append(code(
    """import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5.5, 5.5))
colors = {"Al2O3": "#1f77b4", "TiO2": "#d62728", "ZrO2-m": "#2ca02c"}
for _, r in bulk_df.iterrows():
    for axis in ("a", "b", "c"):
        ax.scatter(r[f"{axis}_expt (A)"], r[f"{axis}_MACE (A)"],
                   color=colors[r["Host"]], s=60, edgecolor="black",
                   label=f"{r['Host']} {axis}" if axis == "a" else None)

lo, hi = 2.8, 13.5
ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Experimental lattice constant (A)")
ax.set_ylabel("MACE-MPA-0 cell-opt (A)")
ax.set_title("Bulk lattice parity: MACE-MPA-0 vs experiment")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, ls="--", alpha=0.4)
fig.tight_layout()
parity_path = os.path.join(ASSETS_DIR, "bulk_lattice_parity.png")
fig.savefig(parity_path, dpi=150, bbox_inches="tight")
plt.close(fig)
display_inline(parity_path)
print(f"Saved: {os.path.abspath(parity_path)}")
"""
))


# ---------------------------------------------------------------------------
# Section 9: Clean host relaxation (all 6 in one batch)
# ---------------------------------------------------------------------------

cells.append(md(
    """---

## Clean host relaxation: all six in one BGR call

Six structures - three zeolite bulks and three oxide slabs - submitted as a single BGR request with `cellopt=False`. Oxide slabs carry an `active_mask` that freezes the bottom half of each slab (standard idiom: lock bulk-like layers, let the top surface rearrange). Zeolite bulks relax without constraint.

Caches as `clean_hosts.json`.
"""
))

cells.append(code(
    """HOST_NAMES = list(HOSTS.keys())
IS_SLAB = {
    "H-CHA": False, "H-SAPO-34": False, "H-MFI": False,
    "Al2O3(0001)": True, "TiO2(110)": True, "ZrO2(-1,1,1)": True,
}

active_masks: dict[str, list[bool] | None] = {}
for name, atoms in HOSTS.items():
    if IS_SLAB[name]:
        active_masks[name] = make_active_mask(atoms, bottom_fraction=0.5)
    else:
        active_masks[name] = None  # zeolite bulks relax unconstrained

clean_atoms_list = [
    ase_to_atomic_data(HOSTS[name], structure_id=f"clean_{name.replace('(', '_').replace(')', '')}",
                       active_mask=active_masks[name])
    for name in HOST_NAMES
]

reply_clean = run_bgr_or_load_cache(
    clean_atoms_list,
    server_url=BGR_SERVER,
    cache_dir=CACHE_DIR,
    label="clean_hosts",
    endpoint_live=BGR_LIVE,
    cellopt=False,
    opttol=OPTTOL,
)

clean_rows = []
E_HOST = {}
HOST_RELAXED: dict[str, ase.Atoms] = {}
for name, opt in zip(HOST_NAMES, reply_clean.atoms):
    relaxed = atomic_data_to_ase(opt)
    HOST_RELAXED[name] = relaxed
    E_HOST[name] = float(opt.energy)
    fmax = float(np.max(np.linalg.norm(np.array(opt.forces).reshape(-1, 3), axis=1)))
    clean_rows.append({
        "Host": name,
        "atoms": len(relaxed),
        "converged": opt.converged,
        "n_steps": opt.num_optimization_steps,
        "max |F| (eV/A)": round(fmax, 4),
        "E_host (eV)": round(E_HOST[name], 4),
    })
pd.DataFrame(clean_rows)
"""
))


# ---------------------------------------------------------------------------
# Section 10: H2O placement across multiple starting orientations
# ---------------------------------------------------------------------------

cells.append(md(
    """---

## H2O placement and starting orientations

Per the brief's §9 watch-item: *don't overfit to one starting guess*. For each host we generate four independent starting configurations by rotating the H2O dipole around the placement site, then let MACE relax each. The per-host lowest-energy orientation is the one we report.

Placement rules:

- **Zeolite bulks (H-CHA, H-SAPO-34, H-MFI)** - H2O is placed near the Bronsted proton (O of H2O ~2.5 A from the framework H) to probe the acid-water interaction that Plessow/Fischer/Anderson all benchmark.
- **Oxide slabs (Al2O3, TiO2, ZrO2)** - H2O is placed above the most-central top-surface metal atom at 2.4 A along the surface normal, using :func:`helpers.surfaces.place_adsorbate`.

Orientations are 0, 90, 180, 270 degrees about the adsorbate's z-axis, applied to a base water with its dipole pointing up.
"""
))

cells.append(code(
    """ORIENTATIONS_DEG = [0, 90, 180, 270]


def _rotate_about_z(atoms: ase.Atoms, deg: float) -> ase.Atoms:
    out = atoms.copy()
    if abs(deg) > 1e-6:
        out.rotate(deg, "z", center=atoms.positions[0])  # rotate about the bonding O
    return out


def _find_bronsted_proton_index(atoms: ase.Atoms) -> int | None:
    \"\"\"Return index of the Bronsted H (H closest to a framework O near Al).\"\"\"
    symbols = atoms.get_chemical_symbols()
    h_indices = [i for i, s in enumerate(symbols) if s == "H"]
    return h_indices[0] if len(h_indices) == 1 else None


def place_h2o_in_zeolite(host: ase.Atoms, orient_deg: float,
                         o_h_distance: float = 2.5) -> ase.Atoms:
    \"\"\"Place H2O with its oxygen pointing at the Bronsted proton.\"\"\"
    h2o = build_adsorbate("H2O")
    h2o = _rotate_about_z(h2o, orient_deg)
    h_idx = _find_bronsted_proton_index(host)
    if h_idx is None:
        # No Bronsted H (pure siliceous): place at cell centre.
        target = host.cell.array.sum(axis=0) / 2.0
        direction = np.array([0.0, 0.0, 1.0])
    else:
        # Point from proton outward along the proton-to-cell-centre vector
        cell_centre = host.cell.array.sum(axis=0) / 2.0
        v = cell_centre - host.positions[h_idx]
        direction = v / (np.linalg.norm(v) + 1e-12)
        target = host.positions[h_idx] + o_h_distance * direction
    # Translate H2O so the oxygen (atom 0) sits at *target*
    h2o.translate(target - h2o.positions[0])
    combined = host.copy() + h2o
    return combined


def place_h2o_on_slab(slab: ase.Atoms, orient_deg: float, height: float = 2.4) -> tuple[ase.Atoms, list[bool]]:
    \"\"\"Place H2O above the central top-surface site of an oxide slab.\"\"\"
    z = slab.positions[:, 2]
    top_mask = z > (z.min() + 0.75 * (z.max() - z.min()))  # top 25%
    top_positions = slab.positions[top_mask]
    if len(top_positions) == 0:
        raise ValueError("No atoms in top 25% of slab.")
    site = find_central_site(top_positions, slab.cell.array)
    h2o = build_adsorbate("H2O")
    h2o = _rotate_about_z(h2o, orient_deg)
    combined, mask = place_adsorbate(slab, h2o, site, height=height, frozen_fraction=0.5)
    return combined, mask


# Build 24 (host x orientation) configurations
CONFIGS: dict[tuple[str, int], dict] = {}
for name, atoms in HOSTS.items():
    for deg in ORIENTATIONS_DEG:
        if IS_SLAB[name]:
            combined, mask = place_h2o_on_slab(HOST_RELAXED[name], deg)
        else:
            combined = place_h2o_in_zeolite(HOST_RELAXED[name], deg)
            mask = None
        CONFIGS[(name, deg)] = {"atoms": combined, "mask": mask}

print(f"Built {len(CONFIGS)} host+H2O configurations")
"""
))

cells.append(md(
    """### Visualise the four H-CHA orientations (sanity check)

A quick look at one host - H-CHA - in its four starting H2O orientations. The water oxygen sits ~2.5 A from the Bronsted proton; the hydrogens rotate around it.
"""
))

cells.append(code(
    """display_widgets_row(
    [(f"{ORIENTATIONS_DEG[i]} deg", CONFIGS[("H-CHA", ORIENTATIONS_DEG[i])]["atoms"])
     for i in range(4)],
    width="230px", height="230px",
)
"""
))


# ---------------------------------------------------------------------------
# Section 11: Host + H2O batch relaxation (24 structures in one call)
# ---------------------------------------------------------------------------

cells.append(md(
    """## Host + H2O batch relaxation

All 24 configurations submitted as a single BGR call. Caches as `host_h2o_batch.json`.
"""
))

cells.append(code(
    """batch_keys = list(CONFIGS.keys())
batch_atoms_data = [
    ase_to_atomic_data(
        CONFIGS[k]["atoms"],
        structure_id=f"{k[0].replace('(', '_').replace(')', '')}_orient{k[1]}",
        active_mask=CONFIGS[k]["mask"],
    )
    for k in batch_keys
]

reply_h2o_batch = run_bgr_or_load_cache(
    batch_atoms_data,
    server_url=BGR_SERVER,
    cache_dir=CACHE_DIR,
    label="host_h2o_batch",
    endpoint_live=BGR_LIVE,
    cellopt=False,
    opttol=OPTTOL,
)

h2o_rows = []
E_HOST_H2O: dict[tuple[str, int], float] = {}
for k, opt in zip(batch_keys, reply_h2o_batch.atoms):
    E_HOST_H2O[k] = float(opt.energy)
    fmax = float(np.max(np.linalg.norm(np.array(opt.forces).reshape(-1, 3), axis=1)))
    h2o_rows.append({
        "Host": k[0], "Orient (deg)": k[1],
        "converged": opt.converged,
        "n_steps": opt.num_optimization_steps,
        "max |F| (eV/A)": round(fmax, 4),
        "E_host+H2O (eV)": round(E_HOST_H2O[k], 4),
    })
h2o_df = pd.DataFrame(h2o_rows)
h2o_df.pivot_table(index="Host", columns="Orient (deg)", values="E_host+H2O (eV)").round(3)
"""
))


# ---------------------------------------------------------------------------
# Serialise
# ---------------------------------------------------------------------------

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (alchemi-playbook)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).resolve().parent.parent / "alchemi-mace-water-sorbents.ipynb"
out_path.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {len(cells)} cells to {out_path}")
