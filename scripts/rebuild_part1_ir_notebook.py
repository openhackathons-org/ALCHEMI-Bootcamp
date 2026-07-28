#!/usr/bin/env python3
"""Build the unified Part 1 ALCHEMI Toolkit notebook."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys


BOOTCAMP_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    BOOTCAMP_ROOT / "part-1-scalable-atomistic-workflows" / "alchemi-water-ir.ipynb"
)
PART_DIR = NOTEBOOK.parent
sys.path.insert(0, str(PART_DIR))

from aux.ui import (  # noqa: E402
    callout_html,
    lesson_summary_html,
    notebook_hero_html,
    process_diagram_html,
    stage_card_html,
)


def source(text: str) -> list[str]:
    return (text.strip("\n") + "\n").splitlines(keepends=True)


def marked_python_source(path: Path, start: str, end: str) -> str:
    """Read one executable teaching excerpt from a maintained Python module."""

    text = path.read_text(encoding="utf-8")
    try:
        body = text.split(start, 1)[1].split(end, 1)[0]
    except IndexError as exc:
        raise ValueError(f"Missing source markers in {path}") from exc
    return body.strip("\n") + "\n"


def markdown(
    cell_id: str,
    text: str,
    *,
    attachments: dict[str, dict[str, str]] | None = None,
) -> dict:
    cell = {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source(text),
    }
    if attachments:
        cell["attachments"] = attachments
    return cell


def code(cell_id: str, text: str, *, source_hidden: bool = False) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": (
            {
                "jupyter": {"source_hidden": True},
                "tags": ["remove-input"],
            }
            if source_hidden
            else {}
        ),
        "outputs": [],
        "source": source(text),
    }


def stage_markdown(
    cell_id: str,
    *,
    stage: int,
    title: str,
    outcome: str,
    body: str = "",
    before: str | None = None,
    compute_time: str | None = None,
    total: int = 7,
) -> dict:
    """Build an accessible stage heading plus the shared progress card."""

    pieces = [
        stage_card_html(
            stage=stage,
            total=total,
            title=title,
            outcome=outcome,
            state="ready",
            compute_time=compute_time,
        ),
    ]
    if before:
        pieces.append(callout_html(before, kind="before"))
    if body.strip():
        pieces.append(body.strip())
    return markdown(cell_id, "\n\n".join(pieces))


def main(output_path: Path = NOTEBOOK) -> None:
    sevennet_config_source = marked_python_source(
        PART_DIR / "aux" / "models" / "sevennet.py",
        "# BEGIN NOTEBOOK MODEL CONFIG",
        "# END NOTEBOOK MODEL CONFIG",
    )
    sevennet_wrapper_source = marked_python_source(
        PART_DIR / "aux" / "models" / "sevennet.py",
        "# BEGIN NOTEBOOK WRAPPER",
        "# END NOTEBOOK WRAPPER",
    )
    reference_figure = (
        BOOTCAMP_ROOT
        / "part-1-scalable-atomistic-workflows"
        / "reference"
        / "artifacts"
        / "b97_3c_ir_reference.png"
    )
    reference_figure_attachment = {
        "b97_3c_ir_reference.png": {
            "image/png": base64.b64encode(reference_figure.read_bytes()).decode()
        }
    }
    cells = [
        markdown(
            "title",
            notebook_hero_html(
                image_path=(
                    "assets/images/banner_candidates/"
                    "water-ir-v2-04-trajectory-to-spectrum.png"
                ),
                image_alt=(
                    "Two complete bent water molecules suspended in a dark "
                    "computational field, linked by a hydrogen bond beside a "
                    "green vibrational-signal motif"
                ),
                title="From one structure to a scalable atomistic workflow",
                subtitle=(
                    "Evaluate one structure, scale to batches, complete and check "
                    "the model, then bring in a new model and run dynamics."
                ),
            )
            + "\n\n"
            + lesson_summary_html(
                do=(
                    "Evaluate one structure, complete AIMNet2 with Coulomb and D3, "
                    "check it on intermolecular interactions, adapt SevenNet-Omni "
                    "for Cu(111), and run a four-system IR calculation."
                ),
                learn=(
                    "AtomicData, Batch, model configuration, neighbors, model "
                    "composition, a custom model adapter, FIRE2, fused NVT to NVE, "
                    "hooks, inflight batching, periodic PME, DomainParallel, and "
                    "DistributedPipeline."
                ),
                need=(
                    "One CUDA GPU and the tutorial environment. The checked run "
                    "took 13 min 1 s of notebook wall time on one H100 PCIe; "
                    "Stage 6 accounted for 9 min 27 s."
                ),
            )
            + "\n\n"
            + callout_html(
                "About the spectra: the harmonic comparison and the finite-temperature MD spectrum answer different questions. For frequencies, the complete AIMNet checkpoint-base + Coulomb + D3 model and B97-3c use the same finite-difference displacement and projected normal-mode protocol. Their intensities come from different dipole models and are not scored against each other. Experiment contributes band positions only. The MD spectrum remains a separate dynamics example and is not used as a DFT accuracy score.",
                kind="note",
            )
            + "\n\n"
            + callout_html(
                "Follow one common Toolkit data path while the calculation grows: one structure, many structures, a completed model, a new model domain, and repeated dynamics updates.",
                kind="check",
            ),
        ),
        markdown(
            "roadmap",
            r"""
## Notebook map

1. **One structure, one result:** ASE → `AtomicData` → `Batch` → energy, forces, and charges.
2. **Same physics, one batch:** agreement, model-aware neighbors, CPU/GPU crossover, and batch layout.
3. **Complete and check the potential:** AIMNet core + Coulomb + D3 on 90 NCI Atlas graphs, with DFT-D3 and CCSD(T)/CBS references.
4. **Bring a model for a new domain:** adapt SevenNet-Omni to Toolkit and evaluate fixed Cu(111) adsorption structures.
5. **Prepare dynamics and IR:** return to the charge-predicting molecular model, relax, compare harmonic frequencies, and build fused NVT → NVE stages.
6. **Run and inspect the trajectory:** save raw state first, then check temperature, topology, and predicted-charge IR.
7. **Choose a scaling path:** refill one GPU, load a checked periodic box, exercise the `DomainParallel` API, then compare three fixed-structure passes on 1, 2, and 4 H100s.

### Checked H100 pacing

| Section | Code time on one H100 PCIe |
|---|---:|
| Setup | 23 s |
| Stage 1: one structure | 18 s |
| Stage 2: batching | 19 s |
| Stage 3: NCI calculation | 22 s |
| Stage 4: adapter and single points | 18 s |
| Stage 5: preparation and harmonic check | 1 min 13 s |
| Stage 6: trajectory and analysis | 9 min 27 s |
| Stage 7: scaling paths | 30 s |
| **Complete notebook code** | **12 min 51 s** |
| **Notebook runner wall time** | **13 min 1 s** |

These are pacing measurements from one complete checked run, not benchmark
results. Checkpoint caches were warm. Hardware, software, and cache state can
change the elapsed time.
"""
            + "\n\n"
            + callout_html(
                "Dynamics is the longest pause. The saved trajectory lets you "
                "change plots without rerunning it.",
                kind="note",
            ),
        ),
        markdown(
            "alchemi-orientation",
            r"""
## Where ALCHEMI fits

ALCHEMI is the workflow layer between atomistic structures, accelerated kernels, models, and simulation outputs.

- **ASE** supplies structures.
- **Toolkit Core (`nvalchemi`)** supplies `AtomicData`, `Batch`, model adapters, composition, dynamics, hooks, sinks, and distributed execution.
- **Toolkit-Ops (`nvalchemiops`)** supplies accelerated neighbors, D3, periodic electrostatics, and segmented-operation kernels.
- **AIMNet2** supplies this molecular checkpoint; it is a model in the ecosystem, not Toolkit itself.
- **SevenNet-Omni** supplies the separate materials checkpoint used in the custom-adapter example; its model and weights are also outside Toolkit.
- **PyTorch** supplies tensors, autograd, compilation, and the distributed backend used by Toolkit Core and the models in this notebook.
- **JAX** is a peer array and autodiff system exposed by supported Toolkit-Ops operations. One small comparison below uses its Toolkit-Ops binding; the main workflow remains in PyTorch.
- **Warp** supplies CUDA kernels below selected Toolkit-Ops operations. It is the implementation layer in this example, not a third model interface.
- **`aux/` in this tutorial** supplies local plotting, validation, recording, and the finite all-pairs `DirectCoulombWrapper` used here. Its names are not Toolkit APIs.
"""
            + "\n\n"
            + process_diagram_html(
                title="How data moves through ALCHEMI",
                steps=(
                    "ASE structures",
                    "Core AtomicData + Batch",
                    "model adapters + Toolkit-Ops kernels",
                    "dynamics + hooks",
                    "saved result files",
                ),
                caption=(
                    "Core provides common data structures and simulation tools. You "
                    "choose the model and integrator, and the notebook saves ordinary "
                    "files that can be inspected after the run."
                ),
            )
            + "\n\n"
            + callout_html(
                "The notebook keeps public Toolkit construction, requested outputs, neighbor choices, model composition, dynamics stages, hooks, and saving visible. Data parsing, repeated checks, and plotting stay in aux/.",
                kind="check",
            ),
        ),
        markdown(
            "setup-heading",
            r"""
## Setup

The collapsed cell checks the required package versions and GPU. The next visible cell holds the model and simulation settings used throughout the notebook; public Toolkit imports remain visible after it.
""",
        ),
        code(
            "setup",
            """
from __future__ import annotations

import os
import sys
from pathlib import Path

PART_NAME = "part-1-scalable-atomistic-workflows"


def locate_part_directory() -> Path:
    # Support kernels started from the repository, the notebook directory,
    # or any child directory. An environment override covers other launchers.
    override = os.environ.get("ALCHEMI_BOOTCAMP_ROOT")
    starts = [Path(override).expanduser()] if override else []
    cwd = Path.cwd().resolve()
    starts.extend((cwd, *cwd.parents))
    for start in starts:
        root = start.resolve()
        candidate = root if root.name == PART_NAME else root / PART_NAME
        if (candidate / "aux").is_dir() and (
            candidate / "alchemi-water-ir.ipynb"
        ).is_file():
            return candidate
    raise RuntimeError(
        "Could not find the Part 1 tutorial. Start the kernel from this "
        "repository, or set ALCHEMI_BOOTCAMP_ROOT to its root directory."
    )


PART_DIR = locate_part_directory()
ROOT = PART_DIR.parent
for search_path in (ROOT, PART_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from aux.ui import NotebookProgress, readable_table
from aux.domain.config import DOMAIN_METHODOLOGY

setup_progress = NotebookProgress(
    title="Check the tutorial environment", total=3, unit="checks"
)

import logging
import math
import warnings
from importlib import metadata
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from aimnet.calculators import AIMNet2Calculator
from ase import Atoms, units
from ase.io import write
from ase.visualize import view
from IPython.display import display
from torch import nn

# JAX and PyTorch share this teaching kernel. Disable JAX's bulk GPU-memory
# reservation before JAX is imported; this does not cap either workload.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORMS"] = "cuda"

# Compiler failures still raise; the external job transcript retains stderr.
torch._logging.set_logs(dynamo=logging.ERROR, inductor=logging.CRITICAL)
# Full float32 matmul is deliberate for the compiled/eager correctness check.
# Hide Inductor's one-time TF32 speed suggestion; the visible cell states why.
warnings.filterwarnings(
    "ignore",
    message=r"TensorFloat32 tensor cores.*not enabled.*",
    category=UserWarning,
)
# These two messages come from known upstream inspection paths. The model
# checkpoint is verified below, and every parameter is frozen before use.
warnings.filterwarnings(
    "ignore",
    message=r"Converting a tensor with requires_grad=True to a scalar.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"invalid escape sequence.*",
    category=SyntaxWarning,
    module=r"physicsnemo[.]utils[.]logging[.]launch",
)
warnings.filterwarnings(
    "ignore",
    message=r"Sets are not currently considered sequences.*",
)
logging.getLogger(
    "nvalchemi.distributed._core.shard_wrappers"
).setLevel(logging.WARNING)
setup_progress.advance(message="Scientific Python and PyTorch imported")

from aux.harmonic_config import (
    HARMONIC_CHARGE_NEUTRALITY_TOLERANCE_E,
    HARMONIC_DISPLACEMENT_STEPS_BOHR,
    HARMONIC_FIRE_INITIAL_DT,
    HARMONIC_FIRE_MAX_STEPS,
    HARMONIC_FMAX_EV_A,
    HARMONIC_FREQUENCY_STEP_TOLERANCE_CM1,
    HARMONIC_HESSIAN_ANTISYMMETRY_REL_MAX,
    HARMONIC_IMAGINARY_FLOOR_CM1,
    HARMONIC_INTENSITY_STEP_ABS_TOLERANCE_KM_MOL,
    HARMONIC_INTENSITY_STEP_REL_TOLERANCE,
    HARMONIC_MODE_OVERLAP_MIN,
    HARMONIC_SELECTED_STEP_BOHR,
)
from aux.composition_config import (
    COMPILED_EAGER_CHARGE_TOLERANCE_E,
    COMPILED_EAGER_ENERGY_TOLERANCE_EV,
    COMPILED_EAGER_FORCE_TOLERANCE_EV_A,
    COMPILED_REPEAT_CHARGE_TOLERANCE_E,
    COMPILED_REPEAT_ENERGY_TOLERANCE_EV,
    COMPILED_REPEAT_FORCE_TOLERANCE_EV_A,
    COMPONENT_CLOSURE_TOLERANCE_EV,
    COMPOSITION_ANALYTIC_COULOMB_ENERGY_TOLERANCE_EV,
    COMPOSITION_ANALYTIC_COULOMB_FORCE_TOLERANCE_EV_A,
    COMPOSITION_CHARGE_AGREEMENT_TOLERANCE_E,
    COMPOSITION_ENERGY_AGREEMENT_TOLERANCE_EV,
    COMPOSITION_FD_ENERGY_ROUTE,
    COMPOSITION_FD_FORCE_TOLERANCE_EV_A,
    COMPOSITION_FD_STEP_A,
    COMPOSITION_FORCE_AGREEMENT_TOLERANCE_EV_A,
    COMPOSITION_INTERACTION_AGREEMENT_TOLERANCE_EV,
    FULL_SERIAL_BATCH_TOLERANCE_EV,
    RESIDUAL_SERIAL_BATCH_TOLERANCE_EV,
)
from aux.models.sevennet_config import (
    D3_REFERENCE_CUTOFF_A,
    D3_REFERENCE_CUTOFF_BOHR,
    D3_REFERENCE_SMOOTHING_FRACTION,
    PBE_D3_BJ_A1,
    PBE_D3_BJ_A2_BOHR,
    PBE_D3_BJ_S6,
    PBE_D3_BJ_S8,
    SEVENNET_CHECKPOINT_DOI,
    SEVENNET_CHECKPOINT_SHA256,
    SEVENNET_CHECKPOINT_URL,
    SEVENNET_MODALITY,
    SEVENNET_MODEL_NAME,
    SEVENNET_PACKAGE_VERSION,
    SEVENNET_REFERENCE_METHOD,
    SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM,
    SEVENNET_REPEAT_FORCE_TOL_EV_A,
)
from aux.runtime import find_bootcamp_root, verify_toolkit_pins
from aux.workflow_config import (
    COVALENT_OH_CUTOFF_A,
    ENERGY_EXCURSION_ADVISORY_MEV_PER_ATOM,
    HBOND_ANGLE_CUTOFF_DEG,
    HBOND_H_ACCEPTOR_CUTOFF_A,
    HBOND_OO_CUTOFF_A,
    IR_CAPTURE_CHARGE_TOLERANCE_E,
    IR_CHARGE_NEUTRALITY_TOLERANCE_E,
    IR_DIPOLE_ORIGIN_TOLERANCE_E_ANGSTROM,
    IR_FIRE_INITIAL_DT,
    IR_INITIAL_VELOCITY_RANDOM_SEEDS,
    IR_NVT_FRICTION_PER_FS,
    IR_NVT_RANDOM_SEED,
    IR_PRODUCTION_STATUS,
    IR_WELCH_OVERLAP_FRACTION,
    IR_WELCH_SEGMENT_TIME_FS,
    IR_WARMUP_STATUS,
    MASS_ONLY_CHARGE_TOLERANCE_E,
    MASS_ONLY_ENERGY_TOLERANCE_EV,
    MASS_ONLY_FORCE_TOLERANCE_EV_A,
    MASS_ONLY_POSITION_ATOL_A,
    MASS_ONLY_POSITION_RTOL,
    OXYGEN_CONNECTIVITY_CUTOFF_A,
    PAIR_TEMPERATURE_RELATIVE_TOLERANCE,
)
import nvalchemi

assert find_bootcamp_root(ROOT) == ROOT
RUN_ID = os.environ.get("ALCHEMI_RUN_ID", os.environ.get("SLURM_JOB_ID", "interactive"))
if not RUN_ID.replace("-", "").replace("_", "").isalnum():
    raise ValueError("ALCHEMI_RUN_ID may contain only letters, numbers, '-' and '_'")
OUTPUT_DIR = PART_DIR / "outputs" / f"run-{RUN_ID}"
REFERENCE_ROOT = PART_DIR / "reference" / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CORE_COMMIT = "331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409"
OPS_COMMIT = "e8e7a7464f6745277a156a3d6f433d06b58c60e3"
EXPECTED_AIMNET_VERSION = "0.2.0"
EXPECTED_JAX_VERSION = "0.9.0.1"
EXPECTED_TORCH_PREFIX = "2.12.0"
requirements = (ROOT / "build" / "requirements.txt").read_text()
assert CORE_COMMIT in requirements and OPS_COMMIT in requirements
installed_pins = verify_toolkit_pins(CORE_COMMIT, OPS_COMMIT)
assert nvalchemi.version == "0.2.0"
assert metadata.version("aimnet") == EXPECTED_AIMNET_VERSION
assert metadata.version("jax") == EXPECTED_JAX_VERSION
assert metadata.version("sevenn") == SEVENNET_PACKAGE_VERSION
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX)
setup_progress.advance(message="Toolkit commits and package versions verified")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("The complete 25,000-step tutorial requires a CUDA GPU.")

D3_PARAMETER_FILE = Path(os.environ.get(
    "ALCHEMI_D3_PARAM_FILE",
    Path.home() / ".cache" / "nvalchemiops" / "dftd3_parameters.pt",
))
EXPECTED_D3_PARAMETER_SHA256 = (
    "b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84"
)
setup_progress.complete("Package versions and CUDA device verified")

display(readable_table(
    pd.DataFrame([
        ("GPU", torch.cuda.get_device_name(DEVICE)),
        ("Toolkit Core", f"{nvalchemi.version} · {CORE_COMMIT[:12]}"),
        ("Toolkit-Ops", installed_pins["Ops"][:12]),
        ("AIMNet", metadata.version("aimnet")),
        ("SevenNet", metadata.version("sevenn")),
        ("PyTorch", torch.__version__),
        ("JAX", metadata.version("jax")),
        ("Warp", metadata.version("warp-lang")),
    ], columns=["Runtime", "Checked version"]),
    label="Tutorial environment",
    show_index=False,
))
""",
            source_hidden=True,
        ),
        code(
            "tutorial-settings",
            """
settings_progress = NotebookProgress(
    title="Show the tutorial settings", total=1, unit="group"
)

MODEL_CHECKPOINT = os.environ.get(
    "ALCHEMI_AIMNET_CHECKPOINT", "aimnet2-b973c-2025-d3_0"
)
DIMER_DISTANCES_A = np.array([2.50, 2.70, 2.90, 3.20, 3.50, 3.90, 4.40, 5.00])
D3_CUTOFF_A = DOMAIN_METHODOLOGY.d3_cutoff_a
SURFACE_D3_CUTOFF_A = D3_REFERENCE_CUTOFF_A
NEIGHBOR_SKIN_A = 0.5
AIMNET_MATMUL_PRECISION = "highest"
torch.set_float32_matmul_precision(AIMNET_MATMUL_PRECISION)
FIRE_MAX_STEPS = 5_000
FIRE_FMAX_EV_A = 0.01
DT_FS = 0.5
TEMPERATURE_K = 75.0
WARMUP_STEPS = 5_000
PRODUCTION_STEPS = 20_000
TOTAL_DYNAMICS_STEPS = WARMUP_STEPS + PRODUCTION_STEPS
H_TO_D_COARSE_MASS_PATH_STEPS = 65
H_TO_D_FINE_MASS_PATH_STEPS = 129
H_TO_D_DEGENERACY_TOLERANCE_CM1 = 2.0

settings_progress.complete("Model, relaxation, and dynamics settings ready")
settings_table = pd.DataFrame([
    ("AIMNet checkpoint", MODEL_CHECKPOINT),
    ("D3 cutoff", f"{D3_CUTOFF_A:g} Å"),
    ("Neighbor skin", f"{NEIGHBOR_SKIN_A:g} Å"),
    ("FIRE2 target", f"{FIRE_FMAX_EV_A:g} eV/Å"),
    ("MD timestep", f"{DT_FS:g} fs"),
    ("Temperature", f"{TEMPERATURE_K:g} K"),
    ("NVT → NVE", f"{WARMUP_STEPS:,} → {PRODUCTION_STEPS:,} steps"),
    ("Production duration", f"{PRODUCTION_STEPS * DT_FS / 1_000:g} ps"),
    ("Requested outputs", "energy, forces, charges"),
], columns=["Setting", "Value"])
display(readable_table(
    settings_table,
    label="Tutorial settings",
    show_index=False,
))

""",
        ),
        code(
            "imports",
            """
imports_progress = NotebookProgress(
    title="Load public Toolkit APIs", total=1, unit="group"
)

# These are the NVIDIA ALCHEMI classes and operations used below.
from nvalchemi.data import AtomicData, Batch, InMemoryDataset
from nvalchemi.distributed import DomainConfig, DomainParallel
from nvalchemi.dynamics import (
    BaseDynamics,
    ConvergenceHook,
    FIRE2,
    FusedStage,
    HostMemory,
    NVE,
    NVTLangevin,
    SizeAwareSampler,
    ZarrData,
    initialize_velocities,
)
from nvalchemi.dynamics.hooks import (
    LoggingHook,
    NaNDetectorHook,
)
from nvalchemi.hooks import Hook
from nvalchemi.models import (
    AIMNet2Wrapper,
    DFTD3ModelWrapper,
    PMEModelWrapper,
    PipelineGroup,
    PipelineModelWrapper,
    PipelineStep,
)
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)
from nvalchemi.neighbors import compute_neighbors
from nvalchemiops.torch import segmented_sum
from nvalchemiops.torch.interactions.electrostatics import (
    estimate_pme_parameters,
)
imports_progress.complete("Public Toolkit APIs loaded")
""",
        ),
        code(
            "helper-imports",
            """
helper_imports_progress = NotebookProgress(
    title="Load tutorial helpers", total=1, unit="group"
)

# Notebook-specific structures, recording, analysis, and presentation.
from aux.analysis import (
    comparison_display_table,
    dimer_interaction_energy_table,
    first_atomic_data_table,
    first_model_result_tables,
    h_to_d_mode_mapping_table,
    ir_comparison_table,
    ir_spectrum_metrics,
    reference_comparison_metrics,
    topology_time_series,
)
from aux.artifacts import (
    graph_atoms_from_batch,
    load_ir_trajectory,
    save_ir_trajectory,
    sha256_file,
    write_structure_artifacts,
)
from aux.adsorption import (
    ADSLAB_KEYS,
    ADSORBATES,
    DEFAULT_DATA_DIR,
    assemble_adsorption_results,
    build_placement_table,
    build_structure_inventory_table,
    build_full_force_table,
    load_adsorption_methodology,
    load_initial_structure_set,
    split_for_batches,
    summarize_adslab_force_regions,
)
from aux.adsorption_visualization import adsorption_widget_grid
from aux.benchmarking import (
    benchmark_device_sweep,
    build_benchmark_batch,
    compare_mixed_and_bucketed,
    first_and_warm_call_rows,
    neighbor_storage_table,
    plot_device_sweep,
)
from aux.framework_comparison import segmented_sum_comparison_table
from aux.capture import PredictedChargeIRHook
from aux.checkpoint import (
    aimnet_model_card_table,
    checkpoint_card,
    resolve_checkpoint_path,
    verify_checkpoint_identities,
)
from build.prewarm_aimnet import CHECKPOINT_IDENTITIES
from aux.composition_checks import (
    build_composition_check_table,
    central_difference_force,
    compare_composition_outputs,
    compare_two_particle_coulomb,
)
from aux.diagnostics import (
    analyze_production_trajectory,
    build_production_diagnostics_display_tables,
    mass_only_invariance,
)
from aux.electrostatics import DirectCoulombWrapper
from aux.experimental_reference import load_experimental_water_fundamentals
from aux.harmonic_ir import (
    ANGSTROM_PER_BOHR,
    symmetric_cartesian_displacements,
)
from aux.harmonic_workflow import (
    analyze_harmonic_step_series,
    build_harmonic_archive_arrays,
    build_harmonic_mode_comparison_table,
    collect_harmonic_displacement_result,
    empty_harmonic_mode_comparison_table,
)
from aux.inflight import (
    inflight_trace_table,
    prepare_inflight_dimer_source,
    register_inflight_trace,
)
from aux.domain import load_prebuilt_domain_box
from aux.domain.display import (
    compact_box_summary_table,
    domain_agreement_display_table,
    molecule_charge_display_tables,
)
from aux.domain.packing import box_summary_table, molecule_charge_tables
from aux.domain.results import load_domain_lesson_view
from aux.ir_display import (
    harmonic_mode_comparison_display_table,
    mass_invariance_display_table,
    monomer_mode_mapping_display_table,
    prepare_monomer_reference_display,
)
from aux.hooks import (
    NotebookStageProgressHook,
    StageStepCounterHook,
    add_stage_step_counters,
    converge_after_steps,
)
from aux.models.sevennet_checkpoint import (
    load_raw_sevennet_omni,
    resolve_sevennet_checkpoint,
)
from aux.models.sevennet import (
    _SevenNetAdapterBase,
    _map_sevennet_outputs,
    _model_device_and_dtype,
    _toolkit_batch_to_sevennet_graph,
)
from aux.models.sevennet_checks import (
    build_sevennet_model_card,
    build_sevennet_settings_table,
    summarize_sevennet_task_outputs,
)
from aux.models.sevennet_lesson import finalize_sevennet_lesson
from aux.numerical_checks import (
    assert_tensor_fields_unchanged,
    build_difference_check_table,
    clone_selected_outputs,
    max_absolute_differences,
    snapshot_tensor_fields,
)
from aux.nci_atlas import (
    assemble_nci_comparison_curves,
    build_graph_index,
    load_nci_atlas_subset,
    reduce_fragment_energies,
    rows_to_atoms,
)
from aux.nci_config import (
    NCI_COMPLETE_MAE_LIMIT_KCAL_MOL,
    NCI_VALIDATION,
    nci_validation_settings_table,
)
from aux.nci_validation import (
    build_nci_force_check_table,
    check_nci_interaction_component_sum,
    check_nci_force,
    nci_force_check_record,
)
from aux.notebook_reporting import build_part1_notebook_report
from aux.nci_plotting import plot_nci_interaction_curves
from aux.plotting import (
    plot_dimer_interaction_energies,
    plot_domain_decomposition,
    plot_harmonic_monomer_comparison,
    plot_monomer_ir_comparison,
    plot_topology_timeline,
)
from aux.precision import (
    precision_display_table,
    summarize_model_precision,
    validate_precision_observation,
)
from aux.reference import (
    load_psi4_b973c_ir_artifact,
)
from aux.reference_data import load_verified_b97_3c_dimer_reference
from aux.run_output import (
    save_water_run_outputs,
)
from aux.structures import (
    make_ir_structures,
    make_water_dimer,
    make_water_dimer_scan,
    make_water_monomer,
)
from aux.ui import callout, figure_with_alt, readable_table
helper_imports_progress.complete("Tutorial helpers loaded")
""",
            source_hidden=True,
        ),
        markdown(
            "framework-primer",
            r"""
### PyTorch, JAX, and Warp: two interfaces and one kernel layer

PyTorch and JAX are high-level array and automatic-differentiation systems. Supported Toolkit-Ops operations expose separate bindings for them, so each framework keeps its own array type and gradient tools. The Toolkit Core APIs and model workflow used in this notebook follow PyTorch.

Warp is lower level. Toolkit-Ops uses Warp to implement selected CUDA kernels, then returns the result through the PyTorch or JAX binding that called it. Most Toolkit users call the binding rather than write a Warp kernel.

`segmented_sum` expects sorted `int32` segment IDs. A Toolkit `Batch` stores each graph's atoms together, so its `batch_idx` has this form. The small array below mirrors that layout.
"""
            + "\n\n"
            + process_diagram_html(
                title="One Toolkit-Ops operation through two interfaces",
                steps=(
                    "PyTorch tensor or JAX array",
                    "matching Toolkit-Ops binding",
                    "selected Warp/CUDA kernel",
                    "result in the same array system",
                ),
                caption=(
                    "This diagram describes Toolkit-Ops. The remaining Toolkit "
                    "Core workflow in this notebook follows the PyTorch route."
                ),
            )
            + "\n\n"
            + callout_html(
                "PyTorch autograd follows the PyTorch binding, while JAX transformations follow the JAX binding. A Warp tape records Warp kernel launches made while the tape is active; it is not the complete PyTorch, JAX, or model graph.",
                kind="note",
            )
            + "\n\n"
            + callout_html(
                "This small calculation checks APIs and results, not speed. First calls may compile. XLA_PYTHON_CLIENT_PREALLOCATE=false is set before JAX is imported so it can share this notebook kernel with PyTorch; this changes allocation behavior, not the calculation.",
                kind="before",
            ),
        ),
        code(
            "framework-primer-example",
            """
framework_progress = NotebookProgress(
    title="Compare PyTorch, JAX, and Warp", total=4, unit="checks"
)

import jax
import jax.numpy as jnp
import warp as wp
from nvalchemiops.jax.segment_ops import segmented_sum as jax_segmented_sum
from nvalchemiops.segment_ops import segmented_sum as warp_segmented_sum
from nvalchemiops.torch.segment_ops import segmented_sum as torch_segmented_sum

jax_gpu = next((device for device in jax.devices() if device.platform == "gpu"), None)
if jax_gpu is None:
    raise RuntimeError("The Toolkit-Ops JAX binding requires a CUDA-capable JAX device.")
framework_progress.advance(message="PyTorch, JAX, and Warp CUDA paths loaded")

# Four atom-level values belonging to two independent systems.
shared_values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
shared_graph_idx = np.array([0, 0, 1, 1], dtype=np.int32)
expected_totals = np.array([3.0, 7.0], dtype=np.float32)

torch_values = torch.tensor(shared_values, device=DEVICE, requires_grad=True)
torch_graph_idx = torch.tensor(shared_graph_idx, device=DEVICE)
torch_totals = torch_segmented_sum(
    torch_values, torch_graph_idx, num_segments=2
)
(torch_gradient,) = torch.autograd.grad(torch_totals.sum(), torch_values)
framework_progress.advance(message="PyTorch returned values and gradients")

with jax.default_device(jax_gpu):
    jax_values = jnp.asarray(shared_values)
    jax_graph_idx = jnp.asarray(shared_graph_idx)
jax_totals = jax_segmented_sum(jax_values, jax_graph_idx, num_segments=2)
jax_gradient = jax.grad(
    lambda values: jax_segmented_sum(values, jax_graph_idx, 2).sum()
)(jax_values)
jax_totals.block_until_ready()
jax_gradient.block_until_ready()
framework_progress.advance(message="JAX returned the same values and gradients")

# The raw call does not manage Torch's stream as the Torch binding does.
torch.cuda.synchronize(DEVICE)
# Warp takes typed arrays and caller-allocated output storage.
warp_values = wp.from_torch(torch_values.detach(), dtype=wp.float32)
warp_graph_idx = wp.from_torch(torch_graph_idx, dtype=wp.int32)
warp_totals = wp.zeros(2, dtype=wp.float32, device=warp_values.device)
warp_segmented_sum(warp_values, warp_graph_idx, warp_totals)
wp.synchronize_device(warp_values.device)

torch_result = torch_totals.detach().cpu().numpy()
jax_result = np.asarray(jax.device_get(jax_totals))
warp_result = warp_totals.numpy()
for result in (torch_result, jax_result, warp_result):
    np.testing.assert_allclose(result, expected_totals)
np.testing.assert_allclose(torch_gradient.detach().cpu().numpy(), 1.0)
np.testing.assert_allclose(np.asarray(jax.device_get(jax_gradient)), 1.0)

framework_table = pd.DataFrame([
    ("PyTorch binding", "torch.Tensor", str(torch_values.dtype),
     str(torch_values.device), torch_result.tolist(), "returns a tensor"),
    ("JAX binding", "jax.Array", str(jax_values.dtype),
     str(jax_values.device), jax_result.tolist(), "returns an array"),
    ("raw Warp", "wp.array", warp_values.dtype.__name__,
     str(warp_values.device), warp_result.tolist(), "fills supplied output"),
], columns=["Path", "Array", "Dtype", "Device", "Totals", "Output"])
display(readable_table(
    framework_table,
    label="The same reduction through two bindings and raw Warp",
    show_index=False,
))
framework_progress.complete("Both bindings and the raw Warp call returned [3, 7]")
display(callout(
    "PyTorch and JAX also returned the expected gradient [1, 1, 1, 1]. "
    "The raw Warp call is lower level: it uses explicit arrays and output storage.",
    kind="result",
    result_state="pass",
))
""",
        ),
        stage_markdown(
            "stage-1",
            stage=1,
            title="One structure, one result",
            outcome="Turn one ASE water molecule into AtomicData, a Batch, and one energy, force, and charge result.",
            before="Before running: isotope substitution changes nuclear mass, not this potential-energy prediction.",
            compute_time=(
                "18 s on one H100 PCIe in the checked run"
            ),
            body=r"""
- Start with one molecule and inspect the data path before calculating an interaction energy.
- The first value comes from the supplied checkpoint. Later stages show how its declared physical terms are composed and checked.
- Inspect coordinate, checkpoint-weight, and output precision before scaling to larger batches.
""",
        ),
        markdown(
            "atomistic-loop",
            r"""
### One simulation loop, five Toolkit decisions

1. Put structure state in `AtomicData`.
2. Pack independent systems in `Batch`.
3. Request only the model outputs the workflow needs.
4. Let a dynamics stage update the batch; let hooks observe it.
5. Save enough state to inspect or replay the analysis.

Every wrapper exposes a `model_config`: required inputs, available and active outputs, periodic support, and neighbor requirements. `set_config("active_outputs", {...})` selects only the tensors needed for the next calculation.

Toolkit calls each independent atomistic system in a `Batch` a **graph**. Here, graph does not mean a chemical bond diagram or a PyTorch computational graph.

"""
            + "\n\n"
            + process_diagram_html(
                title="The atomistic simulation loop",
                steps=(
                    "AtomicData state",
                    "Batch of independent graphs",
                    "requested model outputs",
                    "dynamics + hooks",
                    "saved outputs",
                ),
                caption=(
                    "These are the main Toolkit choices: how structures are batched, "
                    "which model outputs are requested, how dynamics runs, and what "
                    "is saved."
                ),
            )
            + "\n\n"
            + callout_html(
                "Toolkit provides the shared data structures and simulation machinery. The model and integrator determine the calculation you are running.",
                kind="note",
            ),
        ),
        code(
            "first-structure",
            """
structure_progress = NotebookProgress(
    title="Inspect the first atomistic system", total=1, unit="structure"
)
water = make_water_monomer()
water.info["charge"] = 0

print(water)
display(view(water, viewer="x3d"))
structure_progress.complete("One nonperiodic H2O structure ready")
display(callout(
    "Structure view: one oxygen atom bonded to two hydrogen atoms in a bent, "
    "nonperiodic water molecule.",
    kind="note",
))
""",
        ),
        code(
            "load-aimnet",
            """
checkpoint_path = resolve_checkpoint_path(MODEL_CHECKPOINT)
checkpoint_progress = NotebookProgress(
    title="Load the verified AIMNet2 checkpoint", total=1, unit="checkpoint"
)
aimnet = AIMNet2Wrapper.from_checkpoint(checkpoint_path, device=DEVICE)
aimnet.eval()
for parameter in aimnet.parameters():
    parameter.requires_grad_(False)
aimnet.set_config("active_outputs", {"energy", "forces", "charges"})
model_card = checkpoint_card(aimnet, MODEL_CHECKPOINT, checkpoint_path)
EXPECTED_DEFAULT_CHECKPOINT_SHA256 = CHECKPOINT_IDENTITIES[
    MODEL_CHECKPOINT
]["sha256"]
checkpoint_is_override = "ALCHEMI_AIMNET_CHECKPOINT" in os.environ
if not checkpoint_is_override:
    assert model_card["checkpoint_sha256"] == EXPECTED_DEFAULT_CHECKPOINT_SHA256
checkpoint_progress.complete("Checkpoint loaded and SHA-256 verified")

aimnet_model_card_display = aimnet_model_card_table(
    model_card,
    aimnet_version=metadata.version("aimnet"),
    cutoff_A=aimnet.model_config.neighbor_config.cutoff,
    supports_pbc=aimnet.model_config.supports_pbc,
    optional_inputs=aimnet.model_config.optional_inputs,
    neighbor_convention=str(aimnet.model_config.neighbor_config),
    device=str(DEVICE),
)
display(readable_table(
    aimnet_model_card_display,
    label="AIMNet2 model card",
    show_index=False,
))
display(callout(
    "Toolkit exposes energy, forces, and charges for this wrapper; all three "
    "are active. The model card above records the required inputs and neighbor "
    "convention used by later calls.",
    kind="result",
    result_state="pass",
))
if checkpoint_is_override:
    display(callout(
        "A different checkpoint was selected through "
        "ALCHEMI_AIMNET_CHECKPOINT. The tutorial reference values were "
        "generated with ensemble member 0, so use the checks below before "
        "interpreting results from the replacement model.",
        kind="note",
    ))
""",
        ),
        code(
            "hello-world",
            """
hello_progress = NotebookProgress(
    title="Run the first Toolkit model call", total=3, unit="steps"
)
hello_data = AtomicData.from_atoms(water, device=DEVICE, dtype=torch.float32)
hello_batch = Batch.from_data_list([hello_data], device=DEVICE)
hello_progress.advance(message="AtomicData packed into a one-graph Batch")
compute_neighbors(hello_batch, config=aimnet.model_config.neighbor_config)
hello_progress.advance(message="model-compatible neighbors built")
hello = aimnet(hello_batch)
hello_progress.complete("energy, forces, and charges returned")

hello_data_table = first_atomic_data_table(
    num_atoms=hello_data.num_nodes,
    atomic_numbers=hello_data.atomic_numbers.cpu().numpy(),
    positions_shape=hello_data.positions.shape,
    cell=hello_data.cell,
    positions_dtype=str(hello_data.positions.dtype),
    device=str(hello_data.positions.device),
)
display(readable_table(
    hello_data_table,
    label="First AtomicData object",
    show_index=False,
))
hello_forces = hello["forces"].detach().cpu()
hello_charges = hello["charges"].detach().cpu().reshape(-1)
hello_system_results, hello_atom_results = first_model_result_tables(
    num_graphs=hello_batch.num_graphs,
    energy_eV=hello["energy"].item(),
    symbols=water.get_chemical_symbols(),
    charges_e=hello_charges.numpy(),
    forces_eV_A=hello_forces.numpy(),
)

display(readable_table(
    hello_system_results,
    label="First system-level outputs",
    show_index=False,
))
display(readable_table(
    hello_atom_results,
    label="First model outputs by atom",
    show_index=False,
))
""",
        ),
        markdown(
            "precision-note",
            r"""
### Precision belongs to each tensor

- `float32` uses 4 bytes per value and roughly 7 significant digits; `float64` uses 8 bytes and roughly 15 to 16 digits.
- Atomic numbers and neighbor indices remain integer tensors.
- `Tensor.to(dtype)` returns a converted tensor. `Module.to(dtype)` changes the module's floating parameters and buffers in place.
"""
            + "\n\n"
            + callout_html(
                'Toolkit data can start in float64, but AIMNet2 kernels require float32. `adapt_input` returns float32 coordinates while leaving the batch coordinate dtype unchanged; the full wrapper then converts the batch positions, and any cell tensor, to float32 in place. A float64 coordinate tensor does not turn AIMNet2 into a float64 model, and widening saved float32 weights cannot recover information. The setting torch.set_float32_matmul_precision("highest") chooses the internal precision for eligible float32 matrix multiplications; it does not change model weights or input dtypes.',
                kind="check",
            ),
        ),
        code(
            "inspect-float-precision",
            """
precision_progress = NotebookProgress(
    title="Inspect floating-point precision", total=3, unit="checks"
)

floating_parameters = [
    parameter
    for _, parameter in aimnet.named_parameters()
    if parameter.is_floating_point()
]
floating_buffers = [
    buffer for buffer in aimnet.buffers() if buffer.is_floating_point()
]
parameter_count = sum(parameter.numel() for parameter in floating_parameters)
parameter_storage_mib = sum(
    parameter.numel() * parameter.element_size()
    for parameter in floating_parameters
) / 2**20
float64_parameter_storage_mib = (
    parameter_count * torch.empty((), dtype=torch.float64).element_size() / 2**20
)
parameter_dtypes = sorted({str(parameter.dtype) for parameter in floating_parameters})
buffer_dtypes = sorted({str(buffer.dtype) for buffer in floating_buffers})
assert parameter_dtypes == ["torch.float32"]
precision_progress.advance(message="checkpoint parameter and buffer storage inspected")

# Observe the dtype before input adaptation and after the full wrapper call.
precision_probe_data = AtomicData.from_atoms(
    water, device=DEVICE, dtype=torch.float64
)
precision_probe_batch = Batch.from_data_list([precision_probe_data], device=DEVICE)
compute_neighbors(precision_probe_batch, config=aimnet.model_config.neighbor_config)
precision_dtype_before = precision_probe_batch.positions.dtype
precision_model_input = aimnet.adapt_input(precision_probe_batch)
precision_dtype_after_adapt = precision_probe_batch.positions.dtype
precision_probe = aimnet(precision_probe_batch)
precision_dtype_after_forward = precision_probe_batch.positions.dtype
assert precision_dtype_before == torch.float64
assert precision_dtype_after_adapt == torch.float64
assert precision_model_input["coord"].dtype == torch.float32
assert precision_dtype_after_forward == torch.float32
assert precision_probe["energy"].dtype == torch.float64
assert precision_probe["forces"].dtype == torch.float32
assert precision_probe["charges"].dtype == torch.float32
precision_progress.advance(message="wrapper input conversion observed")

energy_magnitude_eV = abs(float(hello["energy"].detach().cpu().reshape(())))
energy32 = torch.tensor(energy_magnitude_eV, dtype=torch.float32)
energy64 = torch.tensor(energy_magnitude_eV, dtype=torch.float64)
spacing32_eV = float(torch.nextafter(energy32, torch.full_like(energy32, torch.inf)) - energy32)
spacing64_eV = float(torch.nextafter(energy64, torch.full_like(energy64, torch.inf)) - energy64)
weight_sample32 = floating_parameters[0].detach().reshape(-1)[:16].cpu()
weight_sample64 = weight_sample32.to(torch.float64)
assert torch.equal(weight_sample64.to(torch.float32), weight_sample32)
precision_progress.complete("tensor storage, model input, and numerical spacing shown")

display(readable_table(pd.DataFrame([
    {"Quantity": "hello-world coordinates", "Observed": str(hello_data.positions.dtype)},
    {"Quantity": "probe coordinates before wrapper call", "Observed": str(precision_dtype_before)},
    {"Quantity": "coordinates passed to AIMNet", "Observed": str(precision_model_input["coord"].dtype)},
    {"Quantity": "probe coordinates after wrapper call", "Observed": str(precision_dtype_after_forward)},
    {"Quantity": "checkpoint floating parameters", "Observed": ", ".join(parameter_dtypes)},
    {"Quantity": "checkpoint floating buffers", "Observed": ", ".join(buffer_dtypes)},
    {"Quantity": "floating parameter count", "Observed": f"{parameter_count:,}"},
    {"Quantity": "float32 parameter storage only", "Observed": f"{parameter_storage_mib:.1f} MiB"},
    {"Quantity": "same parameter count in float64", "Observed": f"{float64_parameter_storage_mib:.1f} MiB"},
    {"Quantity": "probe energy / forces / charges", "Observed": " / ".join(str(precision_probe[name].dtype) for name in ("energy", "forces", "charges"))},
    {"Quantity": "float32 spacing at |E(H2O)|", "Observed": f"{spacing32_eV:.3e} eV"},
    {"Quantity": "float64 spacing at |E(H2O)|", "Observed": f"{spacing64_eV:.3e} eV"},
    {"Quantity": "float32 matmul setting", "Observed": torch.get_float32_matmul_precision()},
]), label="Precision used by the first model", show_index=False))
display(callout(
    "The probe begins with float64 coordinates; AIMNet receives float32 coordinates and converts the batch positions in place. Energy is float64 because AIMNet preserves atomic reference-energy shifts and accumulates system energy in float64; forces and charges are float32. Check precision per tensor. A float64 copy of a stored weight recovers no new information, and spacing shows numerical resolution rather than model error.",
    kind="result",
    result_state="observed",
))
del precision_model_input, precision_probe, weight_sample32, weight_sample64
""",
        ),
        markdown(
            "batch-mental-model",
            r"""
### The first interaction energy is already a three-graph batch

`E(AB) - E(A) - E(B)` needs three independently evaluated systems: the dimer and its two frozen monomers.

- A **graph** here means one atomistic system evaluated independently. It is not a chemical bond diagram or a PyTorch computational graph.
- `num_nodes_per_graph` stores each graph size.
- `batch_idx` maps every atom to its graph.
- `batch_ptr` stores the atom offsets where each graph begins and ends.

`segmented_sum(values, batch_idx, num_graphs)` is the tensor-friendly way to add atom-level values separately for each graph. It stays inside PyTorch's differentiation path; later we use it to turn predicted atomic charges into one dipole per system.
"""
            + "\n\n"
            + process_diagram_html(
                title="One ragged Batch, three systems",
                steps=(
                    "AB · 6 atoms",
                    "A · 3 atoms",
                    "B · 3 atoms",
                ),
                caption=(
                    "batch_ptr marks graph boundaries; batch_idx labels every atom. "
                    "The numbered steps show storage order, not chemical bonds."
                ),
            ),
        ),
        code(
            "first-prediction",
            """
first_prediction_progress = NotebookProgress(
    title="Evaluate the AIMNet checkpoint base on one water dimer",
    total=1,
    unit="model call",
)
water_dimer = make_water_dimer(oo_distance=2.90)
water_dimer.info["charge"] = 0
dimer_fragments = [water_dimer, water_dimer[:3], water_dimer[3:]]
for atoms in dimer_fragments:
    atoms.info["charge"] = 0
dimer_data = [
    AtomicData.from_atoms(atoms, device=DEVICE, dtype=torch.float64)
    for atoms in dimer_fragments
]
single_batch = Batch.from_data_list(dimer_data, device=DEVICE)
compute_neighbors(single_batch, config=aimnet.model_config.neighbor_config)
first_outputs = aimnet(single_batch)

first_energy = first_outputs["energy"].detach().reshape(-1)
residual_interaction_kJ_mol = float(
    (first_energy[0] - first_energy[1] - first_energy[2]).cpu()
) * 96.48533212331002
charge_sums = segmented_sum(
    first_outputs["charges"].reshape(-1),
    single_batch.batch_idx.to(torch.int32),
    single_batch.num_graphs,
)
first_force_max = float(
    torch.linalg.vector_norm(first_outputs["forces"][:6], dim=1).max().detach().cpu()
)
first_prediction_progress.complete(
    "Checkpoint-base energy, forces, and charges evaluated"
)
display(readable_table(pd.Series({
    "graphs": single_batch.num_graphs,
    "atoms": single_batch.num_nodes,
    "checkpoint_base_interaction_kJ_mol": residual_interaction_kJ_mol,
    "dimer_max_force_eV_A": first_force_max,
    "max_abs_graph_charge_e": float(charge_sums.abs().max().detach().cpu()),
}, name="Value").rename_axis("Output").reset_index(),
    label="First model call",
    show_index=False,
))
display(callout(
    f"Observed checkpoint-base interaction: "
    f"{residual_interaction_kJ_mol:.2f} kJ/mol. The embedded SRCoulomb module "
    "has already subtracted a short-range Coulomb term; the base remains "
    "incomplete until full Coulomb and D3 are added.",
    kind="result",
    result_state="observed",
))
""",
        ),
        stage_markdown(
            "stage-2",
            stage=2,
            title="Same calculation, one batch",
            outcome="Check serial and batched agreement, then measure CPU/GPU and homogeneous/heterogeneous batch behavior.",
            before="Predict the shape of batch_idx and batch_ptr for eight AB/A/B triplets before inspecting them.",
            compute_time=(
                "19 s on one H100 PCIe in the checked run"
            ),
            body=(
                r"""
- **Serial:** one graph and one model call at a time.
- **Batched:** the same graphs and model, one call.
- **Homogeneous:** similar graph sizes. **Heterogeneous:** mixed graph sizes; one mixed call can trade call overhead for irregular atom and neighbor work.

- CPU wins when launch and transfer overhead dominate a small workload.
- GPU wins when many atoms, edges, or graphs keep its parallel lanes occupied.
- Batching increases useful work per launch; mixed graph sizes can make the work less even.
- The crossover below times five separately synchronized warm-call blocks and
  reports the median and interquartile range. Checkpoint loading, batch
  construction, host→device placement, and neighbor construction remain
  outside the timed region.
"""
                + "\n\n"
                + process_diagram_html(
                    title="CPU latency and GPU throughput",
                    steps=(
                        "small batch · launch overhead dominates",
                        "more graphs per call",
                        "parallel GPU lanes stay occupied",
                    ),
                    caption=(
                        "The crossover is measured, not assumed. Heterogeneous graph "
                        "sizes can shift it by changing neighbor-storage utilization."
                    ),
                )
            ),
        ),
        code(
            "build-dimer-scan",
            """
scan_build_progress = NotebookProgress(
    title="Build the dimer interaction inputs", total=2, unit="steps"
)
scan_dimers = make_water_dimer_scan(DIMER_DISTANCES_A)
scan_atoms = []
scan_roles = []
for dimer in scan_dimers:
    fragments = (dimer, dimer[:3], dimer[3:])
    for role, atoms in zip(("AB", "A", "B"), fragments, strict=True):
        atoms = atoms.copy()
        atoms.set_pbc(False)
        atoms.info["charge"] = 0
        scan_atoms.append(atoms)
        scan_roles.append(role)
scan_build_progress.advance(
    message=f"{len(scan_atoms)} frozen AB/A/B structures generated"
)

scan_data = [
    AtomicData.from_atoms(atoms, device=DEVICE, dtype=torch.float32)
    for atoms in scan_atoms
]
scan_build_progress.complete("all structures converted to AtomicData")
print("structures:", len(scan_data), "=", len(DIMER_DISTANCES_A), "× (AB, A, B)")
""",
        ),
        code(
            "serial-batch-agreement",
            """
aimnet.set_config("active_outputs", {"energy", "charges"})

serial_batch_progress = NotebookProgress(
    title="Serial loop vs one Toolkit batch",
    total=len(scan_data) + 1,
    unit="model calls",
)
serial_energy = []
for serial_index, data in enumerate(scan_data, start=1):
    one_graph = Batch.from_data_list([data], device=DEVICE)
    compute_neighbors(one_graph, config=aimnet.model_config.neighbor_config)
    serial_energy.append(aimnet(one_graph)["energy"].detach().reshape(()))
    serial_batch_progress.update(
        done=serial_index,
        message=f"serial graph {serial_index} of {len(scan_data)}",
    )
serial_energy = torch.stack(serial_energy)

scan_batch = Batch.from_data_list(scan_data, device=DEVICE)
compute_neighbors(scan_batch, config=aimnet.model_config.neighbor_config)
batch_outputs = aimnet(scan_batch)
batch_energy = batch_outputs["energy"].detach().reshape(-1)
serial_batch_progress.complete("Serial loop and one batched call complete")

serial_batch_error = float(torch.max(torch.abs(serial_energy - batch_energy)).cpu())
assert serial_batch_error < RESIDUAL_SERIAL_BATCH_TOLERANCE_EV
residual_triplets = batch_energy.cpu().numpy().reshape(-1, 3)
residual_interaction_eV = (
    residual_triplets[:, 0] - residual_triplets[:, 1] - residual_triplets[:, 2]
)

display(readable_table(pd.Series({
    "num_graphs": scan_batch.num_graphs,
    "num_atoms": scan_batch.num_nodes,
    "batch_idx_shape": tuple(scan_batch.batch_idx.shape),
    "batch_ptr": scan_batch.batch_ptr.cpu().tolist(),
    "max_abs_serial_minus_batch_eV": serial_batch_error,
}, name="Value").rename_axis("Field").reset_index(),
    label="Serial and batched evaluation summary",
    show_index=False,
))

# Three public ways to move back from a batch to graph-level data.
first_graph = scan_batch.get_data(0)
roundtrip_graphs = scan_batch.to_data_list()
dimer_only_batch = scan_batch.index_select(list(range(0, scan_batch.num_graphs, 3)))
assert first_graph.num_nodes == 6
assert len(roundtrip_graphs) == scan_batch.num_graphs
assert dimer_only_batch.num_graphs == len(DIMER_DISTANCES_A)
print("get_data(0) atoms :", first_graph.num_nodes)
print("to_data_list size :", len(roundtrip_graphs))
print("index_select dimers:", dimer_only_batch.num_graphs)
display(callout(
    f"Serial and batched checkpoint-base energies agree within {RESIDUAL_SERIAL_BATCH_TOLERANCE_EV:.1e} eV.",
    kind="result",
    result_state="pass",
))
""",
        ),
        markdown(
            "core-neighbor-path",
            r"""
### Use the model-aware neighbor path

Toolkit provides several neighbor-search implementations. At Core level, we do not select one by name. For this small, nonperiodic molecular batch, Toolkit chooses a compatible implementation from the model's `NeighborConfig`.

- **Fixed coordinates:** `compute_neighbors(batch, config=model.model_config.neighbor_config)` uses the declared cutoff, output format, and full- or half-list setting.
- **Moving coordinates:** register the hooks from `model.make_neighbor_hooks()`. They use the same model requirements and the configured skin to decide when neighbors must be rebuilt.

This keeps the neighbor list matched to the model. Direct kernel selection belongs in a separate performance study, not in this tutorial.
"""
            + "\n\n"
            + callout_html(
                "Use the Core path here: one `compute_neighbors(...)` call for fixed structures, then model-generated hooks when coordinates move.",
                kind="note",
            ),
        ),
        code(
            "cpu-gpu-crossover",
            """
benchmark_progress = NotebookProgress(
    title="CPU / GPU fixed-workload sweep", total=6, unit="checks"
)

# Fresh wrappers make the first-call row distinct from the warm-call block.
del aimnet
torch.cuda.empty_cache()
aimnet = AIMNet2Wrapper.from_checkpoint(checkpoint_path, device=DEVICE)
cpu_aimnet = AIMNet2Wrapper.from_checkpoint(checkpoint_path, device="cpu")
for route_model in (aimnet, cpu_aimnet):
    route_model.eval()
    for parameter in route_model.parameters():
        parameter.requires_grad_(False)
    route_model.set_config("active_outputs", {"energy"})
benchmark_progress.advance(message="Fresh CPU and GPU wrappers loaded")

# First versus warm calls on one fixed 32-dimer workload.
timing_atoms = [make_water_dimer(2.90) for _ in range(32)]
for atoms in timing_atoms:
    atoms.info["charge"] = 0
gpu_timing_batch = Batch.from_data_list([
    AtomicData.from_atoms(atoms, device=DEVICE, dtype=torch.float64)
    for atoms in timing_atoms
], device=DEVICE)
cpu_timing_batch = Batch.from_data_list([
    AtomicData.from_atoms(atoms, device="cpu", dtype=torch.float64)
    for atoms in timing_atoms
], device="cpu")
compute_neighbors(gpu_timing_batch, config=aimnet.model_config.neighbor_config)
compute_neighbors(cpu_timing_batch, config=cpu_aimnet.model_config.neighbor_config)
cold_warm = pd.DataFrame(
    first_and_warm_call_rows(aimnet, gpu_timing_batch, warm_calls=20, route="GPU")
    + first_and_warm_call_rows(cpu_aimnet, cpu_timing_batch, warm_calls=20, route="CPU")
)
benchmark_progress.advance(message="First and warm 32-graph calls measured")
display(readable_table(
    cold_warm[[
        "route", "phase", "calls", "wall_ms_per_pass", "structures_per_s"
    ]].round(2),
    label="First and warm CPU/GPU calls",
    show_index=False,
))

# Same structures, coordinates, neighbors, and number of timed calls.
benchmark_rows = []
for sweep_index, batch_size in enumerate((1, 8, 32, 128), start=1):
    atoms_set = [make_water_dimer(2.90) for _ in range(batch_size)]
    for atoms in atoms_set:
        atoms.info["charge"] = 0
    gpu_batch = Batch.from_data_list([
        AtomicData.from_atoms(atoms, device=DEVICE, dtype=torch.float64)
        for atoms in atoms_set
    ], device=DEVICE)
    cpu_batch = Batch.from_data_list([
        AtomicData.from_atoms(atoms, device="cpu", dtype=torch.float64)
        for atoms in atoms_set
    ], device="cpu")
    compute_neighbors(gpu_batch, config=aimnet.model_config.neighbor_config)
    compute_neighbors(cpu_batch, config=cpu_aimnet.model_config.neighbor_config)

    gpu_check = aimnet(gpu_batch)["energy"].detach().cpu().reshape(-1)
    cpu_check = cpu_aimnet(cpu_batch)["energy"].detach().cpu().reshape(-1)
    cpu_gpu_error = float(torch.max(torch.abs(gpu_check - cpu_check)))
    assert cpu_gpu_error < 2e-4
    rows = compare_fixed_workload_devices(
        {"GPU": (aimnet, gpu_batch), "CPU": (cpu_aimnet, cpu_batch)},
        warmup_calls=2,
        measured_calls=20,
    )
    for row in rows:
        row["batch_size"] = batch_size
        row["cpu_gpu_max_abs_energy_ueV"] = cpu_gpu_error * 1e6
    benchmark_rows.extend(rows)
    benchmark_progress.update(
        done=2 + sweep_index,
        message=f"batch size {batch_size}: CPU and GPU measured",
    )

crossover = pd.DataFrame(benchmark_rows)
benchmark_progress.complete("Four matched batch sizes complete")
display(readable_table(
    crossover.sort_values(["batch_size", "route"])[[
        "batch_size", "route", "calls", "wall_ms_per_pass", "structures_per_s",
        "atoms_per_s", "cpu_gpu_max_abs_energy_ueV",
    ]].round(2),
    label="CPU/GPU warm-call crossover",
    show_index=False,
))
display(callout(
    "Small batches are latency-bound; larger batches expose GPU parallelism. This is a synchronized warm-inference crossover, not end-to-end application latency.",
    kind="result",
    result_state="observed",
))
del cpu_aimnet, cpu_batch, gpu_batch, cpu_timing_batch, gpu_timing_batch
""",
        ),
        code(
            "batch-layouts",
            """
# Fixed heterogeneous workload: 24 monomers, 12 dimers, 4 hexamers (216 atoms).
layout_progress = NotebookProgress(
    title="Homogeneous buckets vs one heterogeneous batch",
    total=1,
    unit="fixed-workload comparison",
)
layout_groups = [
    [make_water_monomer() for _ in range(24)],
    [make_water_dimer(2.90) for _ in range(12)],
    [make_ir_structures()[0][2].copy() for _ in range(4)],
]
for atoms in sum(layout_groups, []):
    atoms.info["charge"] = 0

mixed_atoms = sum(layout_groups, [])
mixed_batch = Batch.from_data_list([
    AtomicData.from_atoms(atoms, device=DEVICE, dtype=torch.float64)
    for atoms in mixed_atoms
], device=DEVICE)
bucket_batches = [
    Batch.from_data_list([
        AtomicData.from_atoms(atoms, device=DEVICE, dtype=torch.float64)
        for atoms in group
    ], device=DEVICE)
    for group in layout_groups
]
bucket_graph_indices = [
    np.arange(0, 24),
    np.arange(24, 36),
    np.arange(36, 40),
]
for route_batch in (mixed_batch, *bucket_batches):
    compute_neighbors(route_batch, config=aimnet.model_config.neighbor_config)

neighbor_storage = []
for route, route_batches in (
    ("one heterogeneous batch", [mixed_batch]),
    ("homogeneous buckets", bucket_batches),
):
    valid_slots = sum(int(item.num_neighbors.sum().cpu()) for item in route_batches)
    allocated_slots = sum(int(item.neighbor_matrix.numel()) for item in route_batches)
    neighbor_storage.append({
        "route": route,
        "valid_neighbor_slots": valid_slots,
        "allocated_neighbor_slots": allocated_slots,
        "neighbor_slot_utilization": valid_slots / allocated_slots,
    })

layout_result = compare_mixed_and_bucketed(
    aimnet,
    mixed_batch,
    bucket_batches,
    bucket_graph_indices,
    warmup_passes=BENCHMARK_WARMUP_CALLS,
    measured_passes=BENCHMARK_PASSES_PER_REPEAT,
    measured_repeats=BENCHMARK_REPEATS,
    atol=1e-5,
    rtol=0.0,
)
layout_progress.complete("Both layouts measured over 20 passes")
display(readable_table(
    pd.DataFrame(layout_result["timings"])[[
        "route", "calls_per_pass", "graphs", "atoms", "wall_ms_per_pass",
        "structures_per_s",
    ]].round(2),
    label="Homogeneous and heterogeneous batch timings",
    show_index=False,
))
display(readable_table(
    pd.DataFrame(neighbor_storage).round(3),
    label="Neighbor storage by batch layout",
    show_index=False,
))
display(callout(
    f"Both layouts return the same energies (max |Δ| = {layout_result['max_abs_energy_difference']:.2e} eV). Which layout wins is empirical.",
    kind="result",
    result_state="pass",
))
""",
        ),
        markdown(
            "distributed-pipeline-intro",
            r"""
### Run different workflow stages on different GPUs

A **rank** is one worker process, normally attached to one GPU. `DistributedPipeline` is intended for collections of independent systems that pass through different workflow stages. It does not split one trajectory or one model call across GPUs.

- **`Batch`** packs independent graphs into one model call on one device.
- **`stage_a + stage_b`** builds a `FusedStage`; both stages share one rank and GPU.
- **`stage_a | stage_b`** is the two-rank `DistributedPipeline` shorthand.
- **`SizeAwareSampler`** limits the active work. **`BufferConfig`** sets the maximum batch that can move between ranks.

"""
            + "\n\n"
            + callout_html(
                "No multi-GPU timing is reported for this Toolkit release candidate. It fixes reusable-buffer capacity and waits for an asynchronous send before reusing its storage, but Batch.put still skips non-float32 atom and edge fields such as atomic_numbers. The CPU preflight therefore stops before GPU timing. The classic pipeline also makes every rank join a global completion check (`all_reduce`) after every iteration.",
                kind="result",
                result_state="not_reported",
            )
            + "\n\n"
            + r"""

If two stages work on different batches at the same time, throughput is limited by the slower stage. For stage times `t₁` and `t₂`, the ideal two-GPU speedup approaches `(t₁ + t₂) / max(t₁, t₂)` and cannot exceed 2×. Starting the second stage and finishing the batches already inside the pipeline make the real speedup smaller.

The planned H100 comparison keeps the complete **AIMNet2 B97-3c checkpoint base + finite all-pairs Coulomb + pairwise D3(BJ)** potential and processes 8,192 generated water hexamers through one sustained workflow per route. Each `SizeAwareSampler` owns its complete dataset partition while admitting at most 512 systems at a time:

| Route | Public workflow | Dataset and active batch |
|---|---|---:|
| 1 H100 | one `FusedStage`: FIRE2 + NVT + NVE | 8,192 total; at most 512 active |
| 2 H100s | rank 0 FIRE2 → rank 1 NVT + NVE | 8,192 total; at most 512 active |
| 4 H100s | independent pairs 0→1 and 2→3 | 4,096 per pair; at most 512 active per pair |

The four-GPU route uses one `DistributedPipeline` with two independent stage pairs, 0 → 1 and 2 → 3; rank 1 never sends to rank 2. Each upstream sampler owns a complete 4,096-system partition. The script creates the stages, samplers, and pipeline once, then keeps the same pipeline and transfer buffers active until both partitions are exhausted.

Relative to the one-GPU fused route, the ideal upper bound is 2× for one two-stage pair. Two pairs can also divide the independent systems between them, making the four-GPU upper bound 4×. These are upper bounds, not measured results; communication, unequal stage times, setup, and time when only one stage has work all reduce them.

This notebook does not launch a multi-GPU job. The complete run is implemented in
[`scripts/benchmark_part1_distributed_campaign.py`](../scripts/benchmark_part1_distributed_campaign.py)
and launched with `torchrun`. For each 0 → 1 pair, the benchmark script builds:

| Object | Setting for one 0 → 1 pair |
|---|---|
| `SizeAwareSampler` | complete pair partition; `max_atoms=512 * 18`, `max_edges=None`, `max_batch_size=512` |
| `BufferConfig` | `num_systems=512`, `num_nodes=512 * 18`, `num_edges=0` |
| upstream `FIRE2` | `prior_rank=None`, `next_rank=1`, sampler and buffer attached |
| downstream `FusedStage` | NVT + NVE, `prior_rank=0`, `next_rank=None`, same buffer |
| final NVE hook | `ConvergedSnapshotHook(sink=HostMemory(...))` |

After `torchrun` starts the worker processes, the benchmark script initializes
NCCL and builds those stages. The following function shows the same two-rank
setup using the public API. It is not called by this single-rank notebook:

```python
import torch
from nvalchemi.dynamics import DistributedPipeline, FIRE2, FusedStage

def run_two_rank_pipeline(
    *,
    relaxation: FIRE2,
    dynamics: FusedStage,
    device: torch.device,
) -> None:
    pipeline = DistributedPipeline(
        stages={0: relaxation, 1: dynamics},
        synchronized=False,
        backend="nccl",
        device_id=device,
    )
    with pipeline:
        pipeline.run()
```

The context manager performs Toolkit's pipeline setup and cleanup. Here `synchronized=False` disables the pipeline's optional barrier. It does not remove the per-iteration rank wait in the current `run()` loop.

`SizeAwareSampler(max_edges=None)` and `BufferConfig(num_edges=0)` have different jobs. The sampler limits which systems are active. The transfer buffer does not send neighbor arrays; model hooks rebuild them on the receiving rank. `ConvergedSnapshotHook` writes each finished system to CPU `HostMemory` exactly once.

#### What must be checked before timing

Toolkit 0.2 fixes reusable-buffer capacity and waits for an asynchronous send before reusing its storage. It still copies only float32 segmented fields in `Batch.put`; integer fields such as `atomic_numbers` fail the CPU preflight. Before timing, Toolkit must preserve the complete float, integer, and boolean batch, and the H100 run must verify repeated buffer writes, clean termination, and real overlap between stages. The classic `run()` loop still makes every rank join a global completion check (`all_reduce`) after every iteration, so overlap must be measured rather than assumed. Earlier failures and the replacement-run plan are recorded in the [runtime notes](../RUNTIME_SNAPSHOT.md).
"""
            + "\n\n"
            + callout_html(
                "What would run in parallel: many independent hexamers, not the steps of one trajectory. The 25,000-step IR calculation above stays on one GPU. Multi-GPU timings remain NOT REPORTED until Toolkit transfers the complete batch and the exact unmodified H100 run demonstrates correct results and stage overlap.",
                kind="note",
            ),
        ),
        code(
            "pipeline-campaign-results",
            """
PLANNED_CAMPAIGN_SYSTEMS_TOTAL = 8_192
DISTRIBUTED_PIPELINE_NOT_REPORTED_REASON = (
    "Toolkit 0.2 does not transfer the complete Batch, and stage overlap has "
    "not been demonstrated."
)
campaign_progress = NotebookProgress(
    title="Report the pipeline status", total=1, unit="check"
)
campaign_progress.complete("correctness, overlap, and timing are not reported")
display(callout(
    "NOT REPORTED: " + DISTRIBUTED_PIPELINE_NOT_REPORTED_REASON,
    kind="result",
    result_state="not_reported",
))
""",
        ),
        stage_markdown(
            "stage-3",
            stage=3,
            title="Complete and check the potential",
            outcome="Evaluate 90 NCI Atlas graphs in a few passes, add the checkpoint's Coulomb and D3 terms, and compare the complete interaction curves with DFT-D3 and CCSD(T)/CBS.",
            before="Before running: predict which interaction class will respond most strongly when explicit Coulomb is restored.",
            compute_time="22 s on one H100 PCIe in the checked run",
            body=(
                r"""
**NCI** means noncovalent interaction. NCI Atlas is a benchmark collection of molecular complexes and reference interaction energies. Intermolecular interactions influence solvation, molecular recognition, crystal packing, self-assembly, and the stability of molecular materials. The calculation below spans neutral hydrogen bonding, dispersion-dominated binding, and an ionic hydrogen bond.

For every geometry, the interaction energy is `E(AB) - E(A) - E(B)`. Ten separations for three complexes therefore become 90 graphs: 30 geometries times AB, A, and B. The same batch supports all four model variants.

- **Core:** the learned checkpoint output, deliberately incomplete.
- **Core + D3:** restore the explicit pairwise dispersion correction; electrostatics remain omitted.
- **Core + Coulomb:** restore interactions between the predicted partial charges; D3 remains omitted.
- **Complete model:** core + Coulomb + D3, the checkpoint's intended convention.

These are finite gas-phase complexes, so the electrostatic term is direct nonperiodic all-pairs `1/r`. Ewald and PME are for periodic cells and are not used here. The partial curves show what changes when one term is omitted; only the complete model is compared as the final prediction.
"""
                + "\n\n"
                + process_diagram_html(
                    title="One set, a few model passes",
                    steps=(
                        "30 AB/A/B geometry groups",
                        "one 90-graph Batch",
                        "four AIMNet + four Coulomb calls + one shared D3 call",
                        "interaction curves + two references",
                    ),
                    caption=(
                        "Batching changes the number of model calls, not the structures "
                        "or the interaction-energy definition."
                    ),
                )
                + "\n\n"
                + callout_html(
                    "Reference levels: absolute DFT energies use ωB97M-D3(BJ)/def2-TZVPPD, which is close to but not identical to the checkpoint's ωB97M-D3/def2-TZVPP training level. CCSD(T)/CBS supplies an independent interaction-energy reference. Ensemble spread is model disagreement, not calibrated uncertainty.",
                    kind="note",
                )
            ),
        ),
        code(
            "load-nci-atlas",
            """
nci_data_progress = NotebookProgress(
    title="Build the 90-graph NCI Atlas batch", total=3, unit="steps"
)
NCI_DATA_FILE = PART_DIR / "data" / "nci_atlas" / "nci-atlas-curves.csv.gz"
nci_reference_data = load_nci_atlas_subset(NCI_DATA_FILE)
nci_atoms = rows_to_atoms(nci_reference_data)
nci_graph_index = build_graph_index(nci_reference_data)
nci_data_progress.advance(message="three attributed interaction curves loaded")


def fresh_nci_batch(device=DEVICE):
    data = [
        AtomicData.from_atoms(atoms, device="cpu", dtype=torch.float32)
        for atoms in nci_atoms
    ]
    return Batch.from_data_list(data, device=device)


nci_batch = fresh_nci_batch()
assert nci_batch.num_graphs == 90
nci_data_progress.advance(
    message=f"{nci_batch.num_nodes:,} atoms packed without graph padding"
)
nci_data_progress.complete("30 AB/A/B groups retain their source row order")
display(readable_table(
    nci_reference_data[[
        "subset", "system_name", "interaction_class"
    ]].drop_duplicates(ignore_index=True),
    label="NCI Atlas tutorial subset",
    show_index=False,
))
print("batch_ptr shape:", tuple(nci_batch.batch_ptr.shape))
print("graphs:", nci_batch.num_graphs, "= 3 systems × 10 separations × AB/A/B")
""",
        ),
        code(
            "configure-nci-model",
            """
nci_model_progress = NotebookProgress(
    title="Configure AIMNet, Coulomb, and D3", total=4, unit="components"
)
NCI_CHECKPOINTS = [f"aimnet2-wb97m-d3_{index}" for index in range(4)]
aimnet_checkpoint_identities = verify_checkpoint_identities(
    CHECKPOINT_IDENTITIES
)
NCI_D3_SMOOTHING_FRACTION = DOMAIN_METHODOLOGY.d3_smoothing_fraction
NCI_CHARGE_ATOL_E = 2.0e-4
NCI_ENERGY_ATOL_EV = 3.0e-5
NCI_ENERGY_RTOL = 2.0e-6
NCI_NET_FORCE_ATOL_EV_A = 5.0e-3
NCI_FD_STEP_A = COMPOSITION_FD_STEP_A
NCI_FD_ATOL_EV_A = COMPOSITION_FD_FORCE_TOLERANCE_EV_A
NCI_FD_RTOL = 2.0e-2
NCI_PIPELINE_OFFICIAL_FORCE_ATOL_EV_A = (
    COMPOSITION_FORCE_AGREEMENT_TOLERANCE_EV_A
)
nci_aimnet = AIMNet2Wrapper.from_checkpoint(
    NCI_CHECKPOINTS[0], device=DEVICE, compile_model=False
).eval()
nci_metadata = dict(nci_aimnet.model.metadata)
assert nci_metadata["needs_coulomb"] is True
assert nci_metadata["needs_dispersion"] is True
assert nci_metadata["coulomb_mode"] == "sr_embedded"
assert abs(nci_metadata["coulomb_sr_rc"] - 4.6) < 1.0e-5
nci_aimnet.set_config("active_outputs", {"energy", "charges"})
nci_model_progress.advance(
    message="five checkpoint files verified; ensemble member 0 loaded"
)

nci_d3_params = nci_metadata["d3_params"]
nci_d3 = DFTD3ModelWrapper(
    a1=nci_d3_params["a1"],
    a2=nci_d3_params["a2"],
    s8=nci_d3_params["s8"],
    s6=nci_d3_params.get("s6", 1.0),
    cutoff=D3_CUTOFF_A,
    smoothing_fraction=NCI_D3_SMOOTHING_FRACTION,
    param_file=D3_PARAMETER_FILE,
    # The image does not redistribute this generated parameter cache.
    # Toolkit creates it from its official source when it is absent.
    auto_download=True,
).to(DEVICE).eval()
D3_PARAMETER_SHA256 = sha256_file(D3_PARAMETER_FILE)
assert D3_PARAMETER_SHA256 == EXPECTED_D3_PARAMETER_SHA256
nci_d3.set_config("active_outputs", {"energy"})
nci_model_progress.advance(
    message="pairwise D3(BJ) data checked and checkpoint settings applied"
)

nci_coulomb = DirectCoulombWrapper().to(DEVICE).eval()
nci_coulomb.set_config("active_outputs", {"energy"})
nci_model_progress.advance(message="finite all-pairs Coulomb configured")
nci_model_progress.complete("three components are ready for the same 90 graphs")
display(readable_table(pd.DataFrame([
    {"component": "AIMNet core", "depends on": "positions, elements, total charge"},
    {"component": "all-pairs Coulomb", "depends on": "predicted charges"},
    {"component": "pairwise D3(BJ)", "depends on": "positions, elements"},
]), label="Model components", show_index=False))
""",
        ),
        code(
            "evaluate-nci-components",
            """
nci_evaluation_progress = NotebookProgress(
    title="Evaluate the NCI set in nine batched calls", total=9, unit="calls"
)

# D3 depends only on geometry, so all ensemble members share this one pass.
nci_d3_batch = fresh_nci_batch()
compute_neighbors(nci_d3_batch, config=nci_d3.model_config.neighbor_config)
with torch.no_grad():
    nci_d3_graph_eV = nci_d3(nci_d3_batch)["energy"].reshape(-1).cpu()
nci_evaluation_progress.advance(message="D3: one pass over 90 graphs")


nci_member_residual_eV, nci_member_coulomb_eV, nci_charge_residuals_e = [], [], []
for member_index, checkpoint in enumerate(NCI_CHECKPOINTS):
    wrapper = nci_aimnet if member_index == 0 else AIMNet2Wrapper.from_checkpoint(
        checkpoint, device=DEVICE, compile_model=False
    ).eval()
    member_batch = fresh_nci_batch()
    wrapper.set_config("active_outputs", {"energy", "charges"})
    compute_neighbors(member_batch, config=wrapper.model_config.neighbor_config)
    with torch.no_grad():
        member_outputs = wrapper(member_batch)
    nci_evaluation_progress.advance(
        message=f"AIMNet ensemble member {member_index}"
    )
    member_batch.charges = member_outputs["charges"]
    graph_charge = segmented_sum(
        member_batch.charges,
        member_batch.batch_idx.to(torch.int32),
        member_batch.num_graphs,
    ).reshape(-1)
    torch.testing.assert_close(
        graph_charge,
        member_batch.charge.reshape(-1),
        atol=NCI_CHARGE_ATOL_E,
        rtol=0.0,
    )
    nci_charge_residuals_e.append(float(
        (graph_charge - member_batch.charge.reshape(-1)).abs().max().cpu()
    ))
    with torch.no_grad():
        member_coulomb = nci_coulomb(member_batch)["energy"]
    nci_evaluation_progress.advance(
        message=f"Coulomb from member {member_index} charges"
    )
    nci_member_residual_eV.append(member_outputs["energy"].reshape(-1).cpu())
    nci_member_coulomb_eV.append(member_coulomb.reshape(-1).cpu())

nci_member_residual_eV = torch.stack(nci_member_residual_eV)
nci_member_coulomb_eV = torch.stack(nci_member_coulomb_eV)
assert nci_member_residual_eV.shape == nci_member_coulomb_eV.shape == (4, 90)
nci_charge_conservation_max_abs_e = max(nci_charge_residuals_e)
nci_evaluation_progress.complete("four AIMNet, four Coulomb, and one D3 call complete")
print("largest graph-charge residual / e:", nci_charge_conservation_max_abs_e)
""",
        ),
        code(
            "compose-nci-pipeline",
            """
nci_pipeline_progress = NotebookProgress(
    title="Compose the complete model", total=3, unit="checks"
)
nci_aimnet.set_config("active_outputs", {"energy", "charges"})
nci_coulomb.set_config("active_outputs", {"energy"})
nci_d3.set_config("active_outputs", {"energy", "forces"})
nci_full_model = PipelineModelWrapper(
    groups=[
        PipelineGroup(
            steps=[PipelineStep(model=nci_aimnet), PipelineStep(model=nci_coulomb)],
            use_autograd=True,
        ),
        PipelineGroup(
            steps=[PipelineStep(model=nci_d3)], use_autograd=False
        ),
    ],
    neighbor_adaptation="always",
).to(DEVICE).eval()
nci_full_model.set_config("active_outputs", {"energy", "forces"})
nci_pipeline_progress.advance(message="charge-dependent and independent groups assembled")


def nci_pipeline_outputs(atoms_sequence):
    batch = Batch.from_data_list([
        AtomicData.from_atoms(atoms, device="cpu", dtype=torch.float32)
        for atoms in atoms_sequence
    ], device=DEVICE)
    compute_neighbors(batch, config=nci_full_model.model_config.neighbor_config)
    return nci_full_model(batch), batch


nci_full_outputs, nci_full_batch = nci_pipeline_outputs(nci_atoms)
nci_member0_sum = (
    nci_member_residual_eV[0] + nci_member_coulomb_eV[0] + nci_d3_graph_eV
)
nci_pipeline_energy_cpu = nci_full_outputs["energy"].detach().reshape(-1).cpu()
nci_component_sum_max_abs_eV = float(
    (nci_pipeline_energy_cpu - nci_member0_sum).abs().max()
)
torch.testing.assert_close(
    nci_pipeline_energy_cpu,
    nci_member0_sum,
    atol=NCI_ENERGY_ATOL_EV,
    rtol=NCI_ENERGY_RTOL,
)
nci_pipeline_progress.advance(message="one composed pass matches the component sum")

nci_reversed_outputs, _ = nci_pipeline_outputs(list(reversed(nci_atoms)))
nci_reversed_energy_cpu = (
    nci_reversed_outputs["energy"].detach().reshape(-1).cpu().flip(0)
)
nci_graph_order_max_abs_eV = float(
    (nci_reversed_energy_cpu - nci_pipeline_energy_cpu).abs().max()
)
torch.testing.assert_close(
    nci_reversed_energy_cpu,
    nci_pipeline_energy_cpu,
    atol=NCI_ENERGY_ATOL_EV,
    rtol=NCI_ENERGY_RTOL,
)
nci_pipeline_progress.complete("graph order does not change graph energies")
""",
        ),
        code(
            "check-nci-force",
            """
nci_force_progress = NotebookProgress(
    title="Check one complete-model force", total=3, unit="checks"
)
nci_example_index = nci_graph_index.index[
    (nci_graph_index["system_id"] == DOMAIN_METHODOLOGY.nci_system_id)
    & np.isclose(nci_graph_index["scale"], 1.0)
    & (nci_graph_index["fragment"] == "AB")
].item()
nci_example = nci_atoms[nci_example_index]
nci_example_output, _ = nci_pipeline_outputs([nci_example])
nci_example_force = nci_example_output["forces"].detach()
nci_net_force_eV_A = nci_example_force.sum(dim=0)
nci_net_force_max_abs_eV_A = float(nci_net_force_eV_A.abs().max().cpu())
torch.testing.assert_close(
    nci_net_force_eV_A,
    torch.zeros(3, device=DEVICE),
    atol=NCI_NET_FORCE_ATOL_EV_A,
    rtol=0.0,
)
nci_force_progress.advance(message="net force is zero within tolerance")

# Use AIMNet2's official complete-model calculator as an independent route.
nci_official = AIMNet2Calculator(
    str(resolve_checkpoint_path(NCI_CHECKPOINTS[0])),
    device=str(DEVICE),
    needs_coulomb=True,
    needs_dispersion=True,
    compile_model=False,
    train=False,
)
nci_official.set_lrcoulomb_method("simple")
nci_official.set_dftd3_cutoff(
    cutoff=D3_CUTOFF_A,
    smoothing_fraction=NCI_D3_SMOOTHING_FRACTION,
)


def nci_official_outputs(atoms):
    data = AtomicData.from_atoms(atoms, device=DEVICE, dtype=torch.float64)
    batch = Batch.from_data_list([data], device=DEVICE)
    official_input = {"coord": batch.positions, "numbers": batch.atomic_numbers}
    official_input["charge"] = batch.charge.reshape(-1)
    official_input["mol_idx"] = batch.batch_idx
    return nci_official(official_input, forces=True)


nci_official_base = nci_official_outputs(nci_example)
nci_fd_flat_index = int(
    nci_official_base["forces"].abs().reshape(-1).argmax().cpu()
)
nci_fd_atom_index, nci_fd_axis = divmod(nci_fd_flat_index, 3)
nci_official_analytic_force_eV_A = float(
    nci_official_base["forces"][nci_fd_atom_index, nci_fd_axis]
    .detach().cpu()
)
nci_official_fd_energies_eV = []
for sign in (-1.0, 1.0):
    displaced = nci_example.copy()
    displaced.positions[nci_fd_atom_index, nci_fd_axis] += sign * NCI_FD_STEP_A
    displaced_output = nci_official_outputs(displaced)
    nci_official_fd_energies_eV.append(float(
        displaced_output["energy"].detach().cpu().reshape(())
    ))
nci_official_fd_force_eV_A = -(
    nci_official_fd_energies_eV[1] - nci_official_fd_energies_eV[0]
) / (2 * NCI_FD_STEP_A)
nci_official_fd_error_eV_A = abs(
    nci_official_analytic_force_eV_A - nci_official_fd_force_eV_A
)
np.testing.assert_allclose(
    nci_official_analytic_force_eV_A,
    nci_official_fd_force_eV_A,
    rtol=NCI_FD_RTOL,
    atol=NCI_FD_ATOL_EV_A,
)
nci_force_progress.advance(message="official force matches its energy derivative")

nci_toolkit_analytic_force_eV_A = float(
    nci_example_force[nci_fd_atom_index, nci_fd_axis].cpu()
)
nci_toolkit_official_error_eV_A = abs(
    nci_toolkit_analytic_force_eV_A - nci_official_analytic_force_eV_A
)
np.testing.assert_allclose(
    nci_toolkit_analytic_force_eV_A,
    nci_official_analytic_force_eV_A,
    rtol=0.0,
    atol=NCI_PIPELINE_OFFICIAL_FORCE_ATOL_EV_A,
)
nci_force_progress.complete("Toolkit force matches the official calculator")
print("checked atom and Cartesian axis:", nci_fd_atom_index, nci_fd_axis)
print("official analytic force / eV Å⁻¹:", nci_official_analytic_force_eV_A)
print("official energy finite difference / eV Å⁻¹:", nci_official_fd_force_eV_A)
print("Toolkit pipeline force / eV Å⁻¹:", nci_toolkit_analytic_force_eV_A)
del nci_official, nci_official_base, nci_official_outputs
""",
        ),
        code(
            "analyze-nci-curves",
            """
nci_analysis_progress = NotebookProgress(
    title="Compare the interaction curves with two reference levels",
    total=3,
    unit="steps",
)
EV_TO_KCAL_MOL = 1.0 / (units.kcal / units.mol)
nci_member_curves = reduce_fragment_energies(
    nci_graph_index,
    {
        "core": nci_member_residual_eV,
        "core_plus_d3": nci_member_residual_eV + nci_d3_graph_eV,
        "core_plus_coulomb": nci_member_residual_eV + nci_member_coulomb_eV,
        "full": nci_member_residual_eV + nci_member_coulomb_eV + nci_d3_graph_eV,
    },
    unit_scale=EV_TO_KCAL_MOL,
)
nci_curves = mean_member_curves(
    nci_member_curves,
    ("core", "core_plus_d3", "core_plus_coulomb", "full"),
    spread_component="full",
)
nci_dft = reduce_fragment_energies(
    nci_graph_index,
    {"dft_full": nci_reference_data[
        "wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol"
    ]},
)
nci_d3_interaction = reduce_fragment_energies(
    nci_graph_index, {"d3_interaction": nci_d3_graph_eV},
    unit_scale=EV_TO_KCAL_MOL,
)
nci_dft["dft_no_d3"] = nci_dft["dft_full"] - nci_d3_interaction["d3_interaction"]
nci_cc = extract_repeated_interaction_reference(
    nci_reference_data,
    "ccsd_t_cbs_interaction_energy_kcal_mol",
    output_column="ccsd_t_cbs",
)
for reference in (nci_dft, nci_cc):
    nci_curves = nci_curves.merge(reference, on=list(CURVE_KEY_COLUMNS), validate="one_to_one")
nci_analysis_progress.advance(message="AB - A - B applied to model and references")

nci_metrics = interaction_metrics(
    nci_curves,
    {
        "core vs CC": ("core", "ccsd_t_cbs"),
        "+ Coulomb vs CC": ("core_plus_coulomb", "ccsd_t_cbs"),
        "complete vs CC": ("full", "ccsd_t_cbs"),
        "same-D3 bookkeeping identity": (
            "core_plus_coulomb", "dft_no_d3"
        ),
        "complete vs DFT-D3": ("full", "dft_full"),
        "DFT-D3 vs CC": ("dft_full", "ccsd_t_cbs"),
    },
    mean_columns={"ensemble spread": "full_std"},
)
assert (nci_metrics["complete vs CC"] < NCI_COMPLETE_MAE_LIMIT_KCAL_MOL).all()
assert (
    nci_metrics["complete vs DFT-D3"] < NCI_COMPLETE_MAE_LIMIT_KCAL_MOL
).all()
nci_analysis_progress.advance(message="complete-model errors checked for all three curves")

nci_figure, _ = plot_nci_interaction_curves(nci_curves)
nci_figure.savefig(OUTPUT_DIR / "nci_interaction_curves.png", dpi=180, bbox_inches="tight")
display(figure_with_alt(
    nci_figure,
    alt_text=(
        "Three NCI Atlas interaction-energy curves across neutral hydrogen-bonded, "
        "dispersion-dominated, and ionic hydrogen-bonded complexes. Four AIMNet "
        "component combinations are compared with DFT-D3 and CCSD(T)/CBS."
    ),
))
plt.close(nci_figure)
nci_analysis_progress.complete("curves, ensemble spread, and reference errors shown")
display(readable_table(nci_metrics.round(3).reset_index(),
    label="Mean absolute interaction-energy errors / kcal mol⁻¹", show_index=False))
display(callout(
    "On this curated three-system set, restoring the declared Coulomb and D3 terms brings the complete model within 0.5 kcal/mol MAE of both references for every curve. This is a focused composition check, not broad MLIP validation.",
    kind="result",
    result_state="observed",
))
""",
        ),
        code(
            "build-components",
            """
component_build_progress = NotebookProgress(
    title="Configure the three model components", total=3, unit="components"
)
# Variable-size scans remain eager. Compilation is reserved for the fixed IR
# workload in Stage 5, where its exact energy, charge, and force outputs are
# compared with this model before dynamics.
aimnet.set_config("active_outputs", {"energy", "charges"})
component_build_progress.update(done=1, message="AIMNet checkpoint base + charges ready")

d3_params = model_card["d3_params"]
# Reuse the D3 parameter file verified in Stage 3.
coulomb = DirectCoulombWrapper().to(DEVICE)
component_build_progress.update(done=2, message="finite all-pairs Coulomb ready")
d3 = DFTD3ModelWrapper(
    a1=d3_params["a1"],
    a2=d3_params["a2"],
    s8=d3_params["s8"],
    s6=d3_params.get("s6", 1.0),
    cutoff=D3_CUTOFF_A,
    param_file=D3_PARAMETER_FILE,
    auto_download=False,
).to(DEVICE)
for component in (aimnet, coulomb, d3):
    neighbor_config = component.model_config.neighbor_config
    if neighbor_config is not None:
        neighbor_config.skin = NEIGHBOR_SKIN_A
component_build_progress.complete("pairwise D3(BJ) ready; all three components configured")

display(readable_table(
    pd.DataFrame([
        {
            "Component": "AIMNet checkpoint base",
            "Depends on": "positions, elements, total charge",
            "Cutoff / Å": aimnet.model_config.neighbor_config.cutoff,
        },
        {
            "Component": "finite all-pairs Coulomb",
            "Depends on": "AIMNet predicted charges",
            "Cutoff / Å": None,
        },
        {
            "Component": "pairwise D3(BJ)",
            "Depends on": "positions, elements",
            "Cutoff / Å": D3_CUTOFF_A,
        },
    ]),
    label="Composed water-potential components",
    show_index=False,
))
display(callout(
    f"The checked D3 parameter file has SHA-256 {D3_PARAMETER_SHA256[:16]}….",
    kind="result",
    result_state="pass",
))
""",
        ),
        code(
            "component-ablation",
            """
component_progress = NotebookProgress(
    title="Evaluate model components", total=4, unit="passes"
)

compute_neighbors(scan_batch, config=aimnet.model_config.neighbor_config)
residual_outputs = aimnet(scan_batch)
residual_energy = residual_outputs["energy"].detach().reshape(-1)
scan_batch.charges = residual_outputs["charges"]
component_progress.update(done=1, message="AIMNet checkpoint base + charges")

coulomb.set_config("active_outputs", {"energy"})
coulomb_energy = coulomb(scan_batch)["energy"].detach().reshape(-1)
component_progress.update(done=2, message="finite all-pairs Coulomb")

d3.set_config("active_outputs", {"energy"})
compute_neighbors(scan_batch, config=d3.model_config.neighbor_config)
d3_energy = d3(scan_batch)["energy"].detach().reshape(-1)
component_progress.update(done=3, message="pairwise D3(BJ)")

# Pipeline construction copies each component's current output configuration.
# Restore force-producing outputs before assembling the groups.
aimnet.set_config("active_outputs", {"energy", "charges"})
d3.set_config("active_outputs", {"energy", "forces"})
assert coulomb.direct_derivative_keys() == set()
assert "forces" in d3.model_config.active_outputs
aimnet_step = PipelineStep(model=aimnet)
coulomb_step = PipelineStep(model=coulomb)
d3_step = PipelineStep(model=d3)
model = PipelineModelWrapper(
    groups=[
        PipelineGroup(steps=[aimnet_step, coulomb_step], use_autograd=True),
        PipelineGroup(steps=[d3_step], use_autograd=False),
    ],
    neighbor_adaptation="always",
).to(DEVICE)
model.set_config("active_outputs", {"energy", "forces", "charges"})
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)

compute_neighbors(scan_batch, config=model.model_config.neighbor_config)
full_outputs = model(scan_batch)
full_energy = full_outputs["energy"].detach().reshape(-1)
component_sum = residual_energy + coulomb_energy + d3_energy
component_closure_error = float(torch.max(torch.abs(full_energy - component_sum)).cpu())
assert component_closure_error < COMPONENT_CLOSURE_TOLERANCE_EV
assert torch.isfinite(full_outputs["forces"]).all()
assert float(torch.linalg.vector_norm(full_outputs["forces"], dim=1).max().cpu()) > 0.0
component_progress.complete("Pipeline equals the independently evaluated component sum")

display(callout(
    "Two PipelineGroup objects now return energy, forces, and charges. "
    f"The complete pipeline matched the independently evaluated component sum "
    f"within {component_closure_error:.2e} eV.",
    kind="result",
    result_state="pass",
))
""",
        ),
        code(
            "official-composition-agreement",
            """
agreement_progress = NotebookProgress(
    title="Verify the composed model", total=5, unit="checks"
)
official = AIMNet2Calculator(
    str(checkpoint_path),
    device=str(DEVICE),
    needs_coulomb=True,
    needs_dispersion=True,
    compile_model=False,
    train=False,
)
official.set_lrcoulomb_method("simple")
official.set_dftd3_cutoff(
    cutoff=D3_CUTOFF_A,
    smoothing_fraction=NCI_D3_SMOOTHING_FRACTION,
)
agreement_progress.update(done=1, message="official AIMNet all-pairs Coulomb + D3 configured")

official_outputs = official({
    "coord": scan_batch.positions.detach().clone(),
    "numbers": scan_batch.atomic_numbers.detach().clone(),
    "charge": scan_batch.charge.detach().reshape(-1).clone(),
    "mol_idx": scan_batch.batch_idx.detach().clone(),
}, forces=True)
agreement_progress.update(done=2, message="official calculator evaluated")

# Pure reductions live in aux; both complete-model calls stay visible above.
composition_agreement = compare_composition_outputs(
    full_outputs,
    official_outputs,
    interaction_triplets=len(DIMER_DISTANCES_A),
)
agreement_errors = composition_agreement.as_dict()
# Both routes use float32 model kernels and include large atomic energy
# baselines. One millielectronvolt is a few float32 spacings at this energy
# scale; interaction energies are checked separately after AB - A - B.
assert agreement_errors["energy_eV"] < COMPOSITION_ENERGY_AGREEMENT_TOLERANCE_EV
assert (
    agreement_errors["interaction_energy_eV"]
    < COMPOSITION_INTERACTION_AGREEMENT_TOLERANCE_EV
)
assert (
    agreement_errors["forces_eV_A"]
    < COMPOSITION_FORCE_AGREEMENT_TOLERANCE_EV_A
)
assert agreement_errors["charges_e"] < COMPOSITION_CHARGE_AGREEMENT_TOLERANCE_E
agreement_progress.update(done=3, message="official calculator agreement is within tolerance")

# A two-charge system makes the 1/2 ordered-pair convention, sign, and force
# direction independently checkable without invoking AIMNet or D3.
two_charge_distance_A = 2.0
two_charge_values_e = torch.tensor([0.75, -0.75], device=DEVICE)
two_charge_atoms = Atoms(
    "H2", positions=[[0.0, 0.0, 0.0], [two_charge_distance_A, 0.0, 0.0]]
)
two_charge_atoms.info["charge"] = 0
two_charge_data = AtomicData.from_atoms(
    two_charge_atoms, device=DEVICE, dtype=torch.float64
)
two_charge_batch = Batch.from_data_list([two_charge_data], device=DEVICE)
two_charge_batch.charges = two_charge_values_e
two_charge_batch.positions.requires_grad_(True)
two_charge_energy_tensor = coulomb(two_charge_batch)["energy"].sum()
two_charge_forces = -torch.autograd.grad(
    two_charge_energy_tensor, two_charge_batch.positions
)[0]
coulomb_agreement = compare_two_particle_coulomb(
    positions_angstrom=two_charge_batch.positions,
    charges_e=two_charge_values_e,
    observed_energy_eV=two_charge_energy_tensor,
    observed_forces_eV_A=two_charge_forces,
    coulomb_constant_eV_A_per_e2=coulomb.coulomb_constant,
)
analytic_coulomb_errors = {
    "energy_eV": coulomb_agreement.energy_error_eV,
    "forces_eV_A": coulomb_agreement.forces_error_eV_A,
}
assert (
    analytic_coulomb_errors["energy_eV"]
    < COMPOSITION_ANALYTIC_COULOMB_ENERGY_TOLERANCE_EV
)
assert (
    analytic_coulomb_errors["forces_eV_A"]
    < COMPOSITION_ANALYTIC_COULOMB_FORCE_TOLERANCE_EV_A
)
assert float(two_charge_energy_tensor.detach().cpu()) < 0.0
assert float(two_charge_forces[0, 0].detach().cpu()) > 0.0
agreement_progress.update(done=4, message="analytic two-charge Coulomb check passed")

# Use the independent calculator's total energy for the numerical derivative.
# In Toolkit 0.2, checkpoint loading casts large constant atomic
# reference energies to float32. Their derivative is zero, so analytic forces
# remain accurate, but finite differences of the resulting ~keV totals lose
# the much smaller displacement signal.
FD_STEP_A = COMPOSITION_FD_STEP_A
fd_scan_index = 3  # 3.20 Å, away from a neighbor cutoff
fd_atom_index = 3
fd_axis = 0
graph_start = int(scan_batch.batch_ptr[3 * fd_scan_index].item())
model_force = float(
    full_outputs["forces"][graph_start + fd_atom_index, fd_axis].detach().cpu()
)
official_force = float(
    official_outputs["forces"][graph_start + fd_atom_index, fd_axis]
    .detach().cpu()
)
fd_energies = []
for sign in (-1.0, 1.0):
    displaced = scan_dimers[fd_scan_index].copy()
    displaced.positions[fd_atom_index, fd_axis] += sign * FD_STEP_A
    displaced.info["charge"] = 0
    fd_data = AtomicData.from_atoms(displaced, device=DEVICE, dtype=torch.float64)
    fd_batch = Batch.from_data_list([fd_data], device=DEVICE)
    official_displaced = official({
        "coord": fd_batch.positions.detach().clone(),
        "numbers": fd_batch.atomic_numbers.detach().clone(),
        "charge": fd_batch.charge.detach().reshape(-1).clone(),
        "mol_idx": fd_batch.batch_idx.detach().clone(),
    }, forces=True)
    fd_energies.append(float(
        official_displaced["energy"].detach().cpu().reshape(())
    ))

fd_force = central_difference_force(
    fd_energies[0], fd_energies[1], displacement_angstrom=FD_STEP_A
)
official_fd_force_error = abs(official_force - fd_force)
fd_force_error = abs(model_force - fd_force)
fd_force_tolerance = COMPOSITION_FD_FORCE_TOLERANCE_EV_A
assert official_fd_force_error < fd_force_tolerance
assert fd_force_error < fd_force_tolerance
agreement_progress.complete("Energies, forces, and charges match the official calculator")

composition_check_limits = {
    "energy_eV": COMPOSITION_ENERGY_AGREEMENT_TOLERANCE_EV,
    "interaction_energy_eV": COMPOSITION_INTERACTION_AGREEMENT_TOLERANCE_EV,
    "forces_eV_A": COMPOSITION_FORCE_AGREEMENT_TOLERANCE_EV_A,
    "charges_e": COMPOSITION_CHARGE_AGREEMENT_TOLERANCE_E,
    "analytic_coulomb_energy_eV": (
        COMPOSITION_ANALYTIC_COULOMB_ENERGY_TOLERANCE_EV
    ),
    "analytic_coulomb_forces_eV_A": (
        COMPOSITION_ANALYTIC_COULOMB_FORCE_TOLERANCE_EV_A
    ),
    "finite_difference_force_error_eV_A": COMPOSITION_FD_FORCE_TOLERANCE_EV_A,
}
composition_check_table = build_composition_check_table(
    agreement=composition_agreement,
    coulomb=coulomb_agreement,
    finite_difference_energy_route=COMPOSITION_FD_ENERGY_ROUTE,
    finite_difference_step_A=FD_STEP_A,
    finite_difference_force_eV_A=fd_force,
    reference_analytic_force_eV_A=official_force,
    toolkit_force_eV_A=model_force,
    reference_finite_difference_error_eV_A=official_fd_force_error,
    toolkit_finite_difference_error_eV_A=fd_force_error,
    limits=composition_check_limits,
)
display(readable_table(
    composition_check_table.rename_axis("Check").reset_index().rename(columns={
        "max_abs_difference": "Maximum absolute difference",
        "limit": "Limit",
        "units": "Units",
        "passed": "Passed",
    }),
    label="Composition checks",
    show_index=False,
))
finite_difference_context = composition_check_table.attrs["finite_difference"]
display(readable_table(
    pd.DataFrame([
        ("Energy route", finite_difference_context["finite_difference_energy_route"]),
        ("Displacement / Å", finite_difference_context["finite_difference_step_A"]),
        (
            "Finite-difference force / eV Å⁻¹",
            finite_difference_context["finite_difference_force_eV_A"],
        ),
        (
            "Official analytic force / eV Å⁻¹",
            finite_difference_context["official_analytic_force_eV_A"],
        ),
        (
            "Toolkit force / eV Å⁻¹",
            finite_difference_context["toolkit_force_eV_A"],
        ),
    ], columns=["Setting", "Value"]),
    label="Finite-difference setup and forces",
    show_index=False,
))
display(callout(
    "Toolkit composition agrees with the official calculator; the custom Coulomb wrapper passes an analytic two-charge check; and a finite difference of the independent calculator's total energy agrees with the Toolkit force.",
    kind="result",
    result_state="pass",
))
del official, official_outputs, fd_batch, two_charge_batch
""",
        ),
        code(
            "full-pipeline-agreement",
            """
full_agreement_progress = NotebookProgress(
    title="Complete pipeline: serial / batch agreement",
    total=len(scan_data),
    unit="graphs",
)
serial_full_energy = []
for graph_index, data in enumerate(scan_data, start=1):
    one_graph = Batch.from_data_list([data], device=DEVICE)
    compute_neighbors(one_graph, config=model.model_config.neighbor_config)
    serial_full_energy.append(model(one_graph)["energy"].detach().reshape(()))
    full_agreement_progress.update(
        done=graph_index,
        message=f"graph {graph_index} of {len(scan_data)}",
    )
serial_full_energy = torch.stack(serial_full_energy)
full_pipeline_agreement_error = float(
    torch.max(torch.abs(serial_full_energy - full_energy)).cpu()
)
assert full_pipeline_agreement_error < FULL_SERIAL_BATCH_TOLERANCE_EV
full_agreement_progress.complete("Complete-model energies preserve graph identity")
display(callout(
    f"Complete checkpoint-base + Coulomb + D3 serial/batch max |ΔE| = {full_pipeline_agreement_error:.2e} eV.",
    kind="result",
    result_state="pass",
))
""",
        ),
        code(
            "dimer-ablation-plot",
            """
dimer_plot_progress = NotebookProgress(
    title="Plot three ablations and the full model against B97-3c",
    total=3,
    unit="steps",
)
component_vectors = {}
for name, values in {
    "residual": residual_energy,
    "residual_plus_D3": residual_energy + d3_energy,
    "residual_plus_Coulomb": residual_energy + coulomb_energy,
    "full": full_energy,
}.items():
    triplets = values.cpu().numpy().reshape(-1, 3)
    component_vectors[name] = (triplets[:, 0], triplets[:, 1], triplets[:, 2])

dimer_table = dimer_interaction_energy_table(DIMER_DISTANCES_A, component_vectors)
dft_curve_path = REFERENCE_ROOT / "water_dimer_b97_3c" / "interaction_curve.csv"
dft_curve = load_verified_b97_3c_dimer_reference(dft_curve_path)
dimer_table["B97_3c_interaction_kJ_mol"] = dft_curve["interaction_kJ_mol"]
dimer_plot_progress.advance(message="checksummed B97-3c interaction curve loaded")

fig, axis = plot_dimer_interaction_energies(
    dimer_table,
    component_columns=[
        "residual_interaction_kJ_mol",
        "residual_plus_D3_interaction_kJ_mol",
        "residual_plus_Coulomb_interaction_kJ_mol",
        "full_interaction_kJ_mol",
        "B97_3c_interaction_kJ_mol",
    ],
    labels={
        "residual_interaction_kJ_mol": "checkpoint base (incomplete)",
        "residual_plus_D3_interaction_kJ_mol": "base + D3 (Coulomb omitted)",
        "residual_plus_Coulomb_interaction_kJ_mol": "base + Coulomb (D3 omitted)",
        "full_interaction_kJ_mol": "base + Coulomb + D3",
        "B97_3c_interaction_kJ_mol": "full B97-3c reference",
    },
)
fig.savefig(OUTPUT_DIR / "water_dimer_ablation.png", dpi=180, bbox_inches="tight")
display(figure_with_alt(
    fig,
    alt_text=(
        "Water-dimer interaction energy versus O-O distance for the AIMNet "
        "checkpoint base, base plus D3, base plus all-pairs Coulomb, complete "
        "base plus Coulomb plus D3, and the full B97-3c reference. The "
        "partial and complete model curves are shown against the same reference."
    ),
))
plt.close(fig)
dimer_plot_progress.advance(message="interaction curves plotted and saved")

ablation_mae = pd.Series({
    label: float(np.mean(np.abs(
        dimer_table[column] - dimer_table["B97_3c_interaction_kJ_mol"]
    )))
    for label, column in {
        "checkpoint base (incomplete)": "residual_interaction_kJ_mol",
        "base + D3 (Coulomb omitted)": "residual_plus_D3_interaction_kJ_mol",
        "base + Coulomb (D3 omitted)": "residual_plus_Coulomb_interaction_kJ_mol",
        "base + Coulomb + D3": "full_interaction_kJ_mol",
    }.items()
}, name="MAE_vs_full_B97_3c_kJ_mol")
full_mae = float(ablation_mae.loc["base + Coulomb + D3"])
coulomb_omitted_mae = float(ablation_mae.loc["base + D3 (Coulomb omitted)"])
d3_omitted_mae = float(ablation_mae.loc["base + Coulomb (D3 omitted)"])
if coulomb_omitted_mae > d3_omitted_mae:
    ablation_takeaway = (
        "On this eight-geometry scan, omitting Coulomb leaves the larger error "
        f"({coulomb_omitted_mae:.2f} vs {d3_omitted_mae:.2f} kJ/mol), so "
        "restoring explicit electrostatics changes agreement more than restoring D3."
    )
else:
    ablation_takeaway = (
        "On this eight-geometry scan, omitting D3 leaves the larger error "
        f"({d3_omitted_mae:.2f} vs {coulomb_omitted_mae:.2f} kJ/mol), so "
        "restoring dispersion changes agreement more than restoring electrostatics."
    )
dimer_plot_progress.complete("all four MAEs evaluated against the same reference")
display(readable_table(
    ablation_mae.round(3).rename_axis("Model").reset_index().rename(
        columns={"MAE_vs_full_B97_3c_kJ_mol": "MAE vs full B97-3c / kJ mol⁻¹"}
    ),
    label="MAE against the full B97-3c reference",
    show_index=False,
))
display(callout(
    ablation_takeaway
    + " Adding one term by itself need not improve the error monotonically. "
    + f"The complete-model MAE is {full_mae:.2f} kJ/mol. This deterministic, "
    + "geometry-matched scan is not a held-out or broad transferability benchmark. "
    + "The full B97-3c endpoint also contains ATM and gCP; Toolkit's external "
    + "correction is pairwise C6/C8 D3(BJ), so these curves are AIMNet "
    + "composition ablations, not a term-by-term decomposition of B97-3c.",
    kind="result",
    result_state="observed",
))
""",
        ),
        markdown(
            "composition-note",
            callout_html(
                "B97-3c includes D3(BJ)-ATM and gCP. Toolkit adds the pairwise C6/C8 part of D3, and the public checkpoint metadata does not tell us whether ATM and gCP are represented in exactly the same way. We therefore compare every curve with the full B97-3c reference, but treat AIMNet + Coulomb + D3 as the final model. The partial curves show what changes when a term is omitted; their errors are not standalone accuracy estimates and do not need to improve monotonically.",
                kind="note",
            ),
        ),
        stage_markdown(
            "stage-4",
            stage=4,
            title="Bring a model for a new domain",
            outcome=(
                "Connect a raw materials model to Toolkit, inspect and switch its "
                "tasks, then evaluate five periodic and four finite "
                "structures in two batches."
            ),
            before="Before running: identify which model inputs, neighbor fields, and outputs must be translated at the wrapper.",
            compute_time=(
                "18 s on one H100 PCIe in the checked run"
            ),
            body="",
        ),
        markdown(
            "surface-model-switch",
            r"""
### Switch models when the system changes

The water example uses AIMNet2 inside the domain described for its pretrained checkpoint. A Cu surface is a different problem:

- the published AIMNet2 checkpoint supports H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, and I; Cu is not included;
- its software can process periodic cells, and the paper demonstrates condensed CO₂, but the selected checkpoint was trained on molecular data rather than bulk metals or metal surfaces;
- adding Cu to this checkpoint would therefore be an unsupported model choice, not a Toolkit limitation.

For the surface example we switch to **SevenNet-Omni**. Its training data and published tests include molecules, crystals, surfaces, porous materials, and adsorption benchmarks. We use its explicit `mpa` task, which has a PBE(+U) target and no D3 term, then add Toolkit's pairwise PBE-D3(BJ) correction.

The four adsorbates exercise different graph sizes and binding atoms: C-bound CO, a surface-parallel CO₂ starting pose, N-bound NH₃, and O-bound CH₃OH. They are a compact API panel, not a claim that these four starting poses form a chemical ranking set.

Each adsorbate is repeated in x and y by the 3×3 periodic Cu(111) cell. The subtraction therefore gives a fixed-cell, finite-coverage energy difference that includes interactions with periodic adsorbate images; it is not a zero-coverage adsorption energy.
"""
            + "\n\n"
            + callout_html(
                "This is a model-selection decision. Toolkit keeps AtomicData, Batch, neighbors, composition, and outputs consistent while the model changes.",
                kind="note",
            )
            + "\n\n"
            + callout_html(
                "The nine shipped structures are deterministic starting geometries, not relaxed minima. The calculation below reports fixed-geometry single points and forces for one adsorbate per 3×3 periodic surface cell; it is not an equilibrium or zero-coverage adsorption-energy benchmark.",
                kind="check",
            )
            + "\n\n"
            + callout_html(
                "The SevenNet-Omni paper warns that PBE(+U) tasks can behave poorly when oxygen binds to Co or Ni. We use Cu(111), where its reported diagnostic remained physically smooth, but that does not make this task safe for every transition metal.",
                kind="note",
            )
            + "\n\n"
            + process_diagram_html(
                title="One Toolkit workflow, a model suited to surfaces",
                steps=(
                    "ASE Cu(111) + molecules",
                    "two Toolkit batches",
                    "SevenNet graph adapter",
                    "mpa energy/forces + pairwise PBE-D3(BJ)",
                    "fixed-geometry Eads + force checks",
                    "OVITO inspection",
                ),
                caption=(
                    "Five periodic structures and four finite molecules are "
                    "evaluated as two ragged batches."
                ),
            ),
        ),
        markdown(
            "sevennet-model-config",
            r"""
#### 1. Declare what the raw model needs

`ModelConfig` tells Toolkit what the adapter accepts and returns:

| Setting | Meaning here |
|---|---|
| `outputs={"energy", "forces"}` | SevenNet returns one total energy per graph and one force vector per atom. |
| `supports_pbc=True` | The same adapter accepts the 2D-periodic slabs and finite gas molecules. |
| full directed COO neighbors | SevenNet needs both edge directions at the cutoff stored in its checkpoint. |
| `skin=0` | A single-point example uses the exact model cutoff; moving simulations would rebuild through Toolkit hooks. |
| no charge or spin fields | The selected SevenNet task does not consume or report partial charges. |
| `direct_derivative_keys() -> {"forces"}` | SevenNet differentiates edge vectors internally; Toolkit must preserve those forces. |

The model task is not a cosmetic label. `mpa`, `oc20`, and `odac23` represent different training targets. The notebook passes `mpa` explicitly instead of relying on a default.
"""
            + "\n\n"
            + callout_html(
                "SevenNet-Omni is a local model. We do not attach AIMNet's molecular charge model or invent point charges for the metal surface.",
                kind="note",
            ),
        ),
        markdown(
            "sevennet-input-map",
            r"""
#### 2. Translate one Toolkit batch into one SevenNet graph

| Toolkit data | SevenNet data |
|---|---|
| `atomic_numbers`, `positions` | atomic-number input and `pos` |
| `batch_idx`, `num_nodes_list` | atom ownership and atoms per graph |
| full COO `neighbor_list` | `(2, edges)` `edge_index` |
| integer shifts and each graph's cell | Cartesian `edge_vec = r_target - r_source + shift @ cell` |
| cells with PBC `[True, True, False]` | 2D-periodic slab edges without repeated images across the vacuum direction |
| explicit task name | one `mpa` modality entry per graph |

The helper performs tensor assembly and validation. The class below keeps the Toolkit-facing choices visible: capabilities, neighbor format, one raw model call, and output names.
"""
            + "\n\n"
            + callout_html(
                "A ragged batch may mix atom counts, but not model requirements. Periodic slabs and finite molecules are kept in separate batches so their neighbor work remains predictable.",
                kind="check",
            ),
        ),
        markdown(
            "sevennet-output-map",
            r"""
#### 3. Call the raw model once; return Toolkit fields

`forward(...)` converts the complete batch, calls the raw SevenNet model once, and maps:

- `inferred_total_energy` → `energy` with shape `(graphs, 1)` in eV;
- `inferred_force` → `forces` with shape `(atoms, 3)` in eV/Å.

No unit conversion or learned charge output is added. The adapter validates both shapes before returning them.

For the surface correction we use pairwise PBE-D3(BJ) parameters, the reference 95-bohr cutoff, and no taper. The slab is periodic only in x and y, so that long cutoff does not introduce dispersion between repeated images across the vacuum direction.

`PipelineModelWrapper(..., neighbor_adaptation="always")` handles the two neighbor requests. SevenNet asks for full directed COO edges at its checkpoint cutoff; D3 asks for its own 95-bohr layout. Toolkit builds and adapts the neighbor data needed by each component. The direct component calls below are shown only to expose and check the D3 correction; the combined result comes from the pipeline.
"""
            + "\n\n"
            + callout_html(
                "The `mpa` task plus pairwise PBE-D3(BJ) is a practical PBE-family surface model, not a claim that every training record used one identical DFT setup. SevenNet's own adsorption study uses the same pattern of adding D3 when a selected task does not include it.",
                kind="note",
            ),
        ),
        code(
            "define-sevennet-config",
            """
sevennet_config_progress = NotebookProgress(
    title="Declare the SevenNet Toolkit interface", total=1, unit="configuration"
)
"""
            + sevennet_config_source
            + """
sevennet_config_progress.complete("outputs, periodic support, and neighbors declared")
""",
        ),
        code(
            "define-sevennet-wrapper",
            """
sevennet_adapter_progress = NotebookProgress(
    title="Define the SevenNet-Omni Toolkit adapter", total=3, unit="methods"
)

"""
            + sevennet_wrapper_source
            + """
sevennet_adapter_progress.advance(message="Toolkit batch translated to SevenNet graph")
sevennet_adapter_progress.advance(message="one raw SevenNet call isolated")
sevennet_adapter_progress.complete("energy and force outputs mapped to Toolkit names")
display(callout(
    "Read the two cells in order: the first declares Toolkit capabilities; "
    "the second translates the graph, calls SevenNet once, and maps the outputs.",
    kind="check",
))
""",
        ),
        code(
            "load-sevennet-wrapper",
            """
sevennet_load_progress = NotebookProgress(
    title="Load SevenNet-Omni and compose pairwise PBE-D3(BJ)", total=5, unit="checks"
)

sevennet_checkpoint_path, sevennet_checkpoint_sha256 = (
    resolve_sevennet_checkpoint()
)
assert sevennet_checkpoint_sha256 == SEVENNET_CHECKPOINT_SHA256
sevennet_load_progress.advance(message="official checkpoint size and SHA-256 verified")

raw_sevennet, sevennet_checkpoint_config = load_raw_sevennet_omni(
    sevennet_checkpoint_path,
    device=DEVICE,
)
sevennet_load_progress.advance(message="raw float32 e3nn model loaded")

sevennet_model = SevenNetOmniWrapper(
    raw_sevennet,
    modality=SEVENNET_MODALITY,
).to(DEVICE).eval()
sevennet_config = sevennet_model.model_config
assert sevennet_config.outputs == {"energy", "forces"}
assert sevennet_config.supports_pbc and not sevennet_config.needs_pbc
assert sevennet_config.neighbor_config.format is NeighborListFormat.COO
assert not sevennet_config.neighbor_config.half_list
assert sevennet_model.direct_derivative_keys() == {"forces"}
sevennet_load_progress.advance(message="periodic energy/force adapter connected")

surface_d3 = DFTD3ModelWrapper(
    a1=PBE_D3_BJ_A1,
    a2=PBE_D3_BJ_A2_BOHR,
    s8=PBE_D3_BJ_S8,
    s6=PBE_D3_BJ_S6,
    cutoff=SURFACE_D3_CUTOFF_A,
    smoothing_fraction=D3_REFERENCE_SMOOTHING_FRACTION,
    auto_download=False,
    param_file=D3_PARAMETER_FILE,
).to(DEVICE).eval()
surface_d3.set_config("active_outputs", {"energy", "forces"})
sevennet_load_progress.advance(message="Toolkit pairwise PBE-D3(BJ) correction configured")

# This object is ready for Toolkit energy/force workflows that match its
# ModelConfig. It does not provide charge, stress, or embedding outputs.
# Component calls stay separate below so the D3 contribution remains visible.
surface_model = PipelineModelWrapper(
    groups=[
        PipelineGroup(
            steps=[PipelineStep(model=sevennet_model)], use_autograd=False
        ),
        PipelineGroup(
            steps=[PipelineStep(model=surface_d3)], use_autograd=False
        ),
    ],
    neighbor_adaptation="always",
).to(DEVICE).eval()
surface_model.set_config("active_outputs", {"energy", "forces"})
sevennet_load_progress.complete("custom adapter accepted by Toolkit composition")

surface_model_table = pd.Series({
    "Raw model": SEVENNET_MODEL_NAME,
    "Task": SEVENNET_MODALITY,
    "Training target": SEVENNET_REFERENCE_METHOD,
    "sevenn": metadata.version("sevenn"),
    "Checkpoint SHA-256": sevennet_checkpoint_sha256,
    "Checkpoint record": SEVENNET_CHECKPOINT_DOI,
    "Model cutoff / Å": sevennet_model.cutoff,
    "Neighbor list": "COO, full directed",
    "Periodic systems": sevennet_config.supports_pbc,
    "Outputs": sorted(sevennet_config.active_outputs),
    "D3 parameters": "pairwise PBE-D3(BJ)",
    "D3 cutoff": f"{D3_REFERENCE_CUTOFF_BOHR:.0f} bohr / {SURFACE_D3_CUTOFF_A:.3f} Å",
    "D3 taper": D3_REFERENCE_SMOOTHING_FRACTION,
    "SevenNet code license": "MIT",
    "Checkpoint license": "CC BY 4.0",
}, name="Value").rename_axis("Setting").reset_index()
print("SevenNet-Omni checkpoint SHA-256:", sevennet_checkpoint_sha256)
display(readable_table(
    surface_model_table,
    label="Surface model settings",
    show_index=False,
))
""",
        ),
        code(
            "build-adsorption-panel",
            """
adsorption_build_progress = NotebookProgress(
    title="Build two adsorption batches", total=3, unit="steps"
)

adsorption_structures = load_initial_structure_set()
adsorption_methodology = load_adsorption_methodology()
periodic_structures, finite_structures = split_for_batches(
    adsorption_structures
)
adsorption_build_progress.advance(message="nine versioned ASE structures verified")

def pack_structure_batch(structures):
    data = [
        AtomicData.from_atoms(atoms, device=DEVICE, dtype=torch.float32)
        for atoms in structures.values()
    ]
    return Batch.from_data_list(data, device=DEVICE)


periodic_surface_batch = pack_structure_batch(periodic_structures)
finite_molecule_batch = pack_structure_batch(finite_structures)
adsorption_build_progress.advance(
    message=(
        f"periodic batch: {periodic_surface_batch.num_graphs} graphs / "
        f"{periodic_surface_batch.num_nodes} atoms"
    )
)
adsorption_build_progress.complete(
    f"finite batch: {finite_molecule_batch.num_graphs} graphs / "
    f"{finite_molecule_batch.num_nodes} atoms"
)

adsorption_structure_table = pd.DataFrame([
    {
        "structure": key,
        "role": atoms.info["role"],
        "formula": atoms.get_chemical_formula(),
        "atoms": len(atoms),
        "pbc": tuple(bool(value) for value in atoms.pbc),
        "geometry": atoms.info["geometry_status"],
    }
    for key, atoms in adsorption_structures.items()
])
display(readable_table(
    adsorption_structure_table,
    label="Fixed starting structures",
    show_index=False,
))

placement_table = pd.DataFrame(adsorption_methodology["adsorbates"]).rename(
    columns={
        "name": "molecule",
        "anchor_element": "anchor",
        "height_angstrom": "height_A",
        "orientation": "starting_orientation",
    }
)[["molecule", "site", "anchor", "height_A", "starting_orientation"]]
display(readable_table(
    placement_table,
    label="Four fixed starting placements",
    show_index=False,
))
""",
        ),
        code(
            "view-adsorption-panel",
            """
adsorption_view_progress = NotebookProgress(
    title="Inspect the four surface inputs in OVITO", total=1, unit="view"
)
display(adsorption_widget_grid([
    (name, adsorption_structures[ADSLAB_KEYS[name]])
    for name in ADSORBATES
], columns=2))
adsorption_view_progress.complete("official OVITO widgets displayed")
""",
        ),
        code(
            "run-sevennet-wrapper",
            """
sevennet_run_progress = NotebookProgress(
    title="Evaluate nine structures in batches", total=9, unit="passes"
)

# Pass 1: five 2D-periodic graphs through the custom adapter.
compute_neighbors(
    periodic_surface_batch,
    config=sevennet_config.neighbor_config,
)
sevennet_graph_mapping = build_sevennet_mapping_table(
    sevennet_model, periodic_surface_batch
)
sevennet_graph_mapping_passed = bool(
    sevennet_graph_mapping["exact_match"].all()
)
if not sevennet_graph_mapping_passed:
    raise RuntimeError("SevenNet graph mapping check failed")
periodic_model_outputs = sevennet_model(periodic_surface_batch)
sevennet_run_progress.advance(message="five periodic model graphs evaluated")

# Pass 2: four finite gas references through the same adapter.
compute_neighbors(
    finite_molecule_batch,
    config=sevennet_config.neighbor_config,
)
finite_model_outputs = sevennet_model(finite_molecule_batch)
sevennet_run_progress.advance(message="four finite model graphs evaluated")

# Passes 3-4: repeat the raw path on both ragged layouts. This checks graph
# ownership and output slicing; it is not a DFT accuracy score.
periodic_raw_graph = sevennet_model.adapt_input(periodic_surface_batch)
periodic_raw_outputs = sevennet_model.adapt_output(
    raw_sevennet(periodic_raw_graph), periodic_surface_batch
)
periodic_repeat = build_sevennet_repeat_table(
    periodic_model_outputs,
    periodic_raw_outputs,
    labels=list(periodic_structures),
    atom_counts=periodic_surface_batch.num_nodes_list,
)
periodic_repeat.insert(0, "comparison", "adapter output vs direct raw call")
sevennet_run_progress.advance(message="periodic raw-path repeat measured")

finite_raw_graph = sevennet_model.adapt_input(finite_molecule_batch)
finite_raw_outputs = sevennet_model.adapt_output(
    raw_sevennet(finite_raw_graph), finite_molecule_batch
)
finite_repeat = build_sevennet_repeat_table(
    finite_model_outputs,
    finite_raw_outputs,
    labels=list(finite_structures),
    atom_counts=finite_molecule_batch.num_nodes_list,
)
finite_repeat.insert(0, "comparison", "adapter output vs direct raw call")
sevennet_run_progress.advance(message="finite raw-path repeat measured")

# Pass 5: compare one periodic graph with SevenNet's official ASE calculator.
# Use a separately loaded calculator: giving it raw_sevennet would switch the
# shared model out of batch mode and invalidate the remaining Toolkit calls.
from sevenn.calculator import SevenNetCalculator

co_key = ADSLAB_KEYS["CO"]
co_index = list(periodic_structures).index(co_key)
co_atoms = periodic_structures[co_key].copy()
wrapper_energy_eV = float(
    periodic_model_outputs["energy"].reshape(-1)[co_index].detach().cpu()
)
wrapper_force_blocks = torch.split(
    periodic_model_outputs["forces"],
    tuple(int(count) for count in periodic_surface_batch.num_nodes_list),
)
wrapper_forces_eV_A = (
    wrapper_force_blocks[co_index].detach().cpu().numpy().copy()
)
enabled_accelerators = [
    name
    for name in (
        "SEVENNET_ENABLE_CUEQ",
        "SEVENNET_ENABLE_FLASH",
        "SEVENNET_ENABLE_OEQ",
    )
    if os.environ.get(name) == "1"
]
if enabled_accelerators:
    raise RuntimeError(
        "Official comparison must use the same e3nn backend; unset: "
        + ", ".join(enabled_accelerators)
    )

official_calculator = SevenNetCalculator(
    model=sevennet_checkpoint_path,
    file_type="checkpoint",
    device=str(DEVICE),
    modal=SEVENNET_MODALITY,
    enable_cueq=False,
    enable_flash=False,
    enable_oeq=False,
    compute_atomic_virial=False,
)
try:
    co_atoms.calc = official_calculator
    official_energy_eV = float(co_atoms.get_potential_energy())
    official_forces_eV_A = np.asarray(
        co_atoms.get_forces(apply_constraint=False), dtype=np.float64
    ).copy()
    official_edge_count = int(official_calculator.results["num_edges"])
finally:
    co_atoms.calc = None
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    del official_calculator
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

if wrapper_forces_eV_A.shape != official_forces_eV_A.shape:
    raise RuntimeError("Official and Toolkit force arrays have different shapes")
official_energy_difference_eV = abs(wrapper_energy_eV - official_energy_eV)
official_energy_difference_eV_per_atom = (
    official_energy_difference_eV / len(co_atoms)
)
official_force_difference_eV_A = float(
    np.max(np.abs(wrapper_forces_eV_A - official_forces_eV_A))
)
official_calculator_check = pd.DataFrame([{
    "structure": co_key,
    "atoms": len(co_atoms),
    "official_directed_edges": official_edge_count,
    "adapter_energy_eV": wrapper_energy_eV,
    "official_energy_eV": official_energy_eV,
    "energy_difference_eV": official_energy_difference_eV,
    "energy_difference_eV_per_atom": official_energy_difference_eV_per_atom,
    "max_force_component_difference_eV_A": official_force_difference_eV_A,
}])
official_agreement = pd.DataFrame([{
    "comparison": "custom adapter vs official SevenNetCalculator",
    "structure": co_key,
    "atoms": len(co_atoms),
    "energy_difference_eV": official_energy_difference_eV,
    "energy_difference_eV_per_atom": official_energy_difference_eV_per_atom,
    "max_force_component_difference_eV_A": official_force_difference_eV_A,
}])
sevennet_run_progress.advance(
    message="custom adapter matched the official calculator on CO/Cu(111)"
)

# Passes 6-7: the same two batches through Toolkit's pairwise PBE-D3(BJ) model.
compute_neighbors(
    periodic_surface_batch,
    config=surface_d3.model_config.neighbor_config,
)
periodic_d3_outputs = surface_d3(periodic_surface_batch)
sevennet_run_progress.advance(message="five periodic D3 graphs evaluated")

compute_neighbors(
    finite_molecule_batch,
    config=surface_d3.model_config.neighbor_config,
)
finite_d3_outputs = surface_d3(finite_molecule_batch)
sevennet_run_progress.advance(message="four finite D3 graphs evaluated")

# Passes 8-9: run the actual Toolkit pipeline. The comparison below verifies
# that Toolkit returns the same E/F sum as the two visible component calls.
periodic_pipeline_outputs = surface_model(periodic_surface_batch)
periodic_component_sum = {
    "energy": periodic_model_outputs["energy"] + periodic_d3_outputs["energy"],
    "forces": periodic_model_outputs["forces"] + periodic_d3_outputs["forces"],
}
periodic_pipeline_agreement = build_sevennet_repeat_table(
    periodic_component_sum,
    periodic_pipeline_outputs,
    labels=list(periodic_structures),
    atom_counts=periodic_surface_batch.num_nodes_list,
)
periodic_pipeline_agreement.insert(
    0, "comparison", "pipeline output vs explicit component sum"
)
sevennet_run_progress.advance(message="five periodic pipeline graphs evaluated")

finite_pipeline_outputs = surface_model(finite_molecule_batch)
finite_component_sum = {
    "energy": finite_model_outputs["energy"] + finite_d3_outputs["energy"],
    "forces": finite_model_outputs["forces"] + finite_d3_outputs["forces"],
}
finite_pipeline_agreement = build_sevennet_repeat_table(
    finite_component_sum,
    finite_pipeline_outputs,
    labels=list(finite_structures),
    atom_counts=finite_molecule_batch.num_nodes_list,
)
finite_pipeline_agreement.insert(
    0, "comparison", "pipeline output vs explicit component sum"
)
sevennet_run_progress.complete("four finite pipeline graphs evaluated")

sevennet_numerical_agreement = pd.concat(
    [
        periodic_repeat,
        finite_repeat,
        official_agreement,
        periodic_pipeline_agreement,
        finite_pipeline_agreement,
    ],
    ignore_index=True,
)
sevennet_repeat_max_energy_difference_eV_per_atom = float(
    sevennet_numerical_agreement[
        "energy_difference_eV_per_atom"
    ].max()
)
sevennet_repeat_max_force_difference_eV_A = float(
    sevennet_numerical_agreement[
        "max_force_component_difference_eV_A"
    ].max()
)
if (
    sevennet_repeat_max_energy_difference_eV_per_atom
    >= SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM
):
    raise RuntimeError("SevenNet energy difference exceeded its numerical check")
if sevennet_repeat_max_force_difference_eV_A >= SEVENNET_REPEAT_FORCE_TOL_EV_A:
    raise RuntimeError("SevenNet force difference exceeded its numerical check")

periodic_model_energy, periodic_model_forces = split_model_outputs(
    list(periodic_structures),
    periodic_surface_batch.num_nodes_list,
    periodic_model_outputs,
)
finite_model_energy, finite_model_forces = split_model_outputs(
    list(finite_structures),
    finite_molecule_batch.num_nodes_list,
    finite_model_outputs,
)
periodic_d3_energy, periodic_d3_forces = split_model_outputs(
    list(periodic_structures),
    periodic_surface_batch.num_nodes_list,
    periodic_d3_outputs,
)
finite_d3_energy, finite_d3_forces = split_model_outputs(
    list(finite_structures),
    finite_molecule_batch.num_nodes_list,
    finite_d3_outputs,
)
periodic_combined_energy, periodic_combined_forces = split_model_outputs(
    list(periodic_structures),
    periodic_surface_batch.num_nodes_list,
    periodic_pipeline_outputs,
)
finite_combined_energy, finite_combined_forces = split_model_outputs(
    list(finite_structures),
    finite_molecule_batch.num_nodes_list,
    finite_pipeline_outputs,
)

surface_model_energies = periodic_model_energy | finite_model_energy
surface_d3_energies = periodic_d3_energy | finite_d3_energy
surface_combined_energies = periodic_combined_energy | finite_combined_energy
surface_model_forces = periodic_model_forces | finite_model_forces
surface_d3_forces = periodic_d3_forces | finite_d3_forces
surface_combined_forces = periodic_combined_forces | finite_combined_forces

assert len(surface_model_energies) == len(surface_d3_energies) == 9
assert len(surface_combined_energies) == len(surface_combined_forces) == 9
assert all(np.isfinite(value) for value in surface_model_energies.values())
assert all(np.isfinite(value) for value in surface_d3_energies.values())
assert all(np.isfinite(value) for value in surface_combined_energies.values())
assert all(np.isfinite(value).all() for value in surface_combined_forces.values())
""",
        ),
        code(
            "analyze-adsorption-panel",
            """
adsorption_analysis_progress = NotebookProgress(
    title="Show energies and forces", total=4, unit="result views"
)

surface_energy_table = pd.DataFrame([
    {
        "structure": key,
        "role": atoms.info["role"],
        "SevenNet_mpa_energy_eV": surface_model_energies[key],
        "PBE_D3_BJ_pair_correction_eV": surface_d3_energies[key],
        "pipeline_energy_eV": surface_combined_energies[key],
    }
    for key, atoms in adsorption_structures.items()
])
display(readable_table(
    surface_energy_table.round(6),
    label="All surface-panel energies",
    show_index=False,
))
adsorption_analysis_progress.advance(message="all nine energy outputs shown")
display(callout(
    "The nine total energies verify that every graph returned a finite value. "
    "Do not compare absolute totals across different compositions; only the "
    "balanced E(adslab) - E(clean slab) - E(gas) differences below are used.",
    kind="note",
))

adsorption_results = assemble_adsorption_results(
    model_energies_eV=surface_model_energies,
    d3_energies_eV=surface_d3_energies,
    combined_forces_eV_A={
        ADSLAB_KEYS[name]: surface_combined_forces[ADSLAB_KEYS[name]]
        for name in ADSORBATES
    },
)
display(readable_table(
    adsorption_results[[
        "molecule",
        "model_adsorption_energy_eV",
        "d3_adsorption_energy_eV",
        "adsorption_energy_eV",
        "fmax_eV_A",
        "force_rms_eV_A",
    ]].round(5),
    label="Fixed-geometry, finite-coverage adsorption-energy differences",
    show_index=False,
))
adsorption_analysis_progress.advance(message="four slab + gas subtractions shown")
display(callout(
    "NOT REPORTED: adsorption accuracy or a molecule ranking. This panel has "
    "no matched DFT or experimental reference, no site search, and no geometry "
    "relaxation. The values demonstrate the Toolkit model-wrapper and "
    "composition path at four fixed starting geometries.",
    kind="result",
    result_state="not_reported",
))

force_frames = []
for component, values in (
    ("SevenNet-Omni mpa", surface_model_forces),
    ("pairwise PBE-D3(BJ)", surface_d3_forces),
    ("SevenNet-Omni mpa + pairwise PBE-D3(BJ)", surface_combined_forces),
):
    frame = build_full_force_table(adsorption_structures, values)
    frame.insert(0, "component", component)
    force_frames.append(frame)
surface_component_forces = pd.concat(force_frames, ignore_index=True)
# Save every atom-wise force returned by the Toolkit pipeline. The two
# component summaries remain visible above the final output table.
adsorption_forces = force_frames[-1].drop(columns="component").reset_index(drop=True)
surface_force_summary = (
    surface_component_forces.groupby(
        ["component", "structure", "role"], sort=False, as_index=False
    )
    .agg(
        atoms=("atom_index", "count"),
        fmax_eV_A=("force_norm_eV_A", "max"),
        force_rms_eV_A=(
            "force_norm_eV_A",
            lambda values: float(np.sqrt(np.mean(np.square(values)))),
        ),
    )
)
adslab_force_regions = summarize_adslab_force_regions(adsorption_forces)
surface_force_display = (
    surface_force_summary.pivot(
        index=["structure", "role"],
        columns="component",
        values="fmax_eV_A",
    )
    .reset_index()
)
surface_force_display.columns.name = None
display(readable_table(
    surface_force_display.round(5),
    label="Maximum force for every structure and model component / eV Å⁻¹",
    show_index=False,
))
display(readable_table(
    adslab_force_regions[[
        "structure", "region", "fmax_eV_A", "force_rms_eV_A",
    ]].rename(columns={
        "structure": "Structure",
        "region": "Region",
        "fmax_eV_A": "Maximum force / eV Å⁻¹",
        "force_rms_eV_A": "RMS force / eV Å⁻¹",
    }).round(5),
    label="Combined-model forces on adsorbates and Cu atoms",
    show_index=False,
))
adsorption_analysis_progress.advance(
    message=(
        f"{len(surface_component_forces):,} component rows analyzed; "
        f"{len(adsorption_forces):,} complete-model force rows retained"
    )
)

display(callout(
    "The Toolkit graph mapping matched SevenNet exactly. Across the direct "
    "model, official calculator, and composed pipeline checks, the largest "
    f"energy difference was {sevennet_repeat_max_energy_difference_eV_per_atom:.2e} "
    "eV per atom and the largest force-component difference was "
    f"{sevennet_repeat_max_force_difference_eV_A:.2e} eV/Å. The full check "
    "tables are saved with the run.",
    kind="result",
    result_state="pass",
))
adsorption_analysis_progress.complete("adapter and composed outputs checked")

sevennet_max_edge_vector_mapping_difference_A = float(
    sevennet_graph_mapping.loc[
        sevennet_graph_mapping["component"] == "periodic edge vectors",
        "max_abs_difference",
    ].iloc[0]
)
sevennet_max_force_eV_A = float(
    surface_force_summary.loc[
        surface_force_summary["component"]
        == "SevenNet-Omni mpa + pairwise PBE-D3(BJ)",
        "fmax_eV_A",
    ].max()
)
negative_count = int((adsorption_results["adsorption_energy_eV"] < 0.0).sum())
d3_min = float(adsorption_results["d3_adsorption_energy_eV"].min())
d3_max = float(adsorption_results["d3_adsorption_energy_eV"].max())
display(callout(
    f"At these fixed starting geometries, {negative_count} of 4 combined "
    f"energy differences are negative. D3 changes those differences by "
    f"{d3_min:.3f} to {d3_max:.3f} eV. The largest combined "
    f"atomic force is {sevennet_max_force_eV_A:.3f} eV/Å, which is why these "
    "single points are not adsorption minima. A negative value means the "
    "fixed combined structure is below its fixed slab + gas references. The "
    "3×3 cell still includes lateral periodic-image interactions, and these "
    "electronic energies omit zero-point, thermal, and entropy terms.",
    kind="result",
    result_state="observed",
))
""",
        ),
        stage_markdown(
            "stage-5",
            stage=5,
            title="Prepare dynamics and IR",
            outcome="Return to the charge-predicting molecular model, build the isotope × cluster batch, relax it, check harmonic frequencies, and wire one shared call to an IR recorder.",
            before="Predict which fields change between H₂O and D₂O: atomic number, coordinates, energy, force, charge, or mass.",
            compute_time=(
                "1 min 13 s on one H100 PCIe in the checked run"
            ),
            body=(
                r"""
The surface calculation needed a materials model. IR needs predicted charges at every update and a reference at the checkpoint's target level, so this stage returns to the supplied `aimnet2-b973c-2025-d3_0` molecular checkpoint. The next visible cell reconstructs its finite-molecule Coulomb + D3 convention once; the detailed parity checks run in collapsed cells.

- D keeps atomic number 1; only `atomic_masses` changes.
- H/D pairs begin at identical coordinates.
- Neutral graph dipoles are origin-independent; we check this explicitly.
- `add_node_property(...)` creates per-atom arrays such as forces and velocities; `add_system_property(...)` creates one value per graph, such as energy or status.
- `FIRE2` uses the model's neighbor hooks while atoms move, `ConvergenceHook` checks fmax per graph and stops this standalone batch when every graph passes, and `ZarrData` saves the relaxed batch.
"""
                + "\n\n"
                + process_diagram_html(
                    title="Reusable Toolkit dynamics route",
                    steps=(
                        "configure active outputs",
                        "attach neighbor adaptation",
                        "relax with FIRE2",
                        "assign velocities",
                        "fuse NVT + NVE stages",
                        "attach safety + recording hooks",
                        "persist raw state, then analyze",
                    ),
                    caption=(
                        "This seven-step route is the reusable Toolkit pattern. "
                        "The water-specific isotope and IR checks sit around it, not "
                        "inside hidden dynamics helpers."
                    ),
                )
            ),
        ),
        code(
            "build-ir-batch",
            """
ir_atoms, labels = make_ir_structures()
ir_batch_progress = NotebookProgress(
    title="Build the isotope and cluster batch",
    total=len(ir_atoms) + 1,
    unit="steps",
)
ir_data = []
for structure_index, atoms in enumerate(ir_atoms, start=1):
    atoms.info["charge"] = 0
    data = AtomicData.from_atoms(atoms, device=DEVICE, dtype=torch.float32)
    # Energy is one value per system; forces and velocities are per atom.
    data.add_system_property("energy", torch.zeros(1, 1, dtype=torch.float32, device=DEVICE))
    data.add_node_property("forces", torch.zeros_like(data.positions))
    data.add_node_property("velocities", torch.zeros_like(data.positions))
    ir_data.append(data)
    ir_batch_progress.update(
        done=structure_index,
        message=f"{labels[structure_index - 1]} converted to AtomicData",
    )

batch = Batch.from_data_list(ir_data, device=DEVICE)
# Warp-backed FIRE2 and MD operate on one common vector dtype. AIMNet also
# requires float32, so positions, velocities, and forces stay float32 end to end.
assert all(
    tensor.dtype == torch.float32
    for tensor in (batch.positions, batch.velocities, batch.forces)
)
ir_batch_progress.complete("four systems packed into one ragged Batch")
display(readable_table(
    pd.DataFrame([
        ("Systems", ", ".join(labels)),
        ("Graphs", batch.num_graphs),
        ("Atoms", batch.num_nodes),
        ("batch_idx shape", tuple(batch.batch_idx.shape)),
        ("Graph boundaries", batch.batch_ptr.cpu().tolist()),
    ], columns=["Batch field", "Value"]),
    label="Isotope and cluster Batch",
    show_index=False,
))
display(view(ir_atoms[2], viewer="x3d"))
display(callout(
    "Structure view: six water molecules arranged as a nonperiodic cyclic hydrogen-bonded ring.",
    kind="note",
))
""",
        ),
        code(
            "compile-fixed-ir-model",
            """
compile_progress = NotebookProgress(
    title="Compile the fixed 42-atom IR workload", total=3, unit="checks"
)

# Establish the eager reference on the exact production batch.
assert torch.get_float32_matmul_precision() == AIMNET_MATMUL_PRECISION
eager_model = model
fixed_ir_batch = batch.clone()
compute_neighbors(fixed_ir_batch, config=eager_model.model_config.neighbor_config)
IR_HELD_FIELDS = (
    "positions",
    "atomic_numbers",
    "charge",
    "neighbor_matrix",
    "num_neighbors",
)
IR_COMPARE_OUTPUTS = ("energy", "forces", "charges")
fixed_ir_source = snapshot_tensor_fields(
    fixed_ir_batch,
    field_names=IR_HELD_FIELDS,
)
eager_ir_raw = eager_model(fixed_ir_batch)
# Clone each reference immediately. Pipeline calls may reuse Batch-owned
# buffers, so a later call must not change the values being compared here.
eager_ir_outputs = clone_selected_outputs(
    eager_ir_raw,
    output_names=IR_COMPARE_OUTPUTS,
)
assert_tensor_fields_unchanged(
    fixed_ir_batch,
    fixed_ir_source,
    field_names=IR_HELD_FIELDS,
)
torch.cuda.synchronize()
compile_progress.update(done=1, message="eager energy, forces, and charges recorded")

# Default Torch compilation is applied only to this fixed-topology workload.
# Variable-size scans above stay eager and do not trigger shape-specific
# backward autotuning.
production_backbone = torch.compile(
    aimnet.model.eval(),
    fullgraph=False,
    dynamic=False,
)
production_aimnet = AIMNet2Wrapper(production_backbone, train=False).to(DEVICE)
production_aimnet.set_config("active_outputs", {"energy", "charges"})
production_coulomb = DirectCoulombWrapper().to(DEVICE)
production_d3 = DFTD3ModelWrapper(
    a1=d3_params["a1"],
    a2=d3_params["a2"],
    s8=d3_params["s8"],
    s6=d3_params.get("s6", 1.0),
    cutoff=D3_CUTOFF_A,
    param_file=D3_PARAMETER_FILE,
    auto_download=False,
).to(DEVICE)
for component in (production_aimnet, production_d3):
    component.model_config.neighbor_config.skin = NEIGHBOR_SKIN_A
production_d3.set_config("active_outputs", {"energy", "forces"})

model = PipelineModelWrapper(
    groups=[
        PipelineGroup(
            steps=[production_aimnet, production_coulomb], use_autograd=True
        ),
        PipelineGroup(steps=[production_d3], use_autograd=False),
    ],
    neighbor_adaptation="always",
).to(DEVICE)
model.set_config("active_outputs", {"energy", "forces", "charges"})
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)

# Hold positions and the Toolkit-built source neighbor matrix fixed so this
# check isolates compilation from a separate neighbor-list rebuild.
for field in ("cutoff", "format", "half_list", "skin"):
    assert getattr(model.model_config.neighbor_config, field) == getattr(
        eager_model.model_config.neighbor_config, field
    )
# Full float32 matmul is intentional for this strict numerical check;
# compilation throughput is not measured here.
compiled_ir_raw = model(fixed_ir_batch)
compiled_ir_outputs = clone_selected_outputs(
    compiled_ir_raw,
    output_names=IR_COMPARE_OUTPUTS,
)
assert_tensor_fields_unchanged(
    fixed_ir_batch,
    fixed_ir_source,
    field_names=IR_HELD_FIELDS,
)
torch.cuda.synchronize()
compile_progress.update(done=2, message="compiled forward and force backward synchronized")

compiled_repeat_raw = model(fixed_ir_batch)
compiled_repeat_outputs = clone_selected_outputs(
    compiled_repeat_raw,
    output_names=IR_COMPARE_OUTPUTS,
)
assert_tensor_fields_unchanged(
    fixed_ir_batch,
    fixed_ir_source,
    field_names=IR_HELD_FIELDS,
)
torch.cuda.synchronize()
compiled_ir_eager_agreement = max_absolute_differences(
    compiled_ir_outputs,
    eager_ir_outputs,
    output_names=IR_COMPARE_OUTPUTS,
)
compiled_ir_repeat_agreement = max_absolute_differences(
    compiled_repeat_outputs,
    compiled_ir_outputs,
    output_names=IR_COMPARE_OUTPUTS,
)
del eager_ir_raw, compiled_ir_raw, compiled_repeat_raw
print("compiled - eager:", compiled_ir_eager_agreement)
print("compiled repeat :", compiled_ir_repeat_agreement)
compiled_ir_checks = build_difference_check_table(
    {
        "compiled - eager": compiled_ir_eager_agreement,
        "compiled repeat": compiled_ir_repeat_agreement,
    },
    {
        "compiled - eager": {
            "energy": COMPILED_EAGER_ENERGY_TOLERANCE_EV,
            "forces": COMPILED_EAGER_FORCE_TOLERANCE_EV_A,
            "charges": COMPILED_EAGER_CHARGE_TOLERANCE_E,
        },
        "compiled repeat": {
            "energy": COMPILED_REPEAT_ENERGY_TOLERANCE_EV,
            "forces": COMPILED_REPEAT_FORCE_TOLERANCE_EV_A,
            "charges": COMPILED_REPEAT_CHARGE_TOLERANCE_E,
        },
    },
)
display(readable_table(
    compiled_ir_checks,
    label="Compiled and eager model checks",
    show_index=False,
))
if not bool(compiled_ir_checks["passed"].all()):
    raise RuntimeError("Compiled model numerical check failed")
compile_progress.complete("fixed-input compiled calls match the eager Toolkit pipeline")
display(callout(
    "With positions and the Toolkit source neighbor matrix held fixed, the compiled and eager models return the same energies, forces, and charges for this four-system batch. Full float32 matmul is deliberate in this correctness check; this cell does not benchmark compilation speed. FIRE2 and MD can now reuse the compiled model.",
    kind="result",
    result_state="pass",
))
del fixed_ir_source, fixed_ir_batch
del eager_ir_outputs, compiled_ir_outputs, compiled_repeat_outputs
""",
        ),
        code(
            "inspect-ir-batch",
            """
ir_checks_progress = NotebookProgress(
    title="Check isotope and dipole inputs", total=3, unit="checks"
)
compute_neighbors(batch, config=model.model_config.neighbor_config)
initial_outputs = model(batch)
mass_checks = mass_only_invariance(
    batch,
    initial_outputs,
    position_rtol=MASS_ONLY_POSITION_RTOL,
    position_atol_angstrom=MASS_ONLY_POSITION_ATOL_A,
    energy_tolerance_eV=MASS_ONLY_ENERGY_TOLERANCE_EV,
    force_tolerance_eV_A=MASS_ONLY_FORCE_TOLERANCE_EV_A,
    charge_tolerance_e=MASS_ONLY_CHARGE_TOLERANCE_E,
)
ir_checks_progress.advance(message="complete model evaluated on all four systems")

q = initial_outputs["charges"].reshape(-1)
graph_idx = batch.batch_idx.to(torch.int32)
q_sum = segmented_sum(q, graph_idx, batch.num_graphs)
mu = segmented_sum(q[:, None] * batch.positions, graph_idx, batch.num_graphs)
force_norm = torch.linalg.vector_norm(initial_outputs["forces"], dim=1)
initial_fmax = torch.stack([
    force_norm[graph_idx == graph].max() for graph in range(batch.num_graphs)
])

translations = torch.tensor(
    [[1.2, -0.7, 0.5], [-0.3, 0.9, 1.1], [2.0, 1.0, -1.0], [-1.0, -2.0, 0.4]],
    device=DEVICE,
    dtype=batch.positions.dtype,
)
shifted = batch.positions + translations[batch.batch_idx.long()]
mu_shifted = segmented_sum(q[:, None] * shifted, graph_idx, batch.num_graphs)
origin_error = float((mu - mu_shifted).abs().max().detach().cpu())
assert (
    float(q_sum.abs().max().detach().cpu())
    < IR_CHARGE_NEUTRALITY_TOLERANCE_E
)
assert origin_error < IR_DIPOLE_ORIGIN_TOLERANCE_E_ANGSTROM
ir_checks_progress.advance(message="charge neutrality and dipole origin invariance pass")

display(readable_table(
    pd.DataFrame({
        "system": labels,
        "atoms": batch.num_nodes_per_graph.cpu().numpy(),
        "energy_eV": initial_outputs["energy"].detach().reshape(-1).cpu().numpy(),
        "max_force_eV_A": initial_fmax.detach().cpu().numpy(),
        "charge_sum_e": q_sum.detach().cpu().numpy(),
    }),
    label="Initial IR-batch outputs",
    show_index=False,
))
display(readable_table(
    pd.Series(
        {**mass_checks, "origin_error_eA": origin_error}, name="Value"
    ).rename_axis("Check").reset_index(),
    label="Isotope and dipole checks",
    show_index=False,
))
ir_checks_progress.complete("H/D pairs differ only in atomic mass")
display(callout("H/D energy, force, charge, and coordinates match; only mass changes.", kind="result", result_state="pass"))
unrelaxed_batch = batch.clone()
""",
        ),
        code(
            "relax",
            """
batch = unrelaxed_batch.clone()
relax_progress = NotebookProgress(
    title="Batched FIRE2 relaxation", total=FIRE_MAX_STEPS, unit="steps"
)
relaxer = FIRE2(
    model=model,
    dt=IR_FIRE_INITIAL_DT,
    n_steps=FIRE_MAX_STEPS,
    convergence_hook=ConvergenceHook.from_fmax(threshold=FIRE_FMAX_EV_A),
)
for neighbor_hook in model.make_neighbor_hooks():
    relaxer.register_hook(neighbor_hook)
relaxer.register_hook(NaNDetectorHook(frequency=1))
relax_progress_hook: Hook = NotebookStageProgressHook(
    relax_progress, frequency=25, label="FIRE2"
)
relaxer.register_hook(relax_progress_hook)

t0 = perf_counter()
batch = relaxer.run(batch)
torch.cuda.synchronize()
relax_elapsed_s = perf_counter() - t0

# Reset D coordinates exactly to H: isotope substitution changes mass only.
ptr = batch.batch_ptr.tolist()
batch.positions[ptr[1] : ptr[2]].copy_(batch.positions[ptr[0] : ptr[1]])
batch.positions[ptr[3] : ptr[4]].copy_(batch.positions[ptr[2] : ptr[3]])

# Re-evaluate after the coordinate reset so every persisted field describes
# the same final state.
compute_neighbors(batch, config=model.model_config.neighbor_config)
relaxed_outputs = model(batch)
batch.energy = relaxed_outputs["energy"].to(batch.positions.dtype)
batch.forces = relaxed_outputs["forces"]
batch.charges = relaxed_outputs["charges"]
relaxed_mass_checks = mass_only_invariance(
    batch,
    relaxed_outputs,
    position_rtol=MASS_ONLY_POSITION_RTOL,
    position_atol_angstrom=MASS_ONLY_POSITION_ATOL_A,
    energy_tolerance_eV=MASS_ONLY_ENERGY_TOLERANCE_EV,
    force_tolerance_eV_A=MASS_ONLY_FORCE_TOLERANCE_EV_A,
    charge_tolerance_e=MASS_ONLY_CHARGE_TOLERANCE_E,
)

fmax = [
    float(torch.linalg.vector_norm(batch.forces[a:b], dim=1).max().cpu())
    for a, b in zip(ptr[:-1], ptr[1:], strict=True)
]
if max(fmax) > FIRE_FMAX_EV_A:
    relax_progress.update(done=relaxer.step_count, message="Force criterion not reached", state="action")
    raise RuntimeError("FIRE2 did not reach the specified 0.01 eV/Å criterion")
relax_progress.update(
    done=relaxer.step_count,
    message=f"Converged in {relaxer.step_count} steps ({relax_elapsed_s:.1f} s)",
    state="complete",
)
relaxed_batch = batch.clone()

relaxed_zarr_path = OUTPUT_DIR / "water_ir_relaxed.zarr"
relaxed_sink = ZarrData(relaxed_zarr_path, capacity=relaxed_batch.num_graphs)
relaxed_sink.zero()
relaxed_sink.write(relaxed_batch)
relaxed_replay = relaxed_sink.read()
assert relaxed_replay.num_graphs == relaxed_batch.num_graphs
torch.testing.assert_close(
    relaxed_replay.positions,
    relaxed_batch.positions.cpu(),
    rtol=0.0,
    atol=0.0,
)
display(readable_table(
    pd.DataFrame({"system": labels, "fmax_eV_A": fmax}),
    label="Relaxed-structure force maxima",
    show_index=False,
))
display(readable_table(
    pd.Series(relaxed_mass_checks, name="Value")
    .rename_axis("Check")
    .reset_index(),
    label="Relaxed isotope checks",
    show_index=False,
))

relaxed_atoms = [
    graph_atoms_from_batch(relaxed_batch, graph, label)
    for graph, label in enumerate(labels)
]
write(OUTPUT_DIR / "water_ir_relaxed_start.extxyz", relaxed_atoms)
display(callout(
    "All four structures reached fmax < 0.01 eV/Å. Writing them to Zarr and reading them back reproduced the same coordinates.",
    kind="result",
    result_state="pass",
))
""",
        ),
        markdown(
            "harmonic-intro",
            r"""
### Match the harmonic frequency calculation before running MD

The model and DFT calculations use separately optimized minima. The frequency
comparison aligns the finite-difference displacement size, projection, masses,
units, and mode mapping on both sides:

- optimize one H₂O monomer;
- evaluate the **complete** AIMNet + Coulomb + D3 force at every Cartesian displacement;
- obtain a point-charge dipole derivative from AIMNet's geometry-dependent predicted charges;
- remove translation and rotation, then calculate double-harmonic frequencies and model-specific integrated intensities in km/mol;
- repeat the mode analysis with D masses while keeping the geometry, Hessian, and dipole derivative unchanged.

Each displacement step contains all 18 `+/-` structures in one `Batch`, so three step sizes require three full-model calls. The selected 0.005 bohr step matches the B97-3c reference calculation; 0.010 and 0.020 bohr calculations check numerical stability.
"""
            + "\n\n"
            + process_diagram_html(
                title="Matched monomer harmonic comparison",
                steps=(
                    "tight full-model minimum",
                    "18 displaced structures per Batch",
                    "full forces + predicted charges",
                    "Hessian + dipole derivative",
                    "H and D mass-only mode analysis",
                    "B97-3c sticks + observed positions",
                ),
                caption=(
                    "AIMNet + Coulomb + D3 and B97-3c use the same projected normal-mode "
                    "analysis. Their dipole derivatives come from different "
                    "models, so only frequencies are scored. Experiment contributes positions only."
                ),
            )
            + "\n\n"
            + callout_html(
                "This is a same-target-level comparison because the AIMNet checkpoint was trained to B97-3c data. It checks how closely this model reproduces that target for these monomers; it is not an independent electronic-structure benchmark and does not establish that an input is within the training distribution. Frequencies use the complete AIMNet checkpoint-base + Coulomb + D3 model. AIMNet point-charge dipole intensity and B97-3c electronic-dipole intensity are both shown in km/mol, but their absolute values are not treated as an accuracy comparison.",
                kind="note",
            ),
        ),
        code(
            "harmonic-minimum",
            """
harmonic_minimum_progress = NotebookProgress(
    title="Tighten the water-monomer minimum",
    total=HARMONIC_FIRE_MAX_STEPS,
    unit="FIRE2 steps",
)

harmonic_start_atoms = relaxed_atoms[0].copy()
harmonic_start_atoms.info["charge"] = 0
harmonic_start = AtomicData.from_atoms(
    harmonic_start_atoms,
    device=DEVICE,
    dtype=torch.float32,
)
harmonic_start.add_system_property(
    "energy", torch.zeros(1, 1, dtype=torch.float32, device=DEVICE)
)
harmonic_start.add_node_property("forces", torch.zeros_like(harmonic_start.positions))
harmonic_start.add_node_property(
    "velocities", torch.zeros_like(harmonic_start.positions)
)
harmonic_minimum_batch = Batch.from_data_list([harmonic_start], device=DEVICE)

harmonic_relaxer = FIRE2(
    model=eager_model,
    dt=HARMONIC_FIRE_INITIAL_DT,
    n_steps=HARMONIC_FIRE_MAX_STEPS,
    convergence_hook=ConvergenceHook.from_fmax(threshold=HARMONIC_FMAX_EV_A),
)
for neighbor_hook in eager_model.make_neighbor_hooks():
    harmonic_relaxer.register_hook(neighbor_hook)
harmonic_relaxer.register_hook(NaNDetectorHook(frequency=1))
harmonic_relaxer.register_hook(NotebookStageProgressHook(
    harmonic_minimum_progress,
    frequency=25,
    label="monomer FIRE2",
))
harmonic_minimum_batch = harmonic_relaxer.run(harmonic_minimum_batch)

compute_neighbors(
    harmonic_minimum_batch,
    config=eager_model.model_config.neighbor_config,
)
harmonic_minimum_output = eager_model(harmonic_minimum_batch)
harmonic_fmax_eV_A = float(torch.linalg.vector_norm(
    harmonic_minimum_output["forces"], dim=1
).max().detach().cpu())
harmonic_minimum_passed = harmonic_fmax_eV_A <= HARMONIC_FMAX_EV_A
harmonic_minimum_progress.update(
    done=min(harmonic_relaxer.step_count, HARMONIC_FIRE_MAX_STEPS),
    message=f"final fmax = {harmonic_fmax_eV_A:.2e} eV/Å",
    state="complete",
)

harmonic_geometry_angstrom = (
    harmonic_minimum_batch.positions.detach().cpu().numpy().reshape(3, 3)
)
harmonic_atomic_numbers = (
    harmonic_minimum_batch.atomic_numbers.detach().cpu().numpy().reshape(-1)
)
harmonic_h_masses_u = (
    harmonic_minimum_batch.atomic_masses.detach().cpu().numpy().reshape(-1)
)
harmonic_d_masses_u = (
    relaxed_batch.get_data(1).atomic_masses.detach().cpu().numpy().reshape(-1)
)
harmonic_optimized_atoms = harmonic_start_atoms.copy()
harmonic_optimized_atoms.positions = harmonic_geometry_angstrom
write(OUTPUT_DIR / "water_monomer_harmonic_minimum.extxyz", harmonic_optimized_atoms)

assert np.array_equal(harmonic_atomic_numbers, np.array([8, 1, 1]))
assert np.array_equal(
    harmonic_h_masses_u[harmonic_atomic_numbers == 8],
    harmonic_d_masses_u[harmonic_atomic_numbers == 8],
)
assert np.all(
    harmonic_d_masses_u[harmonic_atomic_numbers == 1]
    > harmonic_h_masses_u[harmonic_atomic_numbers == 1]
)
display(readable_table(pd.Series({
    "FIRE2 steps executed": harmonic_relaxer.step_count,
    "final fmax (eV/Å)": harmonic_fmax_eV_A,
    "required fmax (eV/Å)": HARMONIC_FMAX_EV_A,
    "minimum check": "PASS" if harmonic_minimum_passed else "CHECK FAILED",
}, name="Value").rename_axis("Check").reset_index(),
    label="Harmonic minimum",
    show_index=False,
))
""",
        ),
        code(
            "harmonic-finite-difference",
            """
harmonic_fd_progress = NotebookProgress(
    title="Evaluate all harmonic displacements in three batches",
    total=len(HARMONIC_DISPLACEMENT_STEPS_BOHR),
    unit="model calls",
)
harmonic_displacement_results = []

for pass_index, step_bohr in enumerate(HARMONIC_DISPLACEMENT_STEPS_BOHR, start=1):
    step_bohr = float(step_bohr)
    step_angstrom = step_bohr * ANGSTROM_PER_BOHR
    displaced = symmetric_cartesian_displacements(
        harmonic_geometry_angstrom,
        step_angstrom,
    )
    packed_positions = np.concatenate(
        [displaced.plus_angstrom, displaced.minus_angstrom], axis=0
    )

    displaced_data = []
    for positions in packed_positions:
        atoms = harmonic_optimized_atoms.copy()
        atoms.positions = positions
        atoms.info["charge"] = 0
        data = AtomicData.from_atoms(atoms, device=DEVICE, dtype=torch.float32)
        data.add_system_property(
            "energy", torch.zeros(1, 1, dtype=torch.float32, device=DEVICE)
        )
        data.add_node_property("forces", torch.zeros_like(data.positions))
        displaced_data.append(data)

    displaced_batch = Batch.from_data_list(displaced_data, device=DEVICE)
    compute_neighbors(
        displaced_batch,
        config=eager_model.model_config.neighbor_config,
    )
    displaced_output = eager_model(displaced_batch)

    # The helper handles reshaping, dipoles, and finite differences. The
    # displacement choice, Toolkit Batch, neighbor build, and model call stay here.
    displacement_result = collect_harmonic_displacement_result(
        step_bohr=step_bohr,
        step_angstrom=step_angstrom,
        n_atoms=harmonic_geometry_angstrom.shape[0],
        positions_angstrom=displaced_batch.positions.detach().cpu().numpy(),
        forces_eV_per_angstrom=(
            displaced_output["forces"].detach().cpu().numpy()
        ),
        charges_e=displaced_output["charges"].detach().cpu().numpy(),
        dipole_origin_atom_index=0,
        neutral_tolerance_e=HARMONIC_CHARGE_NEUTRALITY_TOLERANCE_E,
    )
    harmonic_displacement_results.append(displacement_result)
    harmonic_fd_progress.update(
        done=pass_index,
        message=(
            f"{displacement_result.structures_in_call} structures at "
            f"{step_bohr:.3f} bohr"
        ),
    )

harmonic_fd_progress.complete("54 displaced structures evaluated in three model calls")
harmonic_series = analyze_harmonic_step_series(
    harmonic_displacement_results,
    geometry_angstrom=harmonic_geometry_angstrom,
    isotopologues=(
        ("H2O", harmonic_h_masses_u),
        ("D2O", harmonic_d_masses_u),
    ),
    selected_step_bohr=HARMONIC_SELECTED_STEP_BOHR,
    minimum_passed=harmonic_minimum_passed,
    frequency_step_tolerance_cm1=HARMONIC_FREQUENCY_STEP_TOLERANCE_CM1,
    intensity_step_abs_tolerance_km_mol=(
        HARMONIC_INTENSITY_STEP_ABS_TOLERANCE_KM_MOL
    ),
    intensity_step_rel_tolerance=HARMONIC_INTENSITY_STEP_REL_TOLERANCE,
    mode_overlap_min=HARMONIC_MODE_OVERLAP_MIN,
    hessian_antisymmetry_rel_max=HARMONIC_HESSIAN_ANTISYMMETRY_REL_MAX,
    imaginary_floor_cm1=HARMONIC_IMAGINARY_FLOOR_CM1,
)

# Keep short aliases for the later comparison, archive, and run summary.
harmonic_fd_table = harmonic_series.displacement_table
harmonic_h_convergence = harmonic_series.convergence_by_isotopologue["H2O"]
harmonic_d_convergence = harmonic_series.convergence_by_isotopologue["D2O"]
selected_harmonic_estimate = harmonic_series.selected_estimate
aimnet_harmonic = dict(harmonic_series.mode_analyses_by_isotopologue)
harmonic_validation = dict(harmonic_series.validation_checks)
harmonic_validation_table = harmonic_series.validation_table
harmonic_convergence_table = harmonic_series.convergence_table
harmonic_comparison_reported = harmonic_series.comparison_reported

display(readable_table(
    harmonic_validation_table,
    label="Harmonic numerical checks",
    show_index=False,
))
display(callout(
    (
        "The 0.005 bohr harmonic result passed the minimum, displacement-step, "
        "mode-continuity, Hessian-symmetry, and imaginary-mode checks."
        if harmonic_comparison_reported
        else "NOT REPORTED: at least one harmonic numerical check did not pass. "
        "The complete displaced-structure results are retained for inspection."
    ),
    kind="result",
    result_state="pass" if harmonic_comparison_reported else "not_reported",
))
""",
        ),
        code(
            "harmonic-comparison",
            """
harmonic_comparison_progress = NotebookProgress(
    title="Map AIMNet + Coulomb + D3 and B97-3c modes",
    total=3,
    unit="steps",
)
mode_order = ("symmetric_stretch", "bend", "antisymmetric_stretch")
mode_number = {"symmetric_stretch": 1, "bend": 2, "antisymmetric_stretch": 3}
harmonic_comparison_table = empty_harmonic_mode_comparison_table()
harmonic_frequency_mae_cm1 = None
harmonic_figure = None
if harmonic_comparison_reported:
    harmonic_comparison_table = build_harmonic_mode_comparison_table(
        geometry_angstrom=harmonic_geometry_angstrom,
        atomic_numbers=harmonic_atomic_numbers,
        isotopologues=(
            ("H2O", harmonic_h_masses_u),
            ("D2O", harmonic_d_masses_u),
        ),
        model_analyses_by_isotopologue=aimnet_harmonic,
        references_by_isotopologue=references,
        observed_by_mode=observed_by_mode,
        mode_order=mode_order,
        mode_numbers=mode_number,
    )
    harmonic_comparison_progress.advance(message="mode characters matched")
    harmonic_frequency_mae_cm1 = float(
        harmonic_comparison_table[
            "AIMNet+Coulomb+D3_minus_B97-3c_cm-1"
        ].abs().mean()
    )
    harmonic_comparison_progress.advance(
        message="frequencies and km/mol sticks tabulated"
    )
    display(readable_table(
        harmonic_mode_comparison_display_table(harmonic_comparison_table),
        label="AIMNet and B97-3c harmonic modes",
        show_index=False,
    ))
    harmonic_figure, _ = plot_harmonic_monomer_comparison(
        harmonic_comparison_table,
        wavenumber_limits_cm1=(500.0, 4200.0),
    )
    harmonic_figure.savefig(
        OUTPUT_DIR / "water_monomer_harmonic_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    display(figure_with_alt(
        harmonic_figure,
        alt_text=(
            "Side-by-side H2O and D2O harmonic frequency comparisons with "
            "model-specific stick intensities. AIMNet plus Coulomb plus D3 "
            "uses a predicted point-charge dipole; B97-3c uses its electronic "
            "dipole derivative. Experimental gas-phase fundamentals are "
            "position markers only."
        ),
    ))
    plt.close(harmonic_figure)
else:
    harmonic_comparison_progress.advance(
        message="numerical checks did not pass; mode matching skipped"
    )
    harmonic_comparison_progress.advance(message="no table or MAE reported")

harmonic_archive_arrays = build_harmonic_archive_arrays(
    geometry_angstrom=harmonic_geometry_angstrom,
    atomic_numbers=harmonic_atomic_numbers,
    isotopologues=(
        ("H2O", harmonic_h_masses_u),
        ("D2O", harmonic_d_masses_u),
    ),
    minimum_forces_eV_per_angstrom=(
        harmonic_minimum_output["forces"].detach().cpu().numpy()
    ),
    selected_step_bohr=HARMONIC_SELECTED_STEP_BOHR,
    selected_estimate=selected_harmonic_estimate,
    mode_analyses_by_isotopologue=aimnet_harmonic,
    displacement_results=harmonic_displacement_results,
)
harmonic_archive_path = OUTPUT_DIR / "water_monomer_aimnet_harmonic_ir.npz"
np.savez_compressed(harmonic_archive_path, **harmonic_archive_arrays)
harmonic_archive_sha256 = sha256_file(harmonic_archive_path)
harmonic_comparison_progress.complete(
    f"raw harmonic arrays saved; SHA-256 {harmonic_archive_sha256[:12]}"
)

display(callout(
    (
        f"AIMNet + Coulomb + D3 versus B97-3c harmonic frequency MAE for these six "
        f"matched modes: {harmonic_frequency_mae_cm1:.1f} cm⁻¹. "
        "Experimental fundamentals remain position context because they are "
        "anharmonic observables."
        if harmonic_comparison_reported
        else "NOT REPORTED: the harmonic numerical checks did not all pass, "
        "so no frequency MAE is interpreted. The raw arrays and check table "
        "remain saved for inspection."
    ),
    kind="result",
    result_state="observed" if harmonic_comparison_reported else "not_reported",
))
""",
        ),
        markdown(
            "ir-mechanism",
            r"""
### One forward pass, two jobs

`active_outputs = {energy, forces, charges}` → `batch.charges` → `segmented_sum(qᵢ rᵢ)` → total dipole μ(t) → finite difference μ̇(t) → 5 ps Hann-window Welch spectrum.

The total cluster dipole keeps intermolecular cross-correlations. Differencing the total dipole includes both `q·v` and charge-flux `r·dq/dt` contributions.

Toolkit creates a `DynamicsContext` with the current batch and step number and
passes it to each registered `Hook`; learners do not construct this object. The
compact recorder implementation lives in `aux`, while registration and the
data path stay visible below.

**Toolkit APIs used here:** `FusedStage`, `Hook`, `ConvergenceHook`, `DynamicsContext`, `register_hook`, and `register_fused_hook`.

**Tutorial helpers:** `PredictedChargeIRHook`, `StageStepCounterHook`, `NotebookStageProgressHook`, and `converge_after_steps` live in `aux/`; they are not Toolkit APIs.
"""
            + "\n\n"
            + callout_html(
                "The recorder keeps its arrays on the GPU and transfers them once after NVE. Hook registration remains visible below; no second model call is made for charges.",
                kind="check",
            ),
        ),
        stage_markdown(
            "stage-6",
            stage=6,
            title="Run and inspect the trajectory",
            outcome=(
                "Execute the 5,000-step NVT + 20,000-step NVE calculation, "
                "save every production frame, then calculate spectra after the checks."
            ),
            before=(
                "This is the fixed 25,000-step teaching workload. It is long enough "
                "to demonstrate the complete analysis path, but it is not a "
                "trajectory-length convergence study."
            ),
            compute_time=(
                "9 min 27 s on one H100 PCIe in the checked run"
            ),
            body=r"""
- 5,000 steps Langevin NVT at 75 K; damped dynamics are not sampled.
- 20,000 steps NVE at 0.5 fs = 10 ps.
- Dipole, charge, energy, and all 42 atomic positions are saved every NVE step.
- Charge is checked on every saved frame.
- After differentiating the dipole, the 10 ps record contains two complete 5 ps Hann windows with 50% overlap. This is enough for a qualitative teaching spectrum, not stable peak heights or trajectory-length convergence.
- NVE kinetic temperature uses Toolkit's unconstrained 3N convention. The one-time initialization removes drift and rotation, but Langevin dynamics does not constrain those degrees of freedom afterward.
- The 20% pair screen checks this exact protocol only; a pass would not by itself prove matched internal vibrational ensembles.
- Both hexamers must remain connected with assigned O–H distances below 1.25 Å.
- Cyclic-DFT overlays require the initial directed six-water ring in every frame.
- The 1 meV atom⁻¹ energy-excursion line is an advisory, not a stopping rule.
""",
        ),
        markdown(
            "reference-preview",
            r"""
### Reference for the long run

The figure below is the separately computed, unscaled B97-3c/def2-mTZVP double-harmonic reference. The AIMNet and B97-3c calculations use separately optimized minima, while matching displacement size, projection, isotope masses, and units. Because the checkpoint was trained to B97-3c data, this is a same-target-level comparison, not an independent electronic-structure benchmark or proof that these inputs are within the training distribution. Within the B97-3c calculation, H and D reuse one optimized geometry, electronic Hessian, and dipole derivative; only masses change. The final monomer plot adds selected observed gas-phase H₂O/D₂O fundamentals as position markers. Raw DFT sticks remain visible; the smooth DFT curve is the known 5 ps discrete Hann response, not a fitted linewidth.

![Four-panel B97-3c double-harmonic IR reference for H2O, D2O, cyclic (H2O)6, and cyclic (D2O)6, showing raw mode-character sticks and independently normalized 5 ps Hann responses.](attachment:b97_3c_ir_reference.png)
"""
            + "\n\n"
            + callout_html(
                "The completed monomer comparison later in the notebook will show three different quantities: a finite-temperature classical MD spectrum, 0 K harmonic DFT sticks, and experimental gas-phase band positions. Compare frequency regions, not absolute intensities. We report the H/D shift only when the H2O and D2O trajectories have similar temperatures.",
                kind="note",
            ),
            attachments=reference_figure_attachment,
        ),
        code(
            "reference-preflight",
            """
reference_progress = NotebookProgress(
    title="Verify computed and observed IR references", total=2, unit="bundles"
)
reference_dirs = {"H2O": "h2o", "D2O": "d2o", "(H2O)6": "h6", "(D2O)6": "d6"}
references = {
    label: load_psi4_b973c_ir_artifact(REFERENCE_ROOT / directory)
    for label, directory in reference_dirs.items()
}
reference_progress.advance(message="B97-3c harmonic bundles verified")
experimental_fundamentals = load_experimental_water_fundamentals()
experimental_data_sha256 = sha256_file(
    PART_DIR
    / "reference"
    / "experimental_water_fundamentals"
    / "water_gas_phase_fundamentals.csv"
)
experimental_artifact_id = (
    f"experimental-water-fundamentals-{experimental_data_sha256[:16]}"
)
reference_progress.complete("Dinu Table 1 position bundle verified")
mode_number = {"symmetric_stretch": 1, "bend": 2, "antisymmetric_stretch": 3}
observed_by_mode = experimental_fundamentals.set_index(["isotopologue", "mode"])
harmonic_mode_indices = {}
assignment_rows = []
for label in ("H2O", "D2O"):
    assignments = reference_water_monomer_mode_labels(references[label])
    harmonic_mode_indices[label] = tuple(mode_number[mode] for mode in assignments)
    for mode, harmonic_cm1 in zip(
        assignments, references[label].frequencies_cm1, strict=True
    ):
        observed_cm1 = float(observed_by_mode.loc[(label, mode), "wavenumber_cm1"])
        assignment_rows.append({
            "system": label,
            "mode": f"ν{mode_number[mode]} {mode.replace('_', ' ')}",
            "B97-3c_harmonic_cm-1": harmonic_cm1,
            "observed_gas_cm-1": observed_cm1,
            "harmonic_minus_observed_cm-1": harmonic_cm1 - observed_cm1,
        })
print("B97-3c reference preflight: PASS")
print("engine:", references["H2O"].engine_version)
print("method:", references["H2O"].manifest["model_chemistry"])
print("observed monomer markers:", len(experimental_fundamentals))
display(readable_table(
    experimental_fundamentals[[
        "isotopologue", "mode_index", "mode", "wavenumber_cm1", "phase"
    ]],
    label="Observed gas-phase fundamentals",
    show_index=False,
))
display(readable_table(
    pd.DataFrame(assignment_rows).round(2),
    label="B97-3c harmonic modes and observed positions",
    show_index=False,
))
display(callout(
    "Harmonic − observed combines electronic-structure error, the double-harmonic approximation, and reference-condition differences. It is context, not an AIMNet validation score.",
    kind="note",
))
""",
        ),
        markdown(
            "observed-source-links",
            r"""
### Experimental sources

Dinu et al., Table 1 (CC BY 4.0) gives six gas-phase H₂-¹⁶O and D₂-¹⁶O positions from Toth's measurements. Positions only; see References.
""",
        ),
        markdown(
            "fused-stage-intro",
            r"""
### Fuse stages without splitting the model call

`nvt + nve` constructs a Toolkit `FusedStage`. Both integrators stay on the same GPU and operate on one live `Batch`.

"Fused" does not mean that NVT and NVE update the same system at once. Each system's status selects exactly one stage for that update; the shared model evaluation is the part performed together.

On each fused update:

1. Toolkit evaluates the model once for the complete live batch.
2. Each system's `status` selects the integrator that updates it.
3. A `BEFORE_STEP` hook counts the update selected by each system's status.
4. A `ConvergenceHook` marks that stage complete and advances the system's `status` when its counter reaches the target.
5. Reaching the final status means that system has finished the fused workflow.

The status and counters belong to each system, not to the batch as a whole. This makes the number of actual NVT and NVE updates directly checkable. Systems can occupy different stages at the same time. With an ordinary input batch, the same systems remain in the active batch until they finish. The next section adds replacement from a larger dataset.
"""
            + "\n\n"
            + process_diagram_html(
                title="One FusedStage, one shared model evaluation",
                steps=(
                    "live Batch on one GPU",
                    "one model call",
                    "status 0 · NVT update",
                    "status 1 · NVE update",
                    "status 2 · finished",
                ),
                caption=(
                    "The model sees the complete live batch. Each system's status value decides "
                    "which integrator updates each system after that shared call."
                ),
            )
            + "\n\n"
            + callout_html(
                "FusedStage reduces repeated work on one GPU. It is not a multi-GPU API; DistributedPipeline is introduced after inflight batching.",
                kind="note",
            ),
        ),
        code(
            "configure-dynamics",
            """
dynamics_setup_progress = NotebookProgress(
    title="Configure the fused NVT and NVE workflow", total=4, unit="steps"
)
batch = relaxed_batch.clone()
batch["status"] = torch.full(
    (batch.num_graphs, 1),
    IR_WARMUP_STATUS,
    dtype=torch.long,
    device=DEVICE,
)
add_stage_step_counters(batch, ("nvt_steps_done", "nve_steps_done"))

# Paired random draws; different masses then generate the physical velocity scale.
for graph, seed in zip(
    range(batch.num_graphs), IR_INITIAL_VELOCITY_RANDOM_SEEDS, strict=True
):
    start, stop = ptr[graph], ptr[graph + 1]
    local_idx = torch.zeros(stop - start, dtype=torch.int32, device=DEVICE)
    initialize_velocities(
        batch.velocities[start:stop],
        batch.atomic_masses[start:stop],
        torch.tensor([TEMPERATURE_K], device=DEVICE, dtype=batch.positions.dtype),
        local_idx,
        random_seed=seed,
        remove_com=True,
        remove_rotations=True,
        rescale=True,
        positions=batch.positions[start:stop],
    )
dynamics_setup_progress.advance(message="paired H/D velocities initialized")

ir_hook = PredictedChargeIRHook(
    warmup_steps=WARMUP_STEPS,
    n_steps=PRODUCTION_STEPS,
    dt_fs=DT_FS,
    warmup_status=IR_WARMUP_STATUS,
    production_status=IR_PRODUCTION_STATUS,
    charge_tolerance=IR_CAPTURE_CHARGE_TOLERANCE_E,
    compile_reducer=True,
)
nvt = NVTLangevin(
    model=model,
    dt=DT_FS,
    temperature=TEMPERATURE_K,
    friction=IR_NVT_FRICTION_PER_FS,
    random_seed=IR_NVT_RANDOM_SEED,
    convergence_hook=converge_after_steps("nvt_steps_done", WARMUP_STEPS),
)
nve = NVE(
    model=model,
    dt=DT_FS,
    convergence_hook=converge_after_steps("nve_steps_done", PRODUCTION_STEPS),
)
dynamics_setup_progress.advance(message="NVT and NVE stages configured")
dynamics = nvt + nve
assert isinstance(dynamics, FusedStage)
dynamics.register_fused_hook(
    StageStepCounterHook({
        IR_WARMUP_STATUS: "nvt_steps_done",
        IR_PRODUCTION_STATUS: "nve_steps_done",
    })
)
dynamics_setup_progress.advance(message="stages fused with per-system step counters")

for neighbor_hook in model.make_neighbor_hooks():
    dynamics.register_hook(neighbor_hook)
dynamics.register_hook(NaNDetectorHook(frequency=100, extra_keys=["velocities"]))
dynamics.register_hook(ir_hook)

md_log_hook = LoggingHook(
    backend="csv",
    log_path=OUTPUT_DIR / "water_ir_dynamics_log.csv",
    frequency=1_000,
)
dynamics.register_hook(md_log_hook)
dynamics_setup_progress.complete("neighbor, safety, recording, and logging hooks attached")

print("stages:", type(nvt).__name__, "+", type(nve).__name__)
print("fused as:", type(dynamics).__name__)
print("exact updates:", WARMUP_STEPS, "+", PRODUCTION_STEPS, "=", TOTAL_DYNAMICS_STEPS)
print("recorder stage/frequency:", ir_hook.stage, ir_hook.frequency)
print("NaN/Inf safety check: every 100 steps; workload is never shortened")
""",
        ),
        markdown(
            "inflight-intro",
            r"""
### Refill the live batch as systems finish

A dataset can be much larger than the batch that fits on one GPU. Inflight batching keeps only the active batch on the GPU and replaces finished systems instead of waiting for the whole batch to become idle.

`Batch` holds the systems active now. `FusedStage` decides which update each active system receives. Inflight batching adds the next system when a finished one leaves.

- `InMemoryDataset` exposes structures and their sizes.
- `SizeAwareSampler` fills a live-batch budget and assigns a stable `system_id` to every structure it adds.
- `FusedStage(..., sampler=..., sinks=...)` removes systems at the final status, writes them to a sink, and asks the sampler for replacements.
- `HostMemory` collects the finished systems on the CPU.

In this example, `status = 0` means NVT, `status = 1` means NVE, and `status = 2` means finished and moved to `HostMemory`. `system_id` remains stable when the live batch is rebuilt; `batch_idx` does not.

`max_atoms`, `max_edges`, and `max_batch_size` limit the **active** batch, not the total dataset. `refill_frequency=1` asks Toolkit to check for finished systems after every `FusedStage` iteration. It does not change the MD timestep, stage lengths, or model batch size.

The composed AIMNet + Coulomb + D3 model uses matrix-form neighbors that are built after systems enter the active batch, so this example sets `max_edges=None` and budgets the live workload with atoms and graph count.

For this Toolkit release candidate, each structure added to the active batch also carries energy, forces, and charges evaluated at its starting coordinates. This prevents the integrator's first update from using placeholder zeros; later model calls replace those values.

The registered `StageStepCounterHook` runs before each `FusedStage` iteration and increments only the counter selected by the current status. The matching `ConvergenceHook` advances a system's status when that counter reaches its target. Final-status systems go to `HostMemory`, released capacity is refilled, and unfinished systems remain in the live batch.

A larger `refill_frequency` rebuilds the live batch less often, but finished systems wait longer before their slots are reused. The best refill interval depends on model cost, stage lengths, and batch-rebuild overhead; this small scheduling example uses 1 so turnover is easy to inspect.

Refilling happens between `FusedStage` iterations, never in the middle of a model evaluation.
"""
            + "\n\n"
            + process_diagram_html(
                title="Inflight batching on one GPU",
                steps=(
                    "larger dataset",
                    "SizeAwareSampler",
                    "GPU-sized active Batch",
                    "per-system stage status",
                    "HostMemory results",
                    "refill freed capacity",
                ),
                caption=(
                    "Finished systems leave the live batch. New systems use the "
                    "freed graph and atom capacity while unfinished systems remain in the active batch."
                ),
            )
            + "\n\n"
            + callout_html(
                "Track systems with system_id. batch_idx only describes the current packed batch and can change after every refill.",
                kind="check",
            )
            + "\n\n"
            + callout_html(
                "This 2,048-system live run demonstrates one-GPU scheduling and collection, and checks that each system ID appears once. It is not a timing benchmark or a production MD calculation: each system runs only 2 NVT and 3 NVE steps. Multi-GPU timing remains unreported until Toolkit moves every required field, accepts repeated payloads in its transfer buffers, and lets the stages work on different batches at the same time.",
                kind="note",
            ),
        ),
        code(
            "inflight-example",
            """
INFLIGHT_SYSTEMS = 2_048
INFLIGHT_ACTIVE_SYSTEMS = 256
INFLIGHT_NVT_STEPS = 2
INFLIGHT_NVE_STEPS = 3
INFLIGHT_VELOCITY_RANDOM_SEED = 404
INFLIGHT_NVT_FRICTION_PER_FS = 0.01
INFLIGHT_NVT_RANDOM_SEED = 505
INFLIGHT_REFILL_FREQUENCY = 1
INFLIGHT_EXIT_STATUS = 2

inflight_progress = NotebookProgress(
    title="Process a larger dataset through one live GPU batch",
    total=3,
    unit="checks",
)

inflight_source = prepare_inflight_dimer_source(
    scan_dimers=scan_dimers,
    scan_batch=scan_batch,
    full_outputs=full_outputs,
    num_systems=INFLIGHT_SYSTEMS,
    temperature_k=TEMPERATURE_K,
    velocity_seed=INFLIGHT_VELOCITY_RANDOM_SEED,
)
atoms_per_dimer = int(inflight_source.num_nodes_per_graph[0])
assert torch.all(inflight_source.num_nodes_per_graph == atoms_per_dimer)

inflight_dataset = InMemoryDataset(in_memory_batch=inflight_source, device=DEVICE)
inflight_sampler = SizeAwareSampler(
    inflight_dataset,
    max_atoms=INFLIGHT_ACTIVE_SYSTEMS * atoms_per_dimer,
    max_edges=None,
    max_batch_size=INFLIGHT_ACTIVE_SYSTEMS,
    shuffle=False,
)
inflight_sink = HostMemory(capacity=INFLIGHT_SYSTEMS)
inflight_progress.advance(
    message=(
        f"{INFLIGHT_SYSTEMS:,} dimers prepared; at most "
        f"{INFLIGHT_ACTIVE_SYSTEMS} are active"
    )
)

eager_model.set_config("active_outputs", {"energy", "forces", "charges"})
inflight_nvt = NVTLangevin(
    model=eager_model,
    dt=DT_FS,
    temperature=TEMPERATURE_K,
    friction=INFLIGHT_NVT_FRICTION_PER_FS,
    random_seed=INFLIGHT_NVT_RANDOM_SEED,
    convergence_hook=converge_after_steps(
        "nvt_steps_done", INFLIGHT_NVT_STEPS
    ),
    device_type=DEVICE.type,
)
inflight_nve = NVE(
    model=eager_model,
    dt=DT_FS,
    convergence_hook=converge_after_steps(
        "nve_steps_done", INFLIGHT_NVE_STEPS
    ),
    device_type=DEVICE.type,
)
inflight = FusedStage(
    sub_stages=[
        (IR_WARMUP_STATUS, inflight_nvt),
        (IR_PRODUCTION_STATUS, inflight_nve),
    ],
    sampler=inflight_sampler,
    sinks=[inflight_sink],
    refill_frequency=INFLIGHT_REFILL_FREQUENCY,
    device_type=DEVICE.type,
)
inflight.register_fused_hook(
    StageStepCounterHook({
        IR_WARMUP_STATUS: "nvt_steps_done",
        IR_PRODUCTION_STATUS: "nve_steps_done",
    })
)
for neighbor_hook in eager_model.make_neighbor_hooks():
    inflight.register_hook(neighbor_hook)
inflight.register_hook(NaNDetectorHook(frequency=1, extra_keys=["velocities"]))
assert inflight.inflight_mode
assert inflight.exit_status == INFLIGHT_EXIT_STATUS

# No global step limit: the per-system NVT and NVE counters let the complete queue finish.
run_result = inflight.run(batch=None)
assert run_result is None
assert inflight_sampler.exhausted
assert inflight.done
inflight_progress.advance(message="all queued systems reached HostMemory")

completed = inflight_sink.drain()
system_ids = completed.system_id.reshape(-1).to(torch.long)
unique_ids, counts = torch.unique(system_ids, sorted=True, return_counts=True)
expected_ids = torch.arange(INFLIGHT_SYSTEMS, dtype=torch.long)
assert completed.num_graphs == INFLIGHT_SYSTEMS
assert torch.equal(unique_ids, expected_ids)
assert torch.equal(counts, torch.ones_like(counts))
assert torch.all(completed.status.reshape(-1) == inflight.exit_status)
assert torch.all(completed.nvt_steps_done == INFLIGHT_NVT_STEPS)
assert torch.all(completed.nve_steps_done == INFLIGHT_NVE_STEPS)
assert len(inflight_sink) == 0

assert INFLIGHT_SYSTEMS % INFLIGHT_ACTIVE_SYSTEMS == 0
expected_waves = INFLIGHT_SYSTEMS // INFLIGHT_ACTIVE_SYSTEMS
expected_fused_steps = expected_waves * (
    INFLIGHT_NVT_STEPS + INFLIGHT_NVE_STEPS
)
assert inflight.step_count == expected_fused_steps
inflight_progress.complete(
    f"{INFLIGHT_SYSTEMS:,} distinct system IDs returned exactly once"
)

inflight_summary = pd.Series({
    "queued systems": INFLIGHT_SYSTEMS,
    "maximum active systems": INFLIGHT_ACTIVE_SYSTEMS,
    "fused steps": inflight.step_count,
    "completed systems": completed.num_graphs,
    "unique system IDs": unique_ids.numel(),
    "duplicate system IDs": int((counts > 1).sum()),
    "NVT updates per system": int(completed.nvt_steps_done[0].item()),
    "NVE updates per system": int(completed.nve_steps_done[0].item()),
    "final status": int(completed.status[0].item()),
}, name="Value").rename_axis("Field").reset_index(),
    label="Inflight batching result",
    show_index=False,
))
display(callout(
    "Every queued dimer reached NVT, then NVE, and appeared exactly once in the CPU result batch.",
    kind="result",
    result_state="pass",
))
inflight_dataset.close()
del inflight_source, inflight_dataset, completed
""",
        ),
        code(
            "run-dynamics",
            """
run_progress = NotebookProgress(
    title="Run the full NVT → NVE trajectory",
    total=TOTAL_DYNAMICS_STEPS,
    unit="steps",
)
dynamics.register_hook(
    NotebookStageProgressHook(run_progress, frequency=1_000, label="NVT + NVE")
)

# This is the complete production run: 5,000 NVT steps, then 20,000 NVE steps.
torch.cuda.synchronize()
started = perf_counter()
final_batch = dynamics.run(batch)
torch.cuda.synchronize()
elapsed_s = perf_counter() - started

trajectory = ir_hook.result()
stage_counts = ir_hook.stage_counts
run_progress.complete(
    f"{TOTAL_DYNAMICS_STEPS:,} updates complete in {elapsed_s / 60:.2f} min"
)
display(readable_table(
    pd.DataFrame([
        ("Wall time / min", elapsed_s / 60.0),
        ("Dipole array", tuple(trajectory.dipoles_e_angstrom.shape)),
        ("Position array", tuple(trajectory.positions_angstrom.shape)),
        (
            "NVT updates recorded",
            stage_counts[f"status_{IR_WARMUP_STATUS}_warmup_steps"],
        ),
        (
            "NVE updates recorded",
            stage_counts[f"status_{IR_PRODUCTION_STATUS}_production_steps"],
        ),
    ], columns=["Trajectory result", "Value"]),
    label="Completed dynamics run",
    show_index=False,
))
""",
        ),
        code(
            "validate-dynamics-run",
            """
run_validation_progress = NotebookProgress(
    title="Check the completed trajectory", total=1, unit="check set"
)
assert final_batch is not None
assert dynamics.step_count == TOTAL_DYNAMICS_STEPS
assert torch.all(final_batch.status.reshape(-1) == dynamics.exit_status)
assert torch.all(final_batch.nvt_steps_done == WARMUP_STEPS)
assert torch.all(final_batch.nve_steps_done == PRODUCTION_STEPS)
assert trajectory.dipoles_e_angstrom.shape == (PRODUCTION_STEPS, 4, 3)
assert trajectory.positions_angstrom.shape == (PRODUCTION_STEPS, 42, 3)
assert stage_counts == {
    f"status_{IR_WARMUP_STATUS}_warmup_steps": WARMUP_STEPS,
    f"status_{IR_PRODUCTION_STATUS}_production_steps": PRODUCTION_STEPS,
}
run_validation_progress.complete("length, routing, and array shapes verified")
""",
            source_hidden=True,
        ),
        code(
            "persist-trajectory",
            """
persist_progress = NotebookProgress(
    title="Save the raw trajectory before analysis", total=2, unit="groups"
)
trajectory_path = OUTPUT_DIR / "water_ir_trajectory.npz"
trajectory_manifest = save_ir_trajectory(trajectory_path, trajectory, labels)
persist_progress.advance(message="all production arrays saved with a checksum")
structure_manifest = write_structure_artifacts(
    OUTPUT_DIR,
    seed_batch=unrelaxed_batch,
    relaxed_batch=relaxed_batch,
    trajectory=trajectory,
    graph_index=2,
    graph_label="(H2O)6",
    stride=100,
)
persist_progress.complete("seed, relaxed, and sampled structures saved")
display(readable_table(
    pd.DataFrame([
        ("File", trajectory_manifest["file"]),
        ("Frames", trajectory_manifest["frames"]),
        ("Systems", trajectory_manifest["graphs"]),
        ("Atoms", trajectory_manifest["atoms"]),
        ("Time step / fs", trajectory_manifest["dt_fs"]),
        ("SHA-256", trajectory_manifest["sha256"][:16] + "…"),
    ], columns=["Field", "Value"]),
    label="Saved raw trajectory",
    show_index=False,
))
display(callout(
    "Saved all 20,000 production frames and the seed, relaxed, and sampled "
    "trajectory structures before starting spectral analysis.",
    kind="result",
    result_state="pass",
))
""",
        ),
        code(
            "analysis-restart",
            """
restart_progress = NotebookProgress(
    title="Reload the trajectory for analysis", total=1, unit="checksum"
)
# Analysis restart: reload the complete trajectory and verify its checksum.
# This supports post-processing; it does not resume the integrator bit-for-bit.
trajectory_sha_before = sha256_file(trajectory_path)
trajectory, reloaded_labels = load_ir_trajectory(trajectory_path)
trajectory_sha_after = sha256_file(trajectory_path)
assert reloaded_labels == labels
assert trajectory_sha_after == trajectory_sha_before == trajectory_manifest["sha256"]
restart_progress.complete("reloaded arrays match the saved SHA-256")

display(callout(
    f"Analysis restart verified: {trajectory.positions_angstrom.shape[0]:,} "
    f"complete frames, SHA-256 {trajectory_sha_after[:16]}…",
    kind="result",
    result_state="pass",
))
""",
        ),
        stage_markdown(
            "stage-7",
            stage=7,
            title="Keep batches moving",
            outcome="Refill one GPU from a larger dataset, then inspect how the same stages would be arranged in a two-rank DistributedPipeline.",
            before="Predict which limit applies to the live batch and which applies to the full dataset: max_atoms, max_batch_size, or dataset length.",
            compute_time=(
                "30 s on one H100 PCIe in the checked run"
            ),
            body=r"""
- `FusedStage` shares one model call across stages on one GPU.
- Inflight batching replaces finished systems inside a bounded active batch.
- `DistributedPipeline` places different stages on different ranks so they can work on different batches at the same time.
- Attendees run the single-GPU path. The multi-GPU construction and any H100 results are loaded as an offline lesson.
""",
        ),
        code(
            "diagnostics",
            """
diagnostics_progress = NotebookProgress(
    title="Check the production trajectory", total=2, unit="steps"
)
KB_EV_K = 8.617333262145e-5
production_diagnostics = analyze_production_trajectory(
    trajectory,
    labels=labels,
    cluster_graph_indices=(2, 3),
    boltzmann_constant_eV_per_K=KB_EV_K,
    energy_spacing_dtype=np.float32,
    oxygen_connectivity_cutoff_angstrom=OXYGEN_CONNECTIVITY_CUTOFF_A,
    covalent_oh_cutoff_angstrom=COVALENT_OH_CUTOFF_A,
    h_acceptor_cutoff_angstrom=HBOND_H_ACCEPTOR_CUTOFF_A,
    oo_cutoff_angstrom=HBOND_OO_CUTOFF_A,
    hbond_angle_cutoff_deg=HBOND_ANGLE_CUTOFF_DEG,
    energy_excursion_advisory_meV_per_atom=(
        ENERGY_EXCURSION_ADVISORY_MEV_PER_ATOM
    ),
)
cluster_topology_by_graph = {
    item.graph_index: item for item in production_diagnostics.cluster_topologies
}

diagnostic_table = production_diagnostics.diagnostic_table
integrity_table = production_diagnostics.integrity_table
diagnostic_display = build_production_diagnostics_display_tables(
    production_diagnostics
)
nve_temperature = production_diagnostics.nve_temperature_3n_K
energy_drift = production_diagnostics.energy_drift_meV_atom_ps
energy_excursion = production_diagnostics.max_energy_excursion_meV_atom
energy_spacing_mev_atom = production_diagnostics.energy_spacing_meV_atom
cluster_intact = production_diagnostics.cluster_intact
cluster_dft_comparison_valid = (
    production_diagnostics.cluster_dft_comparison_valid
)
energy_within_advisory = production_diagnostics.energy_within_advisory
diagnostics_progress.advance(
    message="temperature, energy, charge, and topology calculated"
)

display(readable_table(
    diagnostic_display.diagnostics.round(5),
    label="Production trajectory diagnostics",
    show_index=False,
))
display(readable_table(
    diagnostic_display.integrity.round(5),
    label="Production trajectory integrity checks",
    show_index=False,
))

display(callout(
    "Model energies are evaluated in float32. At the energy scale in this run, "
    f"one float32 spacing is about {energy_spacing_mev_atom[0]:.2f} meV/atom "
    f"for a monomer and {energy_spacing_mev_atom[2]:.2f} meV/atom for a hexamer. "
    "Treat the fitted drift as a low-precision diagnostic; the 1 meV/atom "
    "excursion check is the more meaningful stability result.",
    kind="note",
))

if not cluster_intact:
    raise RuntimeError("Hexamer fragmented or an assigned O–H bond broke")
diagnostics_progress.complete("charge, temperature, energy, and topology checks complete")

topology_message = (
    "Both hexamers stayed connected with intact assigned O–H bonds. "
    + (
        "The initial cyclic topology also persisted in every frame."
        if cluster_dft_comparison_valid
        else "The initial ring changed, so cyclic-DFT comparisons will not be shown."
    )
    + (
        " The energy excursion stayed within the 1 meV/atom reporting line."
        if energy_within_advisory
        else " The energy excursion needs review."
    )
)
display(callout(
    topology_message,
    kind="result",
    result_state="pass" if cluster_dft_comparison_valid else "not_reported",
))
""",
        ),
        code(
            "topology-timeline",
            """
topology_progress = NotebookProgress(
    title="Build the hydrogen-bond topology timeline", total=3, unit="steps"
)
topology_timelines = {}
for graph, label in ((2, "(H2O)6"), (3, "(D2O)6")):
    topology_timelines[label] = topology_time_series(
        trajectory,
        graph,
        h_acceptor_cutoff_angstrom=HBOND_H_ACCEPTOR_CUTOFF_A,
        oo_cutoff_angstrom=HBOND_OO_CUTOFF_A,
        hbond_angle_cutoff_deg=HBOND_ANGLE_CUTOFF_DEG,
        precomputed_topology=cluster_topology_by_graph[graph],
    )
    topology_progress.advance(message=f"{label} timeline analyzed")
topology_figure, _ = plot_topology_timeline(topology_timelines)
topology_figure.savefig(
    OUTPUT_DIR / "water_ir_topology_timeline.png",
    dpi=180,
    bbox_inches="tight",
)
topology_progress.complete("two-system topology figure saved")
ring_alt_summary = "; ".join(
    f"{label}: the initial ring "
    + (
        "persists in every saved frame"
        if bool(timeline["initial_ring_present"].all())
        else "changes during production"
    )
    for label, timeline in topology_timelines.items()
)
display(figure_with_alt(
    topology_figure,
    alt_text=(
        "Two stacked 10 ps time series compare hydrogen-bond counts and binary "
        "persistence of the initial cyclic ring for cyclic (H2O)6 and cyclic "
        f"(D2O)6. {ring_alt_summary}."
    ),
))
plt.close(topology_figure)
""",
        ),
        markdown(
            "spectrum-note",
            r"""
### Predicted-charge IR proxy

The total cluster dipole is not split into molecule-by-molecule pieces: intermolecular cross-correlations belong in the spectrum. Fixed comparison windows are:

- H₂O: 2800–4000 cm⁻¹
- D₂O: 2000–3100 cm⁻¹

These are region summaries, not bond-specific stretch assignments.
""",
        ),
        code(
            "spectrum",
            """
spectrum_progress = NotebookProgress(
    title="Calculate the predicted-charge IR spectra", total=3, unit="steps"
)
OH_REGION_WINDOWS_CM1 = {"H": (2800.0, 4000.0), "D": (2000.0, 3100.0)}
spectrum_analysis = ir_spectrum_metrics(
    trajectory.dipoles_e_angstrom,
    labels,
    dt_fs=trajectory.dt_fs,
    segment_time_fs=IR_WELCH_SEGMENT_TIME_FS,
    overlap=IR_WELCH_OVERLAP_FRACTION,
    region_windows_cm1=OH_REGION_WINDOWS_CM1,
)
spectra = spectrum_analysis.spectra
metrics = spectrum_analysis.metrics
spectrum_progress.advance(message="Welch spectra calculated for four systems")

comparison_analysis = ir_comparison_table(
    metrics,
    nve_temperature,
    labels,
    pair_temperature_relative_tolerance=PAIR_TEMPERATURE_RELATIVE_TOLERANCE,
    cluster_reference_allowed=cluster_dft_comparison_valid,
)
comparisons = comparison_analysis.table
spectrum_progress.advance(message="temperature and topology checks applied")

display(readable_table(
    metrics.round(2),
    label="Predicted-charge IR region metrics",
))
display(readable_table(
    comparison_display_table(comparisons).round(3),
    label="IR comparison availability",
    missing="NOT REPORTED",
))
shown_count = int(comparisons["reported"].sum())
spectrum_progress.complete(
    f"{shown_count} of {len(comparisons)} comparisons are reportable"
)
display(callout(
    f"{shown_count} of {len(comparisons)} comparisons met the temperature and topology requirements. Other values are labeled NOT REPORTED.",
    kind="result",
    result_state="pass" if shown_count == len(comparisons) else "not_reported",
))
""",
        ),
        markdown(
            "reference-note",
            r"""
### Compare three sources without merging their meanings

- B97-3c/def2-mTZVP frequencies and dipole derivatives are checksummed.
- Selected H₂O/D₂O gas-phase fundamentals are checksummed, source-attributed position markers; no experimental intensity or copied spectrum is bundled.
- Raw DFT sticks stay visible; its smooth curve adds only the known 5 ps Hann response.
- MD and DFT are normalized independently. Experiment contributes positions only.
- The MD route uses classical nuclei and predicted charges; it has no nuclear-quantum, zero-point, or quantum-intensity correction.
- Welch variability comes from two overlapping windows of one correlated 10 ps trajectory, not independent trajectories or ensemble convergence.
- Treat this as a qualitative demonstration of broad spectral regions. Peak heights and window-to-window variability are not converged with respect to trajectory length.
- The main comparison is monomer-only. The cyclic-hexamer overlay is not shown if the original ring changes.
""",
        ),
        code(
            "load-reference",
            """
reference_progress = NotebookProgress(
    title="Prepare the checked reference comparison", total=1, unit="comparison"
)
reference_analysis = reference_comparison_metrics(
    spectra,
    references,
    labels,
    dt_fs=trajectory.dt_fs,
    segment_time_fs=IR_WELCH_SEGMENT_TIME_FS,
    region_windows_cm1=OH_REGION_WINDOWS_CM1,
    cluster_reference_allowed=cluster_dft_comparison_valid,
)
reference_comparisons = reference_analysis.comparisons
reference_metrics = reference_analysis.metrics
reference_progress.complete("MD and B97-3c summaries placed on separate scales")
display(readable_table(
    reference_metrics.round(1),
    label="MD and B97-3c frequency summaries",
))
md_isotope_message = (
    "The paired MD H/D shift met its temperature check."
    if bool(comparisons.loc["H2O_over_D2O_centroid", "reported"])
    else (
        "Inspect the H2O and D2O MD frequency regions separately; their paired "
        "MD isotope shift is not reported because the production temperatures "
        "differ too much. The DFT and experimental isotope positions remain "
        "valid context."
    )
)
display(callout(
    "The monomer panels show MD, harmonic DFT, and experiment on separate scales. "
    + md_isotope_message
    + " Do not calculate an IR MAE from this figure.",
    kind="note",
))
""",
        ),
        code(
            "mode-mapping",
            """
mode_mapping_progress = NotebookProgress(
    title="Map H2O modes to D2O modes", total=2, unit="checks"
)
mode_mapping = h_to_d_mode_mapping_table(
    references,
    coarse_mass_path_steps=H_TO_D_COARSE_MASS_PATH_STEPS,
    fine_mass_path_steps=H_TO_D_FINE_MASS_PATH_STEPS,
    degeneracy_tolerance_cm1=H_TO_D_DEGENERACY_TOLERANCE_CM1,
    covalent_oh_cutoff_angstrom=COVALENT_OH_CUTOFF_A,
    h_acceptor_cutoff_angstrom=HBOND_H_ACCEPTOR_CUTOFF_A,
    oo_cutoff_angstrom=HBOND_OO_CUTOFF_A,
    hbond_angle_cutoff_deg=HBOND_ANGLE_CUTOFF_DEG,
)
mode_mapping_progress.advance(
    message=(
        f"{H_TO_D_COARSE_MASS_PATH_STEPS}-step and "
        f"{H_TO_D_FINE_MASS_PATH_STEPS}-step mass paths evaluated"
    )
)
mode_map_table = mode_mapping.table
monomer_mode_map = mode_map_table.loc[
    mode_map_table["system"] == "monomer",
    [
        "character",
        "H_center_cm-1",
        "D_center_cm-1",
        "H_over_D_center",
        "H_IR_sum_km_mol",
        "D_IR_sum_km_mol",
        "mapping_overlap",
    ],
]
display(readable_table(
    monomer_mode_map.round({
        "H_center_cm-1": 1,
        "D_center_cm-1": 1,
        "H_over_D_center": 3,
        "H_IR_sum_km_mol": 1,
        "D_IR_sum_km_mol": 1,
        "mapping_overlap": 4,
    }),
    label="H2O to D2O monomer mode mapping",
    show_index=False,
))
print(
    "Full 12-row monomer + hexamer mapping will be saved as "
    "water_ir_h_to_d_mode_map.csv."
)
mode_mapping_progress.complete("monomer table shown; full mapping ready to save")
display(callout(
    "The table above reports the three monomer assignments. The complete monomer and hexamer mapping is saved to water_ir_h_to_d_mode_map.csv. Agreement between the 65- and 129-step mass paths does not override the MD topology check, so the hexamer overlay is still omitted if its ring changed.",
    kind="note",
))
""",
        ),
        code(
            "plot",
            """
ir_plot_progress = NotebookProgress(
    title="Render MD, harmonic DFT, and observed positions",
    total=2,
    unit="steps",
)
ir_figure, _ = plot_monomer_ir_comparison(
    reference_comparisons,
    experimental_fundamentals,
    harmonic_mode_indices=harmonic_mode_indices,
    wavenumber_limits_cm1=(500.0, 4200.0),
)
ir_plot_progress.advance(message="MD, B97-3c, and observed positions laid out")
ir_figure.savefig(
    OUTPUT_DIR / "water_ir_dft_mapping.png",
    dpi=180,
    bbox_inches="tight",
)
display(figure_with_alt(
    ir_figure,
    alt_text=(
        "Side-by-side H2O and D2O IR panels with separate rows for "
        "finite-temperature AIMNet2 predicted-charge molecular dynamics, B97-3c "
        "harmonic sticks and 5 ps Hann responses, and observed gas-phase "
        "fundamental positions. MD and DFT are independently normalized; "
        "experiment supplies positions only. The H2O and D2O panels occupy "
        "different frequency regions; a quantitative MD isotope comparison is "
        "reported only when its temperature check passes."
    ),
))
ir_plot_progress.complete("final comparison figure saved and displayed")
plt.close(ir_figure)
""",
        ),
        code(
            "save",
            """
save_progress = NotebookProgress(
    title="Save validated outputs", total=3, unit="groups"
)
water_run_results = WaterRunResults(
    diagnostics=diagnostic_table,
    spectrum_metrics=metrics,
    topology_summary=integrity_table,
    comparisons=comparisons,
    dft_comparison=reference_metrics,
    h_to_d_mode_map=mode_map_table,
    harmonic_displacements=harmonic_fd_table,
    harmonic_convergence=harmonic_convergence_table,
    harmonic_checks=harmonic_validation_table,
    harmonic_comparison=harmonic_comparison_table,
    nci_interaction_curves=nci_curves,
    nci_interaction_metrics=nci_metrics,
    nci_ensemble_curves=nci_member_curves,
    dimer_ablation=dimer_table,
    dimer_ablation_mae=ablation_mae,
    adsorption_results=adsorption_results,
    adsorption_forces=adsorption_forces,
    sevennet_graph_mapping=sevennet_graph_mapping,
    sevennet_numerical_agreement=sevennet_numerical_agreement,
    first_warm_calls=cold_warm,
    cpu_gpu_crossover=crossover,
    batch_layout_timings=layout_result["timings"],
    topology_timelines=topology_timelines,
    spectra=spectra,
)
save_progress.advance(message="calculated result tables collected")

saved_run = save_water_run_outputs(
    OUTPUT_DIR,
    results=water_run_results,
    run_details={
        "run_id": RUN_ID,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(DEVICE),
        "torch": torch.__version__,
        "aimnet": metadata.version("aimnet"),
        "sevennet": metadata.version("sevenn"),
        "toolkit_core_commit": installed_pins["Core"],
        "toolkit_ops_commit": installed_pins["Ops"],
        "checkpoint_source": MODEL_CHECKPOINT,
        "checkpoint_sha256": model_card["checkpoint_sha256"],
        "nci_checkpoints": NCI_CHECKPOINTS,
        "aimnet_checkpoint_identities": aimnet_checkpoint_identities,
        "nci_subset_sha256": sha256_file(NCI_DATA_FILE),
        "sevennet_checkpoint_source": SEVENNET_CHECKPOINT_URL,
        "sevennet_checkpoint_sha256": sevennet_checkpoint_sha256,
        "sevennet_checkpoint_doi": SEVENNET_CHECKPOINT_DOI,
        "sevennet_task": SEVENNET_MODALITY,
        "sevennet_reference_method": SEVENNET_REFERENCE_METHOD,
        "adsorption_structure_manifest_sha256": sha256_file(
            DEFAULT_DATA_DIR / "manifest.json"
        ),
        "d3_parameter_file_sha256": D3_PARAMETER_SHA256,
        "checkpoint_override": checkpoint_is_override,
        "notebook_sha256": sha256_file(PART_DIR / "alchemi-water-ir.ipynb"),
        "dimer_reference_manifest_sha256": sha256_file(
            REFERENCE_ROOT / "water_dimer_b97_3c" / "manifest.json"
        ),
        "harmonic_reference_sha256s": {
            label: sha256_file(REFERENCE_ROOT / directory / "manifest.json")
            for label, directory in reference_dirs.items()
        },
        "aimnet_harmonic_archive": {
            "path": harmonic_archive_path.name,
            "sha256": harmonic_archive_sha256,
        },
        "experimental_reference_bundle": {
            "artifact_id": experimental_artifact_id,
            "manifest_sha256": sha256_file(
                PART_DIR
                / "reference"
                / "experimental_water_fundamentals"
                / "manifest.json"
            ),
            "data_sha256": experimental_data_sha256,
            "checksum_index_sha256": sha256_file(
                PART_DIR
                / "reference"
                / "experimental_water_fundamentals"
                / "SHA256SUMS"
            ),
        },
        "pipeline_campaign_bundle": None,
    },
    settings={
        "model": "AIMNet checkpoint base + predicted-charge all-pairs Coulomb + pairwise D3(BJ)",
        "nci_graphs": len(nci_graph_index),
        "nci_interaction_geometries": len(nci_curves),
        "nci_reference_levels": [
            "ωB97M-D3(BJ)/def2-TZVPPD",
            "CCSD(T)/CBS interaction energies",
        ],
        "nci_validation": NCI_VALIDATION.as_record(),
        "custom_adapter_model": SEVENNET_MODEL_NAME,
        "custom_adapter_task": SEVENNET_MODALITY,
        "custom_adapter_scope": (
            "fixed-geometry 2D-periodic Cu(111) and finite molecular "
            "energy/force single points"
        ),
        "custom_adapter_precision": "float32",
        "custom_adapter_compile": False,
        "custom_adapter_energy_repeat_tolerance_eV_per_atom": (
            SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM
        ),
        "custom_adapter_force_repeat_tolerance_eV_A": (
            SEVENNET_REPEAT_FORCE_TOL_EV_A
        ),
        "custom_adapter_geometry_status": (
            "ASE-generated initial placements; not model-relaxed"
        ),
        "surface_d3_cutoff_A": SURFACE_D3_CUTOFF_A,
        "surface_d3_cutoff_bohr": D3_REFERENCE_CUTOFF_BOHR,
        "surface_d3_smoothing_fraction": (
            D3_REFERENCE_SMOOTHING_FRACTION
        ),
        "surface_d3_parameters": {
            "a1": PBE_D3_BJ_A1,
            "a2_bohr": PBE_D3_BJ_A2_BOHR,
            "s6": PBE_D3_BJ_S6,
            "s8": PBE_D3_BJ_S8,
        },
        "electrostatics": "simple nonperiodic all-pairs 1/r; no cutoff",
        "d3_cutoff_A": D3_CUTOFF_A,
        "d3_parameters": d3_params,
        "compile_mode": "default Torch compile on the fixed 42-atom IR batch",
        "residual_serial_batch_tolerance_eV": (
            RESIDUAL_SERIAL_BATCH_TOLERANCE_EV
        ),
        "full_serial_batch_tolerance_eV": FULL_SERIAL_BATCH_TOLERANCE_EV,
        "component_closure_tolerance_eV": COMPONENT_CLOSURE_TOLERANCE_EV,
        "compiled_eager_energy_tolerance_eV": (
            COMPILED_EAGER_ENERGY_TOLERANCE_EV
        ),
        "compiled_eager_force_tolerance_eV_A": (
            COMPILED_EAGER_FORCE_TOLERANCE_EV_A
        ),
        "compiled_eager_charge_tolerance_e": (
            COMPILED_EAGER_CHARGE_TOLERANCE_E
        ),
        "compiled_repeat_energy_tolerance_eV": (
            COMPILED_REPEAT_ENERGY_TOLERANCE_EV
        ),
        "compiled_repeat_force_tolerance_eV_A": (
            COMPILED_REPEAT_FORCE_TOLERANCE_EV_A
        ),
        "compiled_repeat_charge_tolerance_e": (
            COMPILED_REPEAT_CHARGE_TOLERANCE_E
        ),
        "neighbor_skin_A": NEIGHBOR_SKIN_A,
        "fire_initial_dt": IR_FIRE_INITIAL_DT,
        "temperature_K": TEMPERATURE_K,
        "dt_fs": DT_FS,
        "warmup_steps": WARMUP_STEPS,
        "production_steps": PRODUCTION_STEPS,
        "warmup_status": IR_WARMUP_STATUS,
        "production_status": IR_PRODUCTION_STATUS,
        "initial_velocity_random_seeds": IR_INITIAL_VELOCITY_RANDOM_SEEDS,
        "nvt_friction_per_fs": IR_NVT_FRICTION_PER_FS,
        "nvt_random_seed": IR_NVT_RANDOM_SEED,
        "capture_charge_tolerance_e": IR_CAPTURE_CHARGE_TOLERANCE_E,
        "charge_neutrality_tolerance_e": IR_CHARGE_NEUTRALITY_TOLERANCE_E,
        "dipole_origin_tolerance_e_A": (
            IR_DIPOLE_ORIGIN_TOLERANCE_E_ANGSTROM
        ),
        "mass_only_position_rtol": MASS_ONLY_POSITION_RTOL,
        "mass_only_position_atol_A": MASS_ONLY_POSITION_ATOL_A,
        "mass_only_energy_tolerance_eV": MASS_ONLY_ENERGY_TOLERANCE_EV,
        "mass_only_force_tolerance_eV_A": MASS_ONLY_FORCE_TOLERANCE_EV_A,
        "mass_only_charge_tolerance_e": MASS_ONLY_CHARGE_TOLERANCE_E,
        "spectrum_segment_time_fs": IR_WELCH_SEGMENT_TIME_FS,
        "spectrum_overlap": IR_WELCH_OVERLAP_FRACTION,
        "spectrum_windows_cm1": OH_REGION_WINDOWS_CM1,
        "pair_temperature_relative_tolerance": PAIR_TEMPERATURE_RELATIVE_TOLERANCE,
        "h_to_d_coarse_mass_path_steps": H_TO_D_COARSE_MASS_PATH_STEPS,
        "h_to_d_fine_mass_path_steps": H_TO_D_FINE_MASS_PATH_STEPS,
        "h_to_d_degeneracy_tolerance_cm1": (
            H_TO_D_DEGENERACY_TOLERANCE_CM1
        ),
        "oxygen_connectivity_cutoff_A": OXYGEN_CONNECTIVITY_CUTOFF_A,
        "covalent_OH_cutoff_A": COVALENT_OH_CUTOFF_A,
        "hbond_H_acceptor_cutoff_A": HBOND_H_ACCEPTOR_CUTOFF_A,
        "hbond_OO_cutoff_A": HBOND_OO_CUTOFF_A,
        "hbond_angle_cutoff_deg": HBOND_ANGLE_CUTOFF_DEG,
        "energy_excursion_advisory_meV_atom": ENERGY_EXCURSION_ADVISORY_MEV_PER_ATOM,
        "harmonic_fmax_eV_A": HARMONIC_FMAX_EV_A,
        "harmonic_fire_initial_dt": HARMONIC_FIRE_INITIAL_DT,
        "harmonic_displacement_steps_bohr": (
            HARMONIC_DISPLACEMENT_STEPS_BOHR.tolist()
        ),
        "harmonic_selected_step_bohr": HARMONIC_SELECTED_STEP_BOHR,
        "harmonic_frequency_step_tolerance_cm1": (
            HARMONIC_FREQUENCY_STEP_TOLERANCE_CM1
        ),
        "harmonic_intensity_step_abs_tolerance_km_mol": (
            HARMONIC_INTENSITY_STEP_ABS_TOLERANCE_KM_MOL
        ),
        "harmonic_intensity_step_relative_tolerance": (
            HARMONIC_INTENSITY_STEP_REL_TOLERANCE
        ),
        "harmonic_mode_overlap_min": HARMONIC_MODE_OVERLAP_MIN,
        "harmonic_hessian_antisymmetry_relative_max": (
            HARMONIC_HESSIAN_ANTISYMMETRY_REL_MAX
        ),
        "harmonic_charge_neutrality_tolerance_e": (
            HARMONIC_CHARGE_NEUTRALITY_TOLERANCE_E
        ),
        "harmonic_imaginary_floor_cm1": HARMONIC_IMAGINARY_FLOOR_CM1,
    },
    checks={
        "residual_serial_batch_max_abs_eV": serial_batch_error,
        "full_serial_batch_max_abs_eV": full_pipeline_agreement_error,
        "nci_complete_max_MAE_vs_DFT_D3_kcal_mol": float(
            nci_metrics["complete vs DFT-D3"].max()
        ),
        "nci_complete_max_MAE_vs_CCSD_T_CBS_kcal_mol": float(
            nci_metrics["complete vs CC"].max()
        ),
        "nci_force_check": nci_force_check_record(
            nci_force_check, NCI_VALIDATION
        ),
        "component_closure_max_abs_eV": component_closure_error,
        "official_calculator_agreement": agreement_errors,
        "analytic_coulomb": analytic_coulomb_errors,
        "sevennet_adapter": {
            "graph_mapping_passed": sevennet_graph_mapping_passed,
            "structures": int(len(adsorption_structures)),
            "batches": 2,
            "finite_outputs": True,
            "numerical_max_abs_energy_eV_per_atom": (
                sevennet_repeat_max_energy_difference_eV_per_atom
            ),
            "numerical_max_abs_forces_eV_A": (
                sevennet_repeat_max_force_difference_eV_A
            ),
            "max_combined_fmax_eV_A": sevennet_max_force_eV_A,
            "geometry_status": (
                "ASE-generated initial placements; not model-relaxed"
            ),
            "molecules": list(ADSORBATES),
            "periodic_pbc": [True, True, False],
        },
        "compiled_ir_eager_agreement": compiled_ir_eager_agreement,
        "compiled_ir_repeat_agreement": compiled_ir_repeat_agreement,
        "finite_difference_force_energy_route": COMPOSITION_FD_ENERGY_ROUTE,
        "finite_difference_force_step_A": FD_STEP_A,
        "finite_difference_force_reference_eV_A": fd_force,
        "finite_difference_force_official_analytic_eV_A": official_force,
        "finite_difference_force_official_abs_error_eV_A": (
            official_fd_force_error
        ),
        "finite_difference_force_pipeline_eV_A": model_force,
        "finite_difference_force_pipeline_abs_error_eV_A": fd_force_error,
        "cluster_integrity_passed": cluster_intact,
        "initial_ring_persisted_all_frames": cluster_dft_comparison_valid,
        "energy_excursion_within_advisory": energy_within_advisory,
        "reported_comparisons": comparisons["reported"].to_dict(),
        "fused_stage_route_counts": stage_counts,
        "harmonic_checks": harmonic_validation,
        "harmonic_comparison_reported": harmonic_comparison_reported,
        "harmonic_final_fmax_eV_A": harmonic_fmax_eV_A,
        "harmonic_frequency_MAE_vs_B97_3c_cm1": harmonic_frequency_mae_cm1,
        "harmonic_selected_Hessian_antisymmetry_relative": (
            selected_harmonic_estimate.hessian.max_relative_antisymmetry
        ),
        "harmonic_final_frequency_step_change_cm1": {
            "H2O": float(
                harmonic_h_convergence.frequency_max_abs_change_cm1[-1]
            ),
            "D2O": float(
                harmonic_d_convergence.frequency_max_abs_change_cm1[-1]
            ),
        },
        "harmonic_final_intensity_step_change_km_mol": {
            "H2O": float(
                harmonic_h_convergence.ir_intensity_max_abs_change_km_mol[-1]
            ),
            "D2O": float(
                harmonic_d_convergence.ir_intensity_max_abs_change_km_mol[-1]
            ),
        },
    },
)
run_manifest = saved_run.manifest
spectrum_table = saved_run.spectrum_table
save_progress.advance(message="tables, timelines, and spectrum arrays saved")
save_progress.complete("run summary and file list saved")

print("saved:")
for path in sorted(
    path
    for path in OUTPUT_DIR.iterdir()
    if path.name.startswith("water_")
):
    print(" -", path.relative_to(ROOT))
display(callout(
    f"Saved the adapter checks, raw trajectory, Toolkit Zarr data, timings, spectra, and run summary to {OUTPUT_DIR.relative_to(ROOT)}.",
    kind="result",
    result_state="pass",
))
""",
            source_hidden=True,
        ),
        markdown(
            "try-it-note",
            r"""
### Try it: one geometry you choose

Change one O–O distance, then reuse the same public Toolkit path. This reports the composed model only; no DFT value is invented for a geometry that was not computed in the reference set.
"""
            + "\n\n"
            + callout_html(
                "Edit TRY_OO_DISTANCE_A, predict whether the interaction becomes more or less attractive, then run the cell.",
                kind="before",
            ),
        ),
        code(
            "try-it",
            """
TRY_OO_DISTANCE_A = 3.30
trial_progress = NotebookProgress(
    title="Editable composed-model call", total=1, unit="geometry"
)
trial_dimer = make_water_dimer(TRY_OO_DISTANCE_A)
trial_graphs = [trial_dimer, trial_dimer[:3], trial_dimer[3:]]
for atoms in trial_graphs:
    atoms.info["charge"] = 0

trial_data = [
    AtomicData.from_atoms(atoms, device=DEVICE, dtype=torch.float64)
    for atoms in trial_graphs
]
trial_batch = Batch.from_data_list(trial_data, device=DEVICE)
compute_neighbors(trial_batch, config=eager_model.model_config.neighbor_config)
trial_progress.update(
    done=0,
    message="evaluating with the eager model; no compilation is triggered",
)
trial_outputs = eager_model(trial_batch)
trial_energy = trial_outputs["energy"].detach().cpu().reshape(-1).numpy()
trial_interaction_kJ_mol = (
    trial_energy[0] - trial_energy[1] - trial_energy[2]
) * 96.48533212331002
trial_progress.complete("editable AB/A/B interaction evaluated")

display(callout(
    f"Observed composed-model interaction at {TRY_OO_DISTANCE_A:.2f} Å: "
    f"{trial_interaction_kJ_mol:.2f} kJ/mol.",
    kind="result",
    result_state="observed",
))
""",
        ),
        markdown(
            "results-summary-note",
            r"""
### Results summary

This table collects the main numerical checks and measurements made above. It
updates when you rerun the notebook.

The live one-GPU `DomainParallel` row records the API call made in this
notebook; one GPU means one domain, so it does not claim spatial decomposition.
The same 51,200-atom input on 1, 2, and 4 H100s is **RECORDED** only after
loading the checked H100 result set; otherwise it remains **NOT REPORTED**.
Each GPU count has one warm-up and three measured energy/force passes, not a
trajectory or a capacity search. `DistributedPipeline` correctness, overlap,
and timing are **NOT REPORTED** for this release.
""",
        ),
        code(
            "results-summary",
            """
summary_progress = NotebookProgress(
    title="Build the results summary", total=3, unit="steps"
)

batch_results_match = bool(
    serial_batch_error < RESIDUAL_SERIAL_BATCH_TOLERANCE_EV
    and full_pipeline_agreement_error < FULL_SERIAL_BATCH_TOLERANCE_EV
)
cpu_gpu_throughput = crossover.pivot(
    index="batch_size", columns="route", values="structures_per_s"
).sort_index()
gpu_faster_batches = cpu_gpu_throughput.index[
    cpu_gpu_throughput["GPU"] > cpu_gpu_throughput["CPU"]
]
cpu_gpu_crossover_batch_size = (
    int(gpu_faster_batches[0]) if len(gpu_faster_batches) else None
)
cpu_gpu_largest_batch_size = int(cpu_gpu_throughput.index[-1])
cpu_gpu_largest_batch_speedup = float(
    cpu_gpu_throughput.loc[cpu_gpu_largest_batch_size, "GPU"]
    / cpu_gpu_throughput.loc[cpu_gpu_largest_batch_size, "CPU"]
)
cpu_gpu_max_energy_difference_eV = float(
    crossover["max_abs_energy_difference"].max()
)
summary_progress.advance(message="Read batching and CPU/GPU checks")

campaign_successes = 0
campaign_failures = 0
monomer_shown = bool(
    comparisons.loc["H2O_over_D2O_centroid", "reported"]
)
cluster_rows = comparisons.index != "H2O_over_D2O_centroid"
cluster_shown = bool(comparisons.loc[cluster_rows, "reported"].all())
harmonic_failed_checks = tuple(
    name for name, passed in harmonic_validation.items() if not passed
)
cluster_not_reported_reasons = comparisons.loc[
    cluster_rows & ~comparisons["reported"], "status"
].tolist()
sevennet_status = "PASS"  # Earlier mapping or numerical failures stop the run.
campaign_unavailable_reason = DISTRIBUTED_PIPELINE_NOT_REPORTED_REASON
summary_progress.advance(message="Read campaign and spectral checks")

results_summary, not_reported_count = build_results_summary(
    batch_results_match=batch_results_match,
    serial_batch_error_eV=serial_batch_error,
    full_pipeline_agreement_error_eV=full_pipeline_agreement_error,
    cpu_gpu_crossover_batch_size=cpu_gpu_crossover_batch_size,
    cpu_gpu_largest_batch_size=cpu_gpu_largest_batch_size,
    cpu_gpu_largest_batch_speedup=cpu_gpu_largest_batch_speedup,
    cpu_gpu_max_energy_difference_eV=cpu_gpu_max_energy_difference_eV,
    sevennet_status=sevennet_status,
    sevennet_structure_count=len(adsorption_structures),
    sevennet_batch_count=2,
    sevennet_molecule_count=len(ADSORBATES),
    sevennet_max_edge_vector_mapping_difference_A=(
        sevennet_max_edge_vector_mapping_difference_A
    ),
    sevennet_repeat_max_energy_difference_eV_per_atom=(
        sevennet_repeat_max_energy_difference_eV_per_atom
    ),
    sevennet_repeat_max_force_difference_eV_A=(
        sevennet_repeat_max_force_difference_eV_A
    ),
    sevennet_max_force_eV_A=sevennet_max_force_eV_A,
    nci_geometry_count=len(nci_curves),
    nci_graph_count=len(nci_graph_index),
    nci_max_mae_vs_dft_d3_kcal_mol=float(
        nci_metrics["complete vs DFT-D3"].max()
    ),
    nci_max_mae_vs_ccsd_t_cbs_kcal_mol=float(
        nci_metrics["complete vs CC"].max()
    ),
    harmonic_comparison_reported=harmonic_comparison_reported,
    harmonic_frequency_mae_cm1=harmonic_frequency_mae_cm1,
    harmonic_failed_checks=harmonic_failed_checks,
    inflight_queue_complete=bool(
        inflight_sampler.exhausted
        and inflight.done
        and unique_ids.numel() == INFLIGHT_SYSTEMS
        and not (counts > 1).any().item()
        and inflight_nvt_counts_correct
        and inflight_nve_counts_correct
    ),
    inflight_system_count=INFLIGHT_SYSTEMS,
    inflight_active_system_count=INFLIGHT_ACTIVE_SYSTEMS,
    inflight_nvt_steps=INFLIGHT_NVT_STEPS,
    inflight_nve_steps=INFLIGHT_NVE_STEPS,
    domain_live_api_passed=domain_live_api_passed,
    domain_live_world_size=1,
    domain_live_spatially_decomposed=False,
    domain_live_atom_count=int(domain_result.num_nodes),
    domain_live_energy_per_atom_eV=domain_energy_ev / domain_result.num_nodes,
    domain_live_max_force_eV_A=domain_fmax_ev_a,
    domain_live_charge_sum_e=domain_charge_sum,
    domain_results_available=domain_view.available,
    domain_results_unavailable_reason=(
        None if domain_view.available else domain_view.reason
    ),
    domain_successful_cases=domain_view.successful_case_count,
    domain_failed_cases=domain_view.failed_case_count,
    domain_planned_max_atom_count=DOMAIN_FIXED_ATOM_COUNT,
    domain_measured_max_atom_count=domain_view.measured_max_atom_count,
    campaign_available=False,
    campaign_unavailable_reason=campaign_unavailable_reason,
    campaign_successes=campaign_successes,
    campaign_failures=campaign_failures,
    campaign_systems_total=PLANNED_CAMPAIGN_SYSTEMS_TOTAL,
    monomer_shown=monomer_shown,
    monomer_status=comparisons.loc["H2O_over_D2O_centroid", "status"],
    cluster_shown=cluster_shown,
    cluster_not_reported_reasons=cluster_not_reported_reasons,
)
summary_progress.complete("Summary calculated from the results above")
""",
            source_hidden=True,
        ),
        code(
            "display-results-summary",
            """
summary_display_progress = NotebookProgress(
    title="Display the results summary", total=1, unit="table"
)
display(readable_table(
    results_summary,
    label="Part 1 results summary",
    show_index=False,
    missing="NOT REPORTED",
))

display(callout(
    f"{not_reported_count} result row(s) are not reported. "
    "Read the Measured column for the reason in each row.",
    kind="result",
    result_state="not_reported" if not_reported_count else "pass",
))
summary_display_progress.complete("calculated and recorded results shown")
""",
        ),
        markdown(
            "interpretation",
            r"""
### What you can now do

- Build and inspect `AtomicData` and `Batch` directly.
- Adapt a raw external model with `BaseModelMixin`, `ModelConfig`, Toolkit neighbors, and explicit input/output mappings.
- Read the tasks reported by a multitask checkpoint and reuse one loaded model for a small task sweep.
- Switch from a molecular checkpoint to a surface-capable model when the elements and physical domain change.
- Evaluate 2D-periodic slabs and finite molecular references in two batches, then form fixed-geometry adsorption energies and inspect every force output.
- Decide between serial, mixed, and homogeneous-bucket execution from measured data.
- Evaluate a 90-graph NCI set in four AIMNet, four direct-Coulomb, and one shared D3 call, then reduce AB/A/B graph energies into interaction curves.
- Compare the complete molecular model with near-matched DFT-D3 and independent CCSD(T)/CBS references without treating partial curves as separate production models.
- Keep one GPU busy with `FusedStage`, `SizeAwareSampler`, and inflight replacement.
- Inspect the intended `DistributedPipeline` layout and load prerecorded H100 timings when they are available.
- Compose an AIMNet core with explicit finite-system all-pairs Coulomb and D3 terms.
- Compare complete-model and full B97-3c double-harmonic calculations at separately optimized minima, with matched displacement size, projection, masses, and units.
- Relax and propagate four systems with one Toolkit model call per step.
- Reuse predicted charges for a total-dipole IR proxy without a second inference pass.
- Compare AIMNet MD, harmonic B97-3c, and experimental band positions without putting their intensities on a common scale.
- Keep the complete trajectory even when a temperature or topology check rules out one comparison.
"""
            + "\n\n"
            + callout_html(
                "Part 1 ends with saved output files and an editable Toolkit call. The workflow remains visible in the notebook.",
                kind="check",
            ),
        ),
        markdown(
            "references",
            r"""
## References and licenses

**Models and data**

- [AIMNet2 model cards](https://huggingface.co/isayevlab/aimnet2-2025) and
  [paper](https://doi.org/10.1039/D4SC08572H): MIT checkpoints. Metadata declares
  the external Coulomb and D3 terms; the official [Coulomb
  source](https://github.com/isayevlab/aimnetcentral/blob/main/aimnet/modules/lr.py)
  shows their composition.
- [SevenNet pretrained models](https://sevennet.readthedocs.io/en/latest/user_guide/pretrained.html),
  [paper](https://doi.org/10.1038/s41467-026-70195-8), and
  [checkpoint](https://doi.org/10.6084/m9.figshare.30399814): MIT software. The
  checkpoint is downloaded and checksum-checked, not redistributed.
- [NCI Atlas](https://github.com/Honza-R/NCIAtlas) and
  [paper](https://doi.org/10.1021/acs.jctc.9b01265): the 90-graph subset is
  CC BY 4.0 and retains source IDs.

**Scientific methods and references**

- [B97-3c](https://doi.org/10.1063/1.5012601), [particle mesh
  Ewald](https://doi.org/10.1063/1.470117), and [MD vibrational
  spectra](https://doi.org/10.1039/C3CP44302G) support the DFT, periodic, and IR
  methods.
- [Water-cluster frequencies](https://doi.org/10.1063/1.4936654) provide
  harmonic context.
- [Dinu et al., Table 1](https://doi.org/10.1021/acs.jpca.9b07221), CC BY 4.0. Toth: [stretch](https://doi.org/10.1006/jmsp.1998.7771), [bend](https://doi.org/10.1006/jmsp.1998.7611), and [D₂O](https://doi.org/10.1006/jmsp.1999.7815). Positions only.

**Software**

- NVIDIA [ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) and
  [Toolkit-Ops](https://github.com/NVIDIA/nvalchemi-toolkit-ops): Apache-2.0.
- [Packmol](https://github.com/m3g/packmol): MIT; used offline for the periodic box.
- [Psi4](https://psicode.org/): LGPL-3.0. `dftd3-python` and `mctc-gcp`: LGPL-3.0-or-later.

See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and the data READMEs.
""",
        ),
    ]

    # Keep the learner-facing order independent of where the long cell sources
    # are defined above. This places scaling only after the single-GPU fused
    # workflow has been constructed.
    def place_after(anchor_id: str, moving_ids: tuple[str, ...]) -> None:
        moving_set = set(moving_ids)
        selected = [cell for cell in cells if cell["id"] in moving_set]
        selected_by_id = {cell["id"]: cell for cell in selected}
        if set(selected_by_id) != moving_set:
            missing = sorted(moving_set - set(selected_by_id))
            raise KeyError(f"Cannot reorder missing notebook cells: {missing}")
        cells[:] = [cell for cell in cells if cell["id"] not in moving_set]
        anchor_index = next(
            index for index, cell in enumerate(cells) if cell["id"] == anchor_id
        )
        cells[anchor_index + 1 : anchor_index + 1] = [
            selected_by_id[cell_id] for cell_id in moving_ids
        ]

    def set_background(
        cell_ids: tuple[str, ...], *, hide_outputs: bool = False
    ) -> None:
        """Keep detailed checks runnable without crowding the teaching path."""

        by_id = {cell["id"]: cell for cell in cells}
        for cell_id in cell_ids:
            cell = by_id[cell_id]
            metadata = cell.setdefault("metadata", {})
            jupyter = metadata.setdefault("jupyter", {})
            jupyter["source_hidden"] = True
            if hide_outputs:
                jupyter["outputs_hidden"] = True
            tags = metadata.setdefault("tags", [])
            if "remove-input" not in tags:
                tags.append("remove-input")

    def replace_code_source(cell_id: str, text: str) -> None:
        cell = next(cell for cell in cells if cell["id"] == cell_id)
        if cell["cell_type"] != "code":
            raise TypeError(f"{cell_id} is not a code cell")
        cell["source"] = source(text)

    def append_code_source(cell_id: str, text: str) -> None:
        cell = next(cell for cell in cells if cell["id"] == cell_id)
        if cell["cell_type"] != "code":
            raise TypeError(f"{cell_id} is not a code cell")
        cell["source"].extend(source(text))

    def replace_markdown_source(cell_id: str, text: str) -> None:
        cell = next(cell for cell in cells if cell["id"] == cell_id)
        if cell["cell_type"] != "markdown":
            raise TypeError(f"{cell_id} is not a markdown cell")
        cell["source"] = source(text)

    def insert_before(anchor_id: str, cell: dict) -> None:
        index = next(
            i for i, existing in enumerate(cells) if existing["id"] == anchor_id
        )
        cells.insert(index, cell)

    def insert_after(anchor_id: str, cell: dict) -> None:
        index = next(
            i for i, existing in enumerate(cells) if existing["id"] == anchor_id
        )
        cells.insert(index + 1, cell)

    replace_code_source(
        "framework-primer-example",
        """
framework_progress = NotebookProgress(
    title="Run the PyTorch and JAX bindings", total=3, unit="checks"
)
import jax
import jax.numpy as jnp
from nvalchemiops.jax.segment_ops import segmented_sum as jax_segmented_sum
from nvalchemiops.torch.segment_ops import segmented_sum as torch_segmented_sum

jax_gpu = next((device for device in jax.devices() if device.platform == "gpu"), None)
if jax_gpu is None:
    raise RuntimeError("The Toolkit-Ops JAX binding requires a CUDA-capable JAX device.")
framework_progress.advance(message="PyTorch and JAX CUDA paths loaded")

# Four atom-level values belonging to two independent systems.
shared_values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
shared_graph_idx = np.array([0, 0, 1, 1], dtype=np.int32)
expected_totals = np.array([3.0, 7.0], dtype=np.float32)

torch_values = torch.tensor(shared_values, device=DEVICE, requires_grad=True)
torch_graph_idx = torch.tensor(shared_graph_idx, device=DEVICE)
torch_totals = torch_segmented_sum(torch_values, torch_graph_idx, num_segments=2)
(torch_gradient,) = torch.autograd.grad(torch_totals.sum(), torch_values)
framework_progress.advance(message="PyTorch returned totals and gradients")

with jax.default_device(jax_gpu):
    jax_values = jnp.asarray(shared_values)
    jax_graph_idx = jnp.asarray(shared_graph_idx)
jax_totals = jax_segmented_sum(jax_values, jax_graph_idx, num_segments=2)
jax_gradient = jax.grad(
    lambda values: jax_segmented_sum(values, jax_graph_idx, 2).sum()
)(jax_values)
jax_totals.block_until_ready()
jax_gradient.block_until_ready()
framework_progress.complete("JAX returned the same totals and gradients")
""",
    )
    insert_after(
        "framework-primer-example",
        code(
            "framework-primer-warp",
            """
warp_progress = NotebookProgress(
    title="Call the raw Warp operation", total=2, unit="checks"
)
import warp as wp
from nvalchemiops.segment_ops import segmented_sum as warp_segmented_sum

# The raw call uses typed arrays and caller-allocated output storage.
torch.cuda.synchronize(DEVICE)
warp_values = wp.from_torch(torch_values.detach(), dtype=wp.float32)
warp_graph_idx = wp.from_torch(torch_graph_idx, dtype=wp.int32)
warp_totals = wp.zeros(2, dtype=wp.float32, device=warp_values.device)
warp_segmented_sum(warp_values, warp_graph_idx, warp_totals)
wp.synchronize_device(warp_values.device)
warp_progress.advance(message="raw Warp reduction complete")

framework_table = segmented_sum_comparison_table(
    expected_totals=expected_totals,
    torch_result=torch_totals.detach().cpu().numpy(),
    jax_result=np.asarray(jax.device_get(jax_totals)),
    warp_result=warp_totals.numpy(),
    torch_gradient=torch_gradient.detach().cpu().numpy(),
    jax_gradient=np.asarray(jax.device_get(jax_gradient)),
    torch_dtype=str(torch_values.dtype), torch_device=str(torch_values.device),
    jax_dtype=str(jax_values.dtype), jax_device=str(jax_values.device),
    warp_dtype=warp_values.dtype.__name__, warp_device=str(warp_values.device),
)
display(readable_table(
    framework_table,
    label="The same reduction through two bindings and raw Warp",
    show_index=False,
))
warp_progress.complete("all three routes returned [3, 7]")
display(callout(
    "PyTorch and JAX returned the expected gradient [1, 1, 1, 1]. "
    "The raw Warp call uses explicit arrays and output storage.",
    kind="result", result_state="pass",
))
""",
        ),
    )

    replace_code_source(
        "inspect-float-precision",
        """
precision_progress = NotebookProgress(
    title="Inspect floating-point precision", total=3, unit="checks"
)
precision_summary = summarize_model_precision(
    aimnet,
    reference_energy_eV=float(hello["energy"].detach().cpu().reshape(())),
)
assert precision_summary.parameter_dtypes == ("torch.float32",)
assert precision_summary.widening_preserves_stored_values
precision_progress.advance(message="checkpoint storage inspected")

# Observe the dtype before input adaptation and after the full wrapper call.
precision_probe_data = AtomicData.from_atoms(water, device=DEVICE, dtype=torch.float64)
precision_probe_batch = Batch.from_data_list([precision_probe_data], device=DEVICE)
compute_neighbors(precision_probe_batch, config=aimnet.model_config.neighbor_config)
precision_dtype_before = precision_probe_batch.positions.dtype
precision_model_input = aimnet.adapt_input(precision_probe_batch)
precision_dtype_after_adapt = precision_probe_batch.positions.dtype
precision_probe = aimnet(precision_probe_batch)
precision_dtype_after_forward = precision_probe_batch.positions.dtype
precision_observed_dtypes = validate_precision_observation(
    hello_coordinates_dtype=hello_data.positions.dtype,
    probe_coordinates_before_dtype=precision_dtype_before,
    probe_coordinates_after_adapt_dtype=precision_dtype_after_adapt,
    model_input_coordinates_dtype=precision_model_input["coord"].dtype,
    probe_coordinates_after_forward_dtype=precision_dtype_after_forward,
    probe_output_dtypes={
        name: precision_probe[name].dtype
        for name in ("energy", "forces", "charges")
    },
)
precision_progress.advance(message="wrapper input conversion observed")

precision_table = precision_display_table(
    precision_summary,
    observed_dtypes=precision_observed_dtypes,
    matmul_precision=torch.get_float32_matmul_precision(),
)
display(readable_table(
    precision_table, label="Precision used by the first model", show_index=False
))
precision_progress.complete("storage, input dtypes, and numerical spacing shown")
display(callout(
    "Observed: the probe begins with float64 coordinates; AIMNet receives "
    "float32 coordinates and converts the batch positions in place. "
    "Energy is float64 because AIMNet preserves atomic reference-energy shifts "
    "and accumulates system energy in float64; forces and charges are float32. "
    "Check precision per tensor; spacing is numerical resolution, not model error.",
    kind="result", result_state="observed",
))
del precision_model_input, precision_probe
""",
    )

    replace_code_source(
        "cpu-gpu-crossover",
        """
benchmark_progress = NotebookProgress(
    title="Measure first and warm CPU/GPU calls", total=2, unit="checks"
)
# Fresh wrappers keep first-call setup separate from warm inference.
del aimnet
torch.cuda.empty_cache()
aimnet = AIMNet2Wrapper.from_checkpoint(checkpoint_path, device=DEVICE)
cpu_aimnet = AIMNet2Wrapper.from_checkpoint(checkpoint_path, device="cpu")
for route_model in (aimnet, cpu_aimnet):
    route_model.eval()
    for parameter in route_model.parameters():
        parameter.requires_grad_(False)
    route_model.set_config("active_outputs", {"energy"})
benchmark_progress.advance(message="fresh CPU and GPU wrappers loaded")

def benchmark_batch(atoms_sequence, device):
    data = [
        AtomicData.from_atoms(atoms, device=device, dtype=torch.float64)
        for atoms in atoms_sequence
    ]
    return Batch.from_data_list(data, device=device)

timing_atoms = [make_water_dimer(2.90) for _ in range(32)]
for atoms in timing_atoms:
    atoms.info["charge"] = 0
gpu_timing_batch = benchmark_batch(timing_atoms, DEVICE)
cpu_timing_batch = benchmark_batch(timing_atoms, "cpu")
compute_neighbors(gpu_timing_batch, config=aimnet.model_config.neighbor_config)
compute_neighbors(cpu_timing_batch, config=cpu_aimnet.model_config.neighbor_config)
cold_warm = pd.DataFrame(
    first_and_warm_call_rows(aimnet, gpu_timing_batch, warm_calls=20, route="GPU")
    + first_and_warm_call_rows(cpu_aimnet, cpu_timing_batch, warm_calls=20, route="CPU")
)
benchmark_progress.complete("first and warm 32-graph calls measured")
display(readable_table(
    cold_warm[["route", "phase", "calls", "wall_ms_per_pass", "structures_per_s"]].round(2),
    label="First and warm CPU/GPU calls", show_index=False,
))
""",
    )
    insert_after(
        "cpu-gpu-crossover",
        code(
            "cpu-gpu-sweep",
            """
crossover_progress = NotebookProgress(
    title="Measure the warm CPU/GPU crossover", total=4, unit="batch sizes"
)
benchmark_rows = []
for sweep_index, batch_size in enumerate((1, 8, 32, 128), start=1):
    atoms_set = [make_water_dimer(2.90) for _ in range(batch_size)]
    for atoms in atoms_set:
        atoms.info["charge"] = 0
    gpu_batch = benchmark_batch(atoms_set, DEVICE)
    cpu_batch = benchmark_batch(atoms_set, "cpu")
    compute_neighbors(gpu_batch, config=aimnet.model_config.neighbor_config)
    compute_neighbors(cpu_batch, config=cpu_aimnet.model_config.neighbor_config)

    gpu_energy = aimnet(gpu_batch)["energy"].detach().cpu().reshape(-1)
    cpu_energy = cpu_aimnet(cpu_batch)["energy"].detach().cpu().reshape(-1)
    cpu_gpu_error = float(torch.max(torch.abs(gpu_energy - cpu_energy)))
    assert cpu_gpu_error < 2e-4
    rows = compare_fixed_workload_devices(
        {"GPU": (aimnet, gpu_batch), "CPU": (cpu_aimnet, cpu_batch)},
        warmup_calls=2, measured_calls=20,
    )
    for row in rows:
        row["batch_size"] = batch_size
        row["cpu_gpu_max_abs_energy_ueV"] = cpu_gpu_error * 1e6
    benchmark_rows.extend(rows)
    crossover_progress.update(
        done=sweep_index, message=f"batch size {batch_size} measured"
    )

crossover = pd.DataFrame(benchmark_rows)
display(readable_table(
    crossover.sort_values(["batch_size", "route"])[[
        "batch_size", "route", "calls", "wall_ms_per_pass", "structures_per_s",
        "atoms_per_s", "cpu_gpu_max_abs_energy_ueV",
    ]].round(2),
    label="CPU/GPU warm-call crossover", show_index=False,
))
crossover_progress.complete("four matched batch sizes complete")
display(callout(
    "Small batches are latency-bound; larger batches expose GPU parallelism. "
    "These are synchronized warm model calls, not end-to-end application timing.",
    kind="result", result_state="observed",
))
del cpu_aimnet, cpu_batch, gpu_batch, cpu_timing_batch, gpu_timing_batch
""",
        ),
    )

    insert_before(
        "batch-layouts",
        code(
            "build-batch-layouts",
            """
layout_build_progress = NotebookProgress(
    title="Build one mixed batch and three size buckets", total=2, unit="steps"
)
# Fixed workload: 24 monomers, 12 dimers, 4 hexamers (216 atoms).
layout_groups = [
    [make_water_monomer() for _ in range(24)],
    [make_water_dimer(2.90) for _ in range(12)],
    [make_ir_structures()[0][2].copy() for _ in range(4)],
]
for atoms in sum(layout_groups, []):
    atoms.info["charge"] = 0

mixed_batch = Batch.from_data_list([
    AtomicData.from_atoms(atoms, device=DEVICE, dtype=torch.float64)
    for atoms in sum(layout_groups, [])
], device=DEVICE)
bucket_batches = [
    Batch.from_data_list([
        AtomicData.from_atoms(atoms, device=DEVICE, dtype=torch.float64)
        for atoms in group
    ], device=DEVICE)
    for group in layout_groups
]
layout_build_progress.advance(message="same 40 systems packed two ways")
bucket_graph_indices = [np.arange(0, 24), np.arange(24, 36), np.arange(36, 40)]
for route_batch in (mixed_batch, *bucket_batches):
    compute_neighbors(route_batch, config=aimnet.model_config.neighbor_config)
layout_build_progress.complete("model-compatible neighbors built")
""",
        ),
    )
    replace_code_source(
        "batch-layouts",
        """
layout_progress = NotebookProgress(
    title="Compare mixed and bucketed batches", total=1, unit="comparison"
)
layout_result = compare_mixed_and_bucketed(
    aimnet,
    mixed_batch,
    bucket_batches,
    bucket_graph_indices,
    warmup_passes=2,
    measured_passes=20,
    measured_repeats=BENCHMARK_REPEATS,
    atol=1e-5,
    rtol=0.0,
)
storage = neighbor_storage_table({
    "one heterogeneous batch": [mixed_batch],
    "homogeneous buckets": bucket_batches,
})
layout_progress.complete("five repeated blocks measured for both layouts")
display(readable_table(
    pd.DataFrame(layout_result["timings"])[[
        "route", "calls_per_pass", "median_structures_per_s", "relative_iqr",
    ]].rename(columns={
        "route": "Layout",
        "calls_per_pass": "Calls per workload",
        "median_structures_per_s": "Structures / s",
        "relative_iqr": "Relative IQR",
    }).round(2),
    label="Homogeneous and heterogeneous layouts · repeated timings",
    show_index=False,
))
display(readable_table(
    storage.round(3), label="Neighbor storage by batch layout", show_index=False,
))
display(callout(
    f"Both layouts return the same energies (max |Δ| = "
    f"{layout_result['max_abs_energy_difference']:.2e} eV). Which layout wins "
    "depends on the workload and should be measured.",
    kind="result", result_state="pass",
))
""",
    )

    replace_code_source(
        "compose-nci-pipeline",
        """
nci_pipeline_progress = NotebookProgress(
    title="Compose the complete model", total=3, unit="checks"
)
nci_aimnet.set_config("active_outputs", {"energy", "charges"})
nci_coulomb.set_config("active_outputs", {"energy"})
nci_d3.set_config("active_outputs", {"energy", "forces"})
nci_full_model = PipelineModelWrapper(
    groups=[
        PipelineGroup(
            steps=[PipelineStep(model=nci_aimnet), PipelineStep(model=nci_coulomb)],
            use_autograd=True,
        ),
        PipelineGroup(steps=[PipelineStep(model=nci_d3)], use_autograd=False),
    ],
    neighbor_adaptation="always",
).to(DEVICE).eval()
nci_full_model.set_config("active_outputs", {"energy", "forces"})
nci_pipeline_progress.advance(message="charge-dependent and independent groups assembled")

def nci_pipeline_outputs(atoms_sequence):
    batch = Batch.from_data_list([
        AtomicData.from_atoms(atoms, device="cpu", dtype=torch.float32)
        for atoms in atoms_sequence
    ], device=DEVICE)
    compute_neighbors(batch, config=nci_full_model.model_config.neighbor_config)
    return nci_full_model(batch), batch

nci_full_outputs, nci_full_batch = nci_pipeline_outputs(nci_atoms)
nci_member0_sum = (
    nci_member_residual_eV[0] + nci_member_coulomb_eV[0] + nci_d3_graph_eV
)
nci_pipeline_energy_cpu = nci_full_outputs["energy"].detach().reshape(-1).cpu()
nci_component_sum_max_abs_eV = float(
    (nci_pipeline_energy_cpu - nci_member0_sum).abs().max()
)
torch.testing.assert_close(
    nci_pipeline_energy_cpu, nci_member0_sum,
    atol=NCI_ENERGY_ATOL_EV, rtol=NCI_ENERGY_RTOL,
)
nci_pipeline_progress.advance(message="composed pass matches the component sum")

nci_reversed_outputs, _ = nci_pipeline_outputs(list(reversed(nci_atoms)))
nci_reversed_energy_cpu = nci_reversed_outputs["energy"].detach().reshape(-1).cpu().flip(0)
nci_graph_order_max_abs_eV = float(
    (nci_reversed_energy_cpu - nci_pipeline_energy_cpu).abs().max()
)
torch.testing.assert_close(
    nci_reversed_energy_cpu, nci_pipeline_energy_cpu,
    atol=NCI_ENERGY_ATOL_EV, rtol=NCI_ENERGY_RTOL,
)
nci_pipeline_progress.complete("graph order agrees within numerical tolerance")
""",
    )

    replace_code_source(
        "check-nci-force",
        """
nci_force_progress = NotebookProgress(
    title="Check one complete-model force", total=3, unit="checks"
)
nci_example_index = nci_graph_index.index[
    (nci_graph_index["system_id"] == DOMAIN_METHODOLOGY.nci_system_id)
    & np.isclose(nci_graph_index["scale"], 1.0)
    & (nci_graph_index["fragment"] == "AB")
].item()
nci_example = nci_atoms[nci_example_index]
nci_example_output, _ = nci_pipeline_outputs([nci_example])
nci_example_force = nci_example_output["forces"].detach()
nci_force_progress.advance(message="Toolkit complete-model force evaluated")

# Independent AIMNet2 route with the same Coulomb and D3 settings.
nci_official = AIMNet2Calculator(
    str(resolve_checkpoint_path(NCI_CHECKPOINTS[0])),
    device=str(DEVICE), needs_coulomb=True, needs_dispersion=True,
    compile_model=False, train=False,
)
nci_official.set_lrcoulomb_method("simple")
nci_official.set_dftd3_cutoff(
    cutoff=D3_CUTOFF_A, smoothing_fraction=NCI_D3_SMOOTHING_FRACTION
)
nci_force_progress.advance(message="official complete-model route configured")
nci_force_check = check_nci_force(
    example=nci_example,
    toolkit_forces=nci_example_force,
    official_calculator=nci_official,
    device=DEVICE,
    displacement_angstrom=NCI_FD_STEP_A,
    net_force_atol_eV_A=NCI_NET_FORCE_ATOL_EV_A,
    finite_difference_rtol=NCI_FD_RTOL,
    finite_difference_atol_eV_A=NCI_FD_ATOL_EV_A,
    toolkit_official_atol_eV_A=NCI_PIPELINE_OFFICIAL_FORCE_ATOL_EV_A,
)
nci_fd_atom_index, nci_fd_axis = nci_force_check.atom_index, nci_force_check.axis
nci_net_force_max_abs_eV_A = nci_force_check.net_force_max_abs_eV_A
nci_official_analytic_force_eV_A = nci_force_check.official_analytic_force_eV_A
nci_official_fd_force_eV_A = nci_force_check.official_finite_difference_force_eV_A
nci_official_fd_error_eV_A = nci_force_check.official_finite_difference_error_eV_A
nci_toolkit_analytic_force_eV_A = nci_force_check.toolkit_force_eV_A
nci_toolkit_official_error_eV_A = nci_force_check.toolkit_official_error_eV_A
nci_force_progress.complete("energy derivative and Toolkit force agree")
display(readable_table(
    pd.DataFrame([
        ("Atom and axis", f"{nci_fd_atom_index}, {nci_fd_axis}"),
        ("Official analytic force / eV Å⁻¹", nci_official_analytic_force_eV_A),
        ("Energy finite-difference force / eV Å⁻¹", nci_official_fd_force_eV_A),
        ("Toolkit pipeline force / eV Å⁻¹", nci_toolkit_analytic_force_eV_A),
    ], columns=["Check", "Value"]),
    label="Independent force check",
    show_index=False,
))
del nci_official
""",
    )

    replace_code_source(
        "analyze-nci-curves",
        """
nci_analysis_progress = NotebookProgress(
    title="Compare the interaction curves with two reference levels",
    total=3, unit="steps",
)
EV_TO_KCAL_MOL = 1.0 / (units.kcal / units.mol)
nci_components = {
    "core": nci_member_residual_eV,
    "core_plus_d3": nci_member_residual_eV + nci_d3_graph_eV,
    "core_plus_coulomb": nci_member_residual_eV + nci_member_coulomb_eV,
    "full": nci_member_residual_eV + nci_member_coulomb_eV + nci_d3_graph_eV,
}
nci_comparisons = {
    "core vs CC": ("core", "ccsd_t_cbs"),
    "+ Coulomb vs CC": ("core_plus_coulomb", "ccsd_t_cbs"),
    "complete vs CC": ("full", "ccsd_t_cbs"),
    "same-D3 bookkeeping identity": ("core_plus_coulomb", "dft_no_d3"),
    "complete vs DFT-D3": ("full", "dft_full"),
    "DFT-D3 vs CC": ("dft_full", "ccsd_t_cbs"),
}
nci_member_curves, nci_curves, nci_metrics = assemble_nci_comparison_curves(
    nci_graph_index,
    nci_reference_data,
    nci_components,
    d3_graph_energies_eV=nci_d3_graph_eV,
    dft_total_energy_column="wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol",
    cc_interaction_energy_column="ccsd_t_cbs_interaction_energy_kcal_mol",
    comparisons=nci_comparisons,
    energy_to_kcal_mol=EV_TO_KCAL_MOL,
)
nci_analysis_progress.advance(message="AB - A - B applied to model and references")
assert (nci_metrics["complete vs CC"] < NCI_COMPLETE_MAE_LIMIT_KCAL_MOL).all()
assert (nci_metrics["complete vs DFT-D3"] < NCI_COMPLETE_MAE_LIMIT_KCAL_MOL).all()
nci_analysis_progress.advance(message="complete-model errors checked for all curves")

nci_figure, _ = plot_nci_interaction_curves(nci_curves)
nci_figure.savefig(OUTPUT_DIR / "nci_interaction_curves.png", dpi=180, bbox_inches="tight")
display(figure_with_alt(
    nci_figure,
    alt_text=(
        "Three NCI Atlas interaction-energy curves across neutral hydrogen-bonded, "
        "dispersion-dominated, and ionic hydrogen-bonded complexes. Four AIMNet "
        "component combinations are compared with DFT-D3 and CCSD(T)/CBS."
    ),
))
plt.close(nci_figure)
display(readable_table(
    nci_metrics.round(3).reset_index(),
    label="Mean absolute interaction-energy errors / kcal mol⁻¹", show_index=False,
))
nci_analysis_progress.complete("curves, ensemble spread, and errors shown")
display(callout(
    "The complete model is within 0.5 kcal/mol MAE of both references for "
    "these three systems. The no-D3 DFT column subtracts the model's D3 term "
    "from full DFT-D3, so its error must equal the full comparison. It checks "
    "bookkeeping, not accuracy. The independent CCSD(T)/CBS comparison shows "
    "how Coulomb and D3 change the curves. This is not broad MLIP validation.",
    kind="result", result_state="observed",
))
""",
    )

    replace_code_source(
        "load-sevennet-wrapper",
        """
sevennet_load_progress = NotebookProgress(
    title="Load the SevenNet-Omni adapter", total=3, unit="checks"
)
sevennet_checkpoint_path, sevennet_checkpoint_sha256 = resolve_sevennet_checkpoint()
assert sevennet_checkpoint_sha256 == SEVENNET_CHECKPOINT_SHA256
sevennet_load_progress.advance(message="checkpoint size and SHA-256 verified")
raw_sevennet, sevennet_checkpoint_config = load_raw_sevennet_omni(
    sevennet_checkpoint_path, device=DEVICE,
)
sevennet_load_progress.advance(message="raw float32 e3nn model loaded")
sevennet_model = SevenNetOmniWrapper(
    raw_sevennet, modality=SEVENNET_MODALITY,
).to(DEVICE).eval()
sevennet_config = sevennet_model.model_config
assert sevennet_config.outputs == {"energy", "forces"}
assert sevennet_config.supports_pbc and not sevennet_config.needs_pbc
assert sevennet_config.neighbor_config.format is NeighborListFormat.COO
assert not sevennet_config.neighbor_config.half_list
assert sevennet_model.direct_derivative_keys() == {"forces"}
sevennet_load_progress.complete("periodic energy/force adapter connected")
""",
    )
    insert_after(
        "load-sevennet-wrapper",
        code(
            "compose-sevennet-surface-model",
            """
surface_model_progress = NotebookProgress(
    title="Compose SevenNet-Omni with pairwise PBE-D3(BJ)", total=2, unit="steps"
)
surface_d3 = DFTD3ModelWrapper(
    a1=PBE_D3_BJ_A1, a2=PBE_D3_BJ_A2_BOHR,
    s8=PBE_D3_BJ_S8, s6=PBE_D3_BJ_S6,
    cutoff=SURFACE_D3_CUTOFF_A,
    smoothing_fraction=D3_REFERENCE_SMOOTHING_FRACTION,
    auto_download=False, param_file=D3_PARAMETER_FILE,
).to(DEVICE).eval()
surface_d3.set_config("active_outputs", {"energy", "forces"})
surface_model_progress.advance(message="pairwise PBE-D3(BJ) configured")

surface_model = PipelineModelWrapper(
    groups=[
        PipelineGroup(
            steps=[PipelineStep(model=sevennet_model)], use_autograd=False
        ),
        PipelineGroup(steps=[PipelineStep(model=surface_d3)], use_autograd=False),
    ],
    neighbor_adaptation="always",
).to(DEVICE).eval()
surface_model.set_config("active_outputs", {"energy", "forces"})
surface_model_progress.complete("custom adapter accepted by Toolkit composition")
""",
        ),
    )
    insert_after(
        "compose-sevennet-surface-model",
        code(
            "display-sevennet-surface-model",
            """
surface_model_display_progress = NotebookProgress(
    title="Show the surface model settings", total=1, unit="model card"
)
surface_model_table = build_sevennet_settings_table(
    model_name=SEVENNET_MODEL_NAME, modality=SEVENNET_MODALITY,
    reference_method=SEVENNET_REFERENCE_METHOD,
    package_version=metadata.version("sevenn"),
    checkpoint_sha256=sevennet_checkpoint_sha256,
    checkpoint_record=SEVENNET_CHECKPOINT_DOI,
    model_cutoff_A=sevennet_model.cutoff,
    supports_pbc=sevennet_config.supports_pbc,
    outputs=sevennet_config.active_outputs,
    d3_reference_cutoff_bohr=D3_REFERENCE_CUTOFF_BOHR,
    d3_cutoff_A=SURFACE_D3_CUTOFF_A,
    d3_smoothing_fraction=D3_REFERENCE_SMOOTHING_FRACTION,
)
sevennet_model_card = build_sevennet_model_card(
    model=raw_sevennet,
    wrapper=sevennet_model,
)
surface_model_display = surface_model_table.copy()
surface_model_display.loc[
    surface_model_display["Setting"].eq("Checkpoint SHA-256"), "Value"
] = sevennet_checkpoint_sha256[:16] + "…"
display(readable_table(
    surface_model_display,
    label="Surface model settings",
    show_index=False,
))
display(readable_table(
    sevennet_model_card,
    label="SevenNet-Omni domain and input behavior",
    show_index=False,
))
surface_model_display_progress.complete("model domain and composition shown")
""",
            source_hidden=True,
        ),
    )

    replace_code_source(
        "build-adsorption-panel",
        """
adsorption_build_progress = NotebookProgress(
    title="Load the adsorption starting structures", total=2, unit="checks"
)
adsorption_structures = load_initial_structure_set()
adsorption_methodology = load_adsorption_methodology()
periodic_structures, finite_structures = split_for_batches(adsorption_structures)
adsorption_build_progress.advance(message="nine versioned ASE structures verified")
display(readable_table(
    build_structure_inventory_table(adsorption_structures),
    label="Fixed starting structures", show_index=False,
))
display(readable_table(
    build_placement_table(adsorption_methodology),
    label="Four fixed starting placements", show_index=False,
))
adsorption_build_progress.complete("periodic and finite structures separated")
""",
    )
    insert_after(
        "build-adsorption-panel",
        code(
            "pack-adsorption-batches",
            """
adsorption_pack_progress = NotebookProgress(
    title="Pack periodic and finite adsorption batches", total=2, unit="batches"
)
periodic_surface_batch = Batch.from_data_list([
    AtomicData.from_atoms(atoms, device=DEVICE, dtype=torch.float32)
    for atoms in periodic_structures.values()
], device=DEVICE)
adsorption_pack_progress.advance(
    message=(f"periodic: {periodic_surface_batch.num_graphs} graphs / "
             f"{periodic_surface_batch.num_nodes} atoms")
)
finite_molecule_batch = Batch.from_data_list([
    AtomicData.from_atoms(atoms, device=DEVICE, dtype=torch.float32)
    for atoms in finite_structures.values()
], device=DEVICE)
adsorption_pack_progress.complete(
    f"finite: {finite_molecule_batch.num_graphs} graphs / "
    f"{finite_molecule_batch.num_nodes} atoms"
)
            """,
        ),
    )
    insert_after(
        "pack-adsorption-batches",
        markdown(
            "sevennet-tasks",
            r"""
### One checkpoint, several tasks

SevenNet calls these choices **tasks** or **modalities**, not separate output
heads. A task conditions one loaded checkpoint for a training target. We read
`raw_sevennet.modal_map`, then send the same clean Cu(111) slab and CO/Cu(111)
starting geometry through:

| Task | Target represented here | Why show it |
|---|---|---|
| `mpa` | PBE(+U)-level MPtrj/sAlex and cross-domain data | main tutorial model |
| `oc20` | RPBE OC20 metal-catalyst adsorption data | surface-focused alternative |

This **model sweep** uses one task per call. SevenNet can accept a task label
for each graph, but this adapter keeps the method choice explicit.
"""
            + "\n\n"
            + callout_html(
                "Do not compare raw total energies across tasks: their atomic-energy references differ. Every term in an adsorption or reaction-energy cycle must use the same task and the same correction or composite-model scheme. This probe does not add D3.",
                kind="check",
            )
            + "\n\n"
            + callout_html(
                "Toolkit does not automatically send different graphs to different model objects in one call. Use Batch.index_select(...) to form compatible groups, evaluate them separately, and restore the original order. Grouping changes execution only; routing different terms of one energy cycle to different full models is invalid.",
                kind="note",
            ),
        ),
    )
    insert_after(
        "sevennet-tasks",
        code(
            "compare-sevennet-tasks",
            """
sevennet_task_progress = NotebookProgress(
    title="Inspect and switch SevenNet tasks", total=3, unit="steps",
)
available_sevennet_tasks = sorted(str(name) for name in raw_sevennet.modal_map)
selected_task_targets = {
    "mpa": "PBE(+U)-level MPtrj/sAlex + cross-domain data",
    "oc20": "RPBE OC20 metal-catalyst adsorption data",
}
assert set(selected_task_targets).issubset(available_sevennet_tasks)
display(callout(
    f"The checkpoint reports {len(available_sevennet_tasks)} tasks. "
    "This short comparison evaluates mpa and oc20 on the same two graphs; "
    "the full task list remains in available_sevennet_tasks.",
    kind="note",
))
sevennet_task_progress.advance(message="available checkpoint tasks inspected")
task_probe_keys = ("clean_cu111", ADSLAB_KEYS["CO"])
periodic_order = tuple(periodic_structures)
task_probe_indices = [periodic_order.index(key) for key in task_probe_keys]
task_probe_batch = periodic_surface_batch.index_select(task_probe_indices)
sevennet_task_outputs, task_probe_rows = {}, []

for task, target in selected_task_targets.items():
    task_model = SevenNetOmniWrapper(raw_sevennet, modality=task).to(DEVICE).eval()
    task_batch = task_probe_batch.clone()
    compute_neighbors(task_batch, config=task_model.model_config.neighbor_config)
    task_outputs = task_model(task_batch)
    energies = task_outputs["energy"].detach()
    forces = task_outputs["forces"].detach()
    sevennet_task_outputs[task] = {
        "energy": energies.cpu(), "forces": forces.cpu(),
    }
    task_probe_rows.extend(summarize_sevennet_task_outputs(
        task=task, target=target, structure_keys=task_probe_keys,
        batch=task_batch, outputs=task_outputs,
    ))
    sevennet_task_progress.advance(message=f"{task}: two graphs evaluated")

sevennet_task_summary = pd.DataFrame(task_probe_rows)
display(readable_table(
    sevennet_task_summary,
    label="Same two-graph batch evaluated with two tasks", show_index=False,
))
sevennet_task_progress.complete("energy and force outputs shown for both tasks")
display(callout(
    "Both tasks returned finite energy and force arrays for each structure. "
    "Differences in per-structure maximum force magnitudes show fixed-geometry "
    "method sensitivity, not accuracy; that requires matched references. Raw "
    "energies remain available in sevennet_task_outputs but are not compared. "
    "The main calculation uses mpa.",
    kind="result",
    result_state="observed",
))
""",
        ),
    )

    replace_code_source(
        "run-sevennet-wrapper",
        """
sevennet_run_progress = NotebookProgress(
    title="Evaluate nine structures in two batches", total=4, unit="steps"
)

# One raw-model call for five periodic graphs and one for four finite graphs.
compute_neighbors(periodic_surface_batch, config=sevennet_config.neighbor_config)
periodic_model_outputs = sevennet_model(periodic_surface_batch)
compute_neighbors(finite_molecule_batch, config=sevennet_config.neighbor_config)
finite_model_outputs = sevennet_model(finite_molecule_batch)
sevennet_run_progress.advance(message="SevenNet energies and forces returned")

# D3 needs its own neighbor layout, but uses the same two Toolkit batches.
compute_neighbors(periodic_surface_batch, config=surface_d3.model_config.neighbor_config)
periodic_d3_outputs = surface_d3(periodic_surface_batch)
compute_neighbors(finite_molecule_batch, config=surface_d3.model_config.neighbor_config)
finite_d3_outputs = surface_d3(finite_molecule_batch)
sevennet_run_progress.advance(message="pairwise D3 corrections returned")

# The composed Toolkit model returns the final energy and force fields.
periodic_pipeline_outputs = surface_model(periodic_surface_batch)
finite_pipeline_outputs = surface_model(finite_molecule_batch)
sevennet_run_progress.complete("raw model, D3, and composed outputs returned")
""",
    )
    insert_after(
        "run-sevennet-wrapper",
        code(
            "validate-sevennet-wrapper",
            """
sevennet_check_progress = NotebookProgress(
    title="Validate the custom adapter", total=1, unit="check set"
)
sevennet_lesson = finalize_sevennet_lesson(
    wrapper=sevennet_model,
    raw_model=raw_sevennet,
    checkpoint_path=sevennet_checkpoint_path,
    modality=SEVENNET_MODALITY,
    device=DEVICE,
    periodic_structures=periodic_structures,
    finite_structures=finite_structures,
    periodic_batch=periodic_surface_batch,
    finite_batch=finite_molecule_batch,
    periodic_model_outputs=periodic_model_outputs,
    finite_model_outputs=finite_model_outputs,
    periodic_d3_outputs=periodic_d3_outputs,
    finite_d3_outputs=finite_d3_outputs,
    periodic_pipeline_outputs=periodic_pipeline_outputs,
    finite_pipeline_outputs=finite_pipeline_outputs,
    official_structure_key=ADSLAB_KEYS["CO"],
    energy_tolerance_eV_per_atom=SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM,
    force_tolerance_eV_A=SEVENNET_REPEAT_FORCE_TOL_EV_A,
)
sevennet_graph_mapping = sevennet_lesson.graph_mapping
sevennet_graph_mapping_passed = sevennet_lesson.graph_mapping_passed
official_calculator_check = sevennet_lesson.official_calculator_check
sevennet_numerical_agreement = sevennet_lesson.numerical_agreement
sevennet_repeat_max_energy_difference_eV_per_atom = (
    sevennet_lesson.repeat_max_energy_difference_eV_per_atom
)
sevennet_repeat_max_force_difference_eV_A = (
    sevennet_lesson.repeat_max_force_difference_eV_A
)
surface_model_energies, surface_model_forces = (
    sevennet_lesson.model_energies, sevennet_lesson.model_forces
)
surface_d3_energies, surface_d3_forces = (
    sevennet_lesson.d3_energies, sevennet_lesson.d3_forces
)
surface_combined_energies, surface_combined_forces = (
    sevennet_lesson.combined_energies, sevennet_lesson.combined_forces
)
sevennet_check_progress.complete("adapter, native model, and component sum agree")
""",
            source_hidden=True,
        ),
    )

    replace_code_source(
        "inspect-ir-batch",
        """
ir_checks_progress = NotebookProgress(
    title="Inspect the four-system IR batch", total=2, unit="checks"
)
compute_neighbors(batch, config=model.model_config.neighbor_config)
initial_outputs = model(batch)
mass_checks = mass_only_invariance(
    batch,
    initial_outputs,
    position_rtol=MASS_ONLY_POSITION_RTOL,
    position_atol_angstrom=MASS_ONLY_POSITION_ATOL_A,
    energy_tolerance_eV=MASS_ONLY_ENERGY_TOLERANCE_EV,
    force_tolerance_eV_A=MASS_ONLY_FORCE_TOLERANCE_EV_A,
    charge_tolerance_e=MASS_ONLY_CHARGE_TOLERANCE_E,
)
ir_checks_progress.advance(message="complete model evaluated on all four systems")
graph_idx = batch.batch_idx.to(torch.int32)
force_norm = torch.linalg.vector_norm(initial_outputs["forces"], dim=1)
initial_fmax = torch.stack([
    force_norm[graph_idx == graph].max() for graph in range(batch.num_graphs)
])
display(readable_table(
    pd.DataFrame({
        "system": labels,
        "atoms": batch.num_nodes_per_graph.cpu().numpy(),
        "energy_eV": initial_outputs["energy"].detach().reshape(-1).cpu().numpy(),
        "max_force_eV_A": initial_fmax.detach().cpu().numpy(),
    }),
    label="Initial IR-batch outputs", show_index=False,
))
ir_checks_progress.complete("energy and force outputs inspected")
""",
    )
    insert_after(
        "inspect-ir-batch",
        code(
            "check-ir-dipoles",
            """
dipole_check_progress = NotebookProgress(
    title="Check charges and dipole inputs", total=2, unit="checks"
)
q = initial_outputs["charges"].reshape(-1)
q_sum = segmented_sum(q, graph_idx, batch.num_graphs)
mu = segmented_sum(q[:, None] * batch.positions, graph_idx, batch.num_graphs)
dipole_check_progress.advance(message="charge and dipole reduced by system")

translations = torch.tensor(
    [[1.2, -0.7, 0.5], [-0.3, 0.9, 1.1], [2.0, 1.0, -1.0], [-1.0, -2.0, 0.4]],
    device=DEVICE, dtype=batch.positions.dtype,
)
shifted = batch.positions + translations[batch.batch_idx.long()]
mu_shifted = segmented_sum(q[:, None] * shifted, graph_idx, batch.num_graphs)
origin_error = float((mu - mu_shifted).abs().max().detach().cpu())
assert float(q_sum.abs().max().detach().cpu()) < IR_CHARGE_NEUTRALITY_TOLERANCE_E
assert origin_error < IR_DIPOLE_ORIGIN_TOLERANCE_E_ANGSTROM
display(readable_table(
    mass_invariance_display_table(
        mass_checks,
        dipole_origin_error_e_angstrom=origin_error,
    ),
    label="Isotope and dipole checks", show_index=False,
))
dipole_check_progress.complete("neutral-system dipoles are origin independent")
display(callout(
    "H/D energy, force, charge, and coordinates match; only mass changes.",
    kind="result", result_state="pass",
))
unrelaxed_batch = batch.clone()
""",
        ),
    )

    replace_code_source(
        "relax",
        """
batch = unrelaxed_batch.clone()
relax_progress = NotebookProgress(
    title="Batched FIRE2 relaxation", total=FIRE_MAX_STEPS, unit="steps"
)
relaxer = FIRE2(
    model=model,
    dt=IR_FIRE_INITIAL_DT,
    n_steps=FIRE_MAX_STEPS,
    convergence_hook=ConvergenceHook.from_fmax(threshold=FIRE_FMAX_EV_A),
)
for neighbor_hook in model.make_neighbor_hooks():
    relaxer.register_hook(neighbor_hook)
relaxer.register_hook(NaNDetectorHook(frequency=1))
relax_progress_hook: Hook = NotebookStageProgressHook(
    relax_progress, frequency=25, label="FIRE2"
)
relaxer.register_hook(relax_progress_hook)

t0 = perf_counter()
batch = relaxer.run(batch)
torch.cuda.synchronize()
relax_elapsed_s = perf_counter() - t0
relax_progress.update(
    done=relaxer.step_count,
    message=f"FIRE2 stopped after {relaxer.step_count} steps ({relax_elapsed_s:.1f} s)",
    state="complete",
)
""",
    )
    insert_after(
        "relax",
        code(
            "validate-relaxation",
            """
relax_check_progress = NotebookProgress(
    title="Validate the relaxed structures", total=3, unit="checks"
)
# Isotope substitution changes mass only, so paired coordinates must be exact.
ptr = batch.batch_ptr.tolist()
batch.positions[ptr[1]:ptr[2]].copy_(batch.positions[ptr[0]:ptr[1]])
batch.positions[ptr[3]:ptr[4]].copy_(batch.positions[ptr[2]:ptr[3]])
compute_neighbors(batch, config=model.model_config.neighbor_config)
relaxed_outputs = model(batch)
batch.energy = relaxed_outputs["energy"].to(batch.positions.dtype)
batch.forces = relaxed_outputs["forces"]
batch.charges = relaxed_outputs["charges"]
relax_check_progress.advance(message="final state re-evaluated after isotope pairing")

relaxed_mass_checks = mass_only_invariance(
    batch,
    relaxed_outputs,
    position_rtol=MASS_ONLY_POSITION_RTOL,
    position_atol_angstrom=MASS_ONLY_POSITION_ATOL_A,
    energy_tolerance_eV=MASS_ONLY_ENERGY_TOLERANCE_EV,
    force_tolerance_eV_A=MASS_ONLY_FORCE_TOLERANCE_EV_A,
    charge_tolerance_e=MASS_ONLY_CHARGE_TOLERANCE_E,
)
fmax = [
    float(torch.linalg.vector_norm(batch.forces[a:b], dim=1).max().cpu())
    for a, b in zip(ptr[:-1], ptr[1:], strict=True)
]
if max(fmax) > FIRE_FMAX_EV_A:
    raise RuntimeError("FIRE2 did not reach the specified 0.01 eV/Å criterion")
relax_check_progress.advance(message="force and isotope checks passed")
relaxed_batch = batch.clone()
display(readable_table(
    pd.DataFrame({"system": labels, "fmax_eV_A": fmax}),
    label="Relaxed-structure force maxima", show_index=False,
))
display(readable_table(
    mass_invariance_display_table(relaxed_mass_checks),
    label="Relaxed isotope checks", show_index=False,
))
relax_check_progress.complete("all four structures meet the force criterion")
""",
        ),
    )
    insert_after(
        "validate-relaxation",
        code(
            "save-relaxed-structures",
            """
relax_save_progress = NotebookProgress(
    title="Save and replay the relaxed batch", total=2, unit="checks"
)
relaxed_zarr_path = OUTPUT_DIR / "water_ir_relaxed.zarr"
relaxed_sink = ZarrData(relaxed_zarr_path, capacity=relaxed_batch.num_graphs)
relaxed_sink.zero()
relaxed_sink.write(relaxed_batch)
relaxed_replay = relaxed_sink.read()
assert relaxed_replay.num_graphs == relaxed_batch.num_graphs
torch.testing.assert_close(
    relaxed_replay.positions, relaxed_batch.positions.cpu(), rtol=0.0, atol=0.0
)
relax_save_progress.advance(message="Zarr replay matches the GPU batch")
relaxed_atoms = [
    graph_atoms_from_batch(relaxed_batch, graph, label)
    for graph, label in enumerate(labels)
]
write(OUTPUT_DIR / "water_ir_relaxed_start.extxyz", relaxed_atoms)
relax_save_progress.complete("Zarr and extxyz files written")
display(callout(
    "All structures reached fmax < 0.01 eV/Å, and the Zarr replay reproduced "
    "the saved coordinates exactly.",
    kind="result", result_state="pass",
))
""",
        ),
    )

    insert_before(
        "compile-fixed-ir-model",
        code(
            "build-compiled-ir-model",
            """
compiled_model_progress = NotebookProgress(
    title="Compile the fixed IR model", total=1, unit="model"
)
eager_model = model
production_backbone = torch.compile(
    aimnet.model.eval(), fullgraph=False, dynamic=False
)
production_aimnet = AIMNet2Wrapper(production_backbone, train=False).to(DEVICE)
production_aimnet.set_config("active_outputs", {"energy", "charges"})
production_coulomb = DirectCoulombWrapper().to(DEVICE)
production_d3 = DFTD3ModelWrapper(
    a1=d3_params["a1"],
    a2=d3_params["a2"],
    s8=d3_params["s8"],
    s6=d3_params.get("s6", 1.0),
    cutoff=D3_CUTOFF_A,
    param_file=D3_PARAMETER_FILE,
    auto_download=False,
).to(DEVICE)
for component in (production_aimnet, production_d3):
    component.model_config.neighbor_config.skin = NEIGHBOR_SKIN_A
production_d3.set_config("active_outputs", {"energy", "forces"})

model = PipelineModelWrapper(
    groups=[
        PipelineGroup(
            steps=[production_aimnet, production_coulomb], use_autograd=True
        ),
        PipelineGroup(steps=[production_d3], use_autograd=False),
    ],
    neighbor_adaptation="always",
).to(DEVICE).eval()
model.set_config("active_outputs", {"energy", "forces", "charges"})
for parameter in model.parameters():
    parameter.requires_grad_(False)
compiled_model_progress.complete("compiled AIMNet + Coulomb + D3 pipeline ready")
""",
        ),
    )
    replace_code_source(
        "compile-fixed-ir-model",
        """
compile_progress = NotebookProgress(
    title="Check the compiled model against eager execution", total=3, unit="checks"
)
assert torch.get_float32_matmul_precision() == AIMNET_MATMUL_PRECISION
fixed_ir_batch = batch.clone()
compute_neighbors(fixed_ir_batch, config=eager_model.model_config.neighbor_config)
IR_HELD_FIELDS = (
    "positions", "atomic_numbers", "charge", "neighbor_matrix", "num_neighbors"
)
IR_COMPARE_OUTPUTS = ("energy", "forces", "charges")
fixed_ir_source = snapshot_tensor_fields(
    fixed_ir_batch, field_names=IR_HELD_FIELDS
)
eager_ir_raw = eager_model(fixed_ir_batch)
eager_ir_outputs = clone_selected_outputs(
    eager_ir_raw, output_names=IR_COMPARE_OUTPUTS
)
assert_tensor_fields_unchanged(
    fixed_ir_batch, fixed_ir_source, field_names=IR_HELD_FIELDS
)
torch.cuda.synchronize()
compile_progress.advance(message="eager outputs recorded")

for field in ("cutoff", "format", "half_list", "skin"):
    assert getattr(model.model_config.neighbor_config, field) == getattr(
        eager_model.model_config.neighbor_config, field
    )
""",
    )

    replace_code_source(
        "inflight-example",
        """
INFLIGHT_SYSTEMS = 2_048
INFLIGHT_ACTIVE_SYSTEMS = 256
INFLIGHT_NVT_STEPS = 2
INFLIGHT_NVE_STEPS = 3
inflight_progress = NotebookProgress(
    title="Build the inflight queue", total=1, unit="queue"
)
inflight_source = prepare_inflight_dimer_source(
    scan_dimers=scan_dimers,
    scan_batch=scan_batch,
    full_outputs=full_outputs,
    num_systems=INFLIGHT_SYSTEMS,
    temperature_k=TEMPERATURE_K,
    velocity_seed=404,
)
atoms_per_dimer = int(inflight_source.num_nodes_per_graph[0])
inflight_dataset = InMemoryDataset(in_memory_batch=inflight_source, device=DEVICE)
inflight_sampler = SizeAwareSampler(
    inflight_dataset,
    max_atoms=INFLIGHT_ACTIVE_SYSTEMS * atoms_per_dimer,
    max_edges=None,
    max_batch_size=INFLIGHT_ACTIVE_SYSTEMS,
    shuffle=False,
)
inflight_sink = HostMemory(capacity=INFLIGHT_SYSTEMS)
inflight_progress.complete(
    f"{INFLIGHT_SYSTEMS:,} queued; at most {INFLIGHT_ACTIVE_SYSTEMS} active"
)
""",
    )
    insert_after(
        "inflight-example",
        code(
            "configure-inflight-stage",
            """
inflight_stage_progress = NotebookProgress(
    title="Configure the inflight fused stage", total=2, unit="steps"
)
eager_model.set_config("active_outputs", {"energy", "forces", "charges"})
inflight_nvt = NVTLangevin(
    model=eager_model, dt=DT_FS, temperature=TEMPERATURE_K,
    friction=0.01, random_seed=505,
    convergence_hook=converge_after_steps("nvt_steps_done", INFLIGHT_NVT_STEPS),
    device_type=DEVICE.type,
)
inflight_nve = NVE(
    model=eager_model, dt=DT_FS,
    convergence_hook=converge_after_steps("nve_steps_done", INFLIGHT_NVE_STEPS),
    device_type=DEVICE.type,
)
inflight = FusedStage(
    sub_stages=[
        (IR_WARMUP_STATUS, inflight_nvt),
        (IR_PRODUCTION_STATUS, inflight_nve),
    ],
    sampler=inflight_sampler,
    sinks=[inflight_sink],
    refill_frequency=1,
    device_type=DEVICE.type,
)
inflight_stage_progress.advance(message="NVT and NVE stages connected to the queue")
inflight.register_fused_hook(StageStepCounterHook({
    IR_WARMUP_STATUS: "nvt_steps_done",
    IR_PRODUCTION_STATUS: "nve_steps_done",
}))
inflight_trace = register_inflight_trace(inflight)
for neighbor_hook in eager_model.make_neighbor_hooks():
    inflight.register_hook(neighbor_hook)
inflight.register_hook(NaNDetectorHook(frequency=1, extra_keys=["velocities"]))
assert inflight.inflight_mode
inflight_stage_progress.complete("counters, trace, neighbors, and safety hooks attached")
""",
        ),
    )
    insert_after(
        "configure-inflight-stage",
        code(
            "run-inflight-example",
            """
inflight_run_progress = NotebookProgress(
    title="Run the inflight queue", total=2, unit="steps"
)
run_result = inflight.run(batch=None)
assert run_result is None and inflight_sampler.exhausted and inflight.done
inflight_run_progress.advance(message="every queued system reached HostMemory")

completed = inflight_sink.drain()
assert completed.num_graphs == INFLIGHT_SYSTEMS
inflight_run_progress.complete(f"{completed.num_graphs:,} systems returned to CPU")
""",
        ),
    )
    insert_after(
        "run-inflight-example",
        code(
            "validate-inflight-example",
            """
inflight_validation_progress = NotebookProgress(
    title="Check inflight scheduling and results", total=2, unit="checks"
)
system_ids = completed.system_id.reshape(-1).to(torch.long)
unique_ids, counts = torch.unique(system_ids, sorted=True, return_counts=True)
assert torch.equal(unique_ids, torch.arange(INFLIGHT_SYSTEMS, dtype=torch.long))
assert torch.equal(counts, torch.ones_like(counts))
inflight_nvt_counts_correct = bool(
    torch.all(completed.nvt_steps_done == INFLIGHT_NVT_STEPS).item()
)
inflight_nve_counts_correct = bool(
    torch.all(completed.nve_steps_done == INFLIGHT_NVE_STEPS).item()
)
assert inflight_nvt_counts_correct
assert inflight_nve_counts_correct
inflight_validation_progress.advance(message="stable IDs and update counts checked")
inflight_failure_count = 0
inflight_trace.finalize(
    completed_system_ids=system_ids,
    failure_count=inflight_failure_count,
)
inflight_trace_rows = inflight_trace_table(inflight_trace)
assert int(inflight_trace_rows.iloc[-1]["Active"]) == 0
assert int(inflight_trace_rows.iloc[-1]["Completed"]) == INFLIGHT_SYSTEMS
assert int(inflight_trace_rows.iloc[-1]["Failures"]) == inflight_failure_count
inflight_validation_progress.complete("each stable system ID returned exactly once")

inflight_summary = pd.DataFrame([
    ("queued systems", INFLIGHT_SYSTEMS),
    ("maximum active systems", INFLIGHT_ACTIVE_SYSTEMS),
    ("fused steps", inflight.step_count),
    ("completed systems", completed.num_graphs),
    ("failed systems", inflight_failure_count),
    ("unique system IDs", unique_ids.numel()),
    ("duplicate system IDs", int((counts > 1).sum())),
    ("NVT updates per system", int(completed.nvt_steps_done[0].item())),
    ("NVE updates per system", int(completed.nve_steps_done[0].item())),
], columns=["Measure", "Value"])
display(readable_table(inflight_summary,
    label="Inflight batching result", show_index=False))
display(readable_table(
    inflight_trace_rows.drop(columns="Failures"),
    label="Observed active-batch refills",
    show_index=False,
))
display(callout(
    "The trace shows the active batch refilling until all 2,048 stable system IDs reach the CPU result batch exactly once.",
    kind="result",
    result_state="pass",
))
inflight_dataset.close()
del inflight_source, inflight_dataset, completed
""",
            source_hidden=True,
        ),
    )

    replace_markdown_source(
        "stage-7",
        stage_card_html(
            stage=7,
            total=7,
            title="Choose a scaling path by workload shape",
            outcome=(
                "Keep many independent structures moving, then load one checked "
                "periodic box for the domain-parallel path."
            ),
            state="ready",
            compute_time="30 s on one H100 PCIe in the checked run",
        )
        + "\n\n"
        + callout_html(
            "Match the scaling API to the workload shape.",
            kind="before",
        )
        + "\n\n"
        + r"""
| Toolkit path | Use it when |
|---|---|
| `Batch` | independent systems fit in one model call |
| `FusedStage` | systems in one active batch follow different workflow stages |
| inflight `FusedStage` | the queue is larger than the active GPU batch |
| `DomainParallel` | one periodic system must be divided across GPUs |
| `DistributedPipeline` | independent batches move through stages on different GPUs |

This notebook runs batching, fused stages, inflight execution, and a one-GPU
`DomainParallel` walkthrough. It then loads bundled 1-, 2-, and 4-H100
`DomainParallel` results. `DistributedPipeline` remains an API preview.
""",
    )
    replace_markdown_source(
        "inflight-intro",
        r"""
### Refill the active batch as systems finish

Inflight batching keeps a bounded active batch on the GPU and replaces systems as they finish.

- `InMemoryDataset` exposes the queued structures.
- `SizeAwareSampler` fills the active limits set by `max_atoms`, `max_edges`, and `max_batch_size`.
- `FusedStage(..., sampler=..., sinks=...)` advances each active system according to its status.
- `HostMemory` receives finished systems on the CPU and frees their GPU capacity.
- `system_id` identifies the original system throughout the queue. `batch_idx` only describes its current position inside one packed batch and can change after a refill.

Here status 0 means NVT, status 1 means NVE, and status 2 means finished. `refill_frequency=1` checks for free capacity after every fused update; it does not change the MD timestep or split a model call.
"""
        + "\n\n"
        + process_diagram_html(
            title="Inflight batching on one GPU",
            steps=(
                "queued dataset",
                "SizeAwareSampler",
                "bounded active Batch",
                "per-system stage status",
                "HostMemory results",
                "refill free capacity",
            ),
            caption=(
                "Finished systems leave between model updates. Unfinished systems "
                "remain active while stable system IDs preserve result identity."
            ),
        )
        + "\n\n"
        + callout_html(
            "This 2,048-system run demonstrates scheduling and collection, not production MD or a throughput benchmark: each system receives only two NVT and three NVE updates.",
            kind="note",
        ),
    )

    replace_markdown_source(
        "distributed-pipeline-intro",
        r"""
### Optional API note: assign workflow stages to ranks

A **rank** is one worker process, normally assigned to one GPU.
`DistributedPipeline` is for many independent systems that pass through
different stages. It does not divide one trajectory or one model call across
GPUs.

| Level | What it changes |
|---|---|
| `Batch` | independent systems evaluated in one model call on one device |
| `FusedStage` | different updates share one model call on one GPU |
| inflight batching | finished systems are replaced from a larger queue |
| `DistributedPipeline` | different ranks are assigned different workflow stages |

This is the public two-rank construction:

```python
from nvalchemi.dynamics import DistributedPipeline, FIRE2, FusedStage, HostMemory
from nvalchemi.dynamics.base import BufferConfig
from nvalchemi.dynamics.hooks import ConvergedSnapshotHook

transfer = BufferConfig(
    num_systems=max_batch_size,
    num_nodes=max_atoms_in_transfer,
    num_edges=0,  # neighbor arrays are rebuilt after the transfer
)
finished = HostMemory(capacity=total_systems)
nve.register_hook(ConvergedSnapshotHook(sink=finished))

relaxation = FIRE2(
    model=model,
    dt=0.02,
    sampler=relaxation_sampler,
    prior_rank=None,
    next_rank=1,
    max_batch_size=max_batch_size,
    buffer_config=transfer,
    comm_mode="async_recv",
)
dynamics = FusedStage(
    sub_stages=[(0, nvt), (1, nve)],
    prior_rank=0,
    next_rank=None,
    max_batch_size=max_batch_size,
    buffer_config=transfer,
    comm_mode="async_recv",
)

def run_two_rank_pipeline(device):
    pipeline = DistributedPipeline(
        stages={0: relaxation, 1: dynamics},
        synchronized=False,  # removes the optional barrier only
        backend="nccl",
        device_id=device,
    )
    with pipeline:
        pipeline.run()
```

`SizeAwareSampler` fills the active batch. `BufferConfig` sets the matching
transfer capacity on both ranks, `comm_mode` controls adjacent transfers, and
`ConvergedSnapshotHook` writes finished systems to CPU `HostMemory`.

In Toolkit 0.2, the transfer omits integer atom fields and `run()` still uses a
blocking completion check each iteration. This is construction only: stage
overlap and speedup are not reported.
""",
    )

    replace_code_source(
        "pipeline-campaign-results",
        """
PLANNED_CAMPAIGN_SYSTEMS_TOTAL = 8_192
DISTRIBUTED_PIPELINE_NOT_REPORTED_REASON = (
    "Toolkit 0.2 does not transfer the complete Batch, and stage overlap has "
    "not been demonstrated."
)
campaign_progress = NotebookProgress(
    title="Report the pipeline status", total=1, unit="check"
)
campaign_progress.complete("correctness, overlap, and timing are not reported")
display(callout(
    "NOT REPORTED: " + DISTRIBUTED_PIPELINE_NOT_REPORTED_REASON,
    kind="result",
    result_state="not_reported",
))
""",
    )
    append_code_source(
        "compile-fixed-ir-model",
        """
compiled_ir_raw = model(fixed_ir_batch)
compiled_ir_outputs = clone_selected_outputs(
    compiled_ir_raw, output_names=IR_COMPARE_OUTPUTS
)
assert_tensor_fields_unchanged(
    fixed_ir_batch, fixed_ir_source, field_names=IR_HELD_FIELDS
)
torch.cuda.synchronize()
compile_progress.advance(message="compiled energy and force pass synchronized")

compiled_repeat_raw = model(fixed_ir_batch)
compiled_repeat_outputs = clone_selected_outputs(
    compiled_repeat_raw, output_names=IR_COMPARE_OUTPUTS
)
torch.cuda.synchronize()
compiled_ir_eager_agreement = max_absolute_differences(
    compiled_ir_outputs, eager_ir_outputs, output_names=IR_COMPARE_OUTPUTS
)
compiled_ir_repeat_agreement = max_absolute_differences(
    compiled_repeat_outputs, compiled_ir_outputs, output_names=IR_COMPARE_OUTPUTS
)
compiled_ir_checks = build_difference_check_table(
    {
        "compiled - eager": compiled_ir_eager_agreement,
        "compiled repeat": compiled_ir_repeat_agreement,
    },
    {
        "compiled - eager": {
            "energy": COMPILED_EAGER_ENERGY_TOLERANCE_EV,
            "forces": COMPILED_EAGER_FORCE_TOLERANCE_EV_A,
            "charges": COMPILED_EAGER_CHARGE_TOLERANCE_E,
        },
        "compiled repeat": {
            "energy": COMPILED_REPEAT_ENERGY_TOLERANCE_EV,
            "forces": COMPILED_REPEAT_FORCE_TOLERANCE_EV_A,
            "charges": COMPILED_REPEAT_CHARGE_TOLERANCE_E,
        },
    },
)
if not bool(compiled_ir_checks["passed"].all()):
    raise RuntimeError("Compiled model numerical check failed")
compile_progress.complete("compiled and eager outputs match")
display(readable_table(
    compiled_ir_checks, label="Compiled and eager model checks", show_index=False
))
del fixed_ir_source, fixed_ir_batch
del eager_ir_raw, compiled_ir_raw, compiled_repeat_raw
del eager_ir_outputs, compiled_ir_outputs, compiled_repeat_outputs
""",
    )

    insert_before(
        "configure-dynamics",
        code(
            "initialize-dynamics",
            """
dynamics_state_progress = NotebookProgress(
    title="Initialize the dynamics state", total=2, unit="steps"
)
batch = relaxed_batch.clone()
batch["status"] = torch.full(
    (batch.num_graphs, 1), IR_WARMUP_STATUS,
    dtype=torch.long, device=DEVICE,
)
add_stage_step_counters(batch, ("nvt_steps_done", "nve_steps_done"))
dynamics_state_progress.advance(message="status and per-stage counters added")

# Paired random draws; isotope masses set the physical velocity scale.
for graph, seed in zip(
    range(batch.num_graphs), IR_INITIAL_VELOCITY_RANDOM_SEEDS, strict=True
):
    start, stop = ptr[graph], ptr[graph + 1]
    local_idx = torch.zeros(stop - start, dtype=torch.int32, device=DEVICE)
    initialize_velocities(
        batch.velocities[start:stop],
        batch.atomic_masses[start:stop],
        torch.tensor([TEMPERATURE_K], device=DEVICE, dtype=batch.positions.dtype),
        local_idx,
        random_seed=seed, remove_com=True, remove_rotations=True, rescale=True,
        positions=batch.positions[start:stop],
    )
dynamics_state_progress.complete("paired H/D velocities initialized")
""",
        ),
    )
    replace_code_source(
        "configure-dynamics",
        """
dynamics_setup_progress = NotebookProgress(
    title="Build and fuse the NVT and NVE stages", total=3, unit="steps"
)
ir_hook = PredictedChargeIRHook(
    warmup_steps=WARMUP_STEPS,
    n_steps=PRODUCTION_STEPS,
    dt_fs=DT_FS,
    warmup_status=IR_WARMUP_STATUS,
    production_status=IR_PRODUCTION_STATUS,
    charge_tolerance=IR_CAPTURE_CHARGE_TOLERANCE_E,
    compile_reducer=True,
)
nvt = NVTLangevin(
    model=model, dt=DT_FS, temperature=TEMPERATURE_K,
    friction=IR_NVT_FRICTION_PER_FS, random_seed=IR_NVT_RANDOM_SEED,
    convergence_hook=converge_after_steps("nvt_steps_done", WARMUP_STEPS),
)
dynamics_setup_progress.advance(message="NVT warmup stage configured")
nve = NVE(
    model=model, dt=DT_FS,
    convergence_hook=converge_after_steps("nve_steps_done", PRODUCTION_STEPS),
)
dynamics_setup_progress.advance(message="NVE production stage configured")
dynamics = nvt + nve
assert isinstance(dynamics, FusedStage)
dynamics.register_fused_hook(StageStepCounterHook({
    IR_WARMUP_STATUS: "nvt_steps_done",
    IR_PRODUCTION_STATUS: "nve_steps_done",
}))
dynamics_setup_progress.complete("stages fused with per-system counters")
""",
    )
    insert_before(
        "relax",
        markdown(
            "hooks-quick-note",
            r"""
### Hooks: checks and actions during a run

A hook is a small piece of code that Toolkit runs at a chosen point in the workflow and at a chosen frequency. It receives the current workflow context, including the live `Batch`, so it can inspect results, record them, or update workflow state without adding that work to the integrator itself.

`NaN` means *not a number* and `Inf` means *infinity*. Neither is a valid energy, force, velocity, or coordinate. Once a non-finite value enters an integration step, later trajectory frames are no longer meaningful. `NaNDetectorHook` checks energy and forces by default; here we also ask it to check velocities. If it finds a problem, it stops the run and reports the field, step, and affected graph.

Hooks in this notebook refresh neighbor lists, test convergence, count fused-stage steps, record dipoles for IR, write a CSV log, update progress, and catch non-finite values. `register_hook(...)` adds a regular dynamics hook. `register_fused_hook(...)` is for a hook that needs the complete live batch at the fused-stage level.
"""
            + "\n\n"
            + callout_html(
                "The relaxation below checks every step. The longer dynamics run later checks every 100 steps to limit overhead. A smaller frequency catches a failure sooner; a larger frequency checks less often.",
                kind="note",
            ),
        ),
    )
    insert_after(
        "configure-dynamics",
        code(
            "attach-dynamics-hooks",
            """
dynamics_hooks_progress = NotebookProgress(
    title="Attach dynamics hooks", total=3, unit="hook groups"
)
for neighbor_hook in model.make_neighbor_hooks():
    dynamics.register_hook(neighbor_hook)
dynamics_hooks_progress.advance(message="skin-aware neighbor hooks attached")
dynamics.register_hook(NaNDetectorHook(frequency=100, extra_keys=["velocities"]))
dynamics.register_hook(ir_hook)
dynamics_hooks_progress.advance(message="safety and IR recording hooks attached")
md_log_hook = LoggingHook(
    backend="csv",
    log_path=OUTPUT_DIR / "water_ir_dynamics_log.csv",
    frequency=1_000,
)
dynamics.register_hook(md_log_hook)
dynamics_hooks_progress.complete("CSV logging hook attached")

display(readable_table(
    pd.DataFrame([
        ("Fused stages", f"{type(nvt).__name__} → {type(nve).__name__}"),
        (
            "Updates",
            f"{WARMUP_STEPS:,} NVT + {PRODUCTION_STEPS:,} NVE = "
            f"{TOTAL_DYNAMICS_STEPS:,}",
        ),
        ("IR recorder", "dipole, energy, and positions after every NVE step"),
        ("Neighbor hooks", "update lists when atoms move far enough"),
        ("Safety hook", "stop on NaN or Inf; checked every 100 steps"),
    ], columns=["Hook or stage", "What it does"]),
    label="Fused dynamics and hooks",
    show_index=False,
))
""",
        ),
    )

    # Final learner-facing pass. Keep the first useful Toolkit result close to
    # the opening and reserve detailed qualifications for the cells where they
    # affect interpretation.
    replace_markdown_source(
        "title",
        notebook_hero_html(
            image_path=(
                "assets/images/banner_candidates/"
                "water-ir-v2-04-trajectory-to-spectrum.png"
            ),
            image_alt=(
                "Two water molecules linked by a hydrogen bond beside a green "
                "vibrational-signal motif on a dark computational background"
            ),
            title="From one structure to a scalable atomistic workflow",
            subtitle=(
                "Run one model, scale to batches, compose the required terms, then "
                "match the scaling API to the workload."
            ),
        )
        + "\n\n"
        + lesson_summary_html(
            do=(
                "Start with one water molecule, batch it, compose AIMNet2 with "
                "Coulomb and D3, connect SevenNet-Omni for Cu(111), switch its "
                "task, reuse charges in dynamics, then load a checked periodic "
                "box built from the NCI pair for `DomainParallel`."
            ),
            learn=(
                "AtomicData, Batch, neighbors, model composition, a custom model "
                "adapter, task selection, relaxation, fused dynamics, hooks, "
                "inflight batching, periodic PME, DomainParallel, and the "
                "DistributedPipeline layout."
            ),
            need=(
                "One CUDA GPU and the tutorial environment. The checked run took "
                "13 min 1 s of notebook wall time on one H100 PCIe; Stage 6 "
                "accounted for 9 min 27 s."
            ),
        )
        + "\n\n"
        + callout_html(
            "MD, harmonic DFT, and experiment are compared only where they report compatible quantities.",
            kind="note",
        )
        + "\n\n"
        + callout_html(
            "Follow one Toolkit data path through batching, composition, dynamics, "
            "and domain decomposition.",
            kind="check",
        ),
    )
    replace_markdown_source(
        "roadmap",
        r"""
## Notebook map

1. **Run one structure:** create `AtomicData` and `Batch`; inspect energy, forces, and charges.
2. **Scale it:** compare serial calls, GPU batches, and mixed graph sizes.
3. **Complete the model:** add Coulomb and D3; check interaction curves.
4. **Change domains:** connect SevenNet-Omni, switch tasks, and evaluate Cu(111) adsorption structures.
5. **Prepare dynamics:** relax four systems and connect NVT to NVE.
6. **Run and inspect:** save and check a trajectory; calculate qualitative IR.
7. **Choose a scaling path:** refill one GPU, exercise `DomainParallel` on one GPU without decomposition, then compare the same fixed input on 1, 2, and 4 H100s.

**Learning depth:** work directly with `AtomicData`, `Batch`, model composition,
batching, adapters, hooks, and inflight execution. PME and `DomainParallel` are
walked through live on one GPU without decomposition. Saved H100 results cover
multi-GPU `DomainParallel`; `DistributedPipeline` is an API preview with no
reported correctness or timing result.

### Checked H100 pacing

| Section | Code time on one H100 PCIe |
|---|---:|
| Setup | 23 s |
| Stage 1: one structure | 18 s |
| Stage 2: batching | 19 s |
| Stage 3: NCI calculation | 22 s |
| Stage 4: adapter and single points | 18 s |
| Stage 5: preparation and harmonic check | 1 min 13 s |
| Stage 6: trajectory and analysis | 9 min 27 s |
| Stage 7: scaling paths | 30 s |
| **Complete notebook code** | **12 min 51 s** |
| **Notebook runner wall time** | **13 min 1 s** |

These are pacing measurements from one complete checked run, not benchmark
results. Checkpoint caches were warm. Hardware, software, and cache state can
change the elapsed time.
"""
        + "\n\n"
        + callout_html(
            "Dynamics is the longest pause. Reuse the saved trajectory for plotting.",
            kind="note",
        ),
    )
    replace_markdown_source(
        "alchemi-orientation",
        r"""
## Where ALCHEMI fits

ALCHEMI connects structures, models, GPU operations, simulations, and saved results through one data model.

- **ASE** creates and edits structures.
- **Toolkit Core (`nvalchemi`)** provides `AtomicData`, `Batch`, model adapters, composition, dynamics, hooks, result storage, and distributed execution.
- **Toolkit-Ops (`nvalchemiops`)** provides accelerated neighbors, D3, periodic electrostatics, and segmented operations through supported array frameworks.
- **AIMNet2 and SevenNet-Omni** are external models connected through Toolkit interfaces.
- **PyTorch** carries the main workflow. A short comparison after the first result shows where JAX and Warp fit.
- **Packmol** created the checked Stage 7 base box offline; the notebook only
  loads it.
- **`aux/`** contains tutorial-only preparation, checking, and plotting. Its names are not Toolkit APIs.
"""
        + "\n\n"
        + process_diagram_html(
            title="One data path through the tutorial",
            steps=(
                "ASE structure",
                "AtomicData + Batch",
                "model or GPU operation",
                "dynamics + hooks",
                "local or distributed execution",
                "saved results",
            ),
            caption=(
                "The scientific example changes, but the Toolkit data path and "
                "result handling remain familiar."
            ),
        )
        + "\n\n"
        + callout_html(
            "Public Toolkit construction and workflow choices stay visible. Repeated parsing, checking, and plotting stay in aux/.",
            kind="check",
        ),
    )
    replace_markdown_source(
        "setup-heading",
        r"""
## Setup

The collapsed cells check the environment and declare the settings. The next visible cell imports the public Toolkit APIs used in the lesson.
""",
    )
    replace_markdown_source(
        "framework-primer",
        r"""
### One operation, three software layers

A **segmented sum** adds atom values separately for each structure in a batch. The example below sends the same four values through three routes:

- **PyTorch binding:** PyTorch tensors and PyTorch gradients.
- **JAX binding:** JAX arrays and JAX gradients.
- **raw Warp operation:** typed GPU arrays and explicit output storage.

PyTorch and JAX are array and automatic-differentiation front ends. Warp is
the GPU-kernel layer: a raw Warp call writes into an explicit output array,
while Toolkit-Ops bindings make the same operation feel native in PyTorch or
JAX.

Toolkit Core and the models in the rest of this notebook use PyTorch. The JAX
and raw Warp cells are a short comparison, not three competing workflow paths.
"""
        + "\n\n"
        + callout_html(
            "This tiny example checks the APIs and expected result [3, 7]. It is not a performance benchmark.",
            kind="note",
        ),
    )
    replace_markdown_source(
        "stage-3",
        stage_card_html(
            stage=3,
            total=7,
            title="Complete and check the potential",
            outcome=(
                "Evaluate 90 NCI Atlas graphs in a few passes, restore the "
                "checkpoint's Coulomb and D3 terms, and compare complete "
                "interaction curves with two references."
            ),
            state="ready",
            compute_time="22 s on one H100 PCIe in the checked run",
        )
        + "\n\n"
        + callout_html(
            "Before running: which interaction should change most when explicit electrostatics is restored?",
            kind="before",
        )
        + "\n\n"
        + r"""
**NCI** means noncovalent interaction. The examples cover neutral hydrogen bonding, dispersion-dominated binding, and an ionic hydrogen bond, with links to solvation, molecular recognition, and molecular materials.

Interaction energy is `E(AB) - E(A) - E(B)`. Three complexes × ten separations × AB/A/B form one 90-graph batch. The scale `R/Rₑ` is separation divided by the reference value, so 1.0 marks the reference geometry.

The batch stores one prescribed net charge per graph in `Batch.charge`.
AIMNet predicts geometry-dependent per-atom `charges`; their sum is checked
against that input before Coulomb is evaluated.

The checkpoint has `coulomb_mode="sr_embedded"`. Its base output is
`E_base = E_NN - E_Coulomb^SR`; the subtraction prevents double counting when
full Coulomb is added. We evaluate four combinations:

- **checkpoint base:** `E_base`, incomplete on its own;
- **base + D3:** dispersion added, full Coulomb omitted;
- **base + full Coulomb:** electrostatics restored, D3 omitted;
- **complete model:** `E_complete = E_base + E_Coulomb^full + E_D3`.
"""
        + "\n\n"
        + process_diagram_html(
            title="Ninety graphs, a few batched calls",
            steps=(
                "30 AB/A/B groups",
                "one 90-graph Batch",
                "four AIMNet + four Coulomb calls",
                "one shared D3 pass",
                "interaction curves",
            ),
            caption=(
                "Batching changes the number of calls, not the structures or "
                "the interaction-energy definition."
            ),
        ),
    )
    replace_markdown_source(
        "surface-model-switch",
        r"""
### Change the model when the chemistry changes

The AIMNet2 checkpoint cannot represent Cu, and training on molecular data does not establish accuracy for metal surfaces. That is a model limit, not a Toolkit limit.

For the next example we connect **SevenNet-Omni**, whose published domain includes materials, surfaces, and adsorption tests. Its `mpa` task targets PBE(+U) energies and forces and does not include D3, so Toolkit adds the matching pairwise D3 correction separately.

The panel contains CO, CO₂, NH₃, and CH₃OH on a periodic 3×3 Cu(111) slab. The molecules exercise different sizes and binding atoms. Toolkit still uses `AtomicData`, `Batch`, neighbors, model composition, energies, and forces. `BaseModelMixin` is Toolkit's base class for a custom model adapter; it is the extension point that changes here.
"""
        + "\n\n"
        + callout_html(
            "These are fixed starting geometries. Their single-point energy differences and forces help check the adapter, but they are not relaxed adsorption energies or a chemical ranking.",
            kind="check",
        )
        + "\n\n"
        + process_diagram_html(
            title="Keep the Toolkit workflow; change the model",
            steps=(
                "Cu(111) + molecule",
                "AtomicData + Batch",
                "SevenNet adapter",
                "SevenNet + D3",
                "energies + forces",
            ),
            caption=(
                "Five periodic structures and four finite molecules are evaluated "
                "in two batches because their boundary conditions differ."
            ),
        ),
    )
    replace_markdown_source(
        "sevennet-model-config",
        r"""
#### 1. Declare the model interface

`ModelConfig` tells Toolkit how to prepare a call and what the adapter returns.

| Choice | Meaning here |
|---|---|
| energy and forces | one energy per structure and one force vector per atom |
| periodic support | the adapter accepts both slabs and finite molecules |
| full COO neighbors | a source/target edge-index array containing both directions |
| `skin=0` | no extra distance margin is needed for these single-point calls |
| direct forces | SevenNet returns forces itself rather than asking Toolkit to differentiate energy |

The `mpa` task is passed explicitly because SevenNet-Omni ships several tasks with different training targets.
"""
        + "\n\n"
        + callout_html(
            "The adapter requests only energy and forces. It does not attach AIMNet charges to the metal system.",
            kind="note",
        ),
    )
    replace_markdown_source(
        "sevennet-input-map",
        r"""
#### 2. Translate `Batch` into one SevenNet graph

| Toolkit field | SevenNet input |
|---|---|
| atomic numbers and positions | atom types and `pos` |
| `batch_idx` and atoms per graph | graph membership |
| COO neighbor list | `edge_index` |
| periodic shifts and cells | Cartesian edge vectors |
| task name | one `mpa` label per graph |

The conversion helper handles repetitive tensor assembly. The class below keeps the reusable choices visible: supported outputs, periodic behavior, neighbor format, one external-model call, and Toolkit output names.
"""
        + "\n\n"
        + callout_html(
            "A batch may mix atom counts, but each call must use one compatible model task and neighbor convention.",
            kind="check",
        ),
    )
    replace_markdown_source(
        "sevennet-output-map",
        r"""
#### 3. Return Toolkit fields

`forward(...)` converts the complete batch, calls SevenNet once, and maps:

- `inferred_total_energy` to `energy` with shape `(structures, 1)` in eV;
- `inferred_force` to `forces` with shape `(atoms, 3)` in eV/Å.

The D3 component needs a different cutoff and neighbor layout. `PipelineModelWrapper(..., neighbor_adaptation="always")` prepares the correct neighbors for both components and adds their outputs. The exact D3 settings appear in the model-settings table below.
"""
        + "\n\n"
        + callout_html(
            "Direct component calls are used only to inspect the correction. The composed pipeline supplies the reported result.",
            kind="note",
        ),
    )
    replace_markdown_source(
        "stage-5",
        stage_card_html(
            stage=5,
            total=7,
            title="Prepare dynamics and IR",
            outcome=(
                "Return to the charge-predicting molecular model, build four "
                "systems, relax them, check harmonic frequencies, and prepare "
                "one batched dynamics workflow."
            ),
            state="ready",
            compute_time="1 min 13 s on one H100 PCIe in the checked run",
        )
        + "\n\n"
        + callout_html(
            "Before running: what changes when H is replaced by D if the coordinates and electronic model stay fixed?",
            kind="before",
        )
        + "\n\n"
        + r"""
The Cu example needed a materials model. IR needs charges at every dynamics update, so we return to the supplied AIMNet2 molecular checkpoint and its Coulomb + D3 completion.

The batch contains H₂O, D₂O, cyclic (H₂O)₆, and cyclic (D₂O)₆. H and D use the same atomic number and starting coordinates; only their masses differ. This lets one model call advance all four systems while preserving a clean isotope comparison.

The workflow now changes from independent predictions to repeated updates: compose the model, relax the structures with Toolkit's `FIRE2` geometry optimizer, assign velocities, connect constant-temperature NVT to constant-energy NVE, attach hooks, and save the trajectory.
"""
        + "\n\n"
        + process_diagram_html(
            title="Reusable Toolkit dynamics path",
            steps=(
                "complete model",
                "four-system Batch",
                "FIRE2 relaxation",
                "NVT + NVE",
                "hooks + saved state",
            ),
            caption=(
                "The water and isotope checks support the example. Toolkit owns "
                "the data, model, dynamics, hooks, and saved results."
            ),
        ),
    )
    replace_markdown_source(
        "hooks-quick-note",
        r"""
### Hooks: react while a workflow runs

A hook is code that Toolkit calls at a chosen point and frequency. It receives the current `Batch`, so it can inspect results, update workflow state, or record data without changing the integrator.

`NaN` means *not a number*; `Inf` means a value overflowed to infinity. Either makes later trajectory steps unreliable. `NaNDetectorHook` checks energy and forces, plus velocities in this workflow, and reports the failing field, step, and structure before stopping.

Other hooks here rebuild neighbors, detect convergence, count stage updates, record dipoles, write a log, and update progress. `register_hook(...)` attaches to one stage; `register_fused_hook(...)` sees the complete batch managed by `FusedStage`.
"""
        + "\n\n"
        + callout_html(
            "Relaxation checks every step. The longer dynamics run checks every 100 steps to reduce overhead.",
            kind="note",
        ),
    )
    replace_markdown_source(
        "reference-preview",
        r"""
### Reference before the long run

The figure below is a separately computed B97-3c harmonic reference. AIMNet2 was trained toward B97-3c data, so this checks how well the completed model reproduces that target for these systems; it is not an independent validation of the training domain.

Both calculations use separately optimized minima and the same displacement, projection, masses, and units. Experiment supplies selected gas-phase H₂O and D₂O band positions only. Intensities from AIMNet charges and the DFT dipole model are shown separately and are not scored against each other.

![B97-3c harmonic IR reference for H2O, D2O, cyclic (H2O)6, and cyclic (D2O)6.](attachment:b97_3c_ir_reference.png)
"""
        + "\n\n"
        + callout_html(
            "The later plot shows MD, harmonic DFT, and experimental positions separately. Compare frequency regions, not absolute intensity scales.",
            kind="note",
        ),
    )
    replace_markdown_source(
        "harmonic-intro",
        r"""
### Batch the harmonic check

Before running dynamics, we check whether the complete AIMNet + Coulomb + D3 model gives sensible water-monomer vibrational frequencies.

1. Relax one H₂O monomer with the complete model.
2. Build all 18 positive and negative Cartesian displacements in one `Batch`.
3. Request forces and charges once per displacement size.
4. Convert the force changes into frequencies and the charge changes into a predicted point-charge dipole response.
5. Reuse the same geometry and electronic response with D masses.

Three displacement sizes check numerical stability; the selected value matches the B97-3c reference calculation. The main learner-facing result is the batched model evaluation and frequency comparison. Detailed projection and mode-mapping work runs in collapsed cells.
"""
        + "\n\n"
        + callout_html(
            "B97-3c is the checkpoint's target level, while experiment supplies independent band positions. Neither comparison makes the MD and harmonic intensity scales interchangeable.",
            kind="note",
        ),
    )
    replace_markdown_source(
        "ir-mechanism",
        r"""
### Reuse one model call for dynamics and IR

The model already returns energy, forces, and charges. The dynamics uses the forces; an IR hook reduces the same charges and positions into one total dipole per structure:

`charges + positions` → `dipole over time` → `dipole change` → `spectrum`

The total cluster dipole retains correlations between molecules. Recording happens on the GPU and does not trigger another model call.

Toolkit creates a `DynamicsContext` with the current batch and step number and
passes it to each hook; you do not construct it.

**Toolkit APIs:** `FusedStage`, `Hook`, `ConvergenceHook`, `DynamicsContext`, `register_hook`, and `register_fused_hook`.

**Tutorial helpers:** `PredictedChargeIRHook`, `StageStepCounterHook`, `NotebookStageProgressHook`, and `converge_after_steps`.
"""
        + "\n\n"
        + callout_html(
            "Check that the recorder receives predicted charges from the same forward pass used by the integrator.",
            kind="check",
        ),
    )
    replace_markdown_source(
        "fused-stage-intro",
        r"""
### Connect NVT and NVE on one GPU

NVT holds temperature near a target value; NVE then follows constant-energy dynamics without a thermostat. `nvt + nve` creates a Toolkit `FusedStage`.

For each update:

1. Toolkit evaluates the model once for the complete active batch.
2. Each structure's `status` selects either the NVT or NVE update.
3. A convergence hook advances that structure when its assigned step count is complete.

Status belongs to each structure, so different structures can occupy different stages while sharing the model call. `FusedStage` uses the model attached to its first sub-stage for that shared evaluation. Here `nvt + nve` changes the update rule, not the model. The next section uses this fact to replace finished work from a larger queue.
"""
        + "\n\n"
        + process_diagram_html(
            title="One shared model evaluation",
            steps=(
                "active Batch",
                "one model call",
                "status 0: NVT",
                "status 1: NVE",
                "status 2: finished",
            ),
            caption="Each structure receives one update selected by its status.",
        )
        + "\n\n"
        + callout_html(
            "FusedStage reduces repeated work on one GPU. It is not a multi-GPU API.",
            kind="note",
        ),
    )
    replace_markdown_source(
        "stage-6",
        stage_card_html(
            stage=6,
            total=7,
            title="Run and inspect the trajectory",
            outcome=(
                "Execute the 5,000-step NVT + 20,000-step NVE calculation, save "
                "every production frame, and inspect the trajectory before "
                "calculating spectra."
            ),
            state="ready",
            compute_time="9 min 27 s on one H100 PCIe in the checked run",
        )
        + "\n\n"
        + callout_html(
            "This fixed 25,000-step workload demonstrates the complete path. It is not a trajectory-length convergence study.",
            kind="before",
        )
        + "\n\n"
        + r"""
- NVT warms the systems for 5,000 steps; those frames are not analyzed.
- NVE records 20,000 steps at 0.5 fs, giving a 10 ps production trajectory.
- Dipoles, charges, energies, and coordinates are saved before analysis.
- The following checks report temperature behavior, energy stability, and whether both water rings remain connected.
- Differentiating 20,000 saved dipoles leaves 19,999 current samples: two
  overlapping 5 ps windows fit, and the final roughly 2.5 ps does not make a
  third complete window. This is a qualitative spectrum, not converged peak
  heights.
""",
    )
    replace_markdown_source(
        "reference-note",
        r"""
### Three comparisons, three meanings

- **MD:** a finite-temperature trajectory using classical nuclei and AIMNet predicted charges.
- **Harmonic DFT:** a zero-temperature B97-3c frequency calculation with its own dipole derivatives.
- **Experiment:** selected gas-phase band positions, with no experimental intensity curve bundled here.

MD and DFT are normalized separately. One 10 ps trajectory is enough to demonstrate the workflow, but not to establish converged peak heights or uncertainty.
"""
        + "\n\n"
        + callout_html(
            "Use the final figure to compare frequency regions and isotope shifts that pass their checks. Do not calculate one combined IR error score.",
            kind="check",
        ),
    )
    replace_markdown_source(
        "inflight-intro",
        r"""
### Many small systems: refill the active batch

The queue can be larger than the active GPU `Batch`. Inflight execution
replaces finished structures between `FusedStage` updates.

The 2,048 entries repeat the eight Stage 2 dimers with copied complete-model
outputs and new velocities. This tests scheduling and collection, not 2,048
unique geometries.

- `InMemoryDataset` holds the queue; production sources can load lazily.
- `SizeAwareSampler` sets atom and structure limits only (`max_edges=None`) for
  these homogeneous six-atom dimers.
- `HostMemory` stores completed results on the CPU.
- `system_id` is stable across refills; `batch_idx` is only its current slot.

Status 0 selects NVT, 1 selects NVE, and 2 means finished.
The 256-system active limit is chosen so refills are visible; it is not a
measured GPU capacity.
`refill_frequency=1` checks for free capacity after every fused update without
changing the timestep. `run(batch=None)` asks the sampler for the initial
batch and replacements.

The result hook observes the run without changing it. It records the actual
active count and stable IDs at each refill, then checks the CPU result batch.
`NaNDetectorHook` stops this teaching run if an energy, force, or velocity
becomes non-finite; the terminal row records zero observed failures only after
every expected ID has returned exactly once.
"""
        + "\n\n"
        + process_diagram_html(
            title="Inflight batching on one GPU",
            steps=(
                "queued dataset",
                "size-aware sampler",
                "active Batch",
                "per-system status",
                "CPU results",
            ),
            caption="Finished systems leave; queued systems fill the free capacity.",
        )
        + "\n\n"
        + callout_html(
            "The 2,048-system example demonstrates scheduling and collection. Its two NVT and three NVE updates are intentionally too short for scientific MD.",
            kind="note",
        ),
    )
    replace_markdown_source(
        "distributed-pipeline-intro",
        r"""
### Optional API preview: put stages on different GPUs

`DomainParallel` divides one periodic system. `DistributedPipeline` instead
moves batches of independent systems from one workflow stage to the next.
A **rank** is one worker process, usually attached to one GPU.

```python
from nvalchemi.dynamics import DistributedPipeline, FusedStage
from nvalchemi.dynamics.base import BufferConfig

# Both ranks reserve the same maximum transfer shape.
transfer = BufferConfig(
    num_systems=max_batch_size,
    num_nodes=max_atoms_in_transfer,
    num_edges=0,
)
optimization = FusedStage(
    sub_stages=[(0, fire)],
    sampler=relaxation_sampler,
    prior_rank=None,
    next_rank=1,
    max_batch_size=max_batch_size,
    buffer_config=transfer,
    comm_mode="async_recv",
)
dynamics = FusedStage(
    sub_stages=[(0, nvt), (1, nve)],
    prior_rank=0,
    next_rank=None,
    max_batch_size=max_batch_size,
    buffer_config=transfer,
    comm_mode="async_recv",
)
pipeline = DistributedPipeline(
    stages={0: optimization, 1: dynamics},
    synchronized=False,
    backend="nccl",
    device_id=device,
)
with pipeline:
    pipeline.run()
```

`prior_rank` and `next_rank` connect adjacent workers. `BufferConfig` sets the
largest batch shape they may exchange. The receiving stage rebuilds neighbor
arrays, so this example does not transfer edges. `comm_mode="async_recv"`
allows the receiver to wait without blocking its other work.

For stage times `t1` and `t2`, ideal two-GPU throughput speedup is
`(t1 + t2) / max(t1, t2)`, at most 2×. Two independent stage pairs can at most
double that throughput again. Real speedup is lower when stages are unbalanced
or communication is significant. The prerecorded H100 plan uses 8,192 systems
with at most 512 active per pair so startup does not dominate the comparison.

Toolkit 0.2 does not yet preserve a complete atomistic `Batch` on this transfer
path. This notebook therefore shows the intended public API but does not report
pipeline correctness, overlap, or speed.
""",
    )
    replace_markdown_source(
        "interpretation",
        r"""
### What you can reuse

- Convert structures to `AtomicData`, combine them in `Batch`, and recover per-structure results.
- Choose model-aware neighbors and measure when GPU batching helps.
- Compose dependent and independent model contributions without double counting.
- Connect an external energy-and-force model through the Toolkit model interface.
- Relax and propagate several systems together, then record results through hooks.
- Keep a larger queue moving with inflight batching.
- Use `DomainParallel` for one large periodic system. This notebook walks through the public API with one domain on one GPU, then shows bundled results from three checked energy/force passes for the same 51,200-atom input on 1, 2, and 4 H100s.
- `DistributedPipeline` is the intended API when many independent systems pass
  through different stages. The Toolkit 0.2 API shape is introduced here, but its
  correctness, overlap, and speed remain `NOT REPORTED`.
"""
        + "\n\n"
        + callout_html(
            "Try it: decide whether a new workload is many independent systems "
            "or one oversized periodic system, then choose batching or domain "
            "decomposition. Prepare and check any new base box offline before "
            "using it here. Charged periodic systems and multi-stage multi-GPU "
            "execution are left for later.",
            kind="check",
        )
        + "\n\n"
        + "Next: [Part 2: batched adsorption]"
        "(../part-2-batched-adsorption-toolkit/README.md).",
    )
    replace_code_source(
        "serial-batch-agreement",
        """
aimnet.set_config("active_outputs", {"energy", "charges"})
serial_batch_progress = NotebookProgress(
    title="Compare individual and batched calls",
    total=len(scan_data) + 1,
    unit="model calls",
)
serial_energy = []
for index, data in enumerate(scan_data, start=1):
    one_graph = Batch.from_data_list([data], device=DEVICE)
    compute_neighbors(one_graph, config=aimnet.model_config.neighbor_config)
    serial_energy.append(aimnet(one_graph)["energy"].detach().reshape(()))
    serial_batch_progress.update(
        done=index, message=f"individual graph {index} of {len(scan_data)}"
    )
serial_energy = torch.stack(serial_energy)

scan_batch = Batch.from_data_list(scan_data, device=DEVICE)
compute_neighbors(scan_batch, config=aimnet.model_config.neighbor_config)
batch_outputs = aimnet(scan_batch)
batch_energy = batch_outputs["energy"].detach().reshape(-1)
serial_batch_progress.complete("one batch returned the same graph energies")

serial_batch_error = float((serial_energy - batch_energy).abs().max().cpu())
assert serial_batch_error < RESIDUAL_SERIAL_BATCH_TOLERANCE_EV
residual_triplets = batch_energy.cpu().numpy().reshape(-1, 3)
residual_interaction_eV = (
    residual_triplets[:, 0] - residual_triplets[:, 1] - residual_triplets[:, 2]
)
display(readable_table(pd.Series({
    "graphs": scan_batch.num_graphs,
    "atoms": scan_batch.num_nodes,
    "batch_idx shape": tuple(scan_batch.batch_idx.shape),
    "batch_ptr shape": tuple(scan_batch.batch_ptr.shape),
    "largest energy difference / eV": serial_batch_error,
}, name="Value").rename_axis("Result").reset_index(),
    label="Individual and batched agreement", show_index=False,
))
display(callout(
    "One model call replaced the individual loop without changing any graph energy.",
    kind="result", result_state="pass",
))
""",
    )
    insert_after(
        "serial-batch-agreement",
        code(
            "batch-graph-access",
            """
batch_access_progress = NotebookProgress(
    title="Recover structures from a Batch", total=3, unit="APIs"
)
first_graph = scan_batch.get_data(0)
batch_access_progress.advance(message="one graph selected")
roundtrip_graphs = scan_batch.to_data_list()
batch_access_progress.advance(message="complete list recovered")
dimer_only_batch = scan_batch.index_select(
    list(range(0, scan_batch.num_graphs, 3))
)
batch_access_progress.complete("dimer graphs selected as a new Batch")

assert first_graph.num_nodes == 6
assert len(roundtrip_graphs) == scan_batch.num_graphs
assert dimer_only_batch.num_graphs == len(DIMER_DISTANCES_A)
display(readable_table(
    pd.DataFrame([
        ("get_data(0)", f"{first_graph.num_nodes} atoms"),
        ("to_data_list()", f"{len(roundtrip_graphs)} graphs"),
        ("index_select(...)", f"{dimer_only_batch.num_graphs} dimers"),
    ], columns=["Batch API", "Returned"]),
    label="Three ways to recover graphs",
    show_index=False,
))
""",
        ),
    )
    replace_code_source(
        "cpu-gpu-crossover",
        """
benchmark_progress = NotebookProgress(
    title="Measure first and warm CPU/GPU calls", total=2, unit="checks"
)
del aimnet
torch.cuda.empty_cache()
aimnet = AIMNet2Wrapper.from_checkpoint(checkpoint_path, device=DEVICE)
cpu_aimnet = AIMNet2Wrapper.from_checkpoint(checkpoint_path, device="cpu")
for route_model in (aimnet, cpu_aimnet):
    route_model.eval()
    for parameter in route_model.parameters():
        parameter.requires_grad_(False)
    route_model.set_config("active_outputs", {"energy"})
benchmark_progress.advance(message="fresh CPU and GPU wrappers loaded")

timing_atoms = [make_water_dimer(2.90) for _ in range(32)]
for atoms in timing_atoms:
    atoms.info["charge"] = 0
gpu_timing_batch = build_benchmark_batch(
    timing_atoms, device=DEVICE, dtype=torch.float64,
    atomic_data_type=AtomicData, batch_type=Batch,
)
cpu_timing_batch = build_benchmark_batch(
    timing_atoms, device="cpu", dtype=torch.float64,
    atomic_data_type=AtomicData, batch_type=Batch,
)
compute_neighbors(gpu_timing_batch, config=aimnet.model_config.neighbor_config)
compute_neighbors(cpu_timing_batch, config=cpu_aimnet.model_config.neighbor_config)
cold_warm = pd.DataFrame(
    first_and_warm_call_rows(aimnet, gpu_timing_batch, warm_calls=20, route="GPU")
    + first_and_warm_call_rows(cpu_aimnet, cpu_timing_batch, warm_calls=20, route="CPU")
)
benchmark_progress.complete("first and warm 32-graph calls measured")
display(readable_table(
    cold_warm[["route", "phase", "calls", "wall_ms_per_pass", "structures_per_s"]].round(2),
    label="First and warm CPU/GPU calls", show_index=False,
))
""",
    )
    replace_code_source(
        "cpu-gpu-sweep",
        """
BENCHMARK_BATCH_SIZES = (1, 8, 32, 128)
BENCHMARK_WARMUP_CALLS = 3
BENCHMARK_PASSES_PER_REPEAT = 10
BENCHMARK_REPEATS = 5
crossover_progress = NotebookProgress(
    title="Measure the warm CPU/GPU crossover",
    total=len(BENCHMARK_BATCH_SIZES),
    unit="batch sizes",
)
crossover = benchmark_device_sweep(
    batch_sizes=BENCHMARK_BATCH_SIZES,
    structure_factory=lambda: make_water_dimer(2.90),
    atoms_info={"charge": 0},
    routes={
        "GPU": (aimnet, DEVICE, aimnet.model_config.neighbor_config),
        "CPU": (cpu_aimnet, "cpu", cpu_aimnet.model_config.neighbor_config),
    },
    dtype=torch.float64,
    atomic_data_type=AtomicData,
    batch_type=Batch,
    compute_neighbors=compute_neighbors,
    warmup_calls=BENCHMARK_WARMUP_CALLS,
    measured_calls=BENCHMARK_PASSES_PER_REPEAT,
    measured_repeats=BENCHMARK_REPEATS,
    energy_key="energy",
    energy_atol=2.0e-4,
    energy_rtol=0.0,
    on_batch_complete=lambda done, size: crossover_progress.update(
        done=done, message=f"batch size {size} measured"
    ),
)
crossover_progress.complete("four matched batch sizes complete")
""",
    )
    insert_after(
        "cpu-gpu-sweep",
        code(
            "display-cpu-gpu-sweep",
            """
cpu_gpu_display_progress = NotebookProgress(
    title="Show the CPU/GPU crossover", total=2, unit="outputs"
)
display(readable_table(
    crossover[[
        "batch_size", "route", "wall_ms_per_pass", "median_structures_per_s",
        "max_abs_energy_difference",
    ]].rename(columns={
        "batch_size": "Batch size",
        "route": "Device",
        "wall_ms_per_pass": "Time / ms",
        "median_structures_per_s": "Structures / s",
        "max_abs_energy_difference": "CPU-GPU max |ΔE| / eV",
    }).round(2),
    label="CPU/GPU crossover · five repeated warm-call blocks",
    show_index=False,
))
cpu_gpu_display_progress.advance(message="timing table shown")
crossover_figure, _ = plot_device_sweep(crossover)
display(figure_with_alt(
    crossover_figure,
    alt_text=(
        "CPU and GPU throughput versus batch size for the same water-dimer "
        "energy calculation."
    ),
))
plt.close(crossover_figure)
cpu_gpu_display_progress.complete("throughput plot shown")
display(callout(
    "Points are medians and error bars span the interquartile range. Small "
    "batches emphasize response time; larger batches expose GPU throughput. "
    "The crossover belongs to this model, workload, and hardware.",
    kind="result", result_state="observed",
))
del cpu_aimnet, cpu_timing_batch, gpu_timing_batch
""",
        ),
    )
    replace_code_source(
        "load-nci-atlas",
        """
nci_data_progress = NotebookProgress(
    title="Build the 90-graph NCI Atlas batch", total=3, unit="steps"
)
NCI_DATA_FILE = PART_DIR / "data" / "nci_atlas" / "nci-atlas-curves.csv.gz"
nci_reference_data = load_nci_atlas_subset(NCI_DATA_FILE)
nci_atoms = rows_to_atoms(nci_reference_data)
nci_graph_index = build_graph_index(nci_reference_data)
nci_data_progress.advance(message="three interaction curves loaded")

nci_data = [
    AtomicData.from_atoms(atoms, device="cpu", dtype=torch.float32)
    for atoms in nci_atoms
]
nci_batch_template = Batch.from_data_list(nci_data, device=DEVICE)
nci_batch = nci_batch_template.clone()
assert nci_batch.num_graphs == 90
nci_data_progress.advance(message=f"{nci_batch.num_nodes:,} atoms packed")
nci_data_progress.complete("30 AB/A/B groups retain their source order")
display(readable_table(
    nci_reference_data[["subset", "system_name", "interaction_class"]]
    .drop_duplicates(ignore_index=True),
    label="NCI Atlas tutorial subset", show_index=False,
))
display(callout(
    f"One Batch holds {nci_batch.num_graphs} graphs: "
    "3 systems × 10 separations × AB/A/B.",
    kind="result",
    result_state="pass",
))
""",
    )
    insert_before(
        "configure-nci-model",
        code(
            "prepare-nci-resources",
            """
nci_resource_progress = NotebookProgress(
    title="Verify the NCI model files", total=1, unit="set"
)
NCI_CHECKPOINTS = [f"aimnet2-wb97m-d3_{index}" for index in range(4)]
aimnet_checkpoint_identities = verify_checkpoint_identities(
    CHECKPOINT_IDENTITIES
)
NCI_D3_SMOOTHING_FRACTION = DOMAIN_METHODOLOGY.d3_smoothing_fraction
nci_resource_progress.complete("five AIMNet files and shared settings checked")
""",
            source_hidden=True,
        ),
    )
    replace_code_source(
        "configure-nci-model",
        """
nci_model_progress = NotebookProgress(
    title="Configure AIMNet, Coulomb, and D3", total=3, unit="components"
)
nci_aimnet = AIMNet2Wrapper.from_checkpoint(
    NCI_CHECKPOINTS[0], device=DEVICE, compile_model=False
).eval()
nci_metadata = dict(nci_aimnet.model.metadata)
assert nci_metadata["needs_coulomb"] is True
assert nci_metadata["needs_dispersion"] is True
assert nci_metadata["coulomb_mode"] == "sr_embedded"
nci_aimnet.set_config("active_outputs", {"energy", "charges"})
nci_model_progress.advance(
    message="ensemble member 0 loaded with charge output"
)

nci_d3_params = nci_metadata["d3_params"]
nci_d3 = DFTD3ModelWrapper(
    a1=nci_d3_params["a1"], a2=nci_d3_params["a2"],
    s8=nci_d3_params["s8"], s6=nci_d3_params.get("s6", 1.0),
    cutoff=D3_CUTOFF_A, smoothing_fraction=NCI_D3_SMOOTHING_FRACTION,
    param_file=D3_PARAMETER_FILE,
    # The image does not redistribute this generated parameter cache.
    # Toolkit creates it from its official source when it is absent.
    auto_download=True,
).to(DEVICE).eval()
nci_d3.set_config("active_outputs", {"energy"})
nci_model_progress.advance(
    message="checkpoint D3 settings applied"
)
nci_coulomb = DirectCoulombWrapper().to(DEVICE).eval()
nci_coulomb.set_config("active_outputs", {"energy"})
nci_model_progress.complete("three components and named checks are ready")
""",
    )
    insert_after(
        "configure-nci-model",
        code(
            "display-nci-model-settings",
            """
nci_model_display_progress = NotebookProgress(
    title="Check and show the NCI components", total=2, unit="checks"
)
D3_PARAMETER_SHA256 = sha256_file(D3_PARAMETER_FILE)
assert D3_PARAMETER_SHA256 == EXPECTED_D3_PARAMETER_SHA256
nci_model_display_progress.advance(message="D3 parameter file verified")
nci_validation_settings = nci_validation_settings_table(NCI_VALIDATION)
display(readable_table(pd.DataFrame([
    {
        "component": "AIMNet checkpoint base",
        "implementation": "Toolkit AIMNet2Wrapper",
        "depends on": "positions, elements, charge",
    },
    {
        "component": "all-pairs Coulomb",
        "implementation": "tutorial helper",
        "depends on": "predicted charges",
    },
    {
        "component": "pairwise D3(BJ)",
        "implementation": "Toolkit DFTD3ModelWrapper",
        "depends on": "positions, elements",
    },
]).rename(columns={
    "component": "Component",
    "implementation": "Implementation",
    "depends on": "Depends on",
}), label="Model components", show_index=False))
display(callout(
    "The checks require graph-charge conservation, component-sum agreement, "
    "graph-order agreement, and one independent force derivative. Exact "
    "tolerances remain available in nci_validation_settings.",
    kind="note",
))
nci_model_display_progress.complete("components and check set shown")
""",
            source_hidden=True,
        ),
    )
    replace_code_source(
        "evaluate-nci-components",
        """
nci_evaluation_progress = NotebookProgress(
    title="Evaluate the NCI set in nine batched calls", total=9, unit="calls"
)
nci_d3_batch = nci_batch_template.clone()
compute_neighbors(nci_d3_batch, config=nci_d3.model_config.neighbor_config)
with torch.no_grad():
    nci_d3_graph_eV = nci_d3(nci_d3_batch)["energy"].reshape(-1).cpu()
nci_evaluation_progress.advance(message="one shared D3 pass over 90 graphs")

nci_member_residual_eV, nci_member_coulomb_eV, nci_charge_residuals_e = [], [], []
for member_index, checkpoint in enumerate(NCI_CHECKPOINTS):
    wrapper = nci_aimnet if member_index == 0 else AIMNet2Wrapper.from_checkpoint(
        checkpoint, device=DEVICE, compile_model=False
    ).eval()
    member_batch = nci_batch_template.clone()
    wrapper.set_config("active_outputs", {"energy", "charges"})
    compute_neighbors(member_batch, config=wrapper.model_config.neighbor_config)
    with torch.no_grad():
        member_outputs = wrapper(member_batch)
    nci_evaluation_progress.advance(
        message=f"AIMNet ensemble member {member_index}"
    )
    member_batch.charges = member_outputs["charges"]
    graph_charge = segmented_sum(
        member_batch.charges,
        member_batch.batch_idx.to(torch.int32),
        member_batch.num_graphs,
    ).reshape(-1)
    torch.testing.assert_close(
        graph_charge, member_batch.charge.reshape(-1),
        atol=NCI_VALIDATION.charge_atol_e, rtol=0.0,
    )
    nci_charge_residuals_e.append(float(
        (graph_charge - member_batch.charge.reshape(-1)).abs().max().cpu()
    ))
    with torch.no_grad():
        member_coulomb = nci_coulomb(member_batch)["energy"]
    nci_evaluation_progress.advance(
        message=f"Coulomb from member {member_index} charges"
    )
    nci_member_residual_eV.append(member_outputs["energy"].reshape(-1).cpu())
    nci_member_coulomb_eV.append(member_coulomb.reshape(-1).cpu())

nci_member_residual_eV = torch.stack(nci_member_residual_eV)
nci_member_coulomb_eV = torch.stack(nci_member_coulomb_eV)
assert nci_member_residual_eV.shape == nci_member_coulomb_eV.shape == (4, 90)
nci_charge_conservation_max_abs_e = max(nci_charge_residuals_e)
nci_evaluation_progress.complete("four AIMNet, four Coulomb, and one D3 call complete")
display(callout(
    "All predicted graph charges matched their requested totals; the largest "
    f"absolute residual was {nci_charge_conservation_max_abs_e:.2e} e.",
    kind="result",
    result_state="pass",
))
""",
    )
    insert_before(
        "compose-nci-pipeline",
        markdown(
            "nci-composition-context",
            r"""
### Compose the finite-system model

AIMNet predicts atomic charges. Direct Coulomb consumes them in a `use_autograd=True` group; D3 runs independently. Toolkit can then differentiate the charge-dependent energy.

`PipelineStep` objects inside one `PipelineGroup` run in order, so an earlier
step can supply fields used by the next step. Separate groups represent
contributions that can be evaluated independently and then added.

`DirectCoulombWrapper` is a small tutorial helper around Toolkit-Ops, not a
Toolkit Core class. The Toolkit composition APIs remain visible below.

These finite gas-phase complexes use direct all-pairs Coulomb interactions. Ewald and PME are periodic methods and are not used here. The partial combinations expose omitted terms; they are checks, not separate production models.
"""
            + "\n\n"
            + callout_html(
                'neighbor_adaptation="always" prepares each component\'s neighbors.',
                kind="check",
            ),
        ),
    )
    replace_code_source(
        "compose-nci-pipeline",
        """
nci_pipeline_progress = NotebookProgress(
    title="Compose the complete model", total=2, unit="checks"
)
nci_aimnet.set_config("active_outputs", {"energy", "charges"})
nci_coulomb.set_config("active_outputs", {"energy"})
nci_d3.set_config("active_outputs", {"energy", "forces"})
nci_full_model = PipelineModelWrapper(
    groups=[
        PipelineGroup(
            steps=[PipelineStep(model=nci_aimnet), PipelineStep(model=nci_coulomb)],
            use_autograd=True,
        ),
        PipelineGroup(steps=[PipelineStep(model=nci_d3)], use_autograd=False),
    ],
    neighbor_adaptation="always",
).to(DEVICE).eval()
nci_full_model.set_config("active_outputs", {"energy", "forces"})
nci_pipeline_progress.advance(message="dependent and independent groups assembled")

nci_full_batch = nci_batch_template.clone()
compute_neighbors(nci_full_batch, config=nci_full_model.model_config.neighbor_config)
nci_full_outputs = nci_full_model(nci_full_batch)
nci_pipeline_energy_cpu = nci_full_outputs["energy"].detach().reshape(-1).cpu()
nci_member0_sum = (
    nci_member_residual_eV[0] + nci_member_coulomb_eV[0] + nci_d3_graph_eV
)
nci_component_sum_max_abs_eV = check_nci_interaction_component_sum(
    nci_graph_index,
    nci_pipeline_energy_cpu,
    nci_member0_sum,
    atol_eV=NCI_VALIDATION.interaction_energy_atol_eV,
)
nci_pipeline_progress.complete(
    "AB − A − B interaction energies match the component sum"
)
""",
    )
    insert_after(
        "compose-nci-pipeline",
        code(
            "validate-nci-graph-order",
            """
nci_order_progress = NotebookProgress(
    title="Check NCI graph ordering", total=1, unit="check"
)
reverse_order = list(range(nci_batch_template.num_graphs - 1, -1, -1))
nci_reversed_batch = nci_batch_template.index_select(reverse_order)
compute_neighbors(
    nci_reversed_batch, config=nci_full_model.model_config.neighbor_config
)
nci_reversed_energy = (
    nci_full_model(nci_reversed_batch)["energy"].detach().reshape(-1).cpu().flip(0)
)
nci_order_interactions = reduce_fragment_energies(
    nci_graph_index,
    {
        "original_order": nci_pipeline_energy_cpu,
        "reversed_order": nci_reversed_energy,
    },
)
nci_graph_order_max_abs_eV = float(
    (
        nci_order_interactions["reversed_order"]
        - nci_order_interactions["original_order"]
    ).abs().max()
)
torch.testing.assert_close(
    torch.as_tensor(
        nci_order_interactions["reversed_order"].to_numpy(copy=True)
    ),
    torch.as_tensor(
        nci_order_interactions["original_order"].to_numpy(copy=True)
    ),
    atol=NCI_VALIDATION.interaction_energy_atol_eV,
    rtol=0.0,
)
nci_order_progress.complete(
    "AB − A − B interaction energies are unchanged by graph order"
)
""",
            source_hidden=True,
        ),
    )
    replace_code_source(
        "check-nci-force",
        """
nci_force_progress = NotebookProgress(
    title="Check one complete-model force", total=3, unit="checks"
)
nci_example_index = nci_graph_index.index[
    (nci_graph_index["system_id"] == DOMAIN_METHODOLOGY.nci_system_id)
    & np.isclose(nci_graph_index["scale"], 1.0)
    & (nci_graph_index["fragment"] == "AB")
].item()
nci_example = nci_atoms[nci_example_index]
nci_example_batch = Batch.from_data_list([
    AtomicData.from_atoms(nci_example, device="cpu", dtype=torch.float32)
], device=DEVICE)
compute_neighbors(nci_example_batch, config=nci_full_model.model_config.neighbor_config)
nci_example_force = nci_full_model(nci_example_batch)["forces"].detach()
nci_force_progress.advance(message="Toolkit complete-model force evaluated")

nci_official = AIMNet2Calculator(
    str(resolve_checkpoint_path(NCI_CHECKPOINTS[0])),
    device=str(DEVICE), needs_coulomb=True, needs_dispersion=True,
    compile_model=False, train=False,
)
nci_official.set_lrcoulomb_method("simple")
nci_official.set_dftd3_cutoff(
    cutoff=D3_CUTOFF_A, smoothing_fraction=NCI_D3_SMOOTHING_FRACTION
)
nci_force_progress.advance(message="official AIMNet2 route configured")
nci_force_check = check_nci_force(
    example=nci_example,
    toolkit_forces=nci_example_force,
    official_calculator=nci_official,
    device=DEVICE,
    settings=NCI_VALIDATION,
)
nci_force_table = build_nci_force_check_table(nci_force_check, NCI_VALIDATION)
display(readable_table(
    nci_force_table.round(6), label="Independent force checks", show_index=False,
))
nci_force_progress.complete("energy derivative and Toolkit force agree")
display(callout(
    "The pipeline force agrees with the official calculator, and the official "
    "force agrees with an energy finite difference.",
    kind="result", result_state="pass",
))
del nci_official
""",
    )
    insert_before(
        "analyze-nci-curves",
        markdown(
            "nci-reference-context",
            r"""
### Compare with the reference curves

Here we switch from the opening B97-3c checkpoint to the four-member `aimnet2-wb97m-d3_*` ensemble. Its training level is ωB97M-D3/def2-TZVPP; the DFT curves use ωB97M-D3(BJ)/def2-TZVPPD. The levels are close, not identical.

The plot shows the ensemble mean and member-to-member spread: model disagreement, not calibrated uncertainty. The complete model is compared with DFT-D3 and CCSD(T)/CBS.
"""
            + "\n\n"
            + callout_html(
                "The 0.5 kcal/mol check catches missing or double-counted terms here. It is not a general accuracy guarantee.",
                kind="note",
            ),
        ),
    )
    replace_code_source(
        "analyze-nci-curves",
        """
nci_analysis_progress = NotebookProgress(
    title="Compare interaction curves with two references", total=2, unit="checks"
)
EV_TO_KCAL_MOL = 1.0 / (units.kcal / units.mol)
nci_components = {
    "core": nci_member_residual_eV,
    "core_plus_d3": nci_member_residual_eV + nci_d3_graph_eV,
    "core_plus_coulomb": nci_member_residual_eV + nci_member_coulomb_eV,
    "full": nci_member_residual_eV + nci_member_coulomb_eV + nci_d3_graph_eV,
}
nci_comparisons = {
    "checkpoint base vs CC": ("core", "ccsd_t_cbs"),
    "base + full Coulomb vs CC": ("core_plus_coulomb", "ccsd_t_cbs"),
    "complete vs CC": ("full", "ccsd_t_cbs"),
    "same-D3 bookkeeping identity": ("core_plus_coulomb", "dft_no_d3"),
    "complete vs DFT-D3": ("full", "dft_full"),
    "DFT-D3 vs CC": ("dft_full", "ccsd_t_cbs"),
}
nci_member_curves, nci_curves, nci_metrics = assemble_nci_comparison_curves(
    nci_graph_index,
    nci_reference_data,
    nci_components,
    d3_graph_energies_eV=nci_d3_graph_eV,
    dft_total_energy_column="wb97m_d3bj_def2_tzvppd_total_energy_kcal_mol",
    cc_interaction_energy_column="ccsd_t_cbs_interaction_energy_kcal_mol",
    comparisons=nci_comparisons,
    energy_to_kcal_mol=EV_TO_KCAL_MOL,
)
nci_analysis_progress.advance(message="AB - A - B applied to every curve")
assert (nci_metrics["complete vs CC"] < NCI_COMPLETE_MAE_LIMIT_KCAL_MOL).all()
assert (nci_metrics["complete vs DFT-D3"] < NCI_COMPLETE_MAE_LIMIT_KCAL_MOL).all()
nci_analysis_progress.complete("complete-model errors checked")
""",
    )
    insert_after(
        "analyze-nci-curves",
        code(
            "display-nci-curves",
            """
nci_display_progress = NotebookProgress(
    title="Show the NCI comparison", total=3, unit="outputs"
)
nci_figure, _ = plot_nci_interaction_curves(nci_curves)
nci_figure.savefig(OUTPUT_DIR / "nci_interaction_curves.png", dpi=180, bbox_inches="tight")
display(figure_with_alt(
    nci_figure,
    alt_text=(
        "Interaction-energy curves for neutral hydrogen bonding, dispersion, "
        "and an ionic hydrogen bond, with model components and two references."
    ),
))
plt.close(nci_figure)
nci_display_progress.advance(message="interaction curves shown")
display(readable_table(
    nci_metrics.round(3).reset_index(),
    label="Mean absolute interaction-energy errors / kcal mol⁻¹", show_index=False,
))
nci_display_progress.advance(message="reference error table shown")
nci_execution_checks = pd.DataFrame([
    ("graph charge residual / e", nci_charge_conservation_max_abs_e),
    ("interaction: pipeline minus component sum / eV", nci_component_sum_max_abs_eV),
    ("interaction: reversed-order difference / eV", nci_graph_order_max_abs_eV),
], columns=["Check", "Maximum absolute difference"])
display(readable_table(
    nci_execution_checks, label="Batch and composition checks", show_index=False,
))
nci_display_progress.complete("batch, composition, and reference checks shown")
display(callout(
    "The complete model is within 0.5 kcal/mol MAE of both references for "
    "these three systems. The no-D3 DFT column subtracts the model's D3 term "
    "from full DFT-D3, so its error must equal the full comparison. It checks "
    "bookkeeping, not accuracy. The independent CCSD(T)/CBS comparison shows "
    "how Coulomb and D3 change the curves. This is not broad MLIP validation.",
    kind="result", result_state="observed",
))
""",
        ),
    )
    replace_code_source(
        "reference-preflight",
        """
reference_progress = NotebookProgress(
    title="Load the DFT and experimental references", total=2, unit="sources"
)
reference_dirs = {"H2O": "h2o", "D2O": "d2o", "(H2O)6": "h6", "(D2O)6": "d6"}
references = {
    label: load_psi4_b973c_ir_artifact(REFERENCE_ROOT / directory)
    for label, directory in reference_dirs.items()
}
reference_progress.advance(message="B97-3c harmonic results loaded and checked")
experimental_fundamentals = load_experimental_water_fundamentals()
experimental_data_sha256 = sha256_file(
    PART_DIR / "reference" / "experimental_water_fundamentals"
    / "water_gas_phase_fundamentals.csv"
)
experimental_artifact_id = f"experimental-water-fundamentals-{experimental_data_sha256[:16]}"
reference_display = prepare_monomer_reference_display(
    references, experimental_fundamentals
)
observed_by_mode = reference_display.observed_by_mode
harmonic_mode_indices = reference_display.harmonic_mode_indices
reference_progress.complete("observed H2O and D2O positions matched to DFT modes")

display(readable_table(
    pd.DataFrame([
        ("DFT engine", references["H2O"].engine_version),
        ("DFT method", references["H2O"].manifest["model_chemistry"]),
        ("DFT systems", ", ".join(reference_dirs)),
        ("Observed gas-phase markers", len(experimental_fundamentals)),
        ("Experimental data ID", experimental_artifact_id),
    ], columns=["Reference", "Value"]),
    label="IR reference data",
    show_index=False,
))
display(readable_table(
    reference_display.table,
    label="B97-3c harmonic modes and observed positions", show_index=False,
))
""",
    )
    replace_code_source(
        "mode-mapping",
        """
mode_mapping_progress = NotebookProgress(
    title="Map H2O modes to D2O modes", total=2, unit="checks"
)
mode_mapping = h_to_d_mode_mapping_table(
    references,
    coarse_mass_path_steps=H_TO_D_COARSE_MASS_PATH_STEPS,
    fine_mass_path_steps=H_TO_D_FINE_MASS_PATH_STEPS,
    degeneracy_tolerance_cm1=H_TO_D_DEGENERACY_TOLERANCE_CM1,
    covalent_oh_cutoff_angstrom=COVALENT_OH_CUTOFF_A,
    h_acceptor_cutoff_angstrom=HBOND_H_ACCEPTOR_CUTOFF_A,
    oo_cutoff_angstrom=HBOND_OO_CUTOFF_A,
    hbond_angle_cutoff_deg=HBOND_ANGLE_CUTOFF_DEG,
)
mode_mapping_progress.advance(message="coarse and fine isotope-mass paths agree")
mode_map_table = mode_mapping.table
display(readable_table(
    monomer_mode_mapping_display_table(mode_mapping),
    label="H2O to D2O monomer mode mapping", show_index=False,
))
mode_mapping_progress.complete("three monomer assignments shown")
display(callout(
    "The complete monomer and hexamer mapping is saved for inspection. A good "
    "harmonic mode match does not override a failed MD topology check.",
    kind="note",
))
""",
    )

    # Output metadata is grouped by the stage that produced it. The helper
    # validates every required key and flattens these sections only when the
    # final run file is written.
    cells.extend(
        [
            code(
                "manifest-run-details",
                """
run_details_progress = NotebookProgress(
    title="Collect run details", total=1, unit="group"
)
manifest_run_details = {
    "runtime": {
        "run_id": RUN_ID,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(DEVICE),
        "torch": torch.__version__,
        "aimnet": metadata.version("aimnet"),
        "sevennet": metadata.version("sevenn"),
        "toolkit_core_commit": installed_pins["Core"],
        "toolkit_ops_commit": installed_pins["Ops"],
    },
    "aimnet_checkpoint": {
        "checkpoint_source": MODEL_CHECKPOINT,
        "checkpoint_sha256": model_card["checkpoint_sha256"],
        "checkpoint_override": checkpoint_is_override,
    },
    "nci_data": {
        "nci_checkpoints": NCI_CHECKPOINTS,
        "aimnet_checkpoint_identities": aimnet_checkpoint_identities,
        "nci_subset_sha256": sha256_file(NCI_DATA_FILE),
    },
    "sevennet_checkpoint": {
        "sevennet_checkpoint_source": SEVENNET_CHECKPOINT_URL,
        "sevennet_checkpoint_sha256": sevennet_checkpoint_sha256,
        "sevennet_checkpoint_doi": SEVENNET_CHECKPOINT_DOI,
        "sevennet_task": SEVENNET_MODALITY,
        "sevennet_reference_method": SEVENNET_REFERENCE_METHOD,
    },
    "source_files": {
        "adsorption_structure_manifest_sha256": sha256_file(
            DEFAULT_DATA_DIR / "manifest.json"
        ),
        "d3_parameter_file_sha256": D3_PARAMETER_SHA256,
        "notebook_sha256": sha256_file(PART_DIR / "alchemi-water-ir.ipynb"),
        "dimer_reference_manifest_sha256": sha256_file(
            REFERENCE_ROOT / "water_dimer_b97_3c" / "manifest.json"
        ),
    },
    "ir_references": {
        "harmonic_reference_sha256s": {
            label: sha256_file(REFERENCE_ROOT / directory / "manifest.json")
            for label, directory in reference_dirs.items()
        },
        "aimnet_harmonic_archive": {
            "path": harmonic_archive_path.name,
            "sha256": harmonic_archive_sha256,
        },
        "experimental_reference_bundle": {
            "artifact_id": experimental_artifact_id,
            "manifest_sha256": sha256_file(
                PART_DIR / "reference" / "experimental_water_fundamentals"
                / "manifest.json"
            ),
            "data_sha256": experimental_data_sha256,
            "checksum_index_sha256": sha256_file(
                PART_DIR / "reference" / "experimental_water_fundamentals"
                / "SHA256SUMS"
            ),
        },
    },
    "distributed_run": {
        "pipeline_campaign_bundle": None,
        "domain_decomposition_bundle": domain_view.bundle_record,
    },
}
run_details_progress.complete("Run details collected")
""",
                source_hidden=True,
            ),
            code(
                "manifest-model-settings",
                """
model_settings_progress = NotebookProgress(
    title="Collect model settings", total=1, unit="group"
)
manifest_model_settings = {
    "nci_model": {
        "model": (
            "AIMNet checkpoint base + predicted-charge all-pairs Coulomb "
            "+ pairwise D3(BJ)"
        ),
        "nci_graphs": len(nci_graph_index),
        "nci_interaction_geometries": len(nci_curves),
        "nci_reference_levels": [
            "ωB97M-D3(BJ)/def2-TZVPPD",
            "CCSD(T)/CBS interaction energies",
        ],
        "nci_validation": NCI_VALIDATION.as_record(),
    },
    "sevennet_adapter": {
        "custom_adapter_model": SEVENNET_MODEL_NAME,
        "custom_adapter_task": SEVENNET_MODALITY,
        "custom_adapter_scope": (
            "fixed-geometry 2D-periodic Cu(111) and finite molecular "
            "energy/force single points"
        ),
        "custom_adapter_precision": "float32",
        "custom_adapter_compile": False,
        "custom_adapter_energy_repeat_tolerance_eV_per_atom": (
            SEVENNET_REPEAT_ENERGY_TOL_EV_PER_ATOM
        ),
        "custom_adapter_force_repeat_tolerance_eV_A": (
            SEVENNET_REPEAT_FORCE_TOL_EV_A
        ),
        "custom_adapter_geometry_status": (
            "ASE-generated initial placements; not model-relaxed"
        ),
    },
    "surface_dispersion": {
        "surface_d3_cutoff_A": SURFACE_D3_CUTOFF_A,
        "surface_d3_cutoff_bohr": D3_REFERENCE_CUTOFF_BOHR,
        "surface_d3_smoothing_fraction": D3_REFERENCE_SMOOTHING_FRACTION,
        "surface_d3_parameters": {
            "a1": PBE_D3_BJ_A1,
            "a2_bohr": PBE_D3_BJ_A2_BOHR,
            "s6": PBE_D3_BJ_S6,
            "s8": PBE_D3_BJ_S8,
        },
    },
    "model_composition": {
        "electrostatics": "simple nonperiodic all-pairs 1/r; no cutoff",
        "d3_cutoff_A": D3_CUTOFF_A,
        "d3_parameters": d3_params,
        "compile_mode": "default Torch compile on the fixed 42-atom IR batch",
        "residual_serial_batch_tolerance_eV": (
            RESIDUAL_SERIAL_BATCH_TOLERANCE_EV
        ),
        "full_serial_batch_tolerance_eV": FULL_SERIAL_BATCH_TOLERANCE_EV,
        "component_closure_tolerance_eV": COMPONENT_CLOSURE_TOLERANCE_EV,
        "compiled_eager_energy_tolerance_eV": (
            COMPILED_EAGER_ENERGY_TOLERANCE_EV
        ),
        "compiled_eager_force_tolerance_eV_A": (
            COMPILED_EAGER_FORCE_TOLERANCE_EV_A
        ),
        "compiled_eager_charge_tolerance_e": COMPILED_EAGER_CHARGE_TOLERANCE_E,
        "compiled_repeat_energy_tolerance_eV": (
            COMPILED_REPEAT_ENERGY_TOLERANCE_EV
        ),
        "compiled_repeat_force_tolerance_eV_A": (
            COMPILED_REPEAT_FORCE_TOLERANCE_EV_A
        ),
        "compiled_repeat_charge_tolerance_e": (
            COMPILED_REPEAT_CHARGE_TOLERANCE_E
        ),
    },
}
model_settings_progress.complete("Model settings collected")
""",
                source_hidden=True,
            ),
            code(
                "manifest-workflow-settings",
                """
workflow_settings_progress = NotebookProgress(
    title="Collect workflow settings", total=1, unit="group"
)
manifest_workflow_settings = {
    "dynamics": {
        "neighbor_skin_A": NEIGHBOR_SKIN_A,
        "fire_initial_dt": IR_FIRE_INITIAL_DT,
        "temperature_K": TEMPERATURE_K,
        "dt_fs": DT_FS,
        "warmup_steps": WARMUP_STEPS,
        "production_steps": PRODUCTION_STEPS,
        "warmup_status": IR_WARMUP_STATUS,
        "production_status": IR_PRODUCTION_STATUS,
        "initial_velocity_random_seeds": IR_INITIAL_VELOCITY_RANDOM_SEEDS,
        "nvt_friction_per_fs": IR_NVT_FRICTION_PER_FS,
        "nvt_random_seed": IR_NVT_RANDOM_SEED,
        "capture_charge_tolerance_e": IR_CAPTURE_CHARGE_TOLERANCE_E,
        "charge_neutrality_tolerance_e": IR_CHARGE_NEUTRALITY_TOLERANCE_E,
        "dipole_origin_tolerance_e_A": IR_DIPOLE_ORIGIN_TOLERANCE_E_ANGSTROM,
        "mass_only_position_rtol": MASS_ONLY_POSITION_RTOL,
        "mass_only_position_atol_A": MASS_ONLY_POSITION_ATOL_A,
        "mass_only_energy_tolerance_eV": MASS_ONLY_ENERGY_TOLERANCE_EV,
        "mass_only_force_tolerance_eV_A": MASS_ONLY_FORCE_TOLERANCE_EV_A,
        "mass_only_charge_tolerance_e": MASS_ONLY_CHARGE_TOLERANCE_E,
    },
    "spectrum": {
        "spectrum_segment_time_fs": IR_WELCH_SEGMENT_TIME_FS,
        "spectrum_overlap": IR_WELCH_OVERLAP_FRACTION,
        "spectrum_windows_cm1": OH_REGION_WINDOWS_CM1,
    },
    "isotope_analysis": {
        "pair_temperature_relative_tolerance": (
            PAIR_TEMPERATURE_RELATIVE_TOLERANCE
        ),
        "h_to_d_coarse_mass_path_steps": H_TO_D_COARSE_MASS_PATH_STEPS,
        "h_to_d_fine_mass_path_steps": H_TO_D_FINE_MASS_PATH_STEPS,
        "h_to_d_degeneracy_tolerance_cm1": H_TO_D_DEGENERACY_TOLERANCE_CM1,
    },
    "topology": {
        "oxygen_connectivity_cutoff_A": OXYGEN_CONNECTIVITY_CUTOFF_A,
        "covalent_OH_cutoff_A": COVALENT_OH_CUTOFF_A,
        "hbond_H_acceptor_cutoff_A": HBOND_H_ACCEPTOR_CUTOFF_A,
        "hbond_OO_cutoff_A": HBOND_OO_CUTOFF_A,
        "hbond_angle_cutoff_deg": HBOND_ANGLE_CUTOFF_DEG,
        "energy_excursion_advisory_meV_atom": (
            ENERGY_EXCURSION_ADVISORY_MEV_PER_ATOM
        ),
    },
    "harmonic": {
        "harmonic_fmax_eV_A": HARMONIC_FMAX_EV_A,
        "harmonic_fire_initial_dt": HARMONIC_FIRE_INITIAL_DT,
        "harmonic_displacement_steps_bohr": (
            HARMONIC_DISPLACEMENT_STEPS_BOHR.tolist()
        ),
        "harmonic_selected_step_bohr": HARMONIC_SELECTED_STEP_BOHR,
        "harmonic_frequency_step_tolerance_cm1": (
            HARMONIC_FREQUENCY_STEP_TOLERANCE_CM1
        ),
        "harmonic_intensity_step_abs_tolerance_km_mol": (
            HARMONIC_INTENSITY_STEP_ABS_TOLERANCE_KM_MOL
        ),
        "harmonic_intensity_step_relative_tolerance": (
            HARMONIC_INTENSITY_STEP_REL_TOLERANCE
        ),
        "harmonic_mode_overlap_min": HARMONIC_MODE_OVERLAP_MIN,
        "harmonic_hessian_antisymmetry_relative_max": (
            HARMONIC_HESSIAN_ANTISYMMETRY_REL_MAX
        ),
        "harmonic_charge_neutrality_tolerance_e": (
            HARMONIC_CHARGE_NEUTRALITY_TOLERANCE_E
        ),
        "harmonic_imaginary_floor_cm1": HARMONIC_IMAGINARY_FLOOR_CM1,
    },
    "scaling": {
        "domain_methodology": {
            "source": DOMAIN_METHODOLOGY.as_record(),
            "resolved_values": {
                **DOMAIN_METHODOLOGY.resolved_values(json_compatible=True),
                "pme_alpha_a_inv": PME_ALPHA_A_INV,
                "pme_mesh_dimensions": list(PME_MESH_DIMENSIONS),
                "pme_mesh_spacing_a": list(PME_MESH_SPACING_A),
                "derived_domain_cutoff_a": domain_cutoff_a,
            },
        },
        "inflight_systems": INFLIGHT_SYSTEMS,
        "inflight_active_systems": INFLIGHT_ACTIVE_SYSTEMS,
        "inflight_nvt_steps": INFLIGHT_NVT_STEPS,
        "inflight_nve_steps": INFLIGHT_NVE_STEPS,
        "domain_live_molecules_per_species": (
            DOMAIN_METHODOLOGY.live_molecules_per_species
        ),
        "domain_construction_density_g_cm3": (
            DOMAIN_METHODOLOGY.construction_density_g_cm3
        ),
        "domain_packmol_tolerance_a": DOMAIN_METHODOLOGY.packmol_tolerance_a,
        "domain_packmol_precision_a": DOMAIN_METHODOLOGY.packmol_precision_a,
        "domain_packmol_seed": DOMAIN_METHODOLOGY.packmol_seed,
        "domain_pme_realspace_cutoff_a": (
            DOMAIN_METHODOLOGY.pme_realspace_cutoff_a
        ),
        "domain_pme_mesh_safety_factor": (
            DOMAIN_METHODOLOGY.pme_mesh_safety_factor
        ),
        "domain_pme_alpha_a_inv": PME_ALPHA_A_INV,
        "domain_pme_mesh_dimensions": PME_MESH_DIMENSIONS,
        "domain_pme_mesh_spacing_a": PME_MESH_SPACING_A,
        "domain_pme_accuracy": DOMAIN_METHODOLOGY.pme_accuracy,
        "domain_ewald_reference_accuracy": (
            DOMAIN_METHODOLOGY.ewald_reference_accuracy
        ),
        "domain_halo_skin_a": DOMAIN_METHODOLOGY.domain_halo_skin_a,
        "domain_model_cutoff_a": domain_cutoff_a,
        "domain_compile": domain_config.compile,
    },
}
workflow_settings_progress.complete("Dynamics, analysis, and scaling settings collected")
""",
                source_hidden=True,
            ),
            code(
                "manifest-checks",
                """
checks_progress = NotebookProgress(
    title="Collect numerical checks", total=1, unit="group"
)
manifest_checks = {
    "model_composition": {
        "residual_serial_batch_max_abs_eV": serial_batch_error,
        "full_serial_batch_max_abs_eV": full_pipeline_agreement_error,
        "component_closure_max_abs_eV": component_closure_error,
        "official_calculator_agreement": agreement_errors,
        "analytic_coulomb": analytic_coulomb_errors,
        "compiled_ir_eager_agreement": compiled_ir_eager_agreement,
        "compiled_ir_repeat_agreement": compiled_ir_repeat_agreement,
        "finite_difference_force_energy_route": COMPOSITION_FD_ENERGY_ROUTE,
        "finite_difference_force_step_A": FD_STEP_A,
        "finite_difference_force_reference_eV_A": fd_force,
        "finite_difference_force_official_analytic_eV_A": official_force,
        "finite_difference_force_official_abs_error_eV_A": (
            official_fd_force_error
        ),
        "finite_difference_force_pipeline_eV_A": model_force,
        "finite_difference_force_pipeline_abs_error_eV_A": fd_force_error,
    },
    "nci": {
        "graph_charge_conservation_max_abs_e": (
            nci_charge_conservation_max_abs_e
        ),
        "component_sum_max_abs_eV": nci_component_sum_max_abs_eV,
        "graph_order_max_abs_eV": nci_graph_order_max_abs_eV,
        "nci_complete_max_MAE_vs_DFT_D3_kcal_mol": float(
            nci_metrics["complete vs DFT-D3"].max()
        ),
        "nci_complete_max_MAE_vs_CCSD_T_CBS_kcal_mol": float(
            nci_metrics["complete vs CC"].max()
        ),
        "nci_force_check": nci_force_check_record(
            nci_force_check, NCI_VALIDATION
        ),
    },
    "sevennet": {
        "sevennet_adapter": {
            "graph_mapping_passed": sevennet_graph_mapping_passed,
            "structures": int(len(adsorption_structures)),
            "batches": 2,
            "finite_outputs": True,
            "numerical_max_abs_energy_eV_per_atom": (
                sevennet_repeat_max_energy_difference_eV_per_atom
            ),
            "numerical_max_abs_forces_eV_A": (
                sevennet_repeat_max_force_difference_eV_A
            ),
            "max_combined_fmax_eV_A": sevennet_max_force_eV_A,
            "geometry_status": "ASE-generated initial placements; not model-relaxed",
            "molecules": list(ADSORBATES),
            "periodic_pbc": [True, True, False],
        },
    },
    "dynamics": {
        "cluster_integrity_passed": cluster_intact,
        "initial_ring_persisted_all_frames": cluster_dft_comparison_valid,
        "energy_excursion_within_advisory": energy_within_advisory,
        "reported_comparisons": comparisons["reported"].to_dict(),
        "fused_stage_route_counts": stage_counts,
    },
    "harmonic": {
        "harmonic_checks": harmonic_validation,
        "harmonic_comparison_reported": harmonic_comparison_reported,
        "harmonic_final_fmax_eV_A": harmonic_fmax_eV_A,
        "harmonic_frequency_MAE_vs_B97_3c_cm1": harmonic_frequency_mae_cm1,
        "harmonic_selected_Hessian_antisymmetry_relative": (
            selected_harmonic_estimate.hessian.max_relative_antisymmetry
        ),
        "harmonic_final_frequency_step_change_cm1": {
            "H2O": float(harmonic_h_convergence.frequency_max_abs_change_cm1[-1]),
            "D2O": float(harmonic_d_convergence.frequency_max_abs_change_cm1[-1]),
        },
        "harmonic_final_intensity_step_change_km_mol": {
            "H2O": float(
                harmonic_h_convergence.ir_intensity_max_abs_change_km_mol[-1]
            ),
            "D2O": float(
                harmonic_d_convergence.ir_intensity_max_abs_change_km_mol[-1]
            ),
        },
    },
    "scaling": {
        "inflight_completed_systems": INFLIGHT_SYSTEMS,
        "inflight_unique_system_ids": int(unique_ids.numel()),
        "inflight_duplicate_system_ids": int((counts > 1).sum()),
        "inflight_nvt_counts_correct": inflight_nvt_counts_correct,
        "inflight_nve_counts_correct": inflight_nve_counts_correct,
        "domain_world_size": 1,
        "domain_spatially_decomposed": False,
        "domain_atom_count": int(domain_result.num_nodes),
        "domain_energy_eV": domain_energy_ev,
        "domain_force_max_eV_A": domain_fmax_ev_a,
        "domain_charge_dtype": domain_charge_dtype,
        "domain_charge_target_e": domain_charge_target_e,
        "domain_charge_sum_e": domain_charge_sum,
        "domain_charge_residual_e": domain_charge_residual_e,
        "domain_charge_abs_residual_per_atom": (
            domain_charge_abs_residual_per_atom
        ),
        "domain_charge_finite": domain_charge_finite,
        "domain_elapsed_s": domain_elapsed_s,
        "domain_peak_memory_GB": domain_peak_memory_gb,
        "domain_recorded_results_available": domain_view.available,
        "domain_recorded_successful_cases": domain_view.successful_case_count,
        "domain_recorded_failed_cases": domain_view.failed_case_count,
        "pipeline_recorded_results_available": False,
    },
}
checks_progress.complete("Numerical checks collected")
""",
                source_hidden=True,
            ),
        ]
    )

    insert_after(
        "run-inflight-example",
        markdown(
            "domain-decomposition-intro",
            r"""
### One periodic system: prepare spatial decomposition

Reuse the Stage 3 phenol and N-methylacetamide molecules as independent species
in a checked 3,200-atom periodic box with an input total-charge target of zero
and 128 rigid copies of each. Packmol placed them once; it does not run here,
and the box is neither relaxed nor equilibrated.

The static OVITO render does not alter the data: loading it does not start
OVITO or change coordinates; integer supercells preserve composition and
construction density without giant live packing.

This changes many independent graphs into one crowded periodic graph. The NCI
curves do not validate the dense mixture; its 1:1 composition and 1.0 g cm⁻³
construction density are teaching inputs, not a liquid-property calculation.

The periodic model uses AIMNet2-predicted charges with particle mesh Ewald
(PME), the checkpoint base, and tapered D3(BJ).

**Domain decomposition** gives each GPU a spatial region. A **halo** copies
nearby atoms across region edges so boundary interactions are retained. On
one GPU, `DomainParallel` passes the box through without splitting it.
"""
            + "\n\n"
            + process_diagram_html(
                title="From pairwise interactions to a crowded molecular environment",
                steps=(
                    "checked box → AtomicData + Batch",
                    "partition → run → gather",
                    "saved multi-GPU checks",
                ),
                caption=(
                    "The model on each rank evaluates "
                    "E_composed = E_base + E_PME(q(R)) + E_D3."
                ),
            )
            + "\n\n"
            + callout_html(
                "Periodic support does not establish AIMNet2 accuracy for this "
                "dense mixture. This is an API check, not a material-property "
                "result.",
                kind="note",
            ),
        ),
    )
    insert_after(
        "domain-decomposition-intro",
        code(
            "build-domain-box",
            """
PREBUILT_DOMAIN_BOX_DIR = (
    PART_DIR / "data" / "domain_decomposition" / "prebuilt_base_box"
)
domain_box_progress = NotebookProgress(
    title="Load the checked periodic base box", total=2, unit="checks"
)

domain_box = load_prebuilt_domain_box(
    PREBUILT_DOMAIN_BOX_DIR,
    nci_reference_data,
)
domain_plan = domain_box.plan
domain_atoms = domain_box.atoms
domain_box_progress.advance(message="coordinates, metadata, and checksums verified")

domain_box_details = box_summary_table(domain_plan, domain_atoms)
display(readable_table(
    compact_box_summary_table(domain_box_details),
    label="Checked periodic starting box", show_index=False,
))
preview_figure, preview_axis = plt.subplots(figsize=(8.0, 4.5))
preview_axis.imshow(plt.imread(domain_box.preview_path))
preview_axis.set_axis_off()
display(figure_with_alt(
    preview_figure,
    alt_text=(
        "Static OVITO rendering of the checked periodic base box containing "
        "separate phenol and N-methylacetamide molecules."
    ),
))
plt.close(preview_figure)
domain_box_progress.complete(f"{len(domain_atoms):,} atoms loaded; preview verified")
""",
        ),
    )
    insert_after(
        "build-domain-box",
        code(
            "convert-domain-box",
            """
domain_batch_progress = NotebookProgress(
    title="Convert the box to one Toolkit graph", total=2, unit="steps"
)
domain_data = AtomicData.from_atoms(
    domain_atoms, device="cpu", dtype=torch.float32
)
# These fields complete the input schema. The model replaces the energy and
# force zeros during evaluation. Velocities stay zero and are unused here.
domain_data.add_system_property("energy", torch.zeros(1, 1))
domain_data.add_node_property("forces", torch.zeros(len(domain_atoms), 3))
domain_data.add_node_property("velocities", torch.zeros(len(domain_atoms), 3))
domain_batch_progress.advance(message="AtomicData fields prepared")

# The packing helper writes the input total-charge target to Atoms.info["charge"].
# AtomicData carries that zero target into Batch.charge.
domain_batch = Batch.from_data_list([domain_data], device=DEVICE)
assert domain_batch.num_graphs == 1
assert float(domain_batch.charge.item()) == 0.0
domain_batch_progress.complete(f"one Batch graph with {domain_batch.num_nodes:,} atoms")
display(callout(
    f"Batch contains {domain_batch.num_graphs} graph, "
    f"{domain_batch.num_nodes:,} atoms, and a "
    f"{float(domain_batch.charge.item()):g} e total-charge target.",
    kind="result",
    result_state="pass",
))
""",
        ),
    )
    insert_after(
        "convert-domain-box",
        markdown(
            "domain-model-intro",
            r"""
### Compose the periodic model

Reuse `aimnet2-wb97m-d3_0` and its checkpoint D3(BJ) parameters:

`E_base = E_NN - E_Coulomb^SR`

`E_composed = E_base + E_PME(q(R)) + E_D3`

- The checkpoint's embedded `SRCoulomb` module subtracts short-range Coulomb.
  Adding full PME gives `E_NN + E_Coulomb^LR` without double counting.
- `Batch.charge = 0` supplies the model's total-charge target.
  `AIMNet2Wrapper` returns float32 atomic charges. Its internal correction
  reduces in float32, so re-summing a large system in float64 can expose a
  small residual. Toolkit passes those charges to `PMEModelWrapper` unchanged.
- `hybrid_forces=True` supplies fixed-charge PME forces. The shared
  `use_autograd=True` group adds the response through the predicted charges.
- `DFTD3ModelWrapper` adds D3 once and returns its forces directly.
- `estimate_pme_parameters` uses the fixed real-space cutoff to derive a
  consistent splitting parameter and mesh. `DomainParallel` then rebuilds
  neighbors for each GPU region.

Periodic support does not establish accuracy for this mixture, even though the
composition includes every declared energy term.
""",
        ),
    )
    insert_after(
        "domain-model-intro",
        code(
            "configure-domain-pme",
            """
domain_pme_progress = NotebookProgress(
    title="Choose the periodic electrostatics grid", total=1, unit="model"
)
pme_parameters = estimate_pme_parameters(
    domain_batch.positions,
    domain_batch.cell,
    batch_idx=domain_batch.batch_idx,
    accuracy=DOMAIN_METHODOLOGY.pme_accuracy,
    real_space_cutoff=DOMAIN_METHODOLOGY.pme_realspace_cutoff_a,
    mesh_safety_factor=DOMAIN_METHODOLOGY.pme_mesh_safety_factor,
)
PME_ALPHA_A_INV = float(pme_parameters.alpha.item())
PME_MESH_DIMENSIONS = tuple(pme_parameters.mesh_dimensions)
PME_MESH_SPACING_A = tuple(
    float(value) for value in pme_parameters.mesh_spacing[0].detach().cpu()
)
periodic_pme = PMEModelWrapper(
    cutoff=DOMAIN_METHODOLOGY.pme_realspace_cutoff_a,
    mesh_dimensions=PME_MESH_DIMENSIONS,
    spline_order=DOMAIN_METHODOLOGY.pme_spline_order,
    alpha=PME_ALPHA_A_INV,
    accuracy=DOMAIN_METHODOLOGY.pme_accuracy,
    hybrid_forces=True,  # direct fixed-charge PME force
).to(DEVICE).eval()
periodic_pme.model_config.neighbor_config.skin = NEIGHBOR_SKIN_A
domain_pme_progress.complete("cutoff, splitting parameter, and mesh coupled")
""",
        ),
    )
    insert_after(
        "configure-domain-pme",
        code(
            "compose-domain-model",
            """
domain_model_progress = NotebookProgress(
    title="Compose AIMNet2, PME, and D3", total=2, unit="groups"
)
nci_aimnet.set_config("active_outputs", {"energy", "charges"})
nci_d3.set_config("active_outputs", {"energy", "forces"})
# Freeze checkpoint weights while retaining derivatives with respect to positions.
nci_aimnet.requires_grad_(False)

periodic_model = PipelineModelWrapper(
    groups=[
        PipelineGroup(
            steps=[PipelineStep(model=nci_aimnet), PipelineStep(model=periodic_pme)],
            use_autograd=True,  # add the response of predicted charges to R
        ),
        PipelineGroup(steps=[PipelineStep(model=nci_d3)], use_autograd=False),
    ],
    # Component hooks build their own lists on one GPU; DomainParallel builds
    # rank-local lists for the multi-GPU path.
    neighbor_adaptation="never",
).to(DEVICE).eval()
domain_model_progress.advance(message="AIMNet2 charges connected to PME")
periodic_model.set_config("active_outputs", {"energy", "forces", "charges"})
domain_model_progress.complete("dependent electrostatics and independent D3 connected")
""",
        ),
    )
    insert_after(
        "compose-domain-model",
        code(
            "configure-domain-parallel",
            """
domain_config_progress = NotebookProgress(
    title="Configure the spatial halo", total=1, unit="configuration"
)
# This extends the D3 halo for ghost-atom coordination numbers; atoms do not move.
DOMAIN_HALO_SKIN_A = DOMAIN_METHODOLOGY.domain_halo_skin_a

domain_cutoff_a = float(periodic_model.model_config.neighbor_config.cutoff)
domain_config = DomainConfig(
    cutoff=domain_cutoff_a,
    skin=DOMAIN_HALO_SKIN_A,
    mesh=None,  # one GPU has no process mesh; PME still uses PME_MESH_DIMENSIONS
    grid_dims=DOMAIN_METHODOLOGY.domain_grid_dims,
    compile=False,
)
domain_config_progress.complete(
    f"PME grid {PME_MESH_DIMENSIONS}; domain cutoff {domain_cutoff_a:g} Å; "
    f"multi-rank ghost width {domain_config.effective_ghost_width():g} Å"
)
""",
        ),
    )
    insert_after(
        "configure-domain-parallel",
        code(
            "run-domain-single-gpu",
            """
domain_run_progress = NotebookProgress(
    title="Walk through DomainParallel on one GPU", total=2, unit="steps"
)
# BaseDynamics calls the model and hooks once; it does not move the atoms.
domain_evaluator = BaseDynamics(model=periodic_model, n_steps=1)
for neighbor_hook in periodic_model.make_neighbor_hooks():
    domain_evaluator.register_hook(neighbor_hook)
domain_evaluator.register_hook(NaNDetectorHook(frequency=1))
domain_dynamics = DomainParallel(
    dynamics=domain_evaluator,
    config=domain_config,
    n_steps=1,
    device_type=DEVICE.type,
)
domain_run_progress.advance(message="one rank configured; no spatial decomposition")

torch.cuda.synchronize(DEVICE)
torch.cuda.reset_peak_memory_stats(DEVICE)
domain_started = perf_counter()
with domain_dynamics as domain_run:
    domain_local = domain_run.partition(domain_batch)
    domain_local = domain_run.run(domain_local)
    domain_energy = domain_local.energy.detach().clone()
    domain_result = domain_run.gather(domain_local, dst=0)
torch.cuda.synchronize(DEVICE)
domain_elapsed_s = perf_counter() - domain_started
domain_peak_memory_gb = torch.cuda.max_memory_allocated(DEVICE) / 1.0e9
domain_run_progress.complete("partition, run, and gather completed")
""",
        ),
    )
    insert_after(
        "run-domain-single-gpu",
        code(
            "inspect-domain-molecule-charges",
            """
domain_charge_progress = NotebookProgress(
    title="Resolve predicted charge by molecule", total=2, unit="views"
)
# Toolkit-Ops segmented_sum reduces atom charges by molecule_id, giving one
# predicted charge sum for each molecule in the checked base box.
molecule_ids = torch.as_tensor(
    domain_atoms.arrays["molecule_id"],
    device=domain_result.charges.device,
    dtype=torch.int32,
)
domain_molecule_charge_values = segmented_sum(
    domain_result.charges.reshape(-1),
    molecule_ids,
    domain_plan.molecule_count,
)
domain_molecule_charges, domain_molecule_charge_summary = molecule_charge_tables(
    domain_plan, domain_atoms, domain_molecule_charge_values
)
domain_charge_display = molecule_charge_display_tables(
    domain_molecule_charges,
    domain_molecule_charge_summary,
)
display(readable_table(
    domain_charge_display.summary.round(6),
    label="Predicted molecular charge sums", show_index=False,
))
domain_charge_progress.advance(message="all molecule sums summarized")

display(readable_table(
    domain_charge_display.extremes.round(6),
    label="Most negative and positive molecular charge sums",
    show_index=False,
))
display(callout(
    "The model receives one total-charge target for the box. It does not "
    "require each source molecule to remain exactly neutral. "
    "No additional renormalization is applied to the returned charges before "
    "PME. These model-dependent sums are a diagnostic. They are "
    "not validated intermolecular charge transfer; all 256 values are saved.",
    kind="note",
))
domain_charge_progress.complete("summary and charge extremes displayed")
""",
        ),
    )
    insert_after(
        "inspect-domain-molecule-charges",
        code(
            "inspect-domain-single-gpu",
            """
domain_output_progress = NotebookProgress(
    title="Inspect the one-GPU result", total=2, unit="checks"
)

domain_charge_values = domain_result.charges.detach()
domain_charge_dtype = str(domain_charge_values.dtype).removeprefix("torch.")
domain_charge_target_e = float(domain_batch.charge.to(torch.float64).sum().item())
domain_charge_sum = float(domain_charge_values.to(torch.float64).sum().cpu())
domain_charge_residual_e = domain_charge_sum - domain_charge_target_e
domain_charge_abs_residual_per_atom = (
    abs(domain_charge_residual_e) / domain_charge_values.numel()
)
domain_charge_sum_abs_e = float(
    domain_charge_values.to(torch.float64).abs().sum().cpu()
)
domain_charge_max_abs_e = float(
    domain_charge_values.to(torch.float64).abs().max().cpu()
)
domain_charge_finite = bool(torch.isfinite(domain_charge_values).all())
domain_energy_ev = float(domain_energy.reshape(-1)[0].cpu())
domain_fmax_ev_a = float(
    torch.linalg.vector_norm(domain_result.forces, dim=1).max().detach().cpu()
)
domain_live_api_passed = bool(
    domain_charge_finite
    and np.isfinite([
        domain_energy_ev,
        domain_fmax_ev_a,
        domain_charge_sum,
        domain_charge_residual_e,
        domain_charge_abs_residual_per_atom,
        domain_charge_sum_abs_e,
        domain_charge_max_abs_e,
    ]).all()
)
assert domain_live_api_passed
domain_result_atoms = graph_atoms_from_batch(
    domain_result, 0, "domain-box",
    reference_atoms=domain_atoms, include_results=True,
)
write(OUTPUT_DIR / "domain_box_evaluated.extxyz", domain_result_atoms)
domain_output_progress.advance(message="finite energy, forces, and charge saved")

domain_live_summary = pd.DataFrame([
    ("world size", 1),
    ("spatially decomposed", False),
    ("atoms", domain_result.num_nodes),
    ("raw model energy / atom for this fixed input (eV)",
     domain_energy_ev / domain_result.num_nodes),
    ("maximum force (eV/Å)", domain_fmax_ev_a),
    ("predicted charge dtype", domain_charge_dtype),
    ("input total-charge target (e)", domain_charge_target_e),
    ("total predicted charge (e)", domain_charge_sum),
    ("charge residual (e)", domain_charge_residual_e),
    ("absolute charge residual / atom (e)", domain_charge_abs_residual_per_atom),
    ("sum of absolute atomic charges (e)", domain_charge_sum_abs_e),
    ("maximum absolute atomic charge (e)", domain_charge_max_abs_e),
    ("local API-check wall time (s)", domain_elapsed_s),
    ("local Torch peak allocated after reset (GB)", domain_peak_memory_gb),
], columns=["Measure", "Value"]).round(6)
display(readable_table(domain_live_summary,
    label="LIVE WALKTHROUGH · One GPU, one domain", show_index=False))
display(callout(
    "This confirms the public call sequence and finite outputs on one GPU; "
    "it is not domain decomposition, a speedup, or a capacity measurement. "
    "`BaseDynamics` evaluates the "
    "checked, unequilibrated base box without changing its coordinates. Its "
    "first-call time is not comparable with the warmed H100 measurements. "
    "Toolkit passed the returned float32 charges to PME unchanged. Their "
    "reported sum residual is a numerical diagnostic, not a pass threshold. "
    "The raw energy is not an interaction, cohesive, or liquid-property "
    "energy. A large maximum force means the box would need relaxation or "
    "equilibration before dynamics; it is not evidence of material instability.",
    kind="result", result_state="observed",
))
domain_output_progress.complete("result table displayed")
""",
        ),
    )
    insert_after(
        "inspect-domain-single-gpu",
        markdown(
            "domain-parallel-api",
            r"""
### The multi-GPU API used by the offline run

`torchrun` starts one worker **rank** per GPU. Rank 0 supplies the full `Batch`
to `DomainParallel`.

```python
from time import perf_counter

import torch
import torch.distributed as dist
from nvalchemi.distributed import (
    DistributedManager, DomainConfig, DomainParallel, SpatialPartitioner,
)
from nvalchemi.dynamics import BaseDynamics

DistributedManager.initialize()
manager = DistributedManager()
device = torch.device(manager.device)
mesh = manager.initialize_mesh(
    mesh_shape=(manager.world_size,),
    mesh_dim_names=("domain",),
)

full = ... if manager.rank == 0 else None
periodic_model = ...  # checkpoint base + predicted-charge PME + D3
config = DomainConfig(
    cutoff=domain_cutoff_a,
    skin=DOMAIN_HALO_SKIN_A,
    mesh=mesh,
    grid_dims=None,
    compile=False,
    require_nondegenerate=manager.world_size > 1,
)
if manager.rank == 0:
    layout = SpatialPartitioner(config, full.cell, full.pbc)
    print(layout.cells_per_dim, layout.rank_grid)

hooks = periodic_model.make_neighbor_hooks() if manager.world_size == 1 else []
evaluator = BaseDynamics(model=periodic_model, n_steps=1, hooks=hooks)
with DomainParallel(dynamics=evaluator, config=config, n_steps=1) as domain:
    local = domain.partition(full)

    # Initialization and warm-up are not timed.
    local = domain.run(local, n_steps=1)

    pass_times_s = []
    for _ in range(3):
        dist.barrier()
        torch.cuda.synchronize(device)
        started = perf_counter()
        local = domain.run(local, n_steps=1)
        torch.cuda.synchronize(device)
        elapsed_s = torch.tensor(
            perf_counter() - started,
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(elapsed_s, op=dist.ReduceOp.MAX)
        pass_times_s.append(float(elapsed_s.item()))

    total_energy = local.energy.detach().clone()
    full_result = domain.gather(local, dst=0)

DistributedManager.cleanup()
```

`BaseDynamics` does not integrate. Because `DomainParallel` may wrap equivalent
periodic coordinates, the run checks minimum-image displacement. Each timed
`run(..., n_steps=1)` is one energy-and-force evaluation; the slowest rank sets
the time.

Partition and gather each occur once. An untimed warm-up initializes forces.
Multi-GPU `DomainParallel` rebuilds neighbors inside each region; the one-GPU
path uses the model's ordinary neighbor hooks. `SpatialPartitioner` records the
layout, and `require_nondegenerate=True` rejects a full-structure halo.

| Layout | Role |
|---|---|
| process mesh | assigns one worker process to each GPU |
| spatial grid | assigns atoms to GPU regions |
| PME grid | defines the electrostatics FFT repeated on every GPU |

Toolkit 0.2 restricts this example to an input total-charge target of zero
because it is not copied into each region. Toolkit passes AIMNet2 charges to
PME unchanged, sums total energy across GPUs, and `gather` collects atom-level
fields, including forces, on rank 0. Stable `source_atom_id` values stay
outside the Toolkit `Batch`; the runner uses the recorded `SpatialPartitioner`
assignment to restore force order and checks the result against locally cloned
reference positions. The distributed AIMNet2-to-PME group does not emit
predicted atomic charges. Rank consistency is checked through source-ordered
forces and distributed energies.
""",
        ),
    )
    insert_after(
        "inspect-domain-single-gpu",
        markdown(
            "domain-scaling-plan",
            r"""
### Run the same large system on 1, 2, and 4 GPUs

The offline H100 run uses one checked 51,200-atom supercell; Packmol is not
rerun. The structure, composed AIMNet2 + PME + D3 model, model tensors,
coordinates, forces, cutoffs, and requested outputs stay fixed. Only the
number of spatial regions changes. Model tensors, coordinates, and forces are
float32. The single-rank total energy is float32, while Toolkit's cross-rank
energy reduction returns float64 on two and four GPUs.

| Run | Input | Nodes = ranks = GPUs | Work |
|---|---:|---:|---|
| live API walkthrough | 3,200-atom base box | 1 | one fixed-structure energy/force evaluation |
| recorded comparison | 51,200-atom supercell | 1, 2, 4 | one warm-up, then three measured energy/force evaluations |

The box is partitioned once and gathered once. Warm-up is outside timing. Each
measured pass is one `run(..., n_steps=1)` call for energy and forces, with no
integration update. Periodic images must remain equivalent within the
minimum-image tolerance. All three slowest-rank times and their median are
shown. The first one-GPU pass is visibly a first-use outlier even after the
warm-up, so these are instructional measurements, not a general benchmark.

Four GPUs means four one-H100 nodes. The result set checks 1-GPU forces against
2/4 GPUs and 2-GPU distributed energy against 4. The 1-to-multi energy offset
is diagnostic because Toolkit 0.2 reduces those paths differently. The one-GPU
row records the float32 charges passed to PME. Its residual is reported, not
limited; `1e-4 e` applies only to the separate 3,200-atom PME-versus-Ewald
check. See the
[runbook](COMPUTE_LAB_RUNBOOK.md#5-build-and-check-the-recorded-result-set).
"""
            + "\n\n"
            + callout_html(
                "Check the energy and forces before reading the times. The "
                "same 51,200-atom input must pass on 1, 2, and 4 GPUs.",
                kind="check",
            )
            + "\n\n"
            + r"""
Missing files produce `NOT REPORTED`, never an estimate. Every GPU repeats the
PME FFT and workspace, so this example does not claim that all memory is split
evenly across ranks.
""",
        ),
    )
    insert_after(
        "domain-parallel-api",
        code(
            "domain-parallel-results",
            """
DOMAIN_FIXED_ATOM_COUNT = (
    DOMAIN_METHODOLOGY.fixed_molecules_per_species
    * DOMAIN_METHODOLOGY.atoms_per_composition_unit
)
DOMAIN_REQUIRED_WORLD_SIZES = DOMAIN_METHODOLOGY.campaign_world_sizes
DOMAIN_RESULT_DIR = PART_DIR / "data" / "domain_decomposition" / "recorded"
domain_results_progress = NotebookProgress(
    title="Load recorded H100 domain results", total=1, unit="result set"
)
domain_view = load_domain_lesson_view(
    DOMAIN_RESULT_DIR,
    expected_atom_count=DOMAIN_FIXED_ATOM_COUNT,
    expected_world_sizes=DOMAIN_REQUIRED_WORLD_SIZES,
)
if not domain_view.available:
    domain_results_progress.complete("saved H100 result files are missing")
    display(callout(
        "NOT REPORTED: this notebook copy does not include a complete checked "
        "H100 run. "
        + domain_view.reason,
        kind="result", result_state="not_reported",
    ))
else:
    domain_results_progress.complete("saved files and every attempted case checked")
    display(readable_table(
        domain_view.recorded_run_table,
        label="SAVED H100 RUN · DomainParallel",
        show_index=True,
    ))
""",
        ),
    )
    insert_before(
        "convert-domain-box",
        code(
            "display-domain-methodology",
            """
domain_settings_progress = NotebookProgress(
    title="Show the resolved periodic setup", total=1, unit="table"
)
domain_settings_compact = pd.DataFrame([
    ("Live input", f"{len(domain_atoms):,} atoms; "
     f"{DOMAIN_METHODOLOGY.live_molecules_per_species} molecules/species; "
     f"{DOMAIN_METHODOLOGY.construction_density_g_cm3:g} g/cm³ construction density"),
    ("Offline base-box build", f"Packmol "
     f"{DOMAIN_METHODOLOGY.packmol_tolerance_a:g} Å tolerance; "
     f"{DOMAIN_METHODOLOGY.packmol_precision_a:g} Å precision; "
     f"seed {DOMAIN_METHODOLOGY.packmol_seed}; checksums verified at load"),
    ("PME request", f"{DOMAIN_METHODOLOGY.pme_realspace_cutoff_a:g} Å "
     f"real-space cutoff; {DOMAIN_METHODOLOGY.pme_accuracy:g} estimator target; "
     f"{DOMAIN_METHODOLOGY.pme_mesh_safety_factor:g}× mesh safety factor; "
     f"spline {DOMAIN_METHODOLOGY.pme_spline_order}; "
     f"alpha and grid estimated together"),
    ("PME result", f"alpha {PME_ALPHA_A_INV:.6f} Å⁻¹; "
     f"mesh {' × '.join(str(value) for value in PME_MESH_DIMENSIONS)}; "
     f"spacing {' × '.join(f'{value:.3f}' for value in PME_MESH_SPACING_A)} Å"),
    ("Local terms", f"AIMNet "
     f"{DOMAIN_METHODOLOGY.aimnet_neighbor_cutoff_a:g} Å; D3 "
     f"{DOMAIN_METHODOLOGY.d3_cutoff_a:g} Å with final "
     f"{DOMAIN_METHODOLOGY.d3_smoothing_fraction:.0%} tapered; "
     f"{DOMAIN_METHODOLOGY.domain_halo_skin_a:g} Å additional halo depth"),
], columns=["Choice", "Resolved value"])
display(readable_table(
    domain_settings_compact,
    label=f"Periodic box and model settings · v{DOMAIN_METHODOLOGY.version}",
    show_index=False,
))
domain_settings_progress.complete("five resolved choices displayed")
""",
            source_hidden=True,
        ),
    )
    insert_after(
        "domain-scaling-plan",
        code(
            "display-domain-scaling-methodology",
            """
domain_h100_settings_progress = NotebookProgress(
    title="Show the H100 comparison method", total=1, unit="table"
)
domain_h100_settings = pd.DataFrame([
    ("Fixed input",
     f"{DOMAIN_METHODOLOGY.fixed_molecules_per_species} molecules/species; "
     f"{DOMAIN_METHODOLOGY.fixed_molecules_per_species * DOMAIN_METHODOLOGY.atoms_per_composition_unit:,} "
     "atoms on 1, 2, and 4 H100s"),
    ("Execution", f"partition once; "
     f"{DOMAIN_METHODOLOGY.evaluation_warmup_count} untimed warm-up; "
     f"{DOMAIN_METHODOLOGY.evaluation_pass_count} measured "
     f"`run(..., n_steps=1)` passes; gather once"),
    ("Measured work",
     f"{DOMAIN_METHODOLOGY.measured_model_evaluations_per_pass} complete "
     "energy/force evaluation per pass; report all three slowest-rank times "
     "and their median"),
    ("Energy statistic",
     f"1 GPU: {DOMAIN_METHODOLOGY.evaluation_energy_dtype_single_rank}; "
     f"2/4 GPUs: {DOMAIN_METHODOLOGY.evaluation_energy_dtype_multi_rank}; "
     "use the median of the three measured energies for GPU-layout comparisons"),
    ("Coordinates",
     "no integration update; maximum minimum-image displacement ≤ "
     f"{DOMAIN_METHODOLOGY.evaluation_position_mic_tolerance_a:g} Å"),
    ("One-GPU predicted charges",
     "finite float32 values; record target, sum, residual, absolute residual / atom, "
     "magnitude statistics, shape, and hash; do not renormalize after model "
     "evaluation or apply a residual threshold"),
    ("3,200-atom PME ↔ Ewald charge check",
     f"|Σq - Qtarget| ≤ {DOMAIN_METHODOLOGY.charge_sum_tolerance_e:g} e"),
    ("2/4-GPU repeated energy",
     "(max(Epass) - min(Epass)) / N ≤ "
     f"{DOMAIN_METHODOLOGY.distributed_energy_repeatability_tolerance_ev_per_atom:g} "
     "eV/atom"),
    ("2 → 4-GPU energy",
     f"|ΔEdistributed| / N ≤ "
     f"{DOMAIN_METHODOLOGY.evaluation_energy_tolerance_ev_per_atom:g} eV/atom"),
    ("1 → 2/4-GPU force",
     f"|ΔFᵢ| ≤ {DOMAIN_METHODOLOGY.evaluation_force_atol_ev_a:g} eV/Å + "
     f"{DOMAIN_METHODOLOGY.evaluation_force_rtol:g}×|Fᵢ,₁|"),
    ("1 → multi-GPU energy",
     "show the raw offset; do not use it as an agreement check"),
], columns=["Check", "Declared method"])
display(readable_table(
    domain_h100_settings,
    label="Offline H100 comparison method",
    show_index=False,
))
domain_h100_settings_progress.complete("timing and agreement checks displayed")
""",
            source_hidden=True,
        ),
    )
    insert_after(
        "domain-parallel-results",
        code(
            "display-domain-parallel-results",
            """
domain_display_progress = NotebookProgress(
    title="Display recorded domain results", total=1, unit="result set"
)
if domain_view.available:
    display(readable_table(
        domain_view.run_settings_table.rename(
            columns={"setting": "Setting", "value": "Value"}
        ),
        label="Recorded H100 run settings",
        show_index=False,
    ))
    domain_layout_display = domain_view.layout_table.rename(columns={
        "world_size": "GPUs",
        "nodes": "Nodes",
        "ranks": "Ranks",
        "spatial_grid": "Rank grid",
        "owned_atoms_min": "Fewest owned atoms",
        "owned_atoms_max": "Most owned atoms",
    })
    display(readable_table(
        domain_layout_display,
        label=f"Spatial layout · same {DOMAIN_FIXED_ATOM_COUNT:,}-atom input",
        show_index=False,
    ))
    domain_timing_display = domain_view.timing_table[[
        "world_size",
        "pass_1_s",
        "pass_2_s",
        "pass_3_s",
        "median_time_s",
        "speedup_vs_1gpu",
    ]].rename(columns={
        "world_size": "GPUs",
        "pass_1_s": "Pass 1 / s",
        "pass_2_s": "Pass 2 / s",
        "pass_3_s": "Pass 3 / s",
        "median_time_s": "Median / s",
        "speedup_vs_1gpu": "Speedup vs 1 GPU",
    })
    display(readable_table(
        domain_timing_display.round(3),
        label="Three fixed-structure energy/force passes",
        show_index=False,
    ))
    domain_agreement_display = domain_agreement_display_table(
        domain_view.output_agreement_table
    )
    display(readable_table(
        domain_agreement_display.round(6),
        label="Energy and force agreement",
        show_index=False,
    ))
    domain_charge_row = domain_view.charge_diagnostics_table.iloc[0]
    domain_charge_display = pd.DataFrame([
        ("Atoms", domain_charge_row["atom_count"]),
        ("Charge dtype", domain_charge_row["dtype"]),
        ("Target / e", domain_charge_row["target_sum_e"]),
        ("Predicted sum / e", domain_charge_row["charge_sum_e"]),
        ("Residual / e", domain_charge_row["residual_e"]),
        ("|Residual| per atom / e", domain_charge_row["abs_residual_per_atom_e"]),
        ("Finite", domain_charge_row["finite"]),
    ], columns=["Check", "Value"])
    display(readable_table(
        domain_charge_display.round(9),
        label="One-H100 predicted charges passed to PME",
        show_index=False,
    ))
    domain_electrostatics_row = domain_view.electrostatics_table.iloc[0]
    domain_electrostatics_display = pd.DataFrame([
        ("Atoms", domain_electrostatics_row["atom_count"]),
        ("Predicted sum / e", domain_electrostatics_row["charge_sum_e"]),
        ("Charge residual / e", domain_electrostatics_row["charge_residual_e"]),
        (
            "Energy difference / meV atom⁻¹",
            domain_electrostatics_row["energy_abs_error_meV_per_atom"],
        ),
        (
            "Force RMS difference / eV Å⁻¹",
            domain_electrostatics_row["force_rms_error_eV_A"],
        ),
        (
            "Force max difference / eV Å⁻¹",
            domain_electrostatics_row["force_max_error_eV_A"],
        ),
        ("Checks passed", domain_electrostatics_row["passed"]),
    ], columns=["Check", "Value"])
    display(readable_table(
        domain_electrostatics_display.round(6),
        label="Fixed-charge PME versus Ewald check",
        show_index=False,
    ))
    domain_takeaway = domain_view.takeaway
    assert domain_takeaway["all_fixed_evaluations_succeeded"]
    assert domain_takeaway["positions_pbc_equivalent"]
    assert domain_takeaway["all_output_checks_passed"]
    speedup_rows = domain_view.timing_table.loc[
        domain_view.timing_table["world_size"].gt(1)
    ]
    speedup_text = "; ".join(
        f"{int(row.world_size)} GPUs: {row.speedup_vs_1gpu:.2f}×"
        for row in speedup_rows.itertuples(index=False)
    )
    display(callout(
        f"The same {DOMAIN_FIXED_ATOM_COUNT:,}-atom structure ran through the "
        f"public DomainParallel path on 1, 2, and 4 H100s. Every declared "
        f"force and distributed-energy check passed. There was no integration "
        f"update; the largest minimum-image displacement was "
        f"{domain_takeaway['max_minimum_image_displacement_a']:.2e} Å. "
        f"Observed median speedup "
        f"against one GPU: {speedup_text}. Each median comes from "
        f"three raw energy/force passes shown above. The first one-GPU pass "
        f"contains remaining first-use work and is kept visible. These short "
        f"times apply to this input, model, software, and hardware; they do not "
        f"measure a trajectory or a memory limit. The global PME charge mesh "
        f"and full reciprocal FFT path remain replicated on every rank.",
        kind="result", result_state="pass",
    ))
    domain_figure, _ = plot_domain_decomposition(domain_view.plot_data)
    display(figure_with_alt(
        domain_figure,
        alt_text=(
            "Two-panel H100 domain-decomposition result for one fixed "
            "51,200-atom input. The first panel shows the recorded range of "
            "owned atoms per rank on one, two, and four GPUs. The second shows "
            "all three energy-and-force pass times and their median."
        ),
    ))
    plt.close(domain_figure)
    domain_display_progress.complete("tables and fixed-input plot displayed")
else:
    display(callout(
        "The live one-GPU cell exercised the public DomainParallel call "
        "sequence with one domain and no spatial splitting. The fixed "
        "51,200-atom comparison on 1, 2, and 4 H100s is NOT REPORTED because "
        "the saved checked result files are missing.",
        kind="result", result_state="not_reported",
    ))
    domain_display_progress.complete("no recorded result set to display")
""",
        ),
    )

    replace_code_source(
        "save",
        """
save_progress = NotebookProgress(title="Save validated outputs", total=1, unit="run")
saved_run = save_water_run_outputs(
    OUTPUT_DIR,
    results=water_run_results,
    **manifest_input.as_save_arguments(),
)
run_manifest = saved_run.manifest
spectrum_table = saved_run.spectrum_table
save_progress.complete("tables, spectra, and run summary saved")

saved_inventory = pd.DataFrame({"saved_file": saved_run.relative_files})
key_output_names = (
    "part1_results_summary.csv",
    "nci_interaction_curves.csv",
    "surface_adsorption_energies.csv",
    "water_ir_spectra.csv",
    "water_monomer_harmonic_comparison.csv",
    "water_run_manifest.json",
)
key_outputs = saved_inventory.loc[
    saved_inventory["saved_file"].isin(key_output_names)
].rename(columns={"saved_file": "Key output"})
assert len(key_outputs) == len(key_output_names)
display(readable_table(
    key_outputs,
    label="Key files written by the final report",
    show_index=False,
))
display(callout(
    f"Saved {len(saved_run.relative_files)} report files to "
    f"{OUTPUT_DIR.relative_to(ROOT)}. water_run_manifest.json lists every "
    "report file, trajectory array, and raw Toolkit data file written earlier.",
    kind="result", result_state="pass",
))
""",
    )

    replace_code_source(
        "results-summary",
        """
summary_progress = NotebookProgress(
    title="Build the results summary", total=1, unit="report"
)
notebook_report = build_part1_notebook_report(globals())
results_summary = notebook_report.results_summary
not_reported_count = notebook_report.not_reported_count
water_run_results = notebook_report.water_run_results
manifest_input = notebook_report.manifest_input
summary_progress.complete("summary and saved-report records assembled")
""",
    )

    cells[:] = [cell for cell in cells if cell["id"] != "composition-note"]

    place_after(
        "stage-5",
        (
            "build-components",
            "component-ablation",
            "official-composition-agreement",
            "full-pipeline-agreement",
            "dimer-ablation-plot",
        ),
    )
    set_background(
        (
            "component-ablation",
            "official-composition-agreement",
            "full-pipeline-agreement",
            "dimer-ablation-plot",
        ),
    )
    set_background(
        (
            "tutorial-settings",
            "display-cpu-gpu-sweep",
            "display-nci-curves",
            "analyze-adsorption-panel",
            "view-adsorption-panel",
            "compile-fixed-ir-model",
            "reference-preflight",
            "harmonic-minimum",
            "harmonic-finite-difference",
            "harmonic-comparison",
            "diagnostics",
            "topology-timeline",
            "spectrum",
            "load-reference",
            "mode-mapping",
            "plot",
            "results-summary",
            "display-results-summary",
            "inspect-domain-single-gpu",
            "domain-parallel-results",
            "display-domain-parallel-results",
            "pipeline-campaign-results",
            "prepare-nci-resources",
            "display-nci-model-settings",
            "display-sevennet-surface-model",
            "validate-inflight-example",
        )
    )
    set_background(
        (
            "manifest-run-details",
            "manifest-model-settings",
            "manifest-workflow-settings",
            "manifest-checks",
        ),
        hide_outputs=True,
    )
    cells[:] = [
        cell
        for cell in cells
        if cell["id"]
        not in {
            "manifest-run-details",
            "manifest-model-settings",
            "manifest-workflow-settings",
            "manifest-checks",
        }
    ]

    # Give learners one real Toolkit result before comparing array frameworks.
    place_after(
        "hello-world",
        (
            "framework-primer",
            "framework-primer-example",
            "framework-primer-warp",
        ),
    )

    place_after(
        "relax",
        (
            "validate-relaxation",
            "save-relaxed-structures",
            "reference-preflight",
            "observed-source-links",
            "reference-preview",
            "harmonic-intro",
            "harmonic-minimum",
            "harmonic-finite-difference",
            "harmonic-comparison",
        ),
    )
    place_after(
        "ir-mechanism",
        (
            "fused-stage-intro",
            "initialize-dynamics",
            "configure-dynamics",
            "attach-dynamics-hooks",
        ),
    )
    place_after(
        "attach-dynamics-hooks",
        (
            "stage-6",
            "run-dynamics",
            "validate-dynamics-run",
            "persist-trajectory",
            "analysis-restart",
        ),
    )
    place_after(
        "try-it",
        (
            "stage-7",
            "inflight-intro",
            "inflight-example",
            "configure-inflight-stage",
            "run-inflight-example",
            "validate-inflight-example",
            "domain-decomposition-intro",
            "build-domain-box",
            "convert-domain-box",
            "domain-model-intro",
            "configure-domain-pme",
            "compose-domain-model",
            "display-domain-methodology",
            "configure-domain-parallel",
            "run-domain-single-gpu",
            "inspect-domain-molecule-charges",
            "inspect-domain-single-gpu",
            "domain-parallel-api",
            "domain-scaling-plan",
            "display-domain-scaling-methodology",
            "domain-parallel-results",
            "display-domain-parallel-results",
            "distributed-pipeline-intro",
            "pipeline-campaign-results",
            "results-summary",
            "save",
            "results-summary-note",
            "display-results-summary",
        ),
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "ALCHEMI Main",
                "language": "python",
                "name": "alchemi-main",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.13",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(notebook, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the Part 1 ALCHEMI water IR notebook."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=NOTEBOOK,
        help="Notebook path (defaults to the checked-in Part 1 notebook).",
    )
    arguments = parser.parse_args()
    main(arguments.output)
