"""Build the Part 07 learner notebook from explicit cell sources."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent


def md(cell_id: str, source: str) -> nbformat.NotebookNode:
    return new_markdown_cell(source.strip(), id=cell_id)


def code(
    cell_id: str,
    source: str,
    *,
    hidden: bool = False,
    alt: str | None = None,
) -> nbformat.NotebookNode:
    metadata: dict[str, object] = {}
    if hidden:
        metadata = {
            "jupyter": {"source_hidden": True},
            "tags": ["hide-input"],
        }
    if alt is not None:
        metadata["alt"] = alt
    return new_code_cell(source.strip(), id=cell_id, metadata=metadata)


cells = [
    md(
        "n07-banner",
        """
<img src="../../shared/alchemi-banner-left.png" alt="NVIDIA ALCHEMI: AI for Chemistry and Materials Science" style="display:block;box-sizing:border-box;width:100%;max-width:100%;height:auto;">
""",
    ),
    md(
        "n07-title",
        """
# 07 · Training and fine-tuning

**Goal:** Build a fine-tuning strategy from explicit model, optimizer, loss, validation, reporting, and checkpoint pieces, then restart it from disk.

**Core concepts:** `FineTuningStrategy` selects parameter ownership before optimizer construction. Validation is a separate held-out pass. A restartable checkpoint needs both tensor state and a pickle-free model construction recipe.

This notebook contains **two separate examples**. A dimensionless MLP sandbox makes four optimizer updates easy to inspect. A generated argon example then fits a physically defined Lennard-Jones family from energy and force labels. The examples share training APIs, but they do not produce data for one another.

We reuse [`AtomicData` and `Batch` from Part 01](../01-atomicdata-batch/atomicdata-and-batch.ipynb), [model configuration from Part 03](../03-model-interfaces-composition/model-interfaces-composition.ipynb), and [hook registration from Part 04](../04-hooks/hooks.ipynb). The [Core training excerpt](../00-core-playbook/alchemi-core-playbook.ipynb) is the short prerequisite.

<details>
<summary>Where NVIDIA ALCHEMI fits (recap)</summary>

[ALCHEMI](https://developer.nvidia.com/cuda/cuda-x-libraries/alchemi) brings together Python building blocks, accelerated kernels, and deployable services for atomistic workflows.

- **ALCHEMI Toolkit** is a GPU-first Python framework with a unified, composable API for MLIPs and custom models. It provides GPU-native atomic data and batching, model adapters, MD classes, hooks, model training, and single- to multi-GPU pipelines.  
[GitHub repo](https://github.com/NVIDIA/nvalchemi-toolkit) · [Docs](https://nvidia.github.io/nvalchemi-toolkit/) · Apache 2.0 license

- **Toolkit-Ops** supplies GPU-optimized, batched operations for neighbor lists, dynamics, dispersion, and electrostatics, with PyTorch and JAX bindings.  
[GitHub repo](https://github.com/NVIDIA/nvalchemi-toolkit-ops) · [Docs](https://nvidia.github.io/nvalchemi-toolkit-ops/) · Apache 2.0 license

- **ALCHEMI NIM microservices** package supported atomistic workflows as cloud-ready services. The current catalog includes Batched Geometry Relaxation and Batched Molecular Dynamics. Self-hosting uses an NVIDIA AI Enterprise license.  
[Transparency card](https://docs.nvidia.com/nim/alchemi/alchemi-bgr/1.0.0/ai-transparency-card/overview.html) · [Docs](https://docs.nvidia.com/nim/alchemi/alchemi-bgr/latest/index.html)

</details>
""",
    ),
    code(
        "n07-imports",
        """
# ruff: noqa: I001
import copy
import os
from pathlib import Path
os.environ.setdefault("WARP_CACHE_PATH", str(Path("artifacts/warp-cache").resolve()))
import helpers
import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import display
from nvalchemi.hooks import NeighborListHook
from nvalchemi.hooks.reporting import ReportingOrchestrator, RichReporter
from nvalchemi.models import LennardJonesModelWrapper
from nvalchemi.models.base import BaseModelMixin, ModelConfig, NeighborConfig
from nvalchemi.neighbors import compute_neighbors
from nvalchemi.training import BatchValidationCallback, CheckpointHook, ComposedLossFunction, EnergyMSELoss, FineTuningStrategy, ForceMSELoss, OptimizerConfig, TrainingStage, ValidationConfig, default_training_fn
""",
        hidden=True,
    ),
    code(
        "n07-runtime",
        """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
SEED = 1707
torch.manual_seed(SEED)
{"device": device.type, "dtype": str(dtype), "one process": True}
""",
    ),
    code(
        "n07-presentation",
        """
helpers.configure_presentation()
""",
        hidden=True,
    ),
    md(
        "n07-map-heading",
        """
## Course map

Part 07 turns configured models and hooks into an observed, restartable fit. Green marks Part 07 and the training capability used here.
""",
    ),
    md(
        "n07-map",
        """
<object data="../../shared/curriculum-map-07.svg" type="image/svg+xml" aria-label="ALCHEMI course map with Part 07 and training highlighted" style="display:block;width:100%;max-width:100%;height:auto;">
  <img src="../../shared/curriculum-map-07.svg" alt="ALCHEMI course map with Part 07 and training highlighted" style="display:block;width:100%;max-width:100%;height:auto;">
</object>
""",
    ),
    md(
        "n07-level1-heading",
        """
## Level 1: inspect four fine-tuning updates

The first model receives four fixed atoms as twelve ordered coordinates. Its source task and shifted target task are deterministic scalar regressions. The target is a **dimensionless synthetic score**. It does not represent a physical energy, and lower loss here says nothing about an atomistic model.

The MLP has a `backbone` and a `readout`. We will keep the backbone fixed and let `FineTuningStrategy` place only `main.readout.*` in the optimizer.
""",
    ),
    code(
        "n07-toy-data",
        """
toy_model, toy_train_loader, toy_validation_loader, split_frame = (
    helpers.prepare_toy_transfer(device=device)
)
toy_example = next(iter(toy_train_loader))
""",
    ),
    code(
        "n07-toy-shapes",
        """
{
    "graphs": toy_example.num_graphs,
    "positions": tuple(toy_example.positions.shape),
    "energy target shape": tuple(toy_example.energy.shape),
    "target unit": "dimensionless synthetic score",
    "dtype": str(toy_example.positions.dtype),
    "device": toy_example.device.type,
}
""",
    ),
    md(
        "n07-toy-shape-interpretation",
        """
Each minibatch holds four graphs. Every graph owns four ordered atoms, so positions have shape `[16, 3]` after batching and energy targets have shape `[4, 1]`. This MLP is deliberately sensitive to atom order, translation, and rotation. It is an optimizer sandbox, not a scientific wrapper.
""",
    ),
    code(
        "n07-toy-split-shape",
        """
toy_split_summary = (
    split_frame.groupby("split", sort=False)
    .agg(samples=("sample_id", "count"), target_min=("target", "min"), target_max=("target", "max"))
    .reset_index()
)
""",
    ),
    code(
        "n07-toy-split-display",
        """
display(toy_split_summary)
""",
    ),
    md(
        "n07-toy-split-interpretation",
        """
Twelve generated records are optimizer inputs; four separate records are validation inputs. Both splits follow the same synthetic formula, so validation only checks held-out interpolation within this toy generator.
""",
    ),
    md(
        "n07-toy-ownership-question",
        """
### Which parameters can the optimizer update?

`trainable_patterns` uses globs over fully qualified parameter names. The recorder observes both `requires_grad` and membership in the optimizer after strategy setup.
""",
    ),
    code(
        "n07-toy-observers",
        """
toy_ownership = helpers.ParameterOwnershipRecorder()
toy_history = helpers.TrainingHistory(
    energy_mse_label="energy MSE (score²)",
    force_mse_label=None,
)
toy_validation_recorder: BatchValidationCallback = helpers.ValidationBatchRecorder()
toy_parameters_before = {
    name: value.detach().clone() for name, value in toy_model.named_parameters()
}
""",
    ),
    md(
        "n07-toy-strategy-intro",
        """
`EnergyMSELoss` compares one predicted score with one target score per graph. `OptimizerConfig` delays optimizer construction until the strategy knows which parameters are trainable. `ValidationConfig` re-iterates the held-out loader after every two completed updates.
""",
    ),
    code(
        "n07-toy-loss-optimizer",
        """
toy_loss = EnergyMSELoss(dtype_policy="prediction_to_target")
toy_optimizer_config = OptimizerConfig(
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs={"lr": 2.0e-2},
)
""",
    ),
    code(
        "n07-toy-validation",
        """
toy_validation_config = ValidationConfig(
    validation_data=toy_validation_loader,
    every_n_steps=2,
    batch_callback=toy_validation_recorder,
    name="toy-validation",
)
""",
    ),
    md(
        "n07-reporting-callout",
        """
<div style="display:block;box-sizing:border-box;width:100%;max-width:100%;min-width:0;background:#151A1F;color:#F3F4F6;border:1px solid #30363D;border-radius:8px;padding:0.82rem 0.95rem;line-height:1.45;overflow:hidden;overflow-wrap:anywhere;">
  <div style="color:#76B900;font-size:0.76rem;font-weight:700;letter-spacing:0.06em;margin-bottom:0.3rem;">ALCHEMI TOOLKIT API</div>
  <code style="display:block;color:#FFFFFF;font-size:1rem;font-weight:650;white-space:normal;overflow-wrap:anywhere;">ReportingOrchestrator([RichReporter(...)]) -&gt; training hook</code>
  <div style="display:grid;grid-template-columns:max-content minmax(0,1fr);column-gap:0.65rem;row-gap:0.18rem;min-width:0;color:#CDD2D8;margin-top:0.5rem;font-size:0.92rem;">
    <strong style="color:#F3F4F6;">Accepts</strong><span style="min-width:0;overflow-wrap:anywhere;">one or more reporting sinks and a reporting frequency.</span>
    <strong style="color:#F3F4F6;">Returns</strong><span style="min-width:0;overflow-wrap:anywhere;">a normal hook that receives strategy contexts.</span>
  </div>
</div>
""",
    ),
    code(
        "n07-toy-reporter",
        """
toy_rich_reporter = RichReporter(
    title="Four-update fine-tuning",
    layout="training",
    max_plots=1,
    transient=True,
)
toy_reporting = ReportingOrchestrator([toy_rich_reporter], frequency=1)
""",
    ),
    code(
        "n07-toy-strategy",
        """
toy_strategy = FineTuningStrategy(
    models=toy_model,
    trainable_patterns=("main.readout.*",),
    freeze_mode="requires_grad",
    optimizer_configs=toy_optimizer_config,
    training_fn=default_training_fn,
    loss_fn=toy_loss,
    validation_config=toy_validation_config,
    num_steps=4,
    devices=[device],
    hooks=[toy_ownership, toy_history, toy_reporting],
)
""",
    ),
    code(
        "n07-toy-run",
        """
toy_strategy.run(toy_train_loader)
""",
    ),
    code(
        "n07-toy-ownership-frame",
        """
parameter_ownership = toy_ownership.frame()
""",
    ),
    code(
        "n07-toy-ownership-display",
        """
display(parameter_ownership)
""",
    ),
    md(
        "n07-toy-ownership-interpretation",
        """
Only the two `main.readout.*` rows are both gradient-enabled and optimizer-owned. `FineTuningStrategy` restores the model's original `requires_grad` flags after the run, so this setup-time table is the reliable ownership record.
""",
    ),
    code(
        "n07-toy-update-frame",
        """
toy_parameter_updates = pd.DataFrame(
    {
        "parameter": list(toy_parameters_before),
        "largest absolute update": [
            float((dict(toy_model.named_parameters())[name] - before).abs().max().cpu())
            for name, before in toy_parameters_before.items()
        ],
    }
)
""",
    ),
    code(
        "n07-toy-update-display",
        """
display(toy_parameter_updates)
""",
    ),
    md(
        "n07-toy-update-interpretation",
        """
The observed tensor changes agree with the ownership table: the readout moved and the backbone did not. This is direct evidence about parameter selection, not evidence that the synthetic labels improve a scientific model.
""",
    ),
    code(
        "n07-toy-history-frames",
        """
toy_training_rows = toy_history.training_frame()
toy_validation_rows = toy_history.validation_frame()
toy_validation_batches = toy_validation_recorder.frame()
""",
    ),
    code(
        "n07-toy-training-display",
        """
display(toy_training_rows)
""",
    ),
    code(
        "n07-toy-validation-display",
        """
display(toy_validation_rows)
""",
    ),
    code(
        "n07-toy-callback-display",
        """
display(toy_validation_batches)
""",
    ),
    md(
        "n07-toy-results-interpretation",
        """
The training table reports each completed optimizer update. Validation rows are full held-out passes, while the callback table preserves validation `sample_id` values and output shapes. Report the held-out result whether it improves or not; four updates are too few to support a model-selection claim.

In the fresh CPU execution, held-out MSE was 9.31 after update 2 and 8.49 after update 4. Training minibatch MSE was non-monotonic and ended at 19.06 because each point used a different minibatch. These values demonstrate observation plumbing, not model quality.
""",
    ),
    md(
        "n07-toy-plot-question",
        """
**Question:** Do training and held-out MSE move together during these four readout-only updates?
""",
    ),
    code(
        "n07-toy-plot",
        """
toy_figure = helpers.plot_toy_history(toy_training_rows, toy_validation_rows)
display(
    helpers.render_figure(
        toy_figure,
        alt_text="Training MSE and validation MSE across four readout-only optimizer updates.",
    )
)
plt.close(toy_figure)
""",
        alt="Training MSE and validation MSE across four readout-only optimizer updates.",
    ),
    md(
        "n07-toy-plot-interpretation",
        """
The lines describe two different aggregations: one minibatch per training point and the complete validation split per validation point. Their values should not be compared as if they came from the same samples.
""",
    ),
    code(
        "n07-toy-rich-preview",
        """
RichReporter.preview(
    history={"loss/total": toy_training_rows["total loss"].tolist()},
    steps=toy_training_rows["completed optimizer updates"].tolist(),
    title="Static preview from this run",
    max_plots=1,
)
""",
    ),
    md(
        "n07-toy-reporting-interpretation",
        """
The live reporter was attached through the same hook registry as the ownership and history observers. `preview` produces a static terminal layout from the recorded values, which is useful before a long run or when a notebook cannot preserve a live refresh.

**Go deeper:** compare the pinned official [DDP dummy-MLP example](https://github.com/NVIDIA/nvalchemi-toolkit/blob/8c2c307c/examples/intermediate/06_ddp_mlp_training.py) and [Rich reporting example](https://github.com/NVIDIA/nvalchemi-toolkit/blob/8c2c307c/examples/intermediate/07_rich_training_reporting.py). This notebook does not launch `torchrun` or DDP.
""",
    ),
    md(
        "n07-level2-heading",
        """
## Level 2: fit a generated argon potential

The scientific object changes here. We generate isolated, non-periodic Ar4 structures and label each one with the unshifted 12–6 Lennard-Jones equation

$$E = 4\\epsilon\\sum_{i<j}\\left[\\left(\\frac{\\sigma}{r_{ij}}\\right)^{12}-\\left(\\frac{\\sigma}{r_{ij}}\\right)^6\\right], \\qquad \\mathbf F_i=-\\nabla_i E.$$

The reference values are $\\epsilon=0.0104$ eV, $\\sigma=3.40$ Å, and a 7.0 Å hard cutoff. Generated labels come from the stated 12–6 Lennard-Jones equation. They are not electronic-structure data.
""",
    ),
    code(
        "n07-argon-constants",
        """
REFERENCE_EPSILON_EV = 0.0104
REFERENCE_SIGMA_A = 3.40
CUTOFF_A = 7.0
ARGON_SAMPLES = 36
VALIDATION_SAMPLES = 8
""",
    ),
    code(
        "n07-argon-generate",
        """
argon_records = helpers.generate_argon_records(
    count=ARGON_SAMPLES,
    seed=SEED,
    epsilon_eV=REFERENCE_EPSILON_EV,
    sigma_A=REFERENCE_SIGMA_A,
    cutoff_A=CUTOFF_A,
    dtype=dtype,
    device=device,
)
""",
    ),
    code(
        "n07-argon-split",
        """
argon_train_records, argon_validation_records, argon_split_frame = (
    helpers.split_argon_records(
        argon_records,
        validation_count=VALIDATION_SAMPLES,
        seed=71,
    )
)
""",
    ),
    code(
        "n07-argon-loaders",
        """
argon_train_loader = helpers.make_loader(argon_train_records, batch_size=4)
argon_validation_loader = helpers.make_loader(argon_validation_records, batch_size=4)
argon_example = next(iter(argon_train_loader))
""",
    ),
    code(
        "n07-argon-split-summary",
        """
argon_split_summary = (
    argon_split_frame.groupby("split", sort=False)
    .agg(samples=("sample_id", "count"), energy_min_eV=("energy (eV)", "min"), energy_max_eV=("energy (eV)", "max"))
    .reset_index()
)
""",
    ),
    code(
        "n07-argon-split-display",
        """
display(argon_split_summary)
""",
    ),
    code(
        "n07-argon-shapes",
        """
{
    "sample_id": argon_example.sample_id.flatten().tolist(),
    "graphs": argon_example.num_graphs,
    "atoms": argon_example.num_nodes,
    "positions": tuple(argon_example.positions.shape),
    "energy target shape": tuple(argon_example.energy.shape),
    "force target shape": tuple(argon_example.forces.shape),
    "units": {"positions": "Å", "energy": "eV", "forces": "eV/Å"},
    "dtype": str(argon_example.positions.dtype),
    "device": argon_example.device.type,
}
""",
    ),
    md(
        "n07-argon-scope",
        """
Each structure has four argon atoms. Energy is one total per structure with shape `(1, 1)` before batching; forces have shape `(4, 3)` per structure. The deterministic split holds out eight identities. Training and validation share the same generated potential family, so this is parameter recovery inside a controlled model family.

The fit does not establish transferability to bulk argon, other state points, larger clusters, or electronic-structure reference data. A lower training loss is not evidence of broader scientific accuracy.
""",
    ),
    md(
        "n07-argon-split-question",
        """
**Question:** Does the held-out split cover the same pair-distance range as the training split?
""",
    ),
    code(
        "n07-argon-split-plot",
        """
argon_split_figure = helpers.plot_argon_split(argon_records, argon_split_frame)
display(
    helpers.render_figure(
        argon_split_figure,
        alt_text="Pair distances in generated Ar4 structures, colored by train and validation split.",
    )
)
plt.close(argon_split_figure)
""",
        alt="Pair distances in generated Ar4 structures, colored by train and validation split.",
    ),
    md(
        "n07-argon-split-takeaway",
        """
Both colors span compressed and expanded tetrahedra. The held-out result therefore checks interpolation over this generator, not extrapolation to a new structural regime.

**Go deeper:** revisit [`Batch` ownership and recovery in Part 01](../01-atomicdata-batch/atomicdata-and-batch.ipynb) before replacing these in-memory records with a streamed dataset.
""",
    ),
    md(
        "n07-wrapper-heading",
        """
### A distance-based, neighbor-aware wrapper

`TrainableLennardJones` subclasses `torch.nn.Module` and `BaseModelMixin`. It owns two scalar parameters, `log_epsilon` and `log_sigma`, whose exponentials stay positive. `ModelConfig` requests a half COO neighbor list. The forward pass uses only pair displacement norms and scatters pair energies and equal-and-opposite forces back to systems and atoms.

For periodic inputs, the local helper applies Toolkit's convention `r_j - r_i + shift @ cell`. The generated fit remains non-periodic, but a tested two-atom boundary case protects that sign convention.
""",
    ),
    code(
        "n07-wrapper-construct",
        """
initial_lj = helpers.TrainableLennardJones(
    epsilon_eV=0.0070,
    sigma_A=3.10,
    cutoff_A=CUTOFF_A,
).to(device=device, dtype=dtype)
""",
    ),
    code(
        "n07-wrapper-config",
        """
lj_config = initial_lj.model_config
{
    "BaseModelMixin": isinstance(initial_lj, BaseModelMixin),
    "ModelConfig": isinstance(lj_config, ModelConfig),
    "NeighborConfig": isinstance(lj_config.neighbor_config, NeighborConfig),
    "outputs": sorted(lj_config.outputs),
    "neighbor format": lj_config.neighbor_config.format.value,
    "half list": lj_config.neighbor_config.half_list,
    "cutoff (Å)": lj_config.neighbor_config.cutoff,
}
""",
    ),
    code(
        "n07-wrapper-parameters",
        """
lj_parameter_ownership_before = pd.DataFrame(
    {
        "parameter": [name for name, _ in initial_lj.named_parameters()],
        "shape": [tuple(value.shape) for _, value in initial_lj.named_parameters()],
        "requires_grad": [value.requires_grad for value in initial_lj.parameters()],
        "physical value": [float(initial_lj.epsilon_eV), float(initial_lj.sigma_A)],
        "unit": ["eV", "Å"],
    }
)
""",
    ),
    code(
        "n07-wrapper-parameters-display",
        """
display(lj_parameter_ownership_before)
""",
    ),
    code(
        "n07-wrapper-spec",
        """
checkpoint_recipe = initial_lj.checkpoint_spec()
{
    "spec target": checkpoint_recipe.cls_path,
    "rebuilt type": type(checkpoint_recipe.build()).__name__,
    "pickle required": False,
}
""",
    ),
    md(
        "n07-wrapper-spec-interpretation",
        """
`checkpoint_spec()` stores the importable class path and constructor arguments as JSON-compatible metadata. The checkpoint stores the current parameter tensors separately. Rebuilding the module and then loading its state avoids Python pickle as the architecture recipe.
""",
    ),
    code(
        "n07-wrapper-neighbors",
        """
compute_neighbors(
    argon_example,
    config=initial_lj.model_config.neighbor_config,
)
initial_prediction = initial_lj(argon_example)
""",
    ),
    code(
        "n07-wrapper-output-check",
        """
wrapper_output_check = pd.DataFrame(
    {
        "field": ["energy", "forces", "neighbor_list"],
        "shape": [
            tuple(initial_prediction["energy"].shape),
            tuple(initial_prediction["forces"].shape),
            tuple(argon_example.neighbor_list.shape),
        ],
        "unit": ["eV", "eV/Å", "index pairs"],
        "owner": ["model output", "model output", "Batch edge storage"],
    }
)
""",
    ),
    code(
        "n07-wrapper-output-display",
        """
display(wrapper_output_check)
""",
    ),
    md(
        "n07-invariance-callout",
        """
<div style="display:block;box-sizing:border-box;width:100%;max-width:100%;min-width:0;background:#F2F3F1;color:#1B1E20;border:1px solid #D6D9D4;border-radius:8px;padding:0.78rem 0.95rem;line-height:1.45;overflow:hidden;overflow-wrap:anywhere;">
  <div style="color:#725B22;font-size:0.78rem;font-weight:700;letter-spacing:0.03em;margin-bottom:0.25rem;">💡 Highlight</div>
  <div style="min-width:0;font-size:0.98rem;overflow-wrap:anywhere;">The trainable wrapper depends on neighbor distances, so translation and rotation leave energy unchanged and rotate forces with the structure. This invariant construction is the scientific difference from the ordered-coordinate MLP.</div>
</div>
""",
    ),
    md(
        "n07-loss-heading",
        """
### Combine energy and force objectives

`EnergyMSELoss` has units eV² and `ForceMSELoss` has units eV²/Å². We divide each component by a stated squared scale before summing. The combined loss is an optimization scalar, not a physical observable. Component RMSE values retain the units used for interpretation.
""",
    ),
    code(
        "n07-loss-components",
        """
ENERGY_SCALE_EV = 0.02
FORCE_SCALE_EV_A = 0.02
energy_loss = EnergyMSELoss()
force_loss = ForceMSELoss(normalize_by_atom_count=True)
""",
    ),
    code(
        "n07-loss-composition",
        """
argon_loss = ComposedLossFunction(
    [energy_loss, force_loss],
    weights=[1.0 / ENERGY_SCALE_EV**2, 1.0 / FORCE_SCALE_EV_A**2],
    normalize_weights=False,
)
""",
    ),
    code(
        "n07-argon-observers",
        """
argon_ownership = helpers.ParameterOwnershipRecorder()
first_history = helpers.TrainingHistory(initial_lj)
argon_validation_recorder: BatchValidationCallback = helpers.ValidationBatchRecorder()
argon_neighbor_hook = NeighborListHook(
    initial_lj.model_config.neighbor_config,
    stage=TrainingStage.BEFORE_FORWARD,
)
""",
    ),
    code(
        "n07-argon-validation",
        """
argon_validation_config = ValidationConfig(
    validation_data=argon_validation_loader,
    every_n_steps=7,
    batch_callback=argon_validation_recorder,
    name="argon-validation",
)
""",
    ),
    md(
        "n07-checkpoint-heading",
        """
### Stop at a checkpoint boundary

The training split contains seven four-structure minibatches. The first strategy stops after seven optimizer updates, so the interruption boundary falls at the end of one deterministic pass. `CheckpointHook` writes there, and an explicit `save_checkpoint` call captures the same completed state before the in-memory strategy is set aside.

This is a planned interruption demonstration. It exercises reconstruction, counters, model state, and optimizer state without deliberately crashing the kernel.
""",
    ),
    code(
        "n07-checkpoint-directory",
        """
checkpoint_root = helpers.reset_checkpoint_directory(
    Path("artifacts/checkpoints")
)
""",
    ),
    code(
        "n07-checkpoint-hook",
        """
checkpoint_hook = CheckpointHook(
    checkpoint_root,
    step_interval=7,
    async_save=False,
)
""",
    ),
    code(
        "n07-argon-optimizer",
        """
argon_optimizer_config = OptimizerConfig(
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs={"lr": 3.0e-2},
)
""",
    ),
    code(
        "n07-first-strategy",
        """
first_strategy = FineTuningStrategy(
    models=initial_lj,
    trainable_patterns=("main.log_epsilon", "main.log_sigma"),
    freeze_mode="requires_grad",
    optimizer_configs=argon_optimizer_config,
    training_fn=default_training_fn,
    loss_fn=argon_loss,
    validation_config=argon_validation_config,
    num_steps=7,
    devices=[device],
    hooks=[argon_neighbor_hook, argon_ownership, first_history, checkpoint_hook],
)
""",
    ),
    code(
        "n07-first-run",
        """
first_strategy.run(argon_train_loader)
""",
    ),
    code(
        "n07-manual-checkpoint",
        """
interruption_checkpoint_index = first_strategy.save_checkpoint(
    checkpoint_root
)
""",
    ),
    code(
        "n07-checkpoint-state",
        """
checkpoint_state = pd.DataFrame(
    {
        "state": ["completed updates", "periodic index", "interruption index"],
        "value": [
            first_strategy.step_count,
            checkpoint_hook.last_checkpoint_index,
            interruption_checkpoint_index,
        ],
    }
)
""",
    ),
    code(
        "n07-checkpoint-display",
        """
display(checkpoint_state)
""",
    ),
    md(
        "n07-checkpoint-interpretation",
        """
The periodic and explicit saves receive successive manifest indices. From this point onward, the notebook uses a newly reconstructed strategy rather than the first in-memory object.
""",
    ),
    md(
        "n07-resume-heading",
        """
### Reattach runtime data and resume

Validation loaders and runtime hooks are intentionally not serialized. `FineTuningStrategy.load_checkpoint` rebuilds the model from `checkpoint_spec()`, restores model and optimizer tensors plus counters, and accepts fresh runtime hooks. We then reattach `ValidationConfig` and extend the target to 49 completed updates.
""",
    ),
    code(
        "n07-resume-load",
        """
resume_history = helpers.TrainingHistory()
resumed = FineTuningStrategy.load_checkpoint(
    checkpoint_root,
    map_location=device,
    hooks=[resume_history],
)
resumed_model = resumed.models["main"]
""",
    ),
    code(
        "n07-resume-hooks",
        """
resume_neighbor_hook = NeighborListHook(
    resumed_model.model_config.neighbor_config,
    stage=TrainingStage.BEFORE_FORWARD,
)
resumed_checkpoint_hook = CheckpointHook(
    checkpoint_root,
    step_interval=7,
    async_save=False,
)
resumed.register_hook(resume_neighbor_hook)
resumed.register_hook(resumed_checkpoint_hook)
""",
    ),
    code(
        "n07-resume-config",
        """
resumed.validation_config = argon_validation_config
resumed.num_steps = 49
{
    "restored completed updates": resumed.step_count,
    "target completed updates": resumed.num_steps,
    "model type": type(resumed_model).__name__,
}
""",
    ),
    code(
        "n07-resume-run",
        """
resumed.run(argon_train_loader)
""",
    ),
    code(
        "n07-argon-history-combine",
        """
argon_training_rows = pd.concat(
    [first_history.training_frame(), resume_history.training_frame()],
    ignore_index=True,
)
argon_validation_rows = pd.concat(
    [first_history.validation_frame(), resume_history.validation_frame()],
    ignore_index=True,
)
""",
    ),
    code(
        "n07-argon-result-frame",
        """
argon_fit_result = pd.DataFrame(
    {
        "parameter": ["epsilon", "sigma"],
        "initial": [0.0070, 3.10],
        "fitted": [float(resumed_model.epsilon_eV), float(resumed_model.sigma_A)],
        "reference": [REFERENCE_EPSILON_EV, REFERENCE_SIGMA_A],
        "unit": ["eV", "Å"],
    }
)
""",
    ),
    code(
        "n07-argon-result-display",
        """
display(argon_fit_result)
""",
    ),
    code(
        "n07-argon-validation-tail",
        """
display(argon_validation_rows.tail(4))
""",
    ),
    code(
        "n07-argon-final-metrics",
        """
final_validation_row = argon_validation_rows.iloc[-1]
argon_final_metrics = pd.DataFrame(
    {
        "metric": ["energy RMSE", "force RMSE", "scaled total loss"],
        "value": [
            final_validation_row["energy MSE (eV²)"] ** 0.5,
            final_validation_row["force MSE (eV²/Å²)"] ** 0.5,
            final_validation_row["total loss"],
        ],
        "unit": ["eV", "eV/Å", "dimensionless"],
    }
)
""",
    ),
    code(
        "n07-argon-final-metrics-display",
        """
display(argon_final_metrics)
""",
    ),
    md(
        "n07-argon-results-interpretation",
        """
The fresh CPU execution completed 49 updates, recovered epsilon as 0.010188 eV and sigma as 3.409309 Å, and ended with held-out energy RMSE 0.001028 eV and force RMSE 0.000617 eV/Å. These are the bounded results for this generated split, not accuracy estimates for external argon data.

A validation pass appears at each seven-update boundary and once at training exit, so duplicate boundary rows are expected. The held-out values are reported even if they fail to improve.
""",
    ),
    md(
        "n07-argon-plot-question",
        """
**Question:** Do the physical component errors and fitted epsilon/sigma approach the generated reference together?

The panels report Energy RMSE (eV), Force RMSE (eV/Å), Epsilon (eV), and Sigma (Å) beside the dimensionless scaled loss.
""",
    ),
    code(
        "n07-argon-training-plot",
        """
argon_training_figure = helpers.plot_argon_training(
    argon_training_rows,
    argon_validation_rows,
    reference_epsilon_eV=REFERENCE_EPSILON_EV,
    reference_sigma_A=REFERENCE_SIGMA_A,
)
display(
    helpers.render_figure(
        argon_training_figure,
        alt_text="Generated argon training and validation loss, energy and force RMSE, and fitted epsilon and sigma traces.",
    )
)
plt.close(argon_training_figure)
""",
        alt="Generated argon training and validation loss, energy and force RMSE, and fitted epsilon and sigma traces.",
    ),
    md(
        "n07-argon-plot-interpretation",
        """
Loss and parameter traces answer different questions. The loss panels show agreement with generated labels under the chosen scales. The parameter panel checks recovery inside the same two-parameter family. Neither establishes transferability beyond it.
""",
    ),
    md(
        "n07-transfer-heading",
        """
### Transfer fitted scalars to the built-in wrapper

The trainable helper exists to expose gradients and checkpoint reconstruction. Later simulation and domain workflows can use Toolkit's built-in `LennardJonesModelWrapper`, which computes analytical energy and forces with Toolkit-Ops. We transfer only the fitted scalar values and keep the same cutoff, switching policy, and half-list policy.
""",
    ),
    code(
        "n07-transfer-construct",
        """
fitted_epsilon_eV = float(resumed_model.epsilon_eV.detach().cpu())
fitted_sigma_A = float(resumed_model.sigma_A.detach().cpu())
deployment_lj = LennardJonesModelWrapper(
    epsilon=fitted_epsilon_eV,
    sigma=fitted_sigma_A,
    cutoff=CUTOFF_A,
    switch_width=0.0,
    half_list=True,
)
""",
    ),
    code(
        "n07-transfer-custom",
        """
transfer_batch = next(iter(argon_validation_loader)).to(device)
compute_neighbors(
    transfer_batch,
    config=resumed_model.model_config.neighbor_config,
)
trainable_output = resumed_model(transfer_batch)
""",
    ),
    code(
        "n07-transfer-built-in",
        """
compute_neighbors(
    transfer_batch,
    config=deployment_lj.model_config.neighbor_config,
)
built_in_output = deployment_lj(transfer_batch)
""",
    ),
    code(
        "n07-transfer-check",
        """
transfer_check = pd.DataFrame(
    {
        "field": ["energy", "forces"],
        "shape": [tuple(built_in_output["energy"].shape), tuple(built_in_output["forces"].shape)],
        "maximum absolute difference": [
            float((built_in_output["energy"] - trainable_output["energy"]).abs().max()),
            float((built_in_output["forces"] - trainable_output["forces"]).abs().max()),
        ],
        "unit": ["eV", "eV/Å"],
    }
)
""",
    ),
    code(
        "n07-transfer-display",
        """
display(transfer_check)
""",
    ),
    md(
        "n07-transfer-interpretation",
        """
The fresh CPU check found maximum differences below $1.0\\times10^{-17}$ eV for energy and $1.0\\times10^{-17}$ eV/Å for forces. This checks implementation parity for these structures and settings. The built-in wrapper stores epsilon and sigma as deployment inputs rather than trainable `Parameter` objects. Use it with [Part 05 dynamics](../05-base-dynamics/base-dynamics.ipynb) after choosing a scientifically appropriate system and validation protocol.

**Go deeper:** review [model interfaces and composition in Part 03](../03-model-interfaces-composition/model-interfaces-composition.ipynb) and the official [training checkpoint guide](https://nvidia.github.io/nvalchemi-toolkit/modules/training/checkpoints.html).
""",
    ),
    md(
        "n07-model-boundaries",
        """
## Pretrained and larger-model boundaries in this lock

The pinned `AIMNet2Wrapper` loads its supported checkpoint through `from_checkpoint`, but it does not expose the pickle-free reconstruction specification required by `FineTuningStrategy.from_pretrained_checkpoint`. Passing a raw AIMNet `.pt` file to that fine-tuning constructor would therefore be misleading. This notebook documents the boundary instead of monkey-patching the wrapper.

The official advanced examples show how the same strategy pieces extend to larger architectures. [MACE training](https://github.com/NVIDIA/nvalchemi-toolkit/blob/8c2c307c/examples/advanced/10_mace_training.py) and UMA may be useful references, but they are **not runnable in this frozen environment**. Their model dependencies and supported checkpoint paths must be validated in a compatible environment first.

**Go deeper:** read the official [fine-tuning guide](https://nvidia.github.io/nvalchemi-toolkit/userguide/training_finetuning.html) before adapting a supported pretrained model.
""",
    ),
    md(
        "n07-try-heading",
        """
## Try it: change optimizer ownership

Change `try_pattern` from `main.backbone.*` to `main.readout.*`, run one update on a copied model, and inspect the table. Success means every optimizer-owned row matches the chosen glob. The loss value is not the success criterion.
""",
    ),
    code(
        "n07-try-setup",
        """
try_pattern = "main.backbone.*"
try_model = copy.deepcopy(toy_model)
try_ownership = helpers.ParameterOwnershipRecorder()
""",
    ),
    code(
        "n07-try-strategy",
        """
try_strategy = FineTuningStrategy(
    models=try_model,
    trainable_patterns=(try_pattern,),
    freeze_mode="requires_grad",
    optimizer_configs=toy_optimizer_config,
    training_fn=default_training_fn,
    loss_fn=toy_loss,
    num_steps=1,
    devices=[device],
    hooks=[try_ownership],
)
""",
    ),
    code(
        "n07-try-run",
        """
try_strategy.run(toy_train_loader)
try_ownership_frame = try_ownership.frame()
""",
    ),
    code(
        "n07-try-display",
        """
display(try_ownership_frame)
""",
    ),
    code(
        "n07-try-check",
        """
owned_names = try_ownership_frame.loc[
    try_ownership_frame["optimizer_owned"], "parameter"
].tolist()
assert owned_names and all(name.startswith("main.backbone.") for name in owned_names)
owned_names
""",
    ),
    md(
        "n07-recap",
        """
## Recap

### What you learned

- `FineTuningStrategy` resolves parameter globs before `OptimizerConfig` builds optimizers.
- `EnergyMSELoss`, `ForceMSELoss`, and `ComposedLossFunction` keep component meaning visible while producing one optimization scalar.
- `ValidationConfig`, `BatchValidationCallback`, ordinary hooks, and `ReportingOrchestrator` observe different parts of the lifecycle.
- `checkpoint_spec()`, `CheckpointHook`, `save_checkpoint`, and `load_checkpoint` separate reconstructable configuration from tensor state.
- An invariant neighbor-distance wrapper can fit generated Lennard-Jones energy and force labels, then transfer fitted epsilon/sigma into `LennardJonesModelWrapper`.

### How we will use this

Course foundation: [Core Playbook](../00-core-playbook/alchemi-core-playbook.ipynb)  
Model handoff: [Part 03 · Model interfaces and composition](../03-model-interfaces-composition/model-interfaces-composition.ipynb)  
Simulation handoff: [Part 05 · BaseDynamics and FIRE2](../05-base-dynamics/base-dynamics.ipynb)  
Scale handoff: **[Part 08 · Domain decomposition](../08-domain-decomposition/domain-decomposition.ipynb) is in progress.**

The checkpoint proves restartability for this controlled fit. Human scientific review is still required before treating any fitted potential as suitable for a new argon workflow.
""",
    ),
]

notebook = new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
)
nbformat.validate(notebook)
nbformat.write(notebook, HERE / "training-finetuning.ipynb")
