# N07 worklog — Training and fine-tuning

## 2026-08-12 — initial curriculum brief

Status: planned

- Outcome: assemble a pickle-free `TrainingStrategy` from replaceable loss,
  optimizer, validation, logging, checkpoint, and fine-tuning pieces.
- Core lesson: inspect and extend the components and their hook interfaces.
- Optional work: a complete training run belongs in
  `Advanced — if time permits / homework.`
- Packaging: verify the selected model and dataset licenses before distribution.
- Continuity: reuse Part 02 data interfaces, Part 03 model metadata, and Part 04
  hooks.

## 2026-08-13 — implementation design brief

Status: authoring

- Outcome: build and inspect a restartable Toolkit fine-tuning workflow, first
  with a transparent four-feature regression sandbox and then with generated
  argon energy/force labels from a stated Lennard-Jones reference.
- Prior knowledge: Parts 01–04 for `AtomicData`, `Batch`, model wrappers,
  neighbor construction, and hooks; the final Core training excerpt
  (`core-t63`–`core-t71`) supplies the short prerequisite sequence.
- Sequence: establish parameter ownership on a tiny MLP; show one prediction,
  loss, optimizer, split, validation callback, reporting hook, and several
  updates; then generate distorted Ar4 structures, verify analytical labels,
  build an invariant neighbor-based wrapper, stop at a saved checkpoint, resume
  into a fresh strategy, validate, plot losses and fitted parameters, and
  transfer epsilon/sigma to the built-in `LennardJonesModelWrapper`.
- Visuals: one MLP loss trace, one generated-Ar pair-distance/split view, one
  combined train/validation loss and epsilon/sigma trace, and a compact
  checkpoint-state table. Every plot carries units where applicable and
  exportable alt text.
- Visible public APIs: `FineTuningStrategy`, `OptimizerConfig`,
  `EnergyMSELoss`, `ForceMSELoss`, `ValidationConfig`,
  `BatchValidationCallback`, `CheckpointHook`, `RichReporter`,
  `default_training_fn`, `BaseModelMixin`, `ModelConfig`,
  `NeighborConfig`, `compute_neighbors`, and
  `LennardJonesModelWrapper`.
- Helper boundary: deterministic label generation, the trainable LJ wrapper,
  reusable observation hooks, plotting, and presentation setup remain in local
  tested helpers. Strategy construction, loss/optimizer configuration,
  checkpoint save/load, public neighbor calls, and parameter transfer remain
  visible in the notebook.
- Scientific scope: 36 isolated, non-periodic Ar4 configurations generated
  from a two-parameter 12–6 Lennard-Jones reference. Energies are per structure
  in eV; forces are per atom with shape `[N, 3]` in eV/Å. This identifies a
  controlled potential family, not a transferable argon model or a benchmark
  against electronic-structure data.
- Parameter ownership: MLP trainable patterns select only `readout.*`;
  the Ar wrapper owns trainable scalar log-epsilon and log-sigma while cutoff
  and neighbor policy remain configuration. Transferred built-in LJ scalars are
  deployment inputs, not continued training parameters.
- Device/runtime: one process chooses CUDA when available and otherwise CPU;
  no concurrent GPU work. Target fresh execution is under five minutes on CPU
  for the small deterministic examples.
- Exercise: change the MLP trainable pattern and inspect which named parameters
  can update; success is an explicit ownership table, not a lower validation
  loss claim.
- Checks: tests are written before helpers/notebook; unit checks cover LJ
  invariance, analytical force parity, checkpoint specs, and deterministic
  splits. Contract checks cover API visibility, claims, cell size, links,
  output freshness, alt text, and render width. Final validation includes
  scoped tests, fresh-kernel execution, notebook schema validation, lint/style
  checks, HTML render inspection, and two editorial revision passes.
- Pinned limitation: the AIMNet wrapper in this lock does not expose the
  pickle-free reconstruction specification needed by
  `FineTuningStrategy.from_pretrained_checkpoint`. The lesson documents that
  boundary and does not patch the wrapper or pass a raw `.pt` file to that API.
- Optional scale references: the official DDP dummy-MLP, Rich reporting,
  checkpoint, and advanced MACE examples are linked for architecture context.
  DDP is not launched from the notebook, and MACE/UMA are not presented as
  runnable in this frozen environment.

## 2026-08-13 — implementation and verification

Status: complete; human scientific review requested

- Delivered `notebooks/07-training-finetuning/training-finetuning.ipynb` with
  the two-level sequence, notebook-local helpers, and test-first checks.
- Fresh CPU execution completed in about 20 seconds in the pinned environment.
  The executed review copy included current tables, reporting output,
  checkpoint and resume evidence, three described figures, and transfer-parity
  results. The delivered notebook is output-clean, matching the other lessons.
- The generated-Ar fit recovered epsilon `0.010188` eV and sigma `3.401056` Å
  from the `0.010300` eV and `3.400000` Å reference. Final held-out energy RMSE
  was `0.001028` eV and force RMSE was `0.005159` eV/Å; these are reported as
  checks inside the generated two-parameter family, not as transferability
  evidence.
- Verification: 25 scoped tests passed, including fresh-kernel execution and
  rendered-HTML alt-text checks. Ruff lint and format checks passed, IDE
  diagnostics were clear, notebook schema validation passed, and nbconvert
  produced HTML without missing-alt warnings.
- Two editorial passes checked scientific claims, units, shapes, split
  provenance, parameter ownership, checkpoint semantics, local links, cell
  size, output freshness, plot labels, and wording.
- Human review remains appropriate for the final scientific framing and visual
  pacing before publication; no unresolved implementation defect is known.
