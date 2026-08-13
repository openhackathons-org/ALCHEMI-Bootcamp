# Current-pin 1/2/4-GPU campaign

This runner records the evidence that the notebook deliberately does not
produce. It consumes the checked periodic phenol/N-methylacetamide base box,
builds the same 2 × 2 × 4 supercell for every case, and evaluates the
51,200-atom input with the current pinned `AIMNet2Wrapper`.

The box is a deterministic supercell of a Packmol starting geometry at its
declared construction density. It is not an equilibrated liquid, a density
prediction, or a production thermodynamics setup. The selected model route is
the short-range AIMNet2 adapter only. Its purpose here is decomposition parity,
ownership, and controlled timing.

## Preconditions

- Use the tutorials v3 locked environment on a single node with 4 mutually
  visible GPUs. Multi-node behavior is outside this campaign.
- Confirm that the installed Toolkit and Toolkit-Ops distributions expose VCS
  provenance for the current pins. The runner rejects any other commits.
- Let the official AIMNet registry resolve `aimnet2-wb97m-d3_0`, or pass a
  pre-downloaded checkpoint with `--checkpoint`. Its SHA-256 must match the
  campaign specification.
- Start in this `external/` directory. No command asks for interactive input.

## Run serially

Run these cases one at a time. Do not start the next command until the previous
command exits successfully and its case JSON has been written. This serialization
avoids overlapping GPU work and makes the campaign manifest's partial state
meaningful.

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  run_current_pin_campaign.py \
  --world-size 1 \
  --output-root ../results/current-pin
```

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  run_current_pin_campaign.py \
  --world-size 2 \
  --output-root ../results/current-pin
```

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  run_current_pin_campaign.py \
  --world-size 4 \
  --output-root ../results/current-pin
```

Each case partitions once, performs one untimed warm-up, synchronizes ranks,
records three equal-work evaluations using the maximum elapsed rank time, and
gathers once. The record includes exact input and checkpoint checksums,
installed commits, per-rank owned-atom counts, source-atom IDs, forces, energy,
GPU identity, and runtime versions. Halo counts remain explicitly unavailable
because the public API does not expose them.

## Promotion rule

The output campaign remains partial until all three cases exist. The final case
triggers the same checksum, ownership, and energy/force parity gate used by the
notebook. A failed gate changes the manifest to `INVALID`; it does not produce a
plot. A complete manifest still marks its timing as a short, fixed-input
observation rather than a publishable benchmark.

Until these commands finish under the current pins, the notebook status is
`NOT REPORTED`. Archived H100 records from older pins are methodology only, and
no result is publishable by copying their values into this manifest.
