# Archived framework and kernel topology primer

Status: verified reference material, removed from the Core learner path on
2026-08-13. The likely destination is Part 06 (GPU pipelines and profiling) or
a future Toolkit internals tutorial.

## How ALCHEMI Toolkit reaches accelerated kernels

This tutorial uses ALCHEMI Toolkit through its PyTorch-facing APIs. In the
pinned release, ALCHEMI Toolkit calls Toolkit-Ops Torch bindings. Those
bindings route selected operations to accelerated Warp kernels; some
Toolkit-Ops Torch operations and utilities use documented Torch-native
implementations. Toolkit-Ops also provides separate JAX bindings, but ALCHEMI
Toolkit and this tutorial stay on the Torch branch.

![Highlighted path from ALCHEMI Toolkit PyTorch-facing APIs through Toolkit-Ops Torch bindings and a selected operation to accelerated Warp kernels. A muted JAX route remains separate from ALCHEMI Toolkit, and a secondary branch shows documented Torch-native Toolkit-Ops implementations.](../assets/framework-bindings.svg)

The arrows show API delegation toward an operation implementation. Results
return through the same binding. Green marks the path used throughout the
tutorial; the muted JAX branch is ecosystem context.

Editable source: [`framework-bindings.drawio`](../assets/framework-bindings.drawio)

Content-addressed snapshot:
[`framework-bindings-10dc3443f13552ac.svg`](../assets/framework-bindings-10dc3443f13552ac.svg)

## Verified topology notes

- Pinned Toolkit source `nvalchemi/neighbors.py` imports
  `nvalchemiops.torch.neighbors`; no Toolkit import enters the
  `nvalchemiops.jax` namespace.
- `nvalchemiops.jax` is a separate optional binding namespace for Warp-backed
  primitives.
- Toolkit-Ops also contains documented Torch-native operations and utilities.
- The corrected path is `ALCHEMI Toolkit → Toolkit-Ops Torch bindings →
  selected operation → accelerated Warp kernels`.
- The prior Part 01-style diagram merged framework inputs too early. This
  archived topology keeps the JAX path separate and records the Torch-native
  alternative.

Source records:

- `TOOLKIT_API_REFERENCE.md`, “Framework bindings”;
- `worklog/integration.md`, “Core framework topology and curriculum visual
  correction”;
- pinned Toolkit commit `8c2c307c1c0c76baee6f7a68eb75a45da83ffd18`;
- pinned Toolkit-Ops commit `c1e23460859a784e1d78043bcd1c8af0d1095fa2`.
