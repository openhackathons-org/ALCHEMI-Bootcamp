# Tutorial Image Manifest

This manifest lists the `assets/images/` files that are part of the current
tutorial path. Older comparison drafts were moved to the local archive during
the GitHub-clean cleanup.

## Notebook Banner

- `banner_adsorbml_toolkit_abstract_spheres.png`:
  current first-viewport raster banner: a unified abstract slab scene with
  independent green, yellow, and orange spheres.

## Core Tutorial Figures

- `discovery_funnel.png`:
  current notebook chemical-discovery funnel from broad chemical space through narrowing, candidate structures, accelerated simulation, ranked outputs, and DFT/experiment validation.
- `workflow_gpu_batching_with_gpu.png`:
  current notebook GPU-batching search visual: one unified abstract slab scene
  with independent green, yellow, and orange sphere starts converging into an
  integrated GPU accelerator.
- `alchemi_toolkit_community_ops.png`:
  ALCHEMI community, NIM, Toolkit, and Toolkit-Ops architecture figure with neighbor-list, DFT-D3, electrostatics/PME, batched-kernel, and JAX feature callouts.

## Shared Style Prompt

Dark graphite background, matte atoms/materials, soft top-left studio key
light, subdued fill light, NVIDIA-green accents for active adsorption sites,
cyan accents for water/gas-interface motion, no embedded text, no logos, and
no UI frame. Abstract batching visuals should use independent colored spheres
instead of molecule-like adsorbates. Where visible molecules are intentionally
used, prompt them with simple geometry constraints: linear CO2 and CO, bent
H2O, diatomic N2, and sulfur poisoning as a yellow strongly bound surface
species.

## Visual Review

<mark>TODO - VISUAL REVIEW</mark>: keep selected candidates as flat files in
this directory before linking them from the notebook. Older exploratory
image-generation and OVITO-fusion drafts may still exist locally under ignored
`assets/images/_review_candidates/`, but they are not part of the shipped
notebook path.
