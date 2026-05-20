# Tutorial Image Manifest

This manifest lists the `assets/images/` files that are part of the current
tutorial path. Older comparison drafts were moved to the local archive during
the GitHub-clean cleanup.

## Notebook Banner

- `banner_adsorption_scale_alchemi.jpg`:
  current first-viewport raster banner exported from the selected presentation
  replacement slide and compressed for notebook display.

## Partner Logos

- `logos/nvidia-logo.png`, `logos/eneos-orange.png`, `logos/matlantis.png`, and
  `logos/ovito_logo.png`:
  normalized logo strip used at the top of the notebook and Part 1 README.
  Compatibility copies of the ENEOS and Matlantis assets may remain in
  `assets/` while older notebook buffers are still open.

## Core Tutorial Figures

- `discovery_funnel.png`:
  current notebook chemical-discovery funnel from broad chemical space through narrowing, candidate structures, accelerated simulation, ranked outputs, and DFT/experiment validation.
- `workflow_gpu_batching_with_gpu.png`:
  current notebook GPU-batching search visual: one unified abstract slab scene
  with independent green, yellow, and orange sphere starts converging into an
  integrated GPU accelerator.
- `alchemi_toolkit_architecture.png`:
  current ALCHEMI architecture figure exported from the selected presentation
  replacement slide and compressed for notebook display.
- `adsorption_slab_1x_vs_20x_black.png`:
  compact black-background batching summary used in the final adsorption-screen
  interpretation cell.

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

Selected candidates should be kept as flat files in this directory before
linking them from the notebook. Older exploratory image-generation and
OVITO-fusion drafts may still exist locally under ignored
`assets/images/_review_candidates/`, but they are not part of the shipped
notebook path.
