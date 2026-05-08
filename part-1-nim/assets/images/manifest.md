# Adsorption Phenomenon Showcase Panels

Generated as independent 16:9 PNG panels for assembling tutorial banners,
section cards, or split-view figures. These are phenomenon-class visuals,
not pair-specific validation figures.

## Version 0: Core Tutorial Concepts

- `v0_core/01_catalysis_configuration_search.png`:
  catalyst adsorption configuration search as the central tutorial motif.
- `v0_core/02_catalyst_discovery_workflow.png`:
  many adsorption candidates batched through GPU relaxation and ranked.
- `v0_core/workflow_adsorbml_bgr.png`:
  current notebook batching figure with a central GPU connected to independent adsorption panels.
- `v0_core/phenomenon_local_minima.png`:
  potential-energy surface with multiple local minima and one green global minimum.
- `v0_core/alchemi_toolkit_community_ops.png`:
  ALCHEMI community, NIM, Toolkit, and Toolkit-Ops architecture figure with neighbor-list, DFT-D3, electrostatics/PME, batched-kernel, and JAX feature callouts.
- `v0_core/03_methanol_oxygenate_adsorption.png`:
  methanol and oxygenate adsorption on a catalyst surface.
- `v0_core/04_co_hydrogenation_surface_context.png`:
  CO hydrogenation surface context.
- `v0_core/05_oxide_support_adsorption.png`:
  oxide-support adsorption context.
- `v0_core/06_awh_first_water_binding.png`:
  first water-binding context for porous sorbents.
- `v0_core/07_oer_first_adsorption.png`:
  OER first-adsorption context.
- `v0_core/08_nh3_surface_binding.png`:
  NH3 surface-binding context.
- `v0_core/09_nh3_synthesis_binding_context.png`:
  NH3 synthesis binding context.
- `v0_core/10_adsorption_regime_comparison.png`:
  comparison of adsorption regimes.

## Version 1: Phenomenon Panels

- `v1/01_catalysis_adsorption_landscape.png`:
  heterogeneous catalysis as competing adsorption minima on active sites.
- `v1/02_oer_electrocatalysis_adsorption.png`:
  OER electrocatalysis as oxide-surface adsorption-intermediate science.
- `v1/03_water_harvesting_sorption.png`:
  atmospheric water harvesting as vapor sorption and pore-scale nucleation.
- `v1/04_co2_capture_gas_separation.png`:
  selective CO2 capture as gas separation by preferential adsorption.
- `v1/05_catalyst_poisoning_deactivation.png`:
  catalyst deactivation by overly strong poison adsorption.

## Version 2: Process / Result Split Panels

- `v2_split/01_catalysis_process_result.png`:
  candidate configuration sampling on the left, selected adsorption minimum on the right.
- `v2_split/02_oer_process_result.png`:
  adsorbed OER intermediates on the left, oxygen-evolution outcome on the right.
- `v2_split/03_water_harvesting_process_result.png`:
  pore-scale water uptake on the left, desert droplet accumulation on the right.
- `v2_split/04_co2_capture_process_result.png`:
  mixed-gas adsorption on the left, CO2-depleted outlet / loaded sorbent on the right.
- `v2_split/05_poisoning_process_result.png`:
  clean active sites on the left, blocked poisoned surface on the right.

## Shared Style Prompt

Dark graphite background, matte atoms/materials, soft top-left studio key
light, subdued fill light, NVIDIA-green accents for active adsorption sites,
cyan accents for water/gas-interface motion, no embedded text, no logos, and
no UI frame. Visible molecules were prompted with simple geometry constraints:
linear CO2 and CO, bent H2O, diatomic N2 where used, and sulfur poisoning as
a yellow strongly bound surface species.

## Version 3: Clean Illustration Panels

- `v3_illustration/01_oer_three_intermediates.png`:
  clean OER intermediate illustration.
- `v3_illustration/02_water_harvesting_clean_sorption.png`:
  clean water-sorption illustration.
- `v3_illustration/03_co2_n2_clean_separation.png`:
  clean CO2/N2 separation illustration.

## Version 4: Icon-Style Illustration Panels

These are the least photorealistic and most assembly-friendly panels. They use
cartoon/editorial scientific illustration: simplified silhouettes, matte atoms,
bold outlines, constrained molecule counts, and reduced visual clutter.

- `v4_icon_style/01_oer_three_intermediates_icon.png`:
  three oxide slabs side by side with OH, O, and OOH adsorbed.
- `v4_icon_style/02_water_harvesting_sorption_icon.png`:
  six isolated bent H2O molecules and a regular manufactured porous adsorbent.
- `v4_icon_style/03_co2_n2_separation_icon.png`:
  sparse CO2/N2 separation through a regular porous block.
- `v4_icon_style/04_catalyst_poisoning_icon.png`:
  before/after active-site blocking by a sulfur poison.
- `v4_icon_style/05_d3_dispersion_icon.png`:
  weak nonbonded dispersion attraction between two neutral fragments.
- `v4_icon_style/06_electrostatics_pme_icon.png`:
  long-range electrostatic field lines through a periodic simulation cell.

Detailed generation prompts are recorded in
`v4_icon_style/prompts.md`.
