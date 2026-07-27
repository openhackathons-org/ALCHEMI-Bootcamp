# Part 3: OLED Melting Point predictions with ALCHEMI Toolkit

This notebook predicts the melting point of a molecular crystal with
[NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) and the
Orb-v3 molecular potential. Naphthalene is the example system for a complete
solid-liquid coexistence workflow.

## Deployment

This retained notebook does not run in the remastered Part 1 image. Orb 0.7.0
requires Toolkit-Ops below 0.4, while Part 1 uses Toolkit-Ops 0.4. Keep the
historical Orb environment separate instead of forcing incompatible package
versions into one environment.

The Part 3 environment and validation with the current Toolkit versions still
need to be rebuilt before learner use. No NGC API key is required for the
historical workflow.
