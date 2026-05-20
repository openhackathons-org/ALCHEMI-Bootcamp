# ALCHEMI Bootcamp

This repository contains one hands-on tutorial:

[`part-1-batched-adsorption/alchemi-mace-adsorption-search.ipynb`](part-1-batched-adsorption/alchemi-mace-adsorption-search.ipynb)

The notebook shows how batched, GPU-native atomistic simulation changes the
scale of adsorption-configuration search. It uses NVIDIA ALCHEMI Toolkit with
standard Python structure tools, open MACE checkpoints, OC20Dense validation
records, and OVITO visualization.

## Requirements

- NVIDIA GPU
- ALCHEMI Toolkit-capable Jupyter kernel
- The bundled OC20Dense validation pack:
  `part-1-batched-adsorption/data/reference/oc20dense-validation-pack.tgz`

Generated outputs, expanded reference data, runtime caches, and local review
artifacts are ignored by git.

## License

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

Third-party model/data artifacts keep their own upstream terms, including
MACE-MP/MACE-MPA checkpoints under MIT and OC20/OC20Dense data under CC BY 4.0.
