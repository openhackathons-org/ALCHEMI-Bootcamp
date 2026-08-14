# Third-party notices

This file covers the external software and runtime downloads used by the Core
playbook. The repository vendors only the 3Dmol.js browser bundle described
below. Model checkpoints and generated D3 parameters stay in the user's runtime
cache.

## NVIDIA ALCHEMI Toolkit and Toolkit-Ops

The tutorial pins NVIDIA ALCHEMI Toolkit and Toolkit-Ops to immutable Git
commits in `environment/runtime-pins.toml`.

- Toolkit source and Apache-2.0 license:
  <https://github.com/NVIDIA/nvalchemi-toolkit>
- Toolkit-Ops source and Apache-2.0 license:
  <https://github.com/NVIDIA/nvalchemi-toolkit-ops>

Toolkit uses PyTorch tensors in the Core playbook. PyTorch is distributed under
a BSD-style license. ASE supplies the molecular structures and is distributed
under LGPL-2.1-or-later.

- PyTorch source and license: <https://github.com/pytorch/pytorch>
- ASE source and license: <https://gitlab.com/ase/ase>

## py3Dmol and 3Dmol.js

The Core playbook displays ASE structures with `py3Dmol 2.5.5`, a Python
wrapper that emits a 3Dmol.js viewer as notebook HTML.

- py3Dmol: MIT license, <https://pypi.org/project/py3Dmol/>
- 3Dmol.js: BSD-3-Clause license, <https://github.com/3dmol/3Dmol.js>
- 3Dmol.js paper: <https://doi.org/10.1093/bioinformatics/btu829>

The repository bundles these files from the `3dmol` npm package at version
2.5.5:

| Bundled file | Contents |
|---|---|
| `shared/3dmol-2.5.5/3Dmol-min.js` | Minified 3Dmol.js renderer from `build/3Dmol-min.js` |
| `shared/3dmol-2.5.5/3Dmol-min.js.LICENSE.txt` | Attribution notice referenced by the minified file |
| `shared/3dmol-2.5.5/LICENSE` | BSD-3-Clause license text covering 3Dmol.js and incorporated code |

The npm integrity value and file SHA-256 values are recorded in
`environment/runtime-pins.toml` and checked by `environment/check_runtime.py`.

## AIMNet2 and `aimnet`

The Core playbook uses `aimnet 0.2.0` and ensemble member
`aimnet2_wb97m_d3_0` from the AIMNet2 ωB97M-D3 release.

- source and documentation: <https://github.com/isayevlab/aimnetcentral>
- package: <https://pypi.org/project/aimnet/>
- model card and weights: <https://huggingface.co/isayevlab/aimnet2-wb97m-d3>
- model paper: <https://doi.org/10.1039/D4SC08572H>

The source repository and model card list the software and weights under the
MIT license. Runtime setup downloads the checkpoint from the official model
registry, verifies its SHA-256 value, and stores it outside the repository.

## DFT-D3 reference parameters

Toolkit's D3 setup reads the legacy Grimme DFT-D3 source archive and generates
a parameter table in the user's runtime cache. The repository records and
checks the generated file's SHA-256 value; it does not distribute the generated
table or the legacy archive.

- Toolkit-Ops dispersion documentation:
  <https://nvidia.github.io/nvalchemi-toolkit-ops/examples/dispersion/index.html>
- official Grimme DFT-D3 download page:
  <https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/get_dft-d3>

Toolkit checks legacy archive MD5 `a76c752e587422c239c99109547516d2`.
The legacy `dftd3.f` header states GPL version 1 or later. The generated cache
has SHA-256 `b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84`.

## MACE and MACE-MP

The Core environment installs `mace-torch 0.3.15`. The model-swap example uses
Toolkit's `MACEWrapper.from_checkpoint("medium-0b2", ...)` path. The named
MACE-MP checkpoint downloads into the user's runtime cache on first use.

- MACE source and MIT license: <https://github.com/ACEsuit/mace>
- MACE foundation models and MIT license:
  <https://github.com/ACEsuit/mace-foundations>

The repository does not distribute MACE checkpoint bytes.
