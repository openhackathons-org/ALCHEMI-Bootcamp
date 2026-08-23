# Third-party notices

This file covers external software, scientific data, and runtime downloads used
by the Core notebooks. Full installed-package license texts are collected in
`.licenses/Third_party_attr.txt`. Scientific data attribution and file hashes
are recorded in `SOURCES_AND_LICENSES.md`.

## Material copied into the source repository

| Material | Terms | Local notice |
|---|---|---|
| NCI Atlas curve subset and NCIA250 archive | CC BY 4.0 | `notebooks/00-core-playbook/data/nci_atlas/README.md` |
| Mg-MOF-74 primitive structure | MIT | `notebooks/00-core-playbook/data/mof74/README.md` |
| 3Dmol.js 2.5.5 bundle | BSD-3-Clause with incorporated notices | `shared/3dmol-2.5.5/LICENSE` |
| py3Dmol 2.5.5 license copy | MIT | `shared/py3dmol-2.5.5/LICENSE.txt` |

### NCI Atlas data

The two included data files are attributed to Jan Řezáč and NCI Atlas
contributors. The upstream README links its `Creative Commons CC-BY` statement
to the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
The source, citations, checksums, and change descriptions are in
[`SOURCES_AND_LICENSES.md`](SOURCES_AND_LICENSES.md) and the adjacent data
README.

### py3Dmol and 3Dmol.js

The Core notebooks display structures with
[py3Dmol 2.5.5](https://pypi.org/project/py3Dmol/), an MIT-licensed Python
wrapper, and [3Dmol.js 2.5.5](https://github.com/3dmol/3Dmol.js), a
BSD-3-Clause browser renderer. The repository stores the checked JavaScript
bundle, its minifier notice, the complete 3Dmol.js incorporated-code notices,
and a copy of the py3Dmol MIT license.

Live interactive viewers use the same checked 3Dmol.js bundle. The notices
under `shared/3dmol-2.5.5/` apply to the standalone file and generated viewer
HTML.

3Dmol.js publication: <https://doi.org/10.1093/bioinformatics/btu829>

### RASPA2 Mg-MOF-74 data

The included 54-atom Mg-MOF-74 primitive CIF is derived from the Mg-MOF-74
structure distributed with iRASPA/RASPA2 under the MIT License. The pinned
source revision, source and derivative hashes, primitive-cell transformation,
attribution, and citations are recorded in
[`SOURCES_AND_LICENSES.md`](SOURCES_AND_LICENSES.md) and the adjacent data
README.

## Installed software

### NVIDIA ALCHEMI Toolkit and Toolkit-Ops

The tutorials pin Toolkit and Toolkit-Ops to immutable commits in
`environment/runtime-pins.toml`.

- [Toolkit Apache-2.0 license at the pinned commit](https://github.com/NVIDIA/nvalchemi-toolkit/blob/8c2c307c1c0c76baee6f7a68eb75a45da83ffd18/LICENSE)
- [Toolkit-Ops Apache-2.0 license at the pinned commit](https://github.com/NVIDIA/nvalchemi-toolkit-ops/blob/c1e23460859a784e1d78043bcd1c8af0d1095fa2/LICENSE)

### PyTorch, CUDA, and ASE

Toolkit uses [PyTorch](https://github.com/pytorch/pytorch), distributed under
BSD-3-Clause. [ASE](https://gitlab.com/ase/ase) supplies structure objects under
LGPL-2.1-or-later. CUDA wheels use the
[NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/).

### AIMNet2 and `aimnet`

The Core modules use `aimnet 0.2.0` with registry key
`aimnet2-wb97m-d3_0`, which resolves to `aimnet2_wb97m_d3_0.pt`. Runtime setup
verifies SHA-256
`f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28`
before use. The AIMNet software and corresponding AIMNet2 ωB97M-D3 weights are
licensed under MIT.

- [AIMNetCentral source and MIT license](https://github.com/isayevlab/aimnetcentral)
- [AIMNet2 ωB97M-D3 model card](https://huggingface.co/isayevlab/aimnet2-wb97m-d3)
- [model paper](https://doi.org/10.1039/D4SC08572H)

The runtime cache holds checkpoint bytes. The source distribution carries the
alias, hash, model card, and license notice.

### DFT-D3 reference parameters

Toolkit generates its D3 parameter table by parsing the legacy Grimme DFT-D3
reference archive. The legacy reference implementation is licensed
GPL-1.0-or-later. Runtime setup verifies the archive MD5
`a76c752e587422c239c99109547516d2` and generated-table SHA-256
`b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84`.

- [official Grimme DFT-D3 page](https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3)
- [DFT-D3 implementation license comparison](https://dftd3.readthedocs.io/en/latest/comparison.html)
- [Toolkit-Ops dispersion documentation](https://nvidia.github.io/nvalchemi-toolkit-ops/examples/dispersion/index.html)

Local setup writes the generated table to the runtime cache.

### MACE and MACE-MP

The environment installs `mace-torch 0.3.15`. Runtime setup resolves the
MIT-licensed MACE-MPA-0 `medium-mpa-0` checkpoint through MACE's official
resolver and verifies SHA-256
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`.
Core Module 1 uses this checkpoint for the mixed finite and periodic preview.

- [MACE source and MIT license](https://github.com/ACEsuit/mace)
- [MACE foundation models and MIT license](https://github.com/ACEsuit/mace-foundations)
- [MACE-MPA-0 model file](https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model)

Checkpoint bytes stay outside the source repository. The current MACE
dependency set includes `matscipy` under LGPL-2.1-or-later and
`python-hostlist` under
GPL-2.0-or-later. The `.licenses/` inventory records their installed versions
and license texts.

### RDKit

The environment pins [RDKit 2026.3.4](https://pypi.org/project/rdkit/),
distributed under BSD-3-Clause. Core Module 1 uses RDKit locally to parse
SMILES, add explicit hydrogens, generate a fixed-seed ETKDGv3 conformer, and
optimize it with MMFF or UFF. The source repository contains no RDKit binary or
copied RDKit data.
