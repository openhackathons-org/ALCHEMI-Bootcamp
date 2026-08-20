# Sources and licenses

This file records third-party data, viewer code, model assets, and course
artwork used by the tutorials. Each third-party item keeps the terms listed
here.

## Distribution scope

The source repository contains tutorial code and notebooks, one NCI Atlas data
file, one RASPA2-derived Mg-MOF-74 primitive cell,
the 3Dmol.js browser bundle, a py3Dmol license copy, and course artwork.
Installed Python packages, model checkpoints, the original DFT-D3 archive, and
the generated D3 parameter table live in runtime caches or built images.

A built container has a different distribution scope. The current build
installs Python and CUDA packages, generates the D3 table, and downloads model
assets. Review a container with an image-level software bill of materials and
notice set before publishing it.

## Files included in the source repository

| Material | Location | Terms |
|---|---|---|
| NCI Atlas curve subset | `notebooks/00-core-playbook/data/nci_atlas/nci-atlas-curves.csv.gz` | CC-BY stated upstream; version confirmation pending |
| Mg-MOF-74 primitive structure | `notebooks/00-core-playbook/data/mof74/mg-mof-74-primitive.cif` | MIT |
| 3Dmol.js 2.5.5 browser bundle and notices | `shared/3dmol-2.5.5/` | BSD-3-Clause and the incorporated notices in `LICENSE` |
| py3Dmol 2.5.5 license copy | `shared/py3dmol-2.5.5/LICENSE.txt` | MIT |
| NVIDIA course banner | `shared/alchemi-banner-left.png` | NVIDIA course artwork; excluded from Apache-2.0 |
| Course diagrams, styles, and helper source | `shared/` and `notebooks/00-core-playbook/` | Apache-2.0 unless a nearby notice says otherwise |

## NCI Atlas scientific data

Creator and attribution: Jan Řezáč and NCI Atlas contributors. The files come
from [NCI Atlas](https://github.com/Honza-R/NCIAtlas) at revision
`1816bfc72609d7deb1d4f93ab9e27eb13bb44bec`. The pinned README states
`Creative Commons CC-BY` and the revision contains no license file. Confirm the
exact CC-BY version with the data owner before redistribution. Retain the
attribution and the change descriptions below.

### Three interaction curves

- File: `notebooks/00-core-playbook/data/nci_atlas/nci-atlas-curves.csv.gz`
- SHA-256:
  `7ffbc071e2998cee8e487a2697517187110a05f436920f8611d28d2af5d4d7b7`
- Dataset papers:
  [hydrogen-bonded sets](https://doi.org/10.1021/acs.jctc.9b01265) and
  [D442x10](https://doi.org/10.1039/D2CP01602H)

The tutorial selects three complexes at ten separations, extracts the `AB`,
`A`, and `B` records, and reformats them as compressed CSV. Coordinates,
stored energies, gradients, and source identifiers are unchanged.

The adjacent
[`data/nci_atlas/README.md`](notebooks/00-core-playbook/data/nci_atlas/README.md)
records the same attribution and checksums so that the data can travel with its
notice.

## Mg-MOF-74 structure

The 54-atom Mg-MOF-74 primitive cell is derived from the structure distributed
with [iRASPA/RASPA2](https://github.com/iRASPA/RASPA2), licensed under MIT.
The source is pinned to commit
`fd2398c807e0e0ec4d73c6dd9333539ffb11e98e`.

- Source path: `structures/mofs/cif/MgMOF-74.cif`
- Source SHA-256:
  `84923453752f66781295f70de4d8eaba8ecc8510d676ebcfd547ee8d68e23e43`
- Included primitive CIF SHA-256:
  `e3b5d66082fe703363677c2f8fb4260cba137b34e17649df0c92ec68a86a1117`
- Primitive formula and atom count: `C24H6Mg6O18`, 54 atoms

The included CIF applies the R-centered conventional-to-primitive cell
transformation recorded in the adjacent
[`data/mof74/manifest.json`](notebooks/00-core-playbook/data/mof74/manifest.json).
The source contains 162 atoms; the primitive representation contains 54. The
course uses Mg-MOF-74 as a compact porous-material example for CO₂ adsorption
and carbon-capture research.

## Opening benchmark molecules

Core Module 1 stores canonical SMILES and molecular formulas from these
PubChem compound records: [3672](https://pubchem.ncbi.nlm.nih.gov/compound/3672),
[3034285](https://pubchem.ncbi.nlm.nih.gov/compound/3034285),
[40326](https://pubchem.ncbi.nlm.nih.gov/compound/40326),
[1548943](https://pubchem.ncbi.nlm.nih.gov/compound/1548943),
[445354](https://pubchem.ncbi.nlm.nih.gov/compound/445354),
[51040](https://pubchem.ncbi.nlm.nih.gov/compound/51040),
[65182](https://pubchem.ncbi.nlm.nih.gov/compound/65182), and
[3026](https://pubchem.ncbi.nlm.nih.gov/compound/3026). RDKit adds explicit
hydrogens, generates fixed-seed ETKDGv3 coordinates, and optimizes each local
structure. The repository contains the source identifiers and SMILES, not a
downloaded PubChem coordinate file.

## Molecular viewer

The notebooks use py3Dmol 2.5.5 to create interactive HTML backed by 3Dmol.js
2.5.5.

- [py3Dmol](https://pypi.org/project/py3Dmol/): MIT
- [3Dmol.js](https://github.com/3dmol/3Dmol.js): BSD-3-Clause
- [3Dmol.js paper](https://doi.org/10.1093/bioinformatics/btu829)

The checked browser bundle and its complete incorporated notices are stored in
`shared/3dmol-2.5.5/`. The py3Dmol MIT license is stored in
`shared/py3dmol-2.5.5/LICENSE.txt`. Saved interactive outputs in Core Modules 1
and 3 inline the same checked 3Dmol.js code. These notices apply to the
standalone bundle and those embedded copies.

## Runtime software and model assets

The exact versions, commits, aliases, and file hashes are recorded in
[`environment/runtime-pins.toml`](environment/runtime-pins.toml). The installed
Python package inventory and license texts are under [`.licenses/`](.licenses/).

- NVIDIA ALCHEMI Toolkit 0.2.0 and Toolkit-Ops 0.4.1: Apache-2.0.
- AIMNet software and `aimnet2-wb97m-d3_0` weights: MIT. Runtime setup verifies
  checkpoint SHA-256
  `f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28`.
- MACE software and MACE-MPA-0 `medium-mpa-0` weights: MIT. Core Module 1 uses
  this materials model for the mixed finite and periodic preview. Runtime setup
  verifies checkpoint SHA-256
  `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`.
- RDKit 2026.3.4: BSD-3-Clause. Core Module 1 uses its Python 3.12 wheel for
  offline explicit-hydrogen ETKDGv3 structures followed by MMFF or UFF.
- ASE 3.27.0: LGPL-2.1-or-later.
- PyTorch: BSD-3-Clause. CUDA packages and the CUDA base image use the
  [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/).
- Toolkit's D3 setup parses the legacy Grimme DFT-D3 reference archive, which
  is GPL-1.0-or-later, and creates a checked parameter table in the runtime
  cache. A built container carries that table and needs separate distribution
  review.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for direct links to the
upstream licenses and model cards.

## Course artwork and NVIDIA marks

`shared/alchemi-banner-left.png` has SHA-256
`016f3840bb97e61a3950bd70e587305fe9477831db9763c3d081db0b8a5bbf19`.
Its metadata names Nikita Fedik as author and Canva as the authoring tool.
Asset-level origin for any Canva library elements remains unverified. Treat the
banner and NVIDIA marks as NVIDIA course artwork excluded from Apache-2.0.
NVIDIA retains all rights to its names, logos, and trademarks.
