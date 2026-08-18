# Sources and licenses

This file records third-party data, viewer code, model assets, and course
artwork used by the tutorials. Each third-party item keeps the terms listed
here.

## Distribution scope

The source repository contains tutorial code and notebooks, two NCI Atlas data
files, the 3Dmol.js browser bundle, a py3Dmol license copy, and course artwork.
Installed Python packages, model checkpoints, the original DFT-D3 archive, and
the generated D3 parameter table live in runtime caches or built images.

A built container has a different distribution scope. The current build
installs Python and CUDA packages, generates the D3 table, and downloads model
assets. Review a container with an image-level software bill of materials and
notice set before publishing it.

## Files included in the source repository

| Material | Location | Terms |
|---|---|---|
| NCI Atlas curve subset | `notebooks/00-core-playbook/data/nci_atlas/nci-atlas-curves.csv.gz` | CC BY 4.0 |
| NCI Atlas NCIA250 archive | `notebooks/00-core-playbook/data/nci_atlas/NCIA250.zip` | CC BY 4.0 |
| 3Dmol.js 2.5.5 browser bundle and notices | `shared/3dmol-2.5.5/` | BSD-3-Clause and the incorporated notices in `LICENSE` |
| py3Dmol 2.5.5 license copy | `shared/py3dmol-2.5.5/LICENSE.txt` | MIT |
| NVIDIA course banner | `shared/alchemi-banner-left.png` | NVIDIA course artwork; excluded from Apache-2.0 |
| Course diagrams, styles, and helper source | `shared/` and `notebooks/00-core-playbook/` | Apache-2.0 unless a nearby notice says otherwise |

## NCI Atlas scientific data

Creator and attribution: Jan Řezáč and NCI Atlas contributors. The files
come from [NCI Atlas](https://github.com/Honza-R/NCIAtlas) at revision
`1816bfc72609d7deb1d4f93ab9e27eb13bb44bec` and are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Redistribution must
retain the attribution, license link, and the change descriptions below.

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

### NCIA250 equilibrium survey

- File: `notebooks/00-core-playbook/data/nci_atlas/NCIA250.zip`
- [Pinned upstream directory](https://github.com/Honza-R/NCIAtlas/tree/1816bfc72609d7deb1d4f93ab9e27eb13bb44bec/NCIA250)
- SHA-256:
  `34e3c2cec763344dd9be41aa008672c7d052e50db57abe1abc59873d3935c433`
- Source-set papers:
  [D1200](https://doi.org/10.1039/D2CP01602H),
  [HB300SPX](https://doi.org/10.1021/acs.jctc.0c00715),
  [HB375](https://doi.org/10.1021/acs.jctc.9b01265),
  [R739](https://doi.org/10.1021/acs.jctc.0c01341), and
  [SH250](https://doi.org/10.1039/D2CP01600A)

The archive matches the pinned upstream bytes. The tutorial reports statistics
for all 250 complexes and evaluates the 205 complexes whose elements appear in
the pinned AIMNet2 checkpoint metadata.

The adjacent
[`data/nci_atlas/README.md`](notebooks/00-core-playbook/data/nci_atlas/README.md)
records the same attribution and checksums so that the data can travel with its
notice.

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
- MACE software and the optional MACE-MP-0b2 `medium-0b2` checkpoint: MIT. The
  Module 2 wrapper exercise loads MACE after the main AIMNet evaluation.
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
