# Third-party notices

This file summarizes the external software, models, checkpoints, and datasets
used by the current tutorial tree. A software license does not automatically
cover model weights, data, figures, or downloaded runtime files.

## pymatviz and MatterViz

Part 01 uses `pymatviz 0.18.0` and its notebook-native `StructureWidget` to
display ASE molecules with the MatterViz renderer. The shared tutorial assets
include the JavaScript and CSS from `matterviz-anywidget 0.4.0`, together with
the upstream license, so the widget can load without fetching frontend code
during the lesson.

- pymatviz source: <https://github.com/janosh/pymatviz>
- MatterViz source: <https://github.com/janosh/matterviz>
- both projects use the MIT license

The fixed versions and asset SHA-256 values are recorded in
`environment/runtime-pins.toml`.

## PyTorch, JAX, NVIDIA Warp, and Toolkit-Ops

The Part 1 framework primer uses PyTorch and JAX arrays through Toolkit-Ops and
shows the lower-level Warp call used by the same reduction. PyTorch uses a
BSD-3-Clause license. JAX, NVIDIA Warp, and Toolkit-Ops use Apache License 2.0.
The pinned versions and Toolkit-Ops source revision are defined in
`build/requirements.txt`; these software licenses do not cover model weights or
scientific datasets used elsewhere in the tutorial.

## AIMNet2 pretrained models and `aimnet`

Part 1 uses `aimnet 0.2.0`, ensemble member 0 from the AIMNet2 2025 B97-3c
release, and the four-member AIMNet2 ωB97M-D3 ensemble:

- source and documentation: <https://github.com/isayevlab/aimnetcentral>
- package: <https://pypi.org/project/aimnet/>
- AIMNet2 2025 model card and weights: <https://huggingface.co/isayevlab/aimnet2-2025>
- ωB97M-D3 model card and weights: <https://huggingface.co/isayevlab/aimnet2-wb97m-d3>
- both model cards identify their license as MIT
- model paper: <https://doi.org/10.1039/D4SC08572H>

The model cards identify the ensemble weight files, supported elements,
external D3(BJ), and selectable long-range electrostatics. Toolkit supplies the
`AIMNet2Wrapper`; AIMNet supplies the model implementation and checkpoints.
The repository and Docker image do not redistribute these checkpoints. The
image build verifies each official file by size and SHA-256, then removes it in
the same layer. Runtime downloads remain subject to the MIT license shown on
the two model cards. Retain the model-card attribution and license when
redistributing the weights separately.

## DFT-D3 reference parameters

Toolkit-Ops and Toolkit's `DFTD3ModelWrapper` are Apache-2.0 software. The
runtime D3 cache contains standard element-specific D3 reference data used by
the dispersion calculation. The repository and Docker image do not commit or
redistribute that cache. If it is absent, Toolkit downloads the fixed-MD5
legacy Grimme DFT-D3 source archive during the user's first D3 setup and
creates the cache locally; the notebook then checks the cache SHA-256 before
using it.

- Toolkit-Ops source: <https://github.com/NVIDIA/nvalchemi-toolkit-ops>
- Toolkit-Ops dispersion documentation:
  <https://nvidia.github.io/nvalchemi-toolkit-ops/examples/dispersion/index.html>
- official Grimme DFT-D3 download page:
  <https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/get_dft-d3>
- current simple-dftd3 implementation:
  <https://github.com/dftd3/simple-dftd3>

Toolkit currently reads the legacy Bonn archive, not the current
simple-dftd3 repository. Toolkit checks archive MD5
`a76c752e587422c239c99109547516d2`; the tutorial checks generated-cache
SHA-256 `b4828b87b63a43918769d467249492b53f7af94d2ab7ac5ac584a44aa399ec84`.
The legacy `dftd3.f` header states GPL version 1 or later. The current
simple-dftd3 project is LGPL-3.0-or-later, but that separate license must not
be presented as the license of Toolkit's legacy input.

A checksum verifies the file used for a calculation; it does not grant
redistribution rights. Do not add a prewarmed `dftd3_parameters.pt` file to a
release until its exact source and redistribution terms have been recorded.

## Reference calculations and observed water frequencies

The separate B97-3c reference environment uses Psi4 under LGPL-3.0, plus
`dftd3-python`, its `s-dftd3` backend, and `mctc-gcp` under
LGPL-3.0-or-later. The repository stores generated calculation outputs, not
copies of those software packages. See the
[reference README](part-1-scalable-atomistic-workflows/reference/README.md)
for methods, versions, and source links.

The six H2-16O/D2-16O observed band positions are transcribed from Table 1 of
Dinu et al., *J. Phys. Chem. A* 2019, DOI
[10.1021/acs.jpca.9b07221](https://doi.org/10.1021/acs.jpca.9b07221), under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). That table attributes
the underlying measurements to Toth's H2-16O stretch
([10.1006/jmsp.1998.7771](https://doi.org/10.1006/jmsp.1998.7771)), H2-16O bend
([10.1006/jmsp.1998.7611](https://doi.org/10.1006/jmsp.1998.7611)), and D2-16O
([10.1006/jmsp.1999.7815](https://doi.org/10.1006/jmsp.1999.7815)) studies.
Only the six numeric positions and source metadata are included. No table
image, article text, experimental spectrum, or intensity data is redistributed.

## Interaction datasets

The curated NCI Atlas subset used in Part 1 is provided under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and retains source
identifiers and attribution. Two curated files are packaged, each with its own
data README:

- interaction-energy curves (`nci-atlas-curves.csv.gz`):
  [the Part 1 data README](part-1-scalable-atomistic-workflows/data/nci_atlas/README.md)
- shared molecule collection (`ir-molecule-library.extxyz`), which the notebook
  lessons load: [the shared data README](data/nci_atlas/README.md)

Creator: Jan Řezáč and NCI Atlas contributors. Upstream source:
<https://github.com/Honza-R/NCIAtlas> at pinned revision
`1816bfc72609d7deb1d4f93ab9e27eb13bb44bec`. Upstream states its license in
[`README.md`](https://github.com/Honza-R/NCIAtlas/blob/1816bfc72609d7deb1d4f93ab9e27eb13bb44bec/README.md)
and publishes no `LICENSE` file, so that README is the citable license source.

Modifications: subset selection, single-fragment extraction, and reformatting to
CSV and extxyz; coordinates and energies are unchanged.

The archived research notebook's DESS66 input is not included in the Part 1
image. The full repository checkout keeps the upstream DESRES redistribution
notice beside the archive. The source is
[DESS66 version 1.0.0](https://doi.org/10.5281/zenodo.5676284).

## Packmol

Parts 1 and 3 use the conda-forge build of Packmol 21.2.1 to create initial
molecular configurations. Packmol is MIT-licensed software.

- source and license: <https://github.com/m3g/packmol>
- user guide: <https://m3g.github.io/packmol/userguide.shtml>
- package: <https://anaconda.org/conda-forge/packmol/files?version=21.2.1>
- citation: L. Martínez, R. Andrade, E. G. Birgin, and J. M. Martínez,
  *J. Comput. Chem.* **30**, 2157-2164 (2009),
  <https://doi.org/10.1002/jcc.21224>

Packmol places intact input molecules subject to geometric constraints. A
packed box is an initial structure, not an equilibrated liquid or a predicted
thermodynamic state.

## SevenNet-Omni and `sevenn`

Part 1 uses `sevenn 0.13.0` and the official SevenNet-Omni checkpoint:

- source: <https://github.com/MDIL-SNU/SevenNet>
- source release: <https://github.com/MDIL-SNU/SevenNet/tree/v0.13.0>
- pretrained-model documentation: <https://sevennet.readthedocs.io/en/latest/user_guide/pretrained.html>
- checkpoint release: <https://github.com/MDIL-SNU/SevenNet/releases/tag/v0.12.0.cp>
- checkpoint: <https://github.com/MDIL-SNU/SevenNet/releases/download/v0.12.0.cp/checkpoint_sevennet_omni.pth>
- checkpoint size: `103162838` bytes
- checkpoint SHA-256: `ca81bd3aac9fc2696c93dd386615f5a0fe41b92ab9ed7f69fa9526baaa9bab64`
- checkpoint record: <https://doi.org/10.6084/m9.figshare.30399814>
- SevenNet-Omni paper: <https://doi.org/10.1038/s41467-026-70195-8>
- SevenNet software paper: <https://doi.org/10.1021/acs.jctc.4c00190>
- software license: [MIT](https://github.com/MDIL-SNU/SevenNet/blob/v0.13.0/LICENSE)
- checkpoint-record license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The CC BY 4.0 Figshare record links to the GitHub release asset used here, but
the exact 103,162,838-byte GitHub file is hosted outside Figshare and has no
separate model-weight license statement. This repository and its Docker image
therefore do not redistribute the checkpoint. Runtime setup downloads it from
the official release, checks its size and SHA-256, and keeps it only in the
user's cache. Do not redistribute those bytes without explicit upstream
confirmation. SevenNet's code is separately distributed under the MIT license.

Checkpoint-record attribution: Jaesun Kim, Jinmu You, Yutack Park, Yunsung Lim,
Gijin Kim, Deokgi Hong, Seungwu Han, et al. (2025), *Domain-bridging datasets
and DFT results for "Optimizing Cross-Domain Transfer for Universal Machine
Learning Interatomic Potentials"*, Figshare dataset, version 3,
<https://doi.org/10.6084/m9.figshare.30399814.v3>, CC BY 4.0.

Part 1 selects SevenNet-Omni's `mpa` task, which the official documentation
identifies as its general PBE(+U)-level task for systems including surfaces.
The checkpoint output does not include D3. Part 1 composes it with Toolkit's
PBE-D3(BJ) implementation and shows the model and dispersion contributions
separately. Cite the SevenNet software paper and the SevenNet-Omni paper listed
in the upstream repository when using these results.

## Retained Part 2 and Part 3 scientific data

The v2 Docker image targets Part 1 and does not package the retained Part 2/3
notebooks or their scientific data. The development tree still contains an
OC20Dense subset and historical crystal structures and derived files from the
older tutorials. Do not include those files in a public release until:

- the OC20Dense subset has complete CC BY 4.0 attribution and source records;
- each crystal structure has redistribution permission or is replaced with an
  openly redistributable source; and
- derived trajectories, caches, and notebook outputs have been reviewed under
  the same source-data terms.

This is a repository-release issue, not a Part 1 calculation dependency.

## ORB-v3 and `orb-models`

The retained OLED notebook, now Part 3 and historically Part 2, uses
`orb-models 0.7.0` and its `orb_v3_conservative_inf_omat` checkpoint. The
current Part 1 image does not install `orb-models` or its legacy-only `loguru`
dependency because Orb 0.7.0 requires Toolkit-Ops below 0.4. Run the retained
notebook in its separate historical environment. Part 1 no longer uses or
prewarms OrbMol-v2.

- source: <https://github.com/orbital-materials/orb-models>
- release: <https://github.com/orbital-materials/orb-models/tree/v0.7.0>
- Part 3 checkpoint: <https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-inf-omat-20250404.ckpt>
- license: [Apache License 2.0](https://github.com/orbital-materials/orb-models/blob/v0.7.0/LICENSE)

The v0.7.0 package metadata identifies the software as Apache-2.0, and the
upstream README states that Orb models use the same license.

## MACE and MACE foundation checkpoints

The retained legacy adsorption directory uses MACE, but the unified v2 kernel
does not install it. `mace-torch 0.3.15` and the current 0.3.16 release require
`e3nn 0.4.4`; SevenNet requires `e3nn >=0.5` because that release changed the
Clebsch-Gordan coefficient convention. Run the legacy material only in its
historical environment. MACE sources and foundation-model licensing are
published here:

- source: <https://github.com/ACEsuit/mace>
- model registry: <https://github.com/ACEsuit/mace-foundations>

The MACE code and the MACE-MP model are distributed under separate MIT license
notices.

### MACE software

The license supplied with `mace-torch 0.3.15` is reproduced below.

> MIT License
>
> Copyright (c) 2022 ACEsuit/mace
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

### MACE-MP model

The model license is published in the official
[`mace-foundations` repository](https://github.com/ACEsuit/mace-foundations/blob/main/LICENSE)
and applies to the foundation checkpoints used by the retained MACE tutorial.

> MIT License
>
> Copyright (c) 2024 MACE-MP
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
