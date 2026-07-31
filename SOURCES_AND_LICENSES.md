# Sources and licenses

This file records third-party data and visual assets distributed in this
repository. It does not duplicate the changing software dependency inventory.
For the software environment, see
[`.licenses/direct-dependencies.md`](.licenses/direct-dependencies.md) and
[`.licenses/Third_party_attr.txt`](.licenses/Third_party_attr.txt).

Unless stated below, the Apache License 2.0 in this repository does not change
the license or terms that apply to third-party material.

## Distribution scope

This repository distributes tutorial source files, the scientific data and
visual assets identified below, and a Docker build recipe. It does not
distribute a prebuilt container image, installed third-party software
packages, or model checkpoints.

Distributing a prebuilt image would require a separate review and a complete
image-level software bill of materials and notice set covering the installed
operating-system packages, Conda packages, Python packages, bundled binaries,
and CUDA components. In particular, `imageio-ffmpeg` installs an FFmpeg binary
whose GPL terms must be addressed if that binary is redistributed. The
repository-level inventory does not clear a prebuilt image for distribution.

The build recipe references the NVIDIA CUDA base image and downloads
Miniforge, Conda packages, Python packages, and Git dependencies from their
official distribution services. The tutorial downloads MACE-MPA-0 and Orb-v3
model checkpoints when their workflows first run. Those components remain
subject to their own terms:

- NVIDIA CUDA base image and CUDA components:
  [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/)
- Miniforge installer:
  [BSD-3-Clause](https://github.com/conda-forge/miniforge/blob/main/LICENSE)
- OVITO Python module:
  [MIT](https://www.ovito.org/manual/licenses/index.html). The optional
  `ovitos` rendering workflow can call a separately installed OVITO Pro
  executable. OVITO Pro is not distributed by this repository and requires
  its own valid license or entitlement.
- MACE code and MACE-MPA-0 model:
  [MIT](https://github.com/ACEsuit/mace)
- Orb code and Orb-v3 models:
  [Apache-2.0](https://github.com/orbital-materials/orb-models)

The exact reviewed Linux x86_64 software environment is fixed in
`build/conda-linux-64.lock` and
`build/requirements-linux-64.lock.txt`. Its dependency licenses and notices
are recorded under `.licenses/`.

## Scientific data

### Naphthalene crystal structure

- File: `part-2-batched-melting-toolkit/data/naphthalene.cif`
- Source: Crystallography Open Database
  [entry 2311088](https://www.crystallography.net/2311088.cif)
- License: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
- SHA-256:
  `90d674e00952984954ab0382b240dc5733e9bb536e682460a5e394fd7bca509e`
- Original publication: Anna A. Hoser and Anders O. Madsen,
  “Dynamic quantum crystallography: lattice-dynamical models refined against
  diffraction data. II. Applications to L-alanine, naphthalene and xylitol,”
  *Acta Crystallographica Section A* 73 (2017), 102–114,
  [doi:10.1107/S2053273316018994](https://doi.org/10.1107/S2053273316018994).

The source publication identifies this naphthalene dataset as the 100 K
measurement. The COD CIF retains a `_diffrn_ambient_temperature` value of
293(2) K. This metadata discrepancy is documented here and should be resolved
or explained before making temperature-specific claims about the input
structure.

### OC20Dense tutorial subset

- Files:
  `part-1-batched-adsorption-toolkit/data/reference/oc20dense/` and
  `part-1-batched-adsorption-toolkit/data/reference/oc20dense-validation-pack.tgz`
- Source: [Open Catalyst Dataset](https://github.com/Open-Catalyst-Project/Open-Catalyst-Dataset)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Archive SHA-256:
  `c3bd01ca26e68d56f52af7d19a5fb735d2be990e9ecbb050b12115bdb3ade8f4`
- Attribution: Open Catalyst Project contributors, including the OC20 dataset
  described by Chanussot et al., “Open Catalyst 2020 (OC20) Dataset and
  Community Challenges,” *ACS Catalysis* 11 (2021), 6059–6072,
  [doi:10.1021/acscatal.0c04525](https://doi.org/10.1021/acscatal.0c04525).

Only the records used by the tutorial are included. The license requires
attribution when the subset is shared or adapted.

## Logos and brand assets

The ENEOS, Matlantis, and OVITO logos included in the tutorial are used with
permission granted directly for this NVIDIA ALCHEMI tutorial. The UDC logo is
used with permission granted by the project partner. The logos and associated
trademarks remain the property of their respective owners. They are not
licensed under Apache 2.0, and their inclusion does not grant downstream
permission to reuse the marks.

The NVIDIA logo is an NVIDIA trademark and is not licensed under Apache 2.0.
No rights are granted to NVIDIA trademarks or branding.

The tutorial banners, banner artwork, ALCHEMI Toolkit architecture diagram,
adsorption comparison image, and other original tutorial illustrations were
created for this project and are covered by the repository license.
