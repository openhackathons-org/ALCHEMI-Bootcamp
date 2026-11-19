# Direct third-party dependencies

Generated from the dependency declarations in `pyproject.toml` and the Linux environment synchronized from `uv.lock`. License texts and additional metadata are available in [details.json](details.json) and [Third_party_attr.txt](Third_party_attr.txt).

- Source: pyproject.toml and uv.lock (SHA-256 `b79baafd6b098cedf47c10c063e1b7ae4f093ec412735f1b84b66cad8f412cda`)
- Environment: Python 3.12.13 on linux-x86_64
- Packages: 217 installed, 20 direct

## Terms requiring attention

- ASE 3.27.0 is a direct dependency under LGPL-2.1-or-later.
- MACE 0.3.15 is MIT. Its locked dependency tree includes `matscipy` 1.2.0 under LGPL-2.1-or-later and `python-hostlist` 2.3.0 under GPL-2.0-or-later.
- The CUDA packages installed through PyTorch include `cuda-toolkit` 13.0.2 and NVIDIA runtime wheels. Their proprietary components are governed by the [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/); individual package records contain bundled terms when supplied.
- Toolkit builds the D3 parameter cache from the legacy Grimme reference archive under GPL-1.0-or-later. The cache is a runtime asset covered by [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md); it is outside this installed-package inventory.

## Declared packages

| Group | Name | Version | License | Declaration | URL |
| --- | --- | --- | --- | --- | --- |
| runtime | aimnet | 0.2.0 | MIT | aimnet==0.2.0 | https://github.com/isayevlab/aimnetcentral |
| runtime | ase | 3.27.0 | LGPL-2.1-or-later | ase==3.27.0 | https://gitlab.com/ase/ase.git |
| runtime | dm-tree | 0.1.10 | Apache-2.0 | dm-tree==0.1.10 | https://github.com/deepmind/tree |
| runtime | mace-torch | 0.3.15 | MIT | mace-torch==0.3.15 | https://github.com/ACEsuit/mace |
| runtime | matplotlib | 3.11.1 | PSF-2.0 | matplotlib>=3.10,<4 | https://matplotlib.org |
| runtime | nvalchemi-toolkit | 0.3.0 | Apache-2.0 | nvalchemi-toolkit[mace] @ git+https://github.com/NVIDIA/nvalchemi-toolkit.git@31ab73ed0b0bcf44e4769467dff06b7d2674bd37 | git+https://github.com/NVIDIA/nvalchemi-toolkit.git@31ab73ed0b0bcf44e4769467dff06b7d2674bd37 |
| runtime | nvalchemi-toolkit-ops | 0.5.0 | Apache-2.0 | nvalchemi-toolkit-ops[torch-cu13] @ git+https://github.com/NVIDIA/nvalchemi-toolkit-ops.git@6702138e404d40ea79dec62f68c2fa67e858580e | git+https://github.com/NVIDIA/nvalchemi-toolkit-ops.git@6702138e404d40ea79dec62f68c2fa67e858580e |
| runtime | numpy | 2.3.5 | BSD-3-Clause | numpy>=2,<2.4 | https://github.com/numpy/numpy |
| runtime | pandas | 2.3.3 | BSD-3-Clause | pandas>=2.3,<3 | https://github.com/pandas-dev/pandas |
| runtime | py3Dmol | 2.5.5 | MIT | py3dmol==2.5.5 | https://3dmol.org |
| runtime | rdkit | 2026.3.4 | BSD-3-Clause | rdkit==2026.3.4 | https://github.com/kuelumbus/rdkit-pypi |
| runtime | torch | 2.12.0+cu130 | BSD-3-Clause | torch==2.12.0 | https://github.com/pytorch/pytorch |
| runtime | zarr | 3.3.0 | MIT | zarr>=3,<4 | https://github.com/zarr-developers/zarr-python |
| kernel | ipython | 9.16.1 | BSD-3-Clause | ipython>=9,<10 | https://github.com/ipython/ipython |
| kernel | ipykernel | 7.3.0 | BSD-3-Clause | ipykernel>=7,<8 | https://github.com/ipython/ipykernel |
| kernel | ipywidgets | 8.1.9 | BSD-3-Clause | ipywidgets>=8.1,<9 | http://jupyter.org |
| lab | jupyterlab | 4.6.3 | BSD-3-Clause | jupyterlab>=4,<5 | https://github.com/jupyterlab/jupyterlab |
| test | packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | packaging>=26,<27 | https://github.com/pypa/packaging |
| test | pytest | 8.4.2 | MIT | pytest>=8,<9 | https://github.com/pytest-dev/pytest |
| test | ruff | 0.16.2 | MIT | ruff>=0.11,<1 | https://github.com/astral-sh/ruff |
