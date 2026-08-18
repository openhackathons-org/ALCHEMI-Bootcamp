# Installed package license summary

This generated snapshot includes direct and transitive Python packages installed for the locked Linux tutorial environment.

- Source: pyproject.toml and uv.lock (SHA-256 `14afb76b221ae8b8ad9033d285cefda6d60f6700f925c4d32ef6b9889b30e127`)
- Environment: Python 3.12.13 on linux-x86_64
- Packages: 213 installed, 15 direct

## Terms requiring attention

- ASE 3.27.0 is a direct dependency under LGPL-2.1-or-later.
- MACE 0.3.15 is MIT. Its locked dependency tree includes `matscipy` 1.2.0 under LGPL-2.1-or-later and `python-hostlist` 2.3.0 under GPL-2.0-or-later.
- The CUDA packages installed through PyTorch include `cuda-toolkit` 13.0.2 and NVIDIA runtime wheels. Their proprietary components are governed by the [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/); individual package records contain bundled terms when supplied.
- Toolkit builds the D3 parameter cache from the legacy Grimme reference archive under GPL-1.0-or-later. The cache is a runtime asset covered by [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md); it is outside this installed-package inventory.

## Installed packages

| Scope | Name | Version | License | Declared in | URL |
| --- | --- | --- | --- | --- | --- |
| direct | aimnet | 0.2.0 | MIT | pyproject.toml [project.dependencies] | https://github.com/isayevlab/aimnetcentral |
| direct | ase | 3.27.0 | LGPL-2.1-or-later | pyproject.toml [project.dependencies] | https://gitlab.com/ase/ase.git |
| direct | mace-torch | 0.3.15 | MIT | pyproject.toml [project.dependencies] | https://github.com/ACEsuit/mace |
| direct | matplotlib | 3.11.1 | PSF-2.0 | pyproject.toml [project.dependencies] | https://matplotlib.org |
| direct | nvalchemi-toolkit | 0.2.0 | Apache-2.0 | pyproject.toml [project.dependencies] | git+https://github.com/NVIDIA/nvalchemi-toolkit.git@8c2c307c1c0c76baee6f7a68eb75a45da83ffd18 |
| direct | nvalchemi-toolkit-ops | 0.4.1 | Apache-2.0 | pyproject.toml [project.dependencies] | git+https://github.com/NVIDIA/nvalchemi-toolkit-ops.git@c1e23460859a784e1d78043bcd1c8af0d1095fa2 |
| direct | numpy | 2.3.5 | BSD-3-Clause | pyproject.toml [project.dependencies] | https://github.com/numpy/numpy |
| direct | py3Dmol | 2.5.5 | MIT | pyproject.toml [project.dependencies] | https://3dmol.org |
| direct | rich | 14.1.0 | MIT | pyproject.toml [project.dependencies] | https://rich.readthedocs.io/en/latest/ |
| direct | torch | 2.12.0+cu130 | BSD-3-Clause | pyproject.toml [project.dependencies] | https://github.com/pytorch/pytorch |
| direct | zarr | 3.3.0 | MIT | pyproject.toml [project.dependencies] | https://github.com/zarr-developers/zarr-python |
| direct | ipykernel | 7.3.0 | BSD-3-Clause | pyproject.toml [dependency-groups.notebook] | https://github.com/ipython/ipykernel |
| direct | jupyterlab | 4.6.3 | BSD-3-Clause | pyproject.toml [dependency-groups.notebook] | https://github.com/jupyterlab/jupyterlab |
| direct | pytest | 8.4.2 | MIT | pyproject.toml [dependency-groups.test] | https://github.com/pytest-dev/pytest |
| direct | ruff | 0.16.2 | MIT | pyproject.toml [dependency-groups.test] | https://github.com/astral-sh/ruff |
| transitive | absl-py | 2.5.0 | Apache-2.0 |  | https://github.com/abseil/abseil-py |
| transitive | aiobotocore | 3.9.0 | Apache-2.0 |  | https://github.com/aio-libs/aiobotocore |
| transitive | aiohappyeyeballs | 2.7.1 | PSF-2.0 |  | https://github.com/aio-libs/aiohappyeyeballs |
| transitive | aiohttp | 3.14.3 | Apache-2.0 AND MIT |  | https://github.com/aio-libs/aiohttp |
| transitive | aioitertools | 0.13.0 | MIT |  | https://aioitertools.omnilib.dev |
| transitive | aiosignal | 1.4.0 | Apache-2.0 |  | https://github.com/aio-libs/aiosignal |
| transitive | annotated-types | 0.8.0 | MIT |  | https://github.com/annotated-types/annotated-types |
| transitive | antlr4-python3-runtime | 4.9.3 | BSD-3-Clause AND MIT |  | http://www.antlr.org |
| transitive | anyio | 4.14.2 | MIT |  | https://anyio.readthedocs.io/en/latest/ |
| transitive | argon2-cffi | 25.1.0 | MIT |  | https://argon2-cffi.readthedocs.io/ |
| transitive | argon2-cffi-bindings | 25.1.0 | MIT |  | https://tidelift.com/?utm_source=lifter&utm_medium=referral&utm_campaign=hynek |
| transitive | arrow | 1.4.0 | Apache-2.0 |  | https://github.com/arrow-py/arrow |
| transitive | asttokens | 3.0.2 | Apache-2.0 |  | https://asttokens.readthedocs.io/en/latest/index.html |
| transitive | async-lru | 2.3.0 | MIT |  | https://github.com/aio-libs/async-lru |
| transitive | attrs | 26.1.0 | MIT |  | https://www.attrs.org/ |
| transitive | babel | 2.18.0 | BSD-3-Clause |  | https://github.com/python-babel/babel |
| transitive | beartype | 0.22.9 | MIT |  | https://beartype.readthedocs.io |
| transitive | beautifulsoup4 | 4.15.0 | MIT |  | https://www.crummy.com/software/BeautifulSoup/bs4/ |
| transitive | bleach | 6.4.0 | Apache-2.0 |  | https://github.com/mozilla/bleach |
| transitive | botocore | 1.43.56 | Apache-2.0 |  | https://github.com/boto/botocore |
| transitive | certifi | 2026.7.22 | MPL-2.0 |  | https://github.com/certifi/python-certifi |
| transitive | cffi | 2.1.1 | MIT-0 |  | https://cffi.readthedocs.io/ |
| transitive | cftime | 1.6.5 | MIT |  | https://pypi.org/project/cftime/1.6.5/ |
| transitive | charset-normalizer | 3.4.9 | MIT |  | https://charset-normalizer.readthedocs.io/ |
| transitive | click | 8.4.2 | BSD-3-Clause |  | https://github.com/pallets/click/ |
| transitive | cloudpickle | 3.1.2 | BSD-3-Clause |  | https://github.com/cloudpipe/cloudpickle |
| transitive | comm | 0.2.3 | BSD-3-Clause |  | https://github.com/ipython/comm |
| transitive | ConfigArgParse | 1.7.5 | MIT |  | https://github.com/bw2/ConfigArgParse |
| transitive | contourpy | 1.3.3 | BSD-3-Clause |  | https://github.com/contourpy/contourpy |
| transitive | cuda-bindings | 13.3.1 | LicenseRef-NVIDIA-SOFTWARE-LICENSE |  | https://github.com/NVIDIA/cuda-python |
| transitive | cuda-pathfinder | 1.6.0 | Apache-2.0 |  | https://github.com/NVIDIA/cuda-python |
| transitive | cuda-toolkit | 13.0.2 | NVIDIA CUDA Toolkit EULA (proprietary) |  | https://developer.nvidia.com/cuda-toolkit |
| transitive | cuequivariance | 0.11.1 | Apache-2.0 |  | https://pypi.org/project/cuequivariance/0.11.1/ |
| transitive | cuequivariance-torch | 0.11.1 | Apache-2.0 |  | https://pypi.org/project/cuequivariance-torch/0.11.1/ |
| transitive | cycler | 0.12.1 | BSD-3-Clause |  | https://github.com/matplotlib/cycler |
| transitive | debugpy | 1.8.21 | MIT |  | https://github.com/microsoft/debugpy |
| transitive | defusedxml | 0.7.1 | PSF-2.0 |  | https://github.com/tiran/defusedxml |
| transitive | dm-tree | 0.1.10 | Apache-2.0 |  | https://github.com/deepmind/tree |
| transitive | donfig | 0.8.1.post1 | MIT |  | https://github.com/pytroll/donfig |
| transitive | e3nn | 0.4.4 | MIT |  | https://e3nn.org |
| transitive | einops | 0.8.2 | MIT |  | https://github.com/arogozhnikov/einops |
| transitive | executing | 2.2.1 | MIT |  | https://github.com/alexmojaki/executing |
| transitive | fastjsonschema | 2.22.1 | BSD-3-Clause |  | https://github.com/horejsek/python-fastjsonschema |
| transitive | filelock | 3.32.2 | MIT |  | https://github.com/tox-dev/py-filelock |
| transitive | fonttools | 4.63.0 | MIT |  | http://github.com/fonttools/fonttools |
| transitive | fqdn | 1.5.1 | MPL-2.0 |  | https://github.com/ypcrts/fqdn |
| transitive | frozenlist | 1.8.0 | Apache-2.0 |  | https://github.com/aio-libs/frozenlist |
| transitive | fsspec | 2026.7.0 | BSD-3-Clause |  | https://github.com/fsspec/filesystem_spec |
| transitive | gitdb | 4.0.12 | BSD-3-Clause |  | https://github.com/gitpython-developers/gitdb |
| transitive | GitPython | 3.1.59 | BSD-3-Clause |  | https://github.com/gitpython-developers/GitPython |
| transitive | google-crc32c | 1.8.0 | Apache-2.0 |  | https://github.com/googleapis/python-crc32c |
| transitive | h11 | 0.16.0 | MIT |  | https://github.com/python-hyper/h11 |
| transitive | h5py | 3.16.0 | BSD-3-Clause |  | https://github.com/h5py/h5py |
| transitive | hf-xet | 1.6.0 | Apache-2.0 |  | https://github.com/huggingface/xet-core.git |
| transitive | httpcore | 1.0.9 | BSD-3-Clause |  | https://github.com/encode/httpcore |
| transitive | httpx | 0.28.1 | BSD-3-Clause |  | https://github.com/encode/httpx |
| transitive | huggingface_hub | 1.27.0 | Apache-2.0 |  | https://github.com/huggingface/huggingface_hub |
| transitive | hydra-core | 1.3.5 | MIT |  | https://github.com/facebookresearch/hydra |
| transitive | idna | 3.18 | BSD-3-Clause |  | https://github.com/kjd/idna |
| transitive | importlib_metadata | 9.0.0 | Apache-2.0 |  | https://github.com/python/importlib_metadata |
| transitive | iniconfig | 2.3.0 | MIT |  | https://github.com/pytest-dev/iniconfig |
| transitive | ipython | 9.16.1 | BSD-3-Clause |  | https://github.com/ipython/ipython |
| transitive | ipython_pygments_lexers | 1.1.1 | BSD-3-Clause |  | https://github.com/ipython/ipython-pygments-lexers |
| transitive | isoduration | 20.11.0 | ISC |  | https://github.com/bolsote/isoduration |
| transitive | jaxtyping | 0.3.11 | MIT |  | https://github.com/patrick-kidger/jaxtyping |
| transitive | jedi | 0.20.0 | MIT |  | https://jedi.readthedocs.io/en/latest/ |
| transitive | Jinja2 | 3.1.6 | BSD-3-Clause |  | https://github.com/pallets/jinja/ |
| transitive | jmespath | 1.1.0 | MIT |  | https://github.com/jmespath/jmespath.py |
| transitive | json5 | 0.15.0 | Apache-2.0 |  | https://github.com/dpranke/pyjson5 |
| transitive | jsonpointer | 3.1.1 | BSD-3-Clause |  | https://github.com/stefankoegl/python-json-pointer |
| transitive | jsonschema | 4.26.0 | MIT |  | https://github.com/python-jsonschema/jsonschema |
| transitive | jsonschema-specifications | 2025.9.1 | MIT |  | https://github.com/python-jsonschema/jsonschema-specifications |
| transitive | jupyter_builder | 1.2.2 | BSD-3-Clause AND MIT AND ISC |  | https://github.com/jupyterlab/jupyter-builder |
| transitive | jupyter_client | 8.9.1 | BSD-3-Clause |  | https://github.com/jupyter/jupyter_client |
| transitive | jupyter_core | 5.9.1 | BSD-3-Clause |  | https://github.com/jupyter/jupyter_core |
| transitive | jupyter-events | 0.12.1 | BSD-3-Clause |  | https://github.com/jupyter/jupyter_events.git |
| transitive | jupyter-lsp | 2.3.1 | BSD-3-Clause |  | https://jupyterlab-lsp.readthedocs.io/ |
| transitive | jupyter_server | 2.20.0 | BSD-3-Clause |  | https://github.com/jupyter-server/jupyter_server |
| transitive | jupyter_server_terminals | 0.5.4 | BSD-3-Clause |  | https://github.com/jupyter-server/jupyter_server_terminals |
| transitive | jupyterlab_pygments | 0.3.0 | BSD-3-Clause |  | https://github.com/jupyterlab/jupyterlab_pygments.git |
| transitive | jupyterlab_server | 2.28.0 | BSD-3-Clause |  | https://github.com/jupyterlab/jupyterlab_server |
| transitive | kiwisolver | 1.5.0 | BSD-3-Clause |  | https://github.com/nucleic/kiwi |
| transitive | lark | 1.3.1 | MIT |  | https://github.com/lark-parser/lark |
| transitive | lightning-utilities | 0.15.3 | Apache-2.0 |  | https://dev-toolbox.rtfd.io/en/latest/ |
| transitive | lmdb | 2.3.0 | OLDAP-2.8 |  | http://github.com/jnwatson/py-lmdb/ |
| transitive | loguru | 0.7.3 | MIT |  | https://github.com/Delgan/loguru |
| transitive | markdown-it-py | 4.2.0 | MIT |  | https://github.com/executablebooks/markdown-it-py |
| transitive | MarkupSafe | 3.0.3 | BSD-3-Clause |  | https://github.com/pallets/markupsafe/ |
| transitive | matplotlib-inline | 0.2.2 | BSD-3-Clause |  | https://github.com/ipython/matplotlib-inline |
| transitive | matscipy | 1.2.0 | LGPL-2.1-or-later |  | https://github.com/libAtoms/matscipy |
| transitive | mdurl | 0.1.2 | MIT |  | https://github.com/executablebooks/mdurl |
| transitive | mistune | 3.3.4 | BSD-3-Clause |  | https://github.com/lepture/mistune |
| transitive | ml_dtypes | 0.5.4 | Apache-2.0 |  | https://github.com/jax-ml/ml_dtypes |
| transitive | mpmath | 1.3.0 | BSD-3-Clause |  | https://github.com/fredrik-johansson/mpmath |
| transitive | multidict | 6.7.1 | Apache-2.0 |  | https://github.com/aio-libs/multidict |
| transitive | nbclient | 0.11.0 | BSD-3-Clause |  | https://github.com/jupyter/nbclient |
| transitive | nbconvert | 7.17.1 | BSD-3-Clause |  | https://jupyter.org |
| transitive | nbformat | 5.11.0 | BSD-3-Clause |  | https://github.com/jupyter/nbformat.git |
| transitive | nest-asyncio2 | 1.7.2 | BSD-2-Clause |  | https://github.com/Chaoses-Ib/nest-asyncio2 |
| transitive | networkx | 3.6.1 | BSD-3-Clause |  | https://networkx.org/ |
| transitive | notebook_shim | 0.2.4 | BSD-3-Clause |  | https://pypi.org/project/notebook_shim/0.2.4/ |
| transitive | numcodecs | 0.16.5 | MIT |  | https://github.com/zarr-developers/numcodecs |
| transitive | nvidia-cublas | 13.1.1.3 | LicenseRef-NVIDIA-Proprietary |  | https://developer.nvidia.com/cuda-zone |
| transitive | nvidia-cuda-cupti | 13.0.85 | LicenseRef-NVIDIA-Proprietary |  | https://developer.nvidia.com/cuda-zone |
| transitive | nvidia-cuda-nvrtc | 13.0.88 | LicenseRef-NVIDIA-Proprietary |  | https://developer.nvidia.com/cuda-zone |
| transitive | nvidia-cuda-runtime | 13.0.96 | LicenseRef-NVIDIA-Proprietary |  | https://developer.nvidia.com/cuda-zone |
| transitive | nvidia-cudnn-cu13 | 9.20.0.48 | LicenseRef-NVIDIA-Proprietary |  | https://developer.nvidia.com/cuda-zone |
| transitive | nvidia-cufft | 12.0.0.61 | LicenseRef-NVIDIA-Proprietary |  | https://developer.nvidia.com/cuda-zone |
| transitive | nvidia-cufile | 1.15.1.6 | LicenseRef-NVIDIA-Proprietary |  | https://developer.nvidia.com/cuda-zone |
| transitive | nvidia-curand | 10.4.0.35 | LicenseRef-NVIDIA-Proprietary |  | https://developer.nvidia.com/cuda-zone |
| transitive | nvidia-cusolver | 12.0.4.66 | LicenseRef-NVIDIA-Proprietary |  | https://developer.nvidia.com/cuda-zone |
| transitive | nvidia-cusparse | 12.6.3.3 | LicenseRef-NVIDIA-Proprietary |  | https://developer.nvidia.com/cuda-zone |
| transitive | nvidia-cusparselt-cu13 | 0.8.1 | LicenseRef-NVIDIA-Proprietary |  | https://developer.nvidia.com/cusparselt |
| transitive | nvidia-nccl-cu13 | 2.29.7 | BSD-3-Clause |  | https://developer.nvidia.com/cuda-zone |
| transitive | nvidia-nvjitlink | 13.0.88 | LicenseRef-NVIDIA-Proprietary |  | https://developer.nvidia.com/cuda-zone |
| transitive | nvidia-nvshmem-cu13 | 3.4.5 | LicenseRef-NVIDIA-Proprietary |  | https://developer.nvidia.com/cuda-zone |
| transitive | nvidia-nvtx | 13.0.85 | Apache-2.0 |  | https://developer.nvidia.com/cuda-zone |
| transitive | nvidia-physicsnemo | 2.1.1 | Apache-2.0 |  | https://github.com/NVIDIA/physicsnemo |
| transitive | nvtx | 0.2.15 | Apache-2.0 WITH LLVM-exception |  | https://github.com/NVIDIA/NVTX |
| transitive | omegaconf | 2.3.1 | BSD-3-Clause |  | https://github.com/omry/omegaconf |
| transitive | onnx | 1.22.0 | Apache-2.0 |  | https://github.com/onnx/onnx |
| transitive | opt_einsum | 3.4.0 | MIT |  | https://pypi.org/project/opt_einsum/3.4.0/ |
| transitive | opt-einsum-fx | 0.1.4 | MIT |  | https://github.com/Linux-cpp-lisp/opt_einsum_fx |
| transitive | orjson | 3.11.9 | MPL-2.0 AND (Apache-2.0 OR MIT) |  | https://github.com/ijl/orjson |
| transitive | packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |  | https://github.com/pypa/packaging |
| transitive | pandas | 2.3.3 | BSD-3-Clause |  | https://github.com/pandas-dev/pandas |
| transitive | pandocfilters | 1.5.1 | BSD-3-Clause |  | http://github.com/jgm/pandocfilters |
| transitive | parso | 0.8.7 | MIT |  | https://github.com/davidhalter/parso |
| transitive | periodictable | 2.0.2 | LicenseRef-Public-Domain |  | https://github.com/python-periodictable/periodictable |
| transitive | pexpect | 4.9.0 | ISC |  | https://pexpect.readthedocs.io/ |
| transitive | pillow | 12.3.0 | MIT-CMU |  | https://github.com/python-pillow/Pillow |
| transitive | platformdirs | 4.11.2 | MIT |  | https://github.com/tox-dev/platformdirs |
| transitive | plotext | 5.3.2 | MIT |  | https://github.com/piccolomo/plotext |
| transitive | pluggy | 1.6.0 | MIT |  | https://pypi.org/project/pluggy/1.6.0/ |
| transitive | plum-dispatch | 2.9.0 | MIT |  | https://github.com/beartype/plum |
| transitive | prettytable | 3.18.0 | BSD-3-Clause |  | https://github.com/prettytable/prettytable |
| transitive | prometheus_client | 0.26.0 | Apache-2.0 AND BSD-2-Clause |  | https://github.com/prometheus/client_python |
| transitive | prompt_toolkit | 3.0.53 | BSD-3-Clause |  | https://github.com/prompt-toolkit/python-prompt-toolkit |
| transitive | propcache | 0.5.2 | Apache-2.0 |  | https://github.com/aio-libs/propcache |
| transitive | protobuf | 7.35.1 | BSD-3-Clause |  | https://developers.google.com/protocol-buffers/ |
| transitive | psutil | 7.2.2 | BSD-3-Clause |  | https://github.com/giampaolo/psutil |
| transitive | ptyprocess | 0.7.0 | ISC |  | https://github.com/pexpect/ptyprocess |
| transitive | pure_eval | 0.2.3 | MIT |  | http://github.com/alexmojaki/pure_eval |
| transitive | pycparser | 3.0 | BSD-3-Clause |  | https://github.com/eliben/pycparser |
| transitive | pydantic | 2.13.4 | MIT |  | https://github.com/pydantic/pydantic |
| transitive | pydantic_core | 2.46.4 | MIT |  | https://github.com/pydantic/pydantic/tree/main/pydantic-core |
| transitive | Pygments | 2.20.0 | BSD-2-Clause |  | https://github.com/pygments/pygments |
| transitive | pyparsing | 3.3.2 | MIT |  | https://github.com/pyparsing/pyparsing.git |
| transitive | python-dateutil | 2.9.0.post0 | BSD-3-Clause OR Apache-2.0 |  | https://github.com/dateutil/dateutil |
| transitive | python_hostlist | 2.3.0 | GPL-2.0-or-later |  | http://www.nsc.liu.se/~kent/python-hostlist/ |
| transitive | python-json-logger | 4.1.0 | BSD-2-Clause |  | https://nhairs.github.io/python-json-logger |
| transitive | pytz | 2026.3.post1 | MIT |  | https://github.com/stub42/pytz.git |
| transitive | pyvers | 0.2.3 | MIT |  | https://pypi.org/project/pyvers/0.2.3/ |
| transitive | PyYAML | 6.0.3 | MIT |  | https://pyyaml.org/wiki/PyYAMLDocumentation |
| transitive | pyzmq | 27.1.0 | BSD-3-Clause |  | https://github.com/zeromq/pyzmq |
| transitive | referencing | 0.37.0 | MIT |  | https://github.com/python-jsonschema/referencing |
| transitive | requests | 2.34.2 | Apache-2.0 |  | https://github.com/psf/requests |
| transitive | rfc3339-validator | 0.1.4 | MIT |  | https://github.com/naimetti/rfc3339-validator |
| transitive | rfc3986-validator | 0.1.1 | MIT |  | https://github.com/naimetti/rfc3986-validator |
| transitive | rfc3987-syntax | 1.1.0 | MIT |  | https://github.com/willynilly/rfc3987-syntax |
| transitive | rpds-py | 2026.6.3 | MIT |  | https://github.com/crate-py/rpds |
| transitive | s3fs | 2026.7.0 | BSD-3-Clause |  | http://github.com/fsspec/s3fs/ |
| transitive | safetensors | 0.8.0 | Apache-2.0 |  | https://github.com/huggingface/safetensors |
| transitive | scipy | 1.18.0 | BSD-3-Clause |  | https://github.com/scipy/scipy |
| transitive | Send2Trash | 2.1.0 | BSD-3-Clause |  | https://github.com/arsenetar/send2trash.git |
| transitive | setuptools | 81.0.0 | MIT |  | https://github.com/pypa/setuptools |
| transitive | six | 1.17.0 | MIT |  | https://github.com/benjaminp/six |
| transitive | smmap | 5.0.3 | BSD-3-Clause |  | https://github.com/gitpython-developers/smmap |
| transitive | soupsieve | 2.9.2 | MIT |  | https://github.com/facelessuser/soupsieve |
| transitive | stack-data | 0.6.3 | MIT |  | http://github.com/alexmojaki/stack_data |
| transitive | sympy | 1.14.0 | BSD-3-Clause |  | https://github.com/sympy/sympy |
| transitive | tensordict | 0.11.0 | MIT |  | https://github.com/pytorch/tensordict |
| transitive | termcolor | 3.3.0 | MIT |  | https://github.com/termcolor/termcolor |
| transitive | terminado | 0.18.1 | BSD-2-Clause |  | https://github.com/jupyter/terminado |
| transitive | timm | 1.0.28 | Apache-2.0 |  | https://github.com/huggingface/pytorch-image-models |
| transitive | tinycss2 | 1.5.1 | BSD-3-Clause |  | https://www.courtbouillon.org/tinycss2 |
| transitive | torch-ema | 0.3 | MIT |  | https://github.com/fadel/pytorch_ema |
| transitive | torchmetrics | 1.9.0 | Apache-2.0 |  | https://torchmetrics.rtfd.io/en/latest/ |
| transitive | torchvision | 0.27.0 | BSD-3-Clause |  | https://github.com/pytorch/vision |
| transitive | tornado | 6.5.8 | Apache-2.0 |  | https://github.com/tornadoweb/tornado |
| transitive | tqdm | 4.70.0 | MPL-2.0 AND MIT |  | https://github.com/tqdm/tqdm |
| transitive | traitlets | 5.16.1 | BSD-3-Clause |  | https://github.com/ipython/traitlets |
| transitive | treelib | 1.8.0 | Apache-2.0 |  | https://github.com/caesar0301/treelib |
| transitive | triton | 3.7.0 | MIT |  | https://github.com/triton-lang/triton/ |
| transitive | typing_extensions | 4.16.0 | PSF-2.0 |  | https://github.com/python/typing_extensions |
| transitive | typing-inspection | 0.4.3 | MIT |  | https://github.com/pydantic/typing-inspection |
| transitive | tzdata | 2026.3 | Apache-2.0 |  | https://github.com/python/tzdata |
| transitive | uri-template | 1.3.0 | MIT |  | https://gitlab.linss.com/open-source/python/uri-template |
| transitive | urllib3 | 2.7.0 | MIT |  | https://urllib3.readthedocs.io |
| transitive | wadler_lindig | 0.1.7 | Apache-2.0 |  | https://github.com/patrick-kidger/wadler_lindig |
| transitive | warp-lang | 1.16.0 | Apache-2.0 |  | https://github.com/NVIDIA/warp |
| transitive | wcwidth | 0.8.2 | MIT |  | https://github.com/jquast/wcwidth |
| transitive | webcolors | 25.10.0 | BSD-3-Clause |  | https://webcolors.readthedocs.io |
| transitive | webencodings | 0.5.1 | BSD-3-Clause |  | https://github.com/SimonSapin/python-webencodings |
| transitive | websocket-client | 1.9.0 | Apache-2.0 |  | https://github.com/websocket-client/websocket-client/ |
| transitive | wrapt | 2.3.0 | BSD-2-Clause |  | https://github.com/GrahamDumpleton/wrapt |
| transitive | yarl | 1.24.5 | Apache-2.0 |  | https://github.com/aio-libs/yarl |
| transitive | zipp | 4.1.0 | MIT |  | https://github.com/jaraco/zipp |
