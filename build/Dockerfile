# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04@sha256:3b8afffc99c6ca801e6bf31a426400432cc8ba9dea77bd27ccf8cd67f06424eb

# Minimal system deps. The conda env (next layer) handles Qt6/glib/X11 for OVITO.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git g++ ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install a checked Miniforge release. OVITO requires linux/amd64, which is
# also enforced in docker-compose.yml.
ARG MINIFORGE_VERSION=26.3.2-3
ARG MINIFORGE_SHA256=848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-x86_64.sh" && \
    echo "${MINIFORGE_SHA256}  /tmp/miniforge.sh" | sha256sum --check --strict && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

# Recreate the reviewed Linux x86_64 conda environment from exact package URLs.
# build/environment.yml remains the human-maintained input used when refreshing
# this lock and the third-party license inventory.
COPY build/conda-linux-64.lock /tmp/conda-linux-64.lock
RUN /opt/conda/bin/conda create \
        --name alchemi-playbook \
        --file /tmp/conda-linux-64.lock \
        --yes && \
    /opt/conda/bin/conda clean -afy

# Activate the env for all subsequent commands by prepending its bin dir.
# UV_SYSTEM_PYTHON=true lets `uv pip install` target the conda env's
# Python (first on PATH) directly — conda envs aren't venvs in uv's eyes
# (no pyvenv.cfg) but the active Python is what we want to install into.
ENV PATH=/opt/conda/envs/alchemi-playbook/bin:/opt/conda/bin:$PATH \
    CONDA_DEFAULT_ENV=alchemi-playbook \
    UV_SYSTEM_PYTHON=true

WORKDIR /workspace

# Install the reviewed Python environment. Torch is installed first from its
# CUDA 13 wheel index. The complete lock then fixes every PyPI and Git package,
# including the two Toolkit commit URLs.
COPY build/requirements-linux-64.lock.txt /tmp/requirements-linux-64.lock.txt
RUN uv pip install \
        "torch==2.13.0+cu130" \
        --index-url https://download.pytorch.org/whl/cu130 && \
    uv pip install \
        --no-sources-package nvalchemi-toolkit-ops \
        --requirement /tmp/requirements-linux-64.lock.txt

# Fail the build loud if any resolver tug-of-war silently broke an import.
# cuequivariance_ops_torch itself requires the NVIDIA runtime libraries that
# are injected by `docker run --gpus all`, so build-time validation checks the
# installed distribution and runtime smoke tests import the module.
RUN python -c "import importlib.metadata as md; import torch, ovito, ase, pymatgen, nvalchemi, matplotlib, aiohttp, pydantic, ipywidgets; print('imports ok; torch', torch.__version__, 'ovito', ovito.version, 'toolkit', md.version('nvalchemi-toolkit'), 'ops', md.version('nvalchemi-toolkit-ops'), 'cueq_ops', md.version('cuequivariance-ops-torch-cu13'))"

# Ship one notebook kernel name for the whole playbook. Conda/ipykernel/OVITO
# can add default kernelspecs (`python3`, `ovito-pro`); remove those aliases
# plus the old tutorial name so VS Code/Jupyter see only the reproducible
# playbook kernel. Jupyter can also synthesize a native `python3` fallback at
# runtime, and its server API defaults to `python3` unless told otherwise.
# The same config pins the JupyterLab file-browser root to /workspace so users
# land at the repo root and can open either tutorial (both parts are visible).
RUN mkdir -p /opt/conda/envs/alchemi-playbook/etc/jupyter && \
    printf "c.KernelSpecManager.ensure_native_kernel = False\nc.MappingKernelManager.default_kernel_name = \"alchemi-main\"\nc.ServerApp.root_dir = \"/workspace\"\n" > /opt/conda/envs/alchemi-playbook/etc/jupyter/jupyter_config.py && \
    jupyter kernelspec remove -f python3 || true && \
    jupyter kernelspec remove -f ovito-pro || true && \
    jupyter kernelspec remove -f alchemi-barebones || true && \
    python -m ipykernel install --sys-prefix --name alchemi-main --display-name "ALCHEMI Main" && \
    python -c "import json, subprocess; specs=json.loads(subprocess.check_output(['jupyter','kernelspec','list','--json']))['kernelspecs']; assert set(specs) == {'alchemi-main'}, specs; assert specs['alchemi-main']['spec']['display_name'] == 'ALCHEMI Main', specs"

ENV LD_LIBRARY_PATH="/opt/conda/envs/alchemi-playbook/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}" \
    MPLCONFIGDIR=/tmp/mpl \
    OVITO_GUI_MODE=headless

# Bake tutorial content into the locally built image so it can run without a
# bind-mount. Compose uses selective bind-mounts which shadow these at runtime.
COPY README.md LICENSE CHANGELOG.md RUNTIME_SNAPSHOT.md SOURCES_AND_LICENSES.md ./
COPY .licenses ./.licenses
COPY part-1-batched-adsorption-toolkit ./part-1-batched-adsorption-toolkit
COPY part-2-batched-melting-toolkit ./part-2-batched-melting-toolkit

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
