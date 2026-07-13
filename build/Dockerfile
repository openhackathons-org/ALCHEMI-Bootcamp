FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04

# Minimal system deps. The conda env (next layer) handles Qt6/glib/X11 for OVITO.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git g++ ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (open-source conda + mamba). Multiarch installer.
RUN ARCH=$(uname -m) && \
    curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh" && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

# Create the env: Python 3.12 + ovito (from conda.ovito.org, bundles Qt6/glib/fontconfig)
# + uv (fast pip resolver for the rest).
COPY build/environment.yml /tmp/environment.yml
RUN /opt/conda/bin/conda env create -f /tmp/environment.yml && \
    /opt/conda/bin/conda clean -afy

# Activate the env for all subsequent commands by prepending its bin dir.
# UV_SYSTEM_PYTHON=true lets `uv pip install` target the conda env's
# Python (first on PATH) directly — conda envs aren't venvs in uv's eyes
# (no pyvenv.cfg) but the active Python is what we want to install into.
ENV PATH=/opt/conda/envs/alchemi-playbook/bin:/opt/conda/bin:$PATH \
    CONDA_DEFAULT_ENV=alchemi-playbook \
    UV_SYSTEM_PYTHON=true

WORKDIR /workspace

# Install torch from the CUDA 13.0 wheel index first. This satisfies
# nvalchemi-toolkit's transitive torch requirement so the toolkit install
# (next layer) skips pulling the CPU wheel.
RUN uv pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130

# Remaining runtime + tooling deps via uv (faster than conda for these PyPI /
# git packages). Single requirements file -- toolkit and toolkit-ops are
# pinned to explicit commits for reproducibility. `--no-sources-package`
# prevents toolkit's stale internal uv source pin for toolkit-ops from
# overriding the direct URL in requirements.txt.
COPY build/requirements.txt build/overrides.txt /tmp/
RUN uv pip install \
    --no-sources-package nvalchemi-toolkit-ops \
    --overrides /tmp/overrides.txt \
    -r /tmp/requirements.txt

# Fail the build loud if any resolver tug-of-war silently broke an import.
# cuequivariance_ops_torch itself requires the NVIDIA runtime libraries that
# are injected by `docker run --gpus all`, so build-time validation checks the
# installed distribution and runtime smoke tests import the module.
RUN python -c "import importlib.metadata as md, shutil; import torch, ovito, ase, pymatgen, nvalchemi, aimnet, physicsnemo, orb_models, loguru, matplotlib, aiohttp, pydantic, ipywidgets, warp; from nvalchemi.models.aimnet2 import AIMNet2Wrapper; from nvalchemi.models.dftd3 import DFTD3ModelWrapper; from nvalchemi.models.pipeline import PipelineModelWrapper; from nvalchemi.neighbors import compute_neighbors; assert shutil.which('packmol'), 'packmol executable missing'; assert shutil.which('dot'), 'Graphviz dot executable missing'; print('imports ok; torch', torch.__version__, 'ovito', ovito.version, 'toolkit', md.version('nvalchemi-toolkit'), 'ops', md.version('nvalchemi-toolkit-ops'), 'aimnet', md.version('aimnet'), 'physicsnemo', md.version('nvidia-physicsnemo'), 'warp', md.version('warp-lang'), 'orb', md.version('orb-models'), 'cueq_ops', md.version('cuequivariance-ops-torch-cu13'))"

# Ship one notebook kernel name for the whole playbook. Conda/ipykernel/OVITO
# can add default kernelspecs (`python3`, `ovito-pro`); remove those aliases
# plus the old tutorial name so VS Code/Jupyter see only the reproducible
# playbook kernel. Jupyter can also synthesize a native `python3` fallback at
# runtime, and its server API defaults to `python3` unless told otherwise.
# The same config pins the JupyterLab file-browser root to /workspace so users
# land at the repo root and can open all three tutorials.
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

# Bake tutorial content into the image so a `docker save | docker load`
# → `docker run` (no bind-mount) still works. Compose uses selective
# bind-mounts which shadow these at runtime.
COPY README.md LICENSE CHANGELOG.md RUNTIME_SNAPSHOT.md ./
COPY build/requirements.txt ./build/requirements.txt
COPY build/overrides.txt ./build/overrides.txt
COPY part-1-batched-adsorption-toolkit ./part-1-batched-adsorption-toolkit
COPY part-1-water-hydrogen-bonding-toolkit ./part-1-water-hydrogen-bonding-toolkit
COPY part-2-batched-melting-toolkit ./part-2-batched-melting-toolkit
COPY part-3-toolkit-foundations ./part-3-toolkit-foundations

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
