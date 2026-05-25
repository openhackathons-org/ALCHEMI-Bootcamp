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
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cu130

# Remaining runtime + tooling deps via uv (faster than conda for these PyPI /
# git packages). Single requirements file -- toolkit-ops is pinned to git
# HEAD inside it so transitive PyPI resolution can't silently override
# the integrator-fix commits (PR #77, etc.). Copy into /tmp so the file
# doesn't end up inside /workspace (which is the WORKDIR and would leak
# it into the student view).
COPY build/requirements.txt /tmp/
RUN uv pip install -r /tmp/requirements.txt

# Fail the build loud if any resolver tug-of-war silently broke an import.
RUN python -c "import torch, ovito, ase, pymatgen, nvalchemi, matplotlib, aiohttp, pydantic, ipywidgets; print('imports ok; torch', torch.__version__, 'ovito', ovito.version)"

ENV BGR_SERVER=http://localhost:8000 \
    LD_LIBRARY_PATH="/opt/conda/envs/alchemi-playbook/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}" \
    MPLCONFIGDIR=/tmp/mpl \
    OVITO_GUI_MODE=headless

# Bake tutorial content into the image so a `docker save | docker load`
# → `docker run` (no bind-mount) still works. Compose uses selective
# bind-mounts which shadow these at runtime.
COPY README.md LICENSE ./
COPY part-1-nim ./part-1-nim
COPY part-2-toolkit ./part-2-toolkit

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
