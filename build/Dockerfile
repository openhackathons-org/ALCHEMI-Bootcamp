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
    UV_SYSTEM_PYTHON=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

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
COPY build/requirements.txt /tmp/requirements.txt
RUN uv pip install \
    --no-sources-package nvalchemi-toolkit-ops \
    -r /tmp/requirements.txt

# Check package metadata before importing the Part 1 stack.
RUN uv pip check

# Fail the build loudly if a required Part 1 import is broken.
RUN python -c "import importlib.metadata as md, shutil; import torch, jax, ovito, ase, pymatgen, nvalchemi, aimnet, sevenn, physicsnemo, matplotlib, aiohttp, pydantic, ipywidgets, warp; from nvalchemi.distributed import DomainConfig, DomainParallel; from nvalchemi.models import AIMNet2Wrapper, DFTD3ModelWrapper, PMEModelWrapper, PipelineModelWrapper; from nvalchemi.neighbors import compute_neighbors; assert shutil.which('packmol'), 'packmol executable missing'; assert shutil.which('dot'), 'Graphviz dot executable missing'; print('imports ok; torch', torch.__version__, 'jax', jax.__version__, 'ovito', ovito.version, 'toolkit', md.version('nvalchemi-toolkit'), 'ops', md.version('nvalchemi-toolkit-ops'), 'aimnet', md.version('aimnet'), 'sevenn', md.version('sevenn'), 'e3nn', md.version('e3nn'), 'physicsnemo', md.version('nvidia-physicsnemo'), 'warp', md.version('warp-lang'), 'packmol executable', shutil.which('packmol'))"

# Verify the opening AIMNet checkpoint plus the four ensemble members used by
# the NCI stage, then remove the downloaded weights in the same image layer.
# A runtime cache or first-use download supplies them to the learner session.
COPY build/prewarm_aimnet.py /tmp/prewarm_aimnet.py
RUN python /tmp/prewarm_aimnet.py --remove-checkpoints-after-test

# Exercise Toolkit's official D3 setup and fixed checksum during the image
# build, but do not retain the generated parameter tensor in an image layer.
# Learners create it in their own runtime cache on first use.
COPY build/prewarm_d3.py /tmp/prewarm_d3.py
RUN python /tmp/prewarm_d3.py \
      --parameter-file /tmp/part1-dftd3-parameters.pt \
      --output /tmp/part1-d3-check.json && \
    rm -f /tmp/part1-dftd3-parameters.pt /tmp/part1-d3-check.json

# Verify the pinned SevenNet-Omni checkpoint without retaining it in the image.
# The prewarm step runs the official calculator and the tutorial's Toolkit
# adapter on the same periodic Cu(111)+CO structure, then removes the downloaded
# checkpoint in the same image layer.
COPY build/prewarm_sevennet.py /tmp/part1-prewarm-source/build/prewarm_sevennet.py
COPY part-1-scalable-atomistic-workflows/aux/__init__.py /tmp/part1-prewarm-source/part-1-scalable-atomistic-workflows/aux/__init__.py
COPY part-1-scalable-atomistic-workflows/aux/models/__init__.py /tmp/part1-prewarm-source/part-1-scalable-atomistic-workflows/aux/models/__init__.py
COPY part-1-scalable-atomistic-workflows/aux/models/sevennet.py /tmp/part1-prewarm-source/part-1-scalable-atomistic-workflows/aux/models/sevennet.py
COPY part-1-scalable-atomistic-workflows/aux/models/sevennet_checkpoint.py /tmp/part1-prewarm-source/part-1-scalable-atomistic-workflows/aux/models/sevennet_checkpoint.py
COPY part-1-scalable-atomistic-workflows/aux/models/sevennet_config.py /tmp/part1-prewarm-source/part-1-scalable-atomistic-workflows/aux/models/sevennet_config.py
RUN python /tmp/part1-prewarm-source/build/prewarm_sevennet.py \
      --remove-checkpoint-after-test

# Ship one notebook kernel name for the whole playbook. Conda/ipykernel/OVITO
# can add default kernelspecs (`python3`, `ovito-pro`); remove those aliases
# plus the old tutorial name so VS Code/Jupyter see only the reproducible
# playbook kernel. Jupyter can also synthesize a native `python3` fallback at
# runtime, and its server API defaults to `python3` unless told otherwise.
# The same config pins the JupyterLab file-browser root to /workspace.
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
COPY .dockerignore .gitattributes .gitignore ./
COPY README.md LICENSE THIRD_PARTY_NOTICES.md CHANGELOG.md RUNTIME_SNAPSHOT.md ./
COPY TUTORIAL_DESIGN_PRINCIPLES.md ALCHEMI_TUTORIAL_PRINCIPLES.md TOOLKIT_API_CURRICULUM.md ./
COPY build/Dockerfile build/docker-compose.yml build/environment.yml ./build/
COPY build/requirements.txt ./build/requirements.txt
COPY build/prewarm_aimnet.py ./build/prewarm_aimnet.py
COPY build/prewarm_d3.py ./build/prewarm_d3.py
COPY build/prewarm_sevennet.py ./build/prewarm_sevennet.py
COPY part-1-scalable-atomistic-workflows ./part-1-scalable-atomistic-workflows
# Retained Parts 2 and 3 need separate historical environments and contain
# data that is outside the Part 1 release. Include only their status READMEs.
COPY part-2-batched-adsorption-toolkit/README.md ./part-2-batched-adsorption-toolkit/README.md
COPY part-3-batched-melting-toolkit/README.md ./part-3-batched-melting-toolkit/README.md
# Stage 7 and the Compute Lab runbook link to these maintained sources.
# Learners do not launch the Slurm jobs during class, but the complete public
# API, validation, and recorded-result paths must remain inspectable.
COPY scripts/assemble_part1_pipeline_campaign.py \
     scripts/benchmark_part1_distributed_campaign.py \
     scripts/part1_distributed_campaign_contract.py \
     scripts/part1_domain_plan.py \
     scripts/part1_domain_run.py \
     scripts/prepare_nci_subset.py \
     scripts/rebuild_part1_ir_notebook.py \
     scripts/record_part1_campaign_failure.py \
     scripts/render_part1_prebuilt_domain_box.py \
     scripts/review_part1_ir_executed_notebook.py \
     scripts/run_notebook_no_timeout.py \
     scripts/run_part1_distributed_torchrun.sh \
     scripts/run_part1_domain_decomposition.sh \
     scripts/slurm_part1_dimer_reference.sbatch \
     scripts/slurm_part1_distributed_campaign.sbatch \
     scripts/slurm_part1_domain_decomposition.sbatch \
     scripts/slurm_part1_remaster_h100.sbatch \
     scripts/slurm_part1_sevennet_setup.sbatch \
     scripts/validate_part1_ir_run.py \
     ./scripts/
COPY build/part1-source-files.txt ./build/part1-source-files.txt
COPY build/verify_part1_runtime.py ./build/verify_part1_runtime.py

# Run the CPU-capable checks against the baked files and environment. Git
# identity is checked before the image build; the H100 run repeats this command
# with --require-cuda against the finished image.
RUN python build/verify_part1_runtime.py \
    --source-root /workspace \
    --skip-source-check \
    --output build/part1-image-runtime.json

# Parse every final notebook cell and exercise the strict packaged-reference
# loaders after the learner files have been copied. GPU model execution remains
# part of the separate H100 release run.
RUN WARP_CACHE_PATH=/tmp/warp-cache python -m pytest -q \
    part-1-scalable-atomistic-workflows/tests/test_notebook_contract.py \
    part-1-scalable-atomistic-workflows/tests/test_artifact_manifest.py \
    part-1-scalable-atomistic-workflows/tests/test_reference_data.py \
    part-1-scalable-atomistic-workflows/tests/test_experimental_reference.py

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
