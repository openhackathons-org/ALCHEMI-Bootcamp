FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git g++ && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv venv --seed --python 3.12 /opt/venv
ENV VIRTUAL_ENV=/opt/venv UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace
RUN uv pip install "nvalchemi-toolkit[ase,mace,aimnet] @ git+https://github.com/NVIDIA/nvalchemi-toolkit.git"
RUN uv pip install torch --reinstall --index-url https://download.pytorch.org/whl/cu130
RUN uv pip install jupyterlab

COPY . .

EXPOSE 8890
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8890", "--no-browser", "--allow-root"]
