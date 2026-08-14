# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.11.1 AS uv

FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python3.12 \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

RUN sed -i -E 's/^UID_MAX[[:space:]]+.*/UID_MAX 4294967294/' /etc/login.defs \
    && sed -i -E 's/^GID_MAX[[:space:]]+.*/GID_MAX 4294967294/' /etc/login.defs \
    && groupadd --gid "${GROUP_ID}" alchemi \
    && useradd \
        --create-home \
        --gid "${GROUP_ID}" \
        --no-log-init \
        --shell /bin/bash \
        --uid "${USER_ID}" \
        alchemi \
    && mkdir -p /opt/alchemi-v3-runtime /workspace \
    && chown -R alchemi:alchemi /opt/alchemi-v3-runtime /workspace

ENV ALCHEMI_V3_RUNTIME_ROOT=/opt/alchemi-v3-runtime \
    HOME=/home/alchemi

WORKDIR /workspace

USER alchemi

COPY --chown=alchemi:alchemi pyproject.toml uv.lock .python-version ./
COPY --chown=alchemi:alchemi scripts/v3-run scripts/v3-sync ./scripts/
COPY --chown=alchemi:alchemi \
    environment/prewarm_assets.py \
    environment/runtime-pins.toml \
    ./environment/

RUN ./scripts/v3-sync

COPY --chown=alchemi:alchemi . .

RUN ./scripts/v3-run python environment/check_runtime.py \
    && ./scripts/v3-run python -c \
        'import torch; from nvalchemi.models import MACEWrapper; MACEWrapper.from_checkpoint("medium-0b2", device=torch.device("cpu"))'

EXPOSE 8888

CMD ["./scripts/v3-jupyter-container"]
