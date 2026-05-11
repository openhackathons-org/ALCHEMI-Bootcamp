#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/tmp/alchemi-playbook"
BUILD_DIR="${REPO_DIR}/build"
IMAGE="alchemi-playbook"
COMPOSE_PROJECT="alchemi-playbook"

require_x86_64() {
    local arch
    arch="$(uname -m)"
    if [ "$arch" != "x86_64" ]; then
        echo "Error: unified image requires an x86_64 host (this node is ${arch})."
        echo "       OVITO ships only x86_64 builds on conda.ovito.org."
        echo "       Allocate on an x86_64 GPU partition (e.g. computelab umbriel-b200)."
        exit 1
    fi
}

load_env() {
    if [ -f "$BUILD_DIR/.env" ]; then
        set -a
        # shellcheck source=/dev/null
        source "$BUILD_DIR/.env"
        set +a
    fi
}

ngc_login() {
    load_env
    if [ -z "${NGC_API_KEY:-}" ]; then
        echo "Error: NGC_API_KEY not set. Create build/.env with NGC_API_KEY=<key>."
        exit 1
    fi
    echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
}

cmd_start() {
    if [ ! -d "$BUILD_DIR" ]; then
        echo "Error: build dir not found at $BUILD_DIR. Run deploy.sh setup first."
        exit 1
    fi

    require_x86_64
    ngc_login

    cd "$BUILD_DIR"
    docker compose -p "$COMPOSE_PROJECT" up -d --build
    echo "Stack started."
    sleep 3
    cmd_status
}

cmd_restart() {
    cd "$BUILD_DIR"
    docker compose -p "$COMPOSE_PROJECT" restart
    echo "Stack restarted."
    sleep 3
    cmd_status
}

cmd_rebuild() {
    require_x86_64
    ngc_login
    cd "$BUILD_DIR"
    docker compose -p "$COMPOSE_PROJECT" up -d --build --force-recreate
    echo "Stack rebuilt and started."
    sleep 3
    cmd_status
}

cmd_status() {
    cd "$BUILD_DIR"
    echo "=== Service Status ==="
    docker compose -p "$COMPOSE_PROJECT" ps

    echo ""
    echo "=== Jupyter URL ==="
    docker compose -p "$COMPOSE_PROJECT" logs jupyter 2>&1 \
        | grep -oE 'http://127\.0\.0\.1:[0-9]+/lab\?token=[a-z0-9]+' \
        | tail -1 || echo "Jupyter not ready yet."

    echo ""
    echo "=== GPUs visible to Jupyter container ==="
    docker compose -p "$COMPOSE_PROJECT" exec -T jupyter nvidia-smi -L 2>/dev/null \
        | head -8 || echo "Jupyter container not running (or nvidia-smi unavailable)."

    echo ""
    echo "=== BGR Health ==="
    curl -sf http://localhost:8000/v1/health/ready && echo " BGR: healthy" \
        || echo "BGR: not ready"

    echo ""
    echo "=== Grafana ==="
    echo "http://localhost:3000 (admin/admin)"
}

cmd_stop() {
    cd "$BUILD_DIR"
    docker compose -p "$COMPOSE_PROJECT" down
    echo "Stack stopped."
}

case "${1:-}" in
    restart) cmd_restart ;;
    rebuild) cmd_rebuild ;;
    status)  cmd_status ;;
    stop)    cmd_stop ;;
    *)       cmd_start ;;
esac
