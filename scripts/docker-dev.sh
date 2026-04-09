#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/tmp/alchemi-playbooks"
IMAGE="alchemi-playbook-jupyter"
COMPOSE_PROJECT="alchemi-playbook"

load_env() {
    if [ -f "$REPO_DIR/.env" ]; then
        set -a
        # shellcheck source=/dev/null
        source "$REPO_DIR/.env"
        set +a
    fi
}

ngc_login() {
    load_env
    if [ -z "${NGC_API_KEY:-}" ]; then
        echo "Error: NGC_API_KEY not set. Create .env with NGC_API_KEY=<key> in the repo."
        exit 1
    fi
    echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
}

cmd_start() {
    if [ ! -d "$REPO_DIR" ]; then
        echo "Error: Repo not found at $REPO_DIR. Run deploy.sh setup first."
        exit 1
    fi

    ngc_login

    cd "$REPO_DIR"
    docker compose -p "$COMPOSE_PROJECT" up -d --build
    echo "Stack started."
    sleep 3
    cmd_status
}

cmd_restart() {
    cd "$REPO_DIR"
    docker compose -p "$COMPOSE_PROJECT" restart
    echo "Stack restarted."
    sleep 3
    cmd_status
}

cmd_rebuild() {
    ngc_login
    cd "$REPO_DIR"
    docker compose -p "$COMPOSE_PROJECT" up -d --build --force-recreate
    echo "Stack rebuilt and started."
    sleep 3
    cmd_status
}

cmd_status() {
    cd "$REPO_DIR"
    echo "=== Service Status ==="
    docker compose -p "$COMPOSE_PROJECT" ps

    echo ""
    echo "=== Jupyter URL ==="
    docker compose -p "$COMPOSE_PROJECT" logs jupyter 2>&1 \
        | grep -oE 'http://127\.0\.0\.1:[0-9]+/lab\?token=[a-z0-9]+' \
        | tail -1 || echo "Jupyter not ready yet."

    echo ""
    echo "=== BGR Health ==="
    curl -sf http://localhost:8000/v1/health/ready && echo " BGR: healthy" \
        || echo "BGR: not ready"

    echo ""
    echo "=== Grafana ==="
    echo "http://localhost:3000 (admin/admin)"
}

cmd_stop() {
    cd "$REPO_DIR"
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
