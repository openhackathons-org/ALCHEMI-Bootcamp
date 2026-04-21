#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/tmp/alchemi-playbook-part1"
IMAGE="alchemi-playbook-part1-jupyter"
COMPOSE_PROJECT="alchemi-playbook-part1"

load_env() {
    # Preference order: existing NGC_API_KEY > .env next to compose > ~/.config/ngc/api_key
    if [ -n "${NGC_API_KEY:-}" ]; then
        return 0
    fi
    if [ -f "$REPO_DIR/.env" ]; then
        set -a
        # shellcheck source=/dev/null
        source "$REPO_DIR/.env"
        set +a
    fi
    if [ -z "${NGC_API_KEY:-}" ] && [ -f "$HOME/.config/ngc/api_key" ]; then
        NGC_API_KEY="$(tr -d '\n' < "$HOME/.config/ngc/api_key")"
        export NGC_API_KEY
    fi
}

ngc_login() {
    load_env
    if [ -z "${NGC_API_KEY:-}" ]; then
        cat <<'EOF' >&2
Error: NGC_API_KEY not set. Provide one of:
  - ~/.config/ngc/api_key  (one line, chmod 600)
  - <repo>/.env            (NGC_API_KEY=<key>)
  - environment variable NGC_API_KEY exported before running this script
EOF
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
