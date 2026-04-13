#!/usr/bin/env bash
set -euo pipefail

JUPYTER_LOCAL_PORT="${JUPYTER_LOCAL_PORT:-8888}"
GRAFANA_LOCAL_PORT="${GRAFANA_LOCAL_PORT:-3000}"
REMOTE_REPO_DIR="/tmp/alchemi-playbook-part1"
STATE_FILE="/tmp/alchemi-playbook-part1-deploy.env"
REMOTE_SCRIPT="/tmp/docker-dev-part1.sh"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: deploy.sh <command> [args]

Commands:
  setup  <login-host> <compute-node>   First-time: copy repo, build, start stack, open tunnel
  restart                               Pull latest local changes and restart stack
  pull-changes                          Sync remote Jupyter edits to local working directory
  status                                Show Jupyter URL, BGR health, Grafana URL
  stop                                  Close tunnels and stop stack

Environment variables:
  JUPYTER_LOCAL_PORT    Local Jupyter port  (default: 8888)
  GRAFANA_LOCAL_PORT    Local Grafana port  (default: 3000)
EOF
    exit 1
}

save_state() {
    cat > "$STATE_FILE" <<EOF
LOGIN_HOST=$1
COMPUTE_NODE=$2
EOF
}

load_state() {
    if [ ! -f "$STATE_FILE" ]; then
        echo "No active deployment. Run 'deploy.sh setup <login-host> <compute-node>' first."
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$STATE_FILE"
}

remote_exec() {
    ssh -J "$LOGIN_HOST" "$COMPUTE_NODE" "$@"
}

open_tunnels() {
    close_tunnels 2>/dev/null || true
    ssh -f -N \
        -o ExitOnForwardFailure=yes \
        -J "$LOGIN_HOST" \
        -L "${JUPYTER_LOCAL_PORT}:localhost:8888" \
        -L "${GRAFANA_LOCAL_PORT}:localhost:3000" \
        "$COMPUTE_NODE"
    sleep 1
    if lsof -ti:"${JUPYTER_LOCAL_PORT}" > /dev/null 2>&1; then
        echo "Tunnels open:"
        echo "  Jupyter:  localhost:${JUPYTER_LOCAL_PORT} -> ${COMPUTE_NODE}:8888"
        echo "  Grafana:  localhost:${GRAFANA_LOCAL_PORT} -> ${COMPUTE_NODE}:3000"
    else
        echo "Failed to open SSH tunnels."
        return 1
    fi
}

close_tunnels() {
    local pid
    pid=$(lsof -ti:"${JUPYTER_LOCAL_PORT}" 2>/dev/null) || true
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        echo "Tunnels closed."
    fi
}

copy_repo() {
    echo "Copying repo to ${COMPUTE_NODE}:${REMOTE_REPO_DIR}..."
    # Create a tarball excluding .git and large transient files, pipe through SSH
    COPYFILE_DISABLE=1 tar -C "$REPO_DIR" \
        --no-xattrs \
        --exclude='.git' \
        --exclude='.env' \
        --exclude='.claude' \
        --exclude='cached_responses' \
        --exclude='outputs' \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='.ruff_cache' \
        --exclude='.ipynb_checkpoints' \
        --exclude='.DS_Store' \
        --exclude='matplotlib-*' \
        --exclude='tmp*' \
        --exclude='docker-compose.override.yml' \
        -cf - . \
        | ssh -J "$LOGIN_HOST" "$COMPUTE_NODE" \
            "mkdir -p ${REMOTE_REPO_DIR} && tar -C ${REMOTE_REPO_DIR} -xf -"
}

cmd_setup() {
    [ -z "${1:-}" ] || [ -z "${2:-}" ] && usage
    LOGIN_HOST="$1"
    COMPUTE_NODE="$2"
    save_state "$LOGIN_HOST" "$COMPUTE_NODE"

    # Verify .env exists locally
    if [ ! -f "$REPO_DIR/.env" ]; then
        echo "Error: .env not found. Copy .env.example to .env and set your NGC_API_KEY."
        exit 1
    fi

    copy_repo

    echo "Copying .env to ${COMPUTE_NODE}..."
    scp -o ProxyJump="$LOGIN_HOST" "$REPO_DIR/.env" "${COMPUTE_NODE}:${REMOTE_REPO_DIR}/.env"

    echo "Copying docker-dev.sh to ${COMPUTE_NODE}..."
    scp -o ProxyJump="$LOGIN_HOST" "$SCRIPT_DIR/docker-dev.sh" "${COMPUTE_NODE}:${REMOTE_SCRIPT}"

    echo "Building and starting stack on ${COMPUTE_NODE}..."
    remote_exec "bash ${REMOTE_SCRIPT}"

    echo "Opening SSH tunnels..."
    open_tunnels

    echo ""
    echo "Setup complete."
    echo "  JupyterLab: http://localhost:${JUPYTER_LOCAL_PORT}"
    echo "  Grafana:    http://localhost:${GRAFANA_LOCAL_PORT}"
}

cmd_restart() {
    load_state
    echo "Syncing local changes to ${COMPUTE_NODE}..."
    copy_repo
    echo "Rebuilding stack on ${COMPUTE_NODE}..."
    remote_exec "bash ${REMOTE_SCRIPT} rebuild"
}

cmd_status() {
    load_state
    remote_exec "bash ${REMOTE_SCRIPT} status"

    echo ""
    if lsof -ti:"${JUPYTER_LOCAL_PORT}" > /dev/null 2>&1; then
        echo "Tunnel: active (Jupyter localhost:${JUPYTER_LOCAL_PORT}, Grafana localhost:${GRAFANA_LOCAL_PORT})"
    else
        echo "Tunnel: inactive"
    fi
}

cmd_pull_changes() {
    load_state
    echo "Pulling changes from remote..."
    ssh -J "$LOGIN_HOST" "$COMPUTE_NODE" \
        "cd ${REMOTE_REPO_DIR} && tar -cf - *.ipynb helpers/" \
        | tar -C "$REPO_DIR" -xf -
    echo "Remote notebook and helper changes applied locally. Review with 'git diff'."
}

cmd_stop() {
    load_state
    close_tunnels
    echo "Stopping stack on ${COMPUTE_NODE}..."
    remote_exec "bash ${REMOTE_SCRIPT} stop"
    rm -f "$STATE_FILE"
}

case "${1:-}" in
    setup)        shift; cmd_setup "$@" ;;
    restart)      cmd_restart ;;
    pull-changes) cmd_pull_changes ;;
    status)       cmd_status ;;
    stop)         cmd_stop ;;
    *)            usage ;;
esac
