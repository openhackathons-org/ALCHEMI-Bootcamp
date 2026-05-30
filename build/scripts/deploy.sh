#!/usr/bin/env bash
set -euo pipefail

JUPYTER_LOCAL_PORT="${JUPYTER_LOCAL_PORT:-8888}"
GRAFANA_LOCAL_PORT="${GRAFANA_LOCAL_PORT:-3000}"
PROMETHEUS_LOCAL_PORT="${PROMETHEUS_LOCAL_PORT:-}"   # set to e.g. 9090 to forward
BGR_LOCAL_PORT="${BGR_LOCAL_PORT:-}"                 # set to e.g. 8000 to forward

REMOTE_REPO_DIR="/tmp/alchemi-playbook"
REMOTE_BUILD_DIR="${REMOTE_REPO_DIR}/build"
STATE_FILE="/tmp/alchemi-playbook-deploy.env"
REMOTE_SCRIPT="/tmp/docker-dev.sh"

# Path resolution: this script lives at dev/build/scripts/deploy.sh.
# BUILD_DIR -> dev/build, REPO_DIR -> dev/ (the build context root).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$BUILD_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: deploy.sh <command> [args]

Commands:
  setup  <login-host> <compute-node>   First-time: copy repo, build, start stack, open tunnels
  restart                               Pull latest local changes and rebuild stack
  pull-changes                          Sync remote Jupyter edits to local working directory
  status                                Show Jupyter URL, BGR health, Grafana URL
  stop                                  Close tunnels and stop stack

Environment variables:
  JUPYTER_LOCAL_PORT     Local Jupyter port    (default: 8888)
  GRAFANA_LOCAL_PORT     Local Grafana port    (default: 3000)
  PROMETHEUS_LOCAL_PORT  Local Prometheus port (default: unset; set to 9090 to forward)
  BGR_LOCAL_PORT         Local BGR port        (default: unset; set to 8000 to forward)
EOF
    exit 1
}

cleanup_legacy_state() {
    local removed=0
    for f in /tmp/alchemi-playbook-part1-deploy.env \
             /tmp/alchemi-playbook-part2-deploy.env \
             /tmp/docker-dev-part1.sh \
             /tmp/docker-dev-part2.sh; do
        if [ -e "$f" ]; then
            rm -f "$f"
            removed=1
        fi
    done
    if [ "$removed" -eq 1 ]; then
        echo "Removed legacy per-part state files from /tmp."
    fi
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

build_ssh_forwards() {
    # Echoes the SSH -L flags for whichever ports are non-empty.
    printf -- '-L %s:localhost:8888 ' "${JUPYTER_LOCAL_PORT}"
    printf -- '-L %s:localhost:3000 ' "${GRAFANA_LOCAL_PORT}"
    if [ -n "${PROMETHEUS_LOCAL_PORT}" ]; then
        printf -- '-L %s:localhost:9090 ' "${PROMETHEUS_LOCAL_PORT}"
    fi
    if [ -n "${BGR_LOCAL_PORT}" ]; then
        printf -- '-L %s:localhost:8000 ' "${BGR_LOCAL_PORT}"
    fi
}

open_tunnels() {
    close_tunnels 2>/dev/null || true
    # shellcheck disable=SC2046
    ssh -f -N \
        -o ExitOnForwardFailure=yes \
        -J "$LOGIN_HOST" \
        $(build_ssh_forwards) \
        "$COMPUTE_NODE"
    sleep 1
    if lsof -ti:"${JUPYTER_LOCAL_PORT}" > /dev/null 2>&1; then
        echo "Tunnels open:"
        echo "  Jupyter:    localhost:${JUPYTER_LOCAL_PORT} -> ${COMPUTE_NODE}:8888"
        echo "  Grafana:    localhost:${GRAFANA_LOCAL_PORT} -> ${COMPUTE_NODE}:3000"
        [ -n "${PROMETHEUS_LOCAL_PORT}" ] && \
            echo "  Prometheus: localhost:${PROMETHEUS_LOCAL_PORT} -> ${COMPUTE_NODE}:9090"
        [ -n "${BGR_LOCAL_PORT}" ] && \
            echo "  BGR:        localhost:${BGR_LOCAL_PORT} -> ${COMPUTE_NODE}:8000"
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
    echo "Stopping any running stack on ${COMPUTE_NODE} (so bind-mounts release before nuke)..."
    # Handle both old-layout (compose at REPO_DIR root) and new-layout
    # (compose at REPO_DIR/build/) without failing if neither exists yet.
    remote_exec "
        if [ -f ${REMOTE_BUILD_DIR}/docker-compose.yml ]; then
            cd ${REMOTE_BUILD_DIR} && docker compose -p alchemi-playbook down 2>/dev/null || true
        elif [ -f ${REMOTE_REPO_DIR}/docker-compose.yml ]; then
            cd ${REMOTE_REPO_DIR} && docker compose -p alchemi-playbook down 2>/dev/null || true
        fi
        rm -rf ${REMOTE_REPO_DIR}
        mkdir -p ${REMOTE_REPO_DIR}
    "

    echo "Copying repo to ${COMPUTE_NODE}:${REMOTE_REPO_DIR}..."
    COPYFILE_DISABLE=1 tar -C "$REPO_DIR" \
        --no-xattrs \
        --exclude='.git' \
        --exclude='.env' \
        --exclude='build/.env' \
        --exclude='.claude' \
        --exclude='cached_responses' \
        --exclude='outputs' \
        --exclude='logs' \
        --exclude='./part-1-nim/assets' \
        --exclude='./part-2-toolkit/assets' \
        --exclude='./part-3-batched-adsorption/data/reference/oc20dense-validation-pack.tgz' \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='.ruff_cache' \
        --exclude='.ipynb_checkpoints' \
        --exclude='.DS_Store' \
        --exclude='matplotlib-*' \
        --exclude='tmp*' \
        --exclude='docker-compose.override.yml' \
        --exclude='.Trash-*' \
        --exclude='*-with_outputs.ipynb' \
        -cf - . \
        | ssh -J "$LOGIN_HOST" "$COMPUTE_NODE" \
            "tar -C ${REMOTE_REPO_DIR} -xf -"
}

cmd_setup() {
    [ -z "${1:-}" ] || [ -z "${2:-}" ] && usage
    LOGIN_HOST="$1"
    COMPUTE_NODE="$2"

    cleanup_legacy_state
    save_state "$LOGIN_HOST" "$COMPUTE_NODE"

    if [ ! -f "$BUILD_DIR/.env" ]; then
        echo "Error: .env not found at ${BUILD_DIR}/.env."
        echo "       Copy build/.env.example to build/.env and set your NGC_API_KEY."
        exit 1
    fi

    copy_repo

    echo "Copying .env to ${COMPUTE_NODE}..."
    scp -o ProxyJump="$LOGIN_HOST" "$BUILD_DIR/.env" "${COMPUTE_NODE}:${REMOTE_BUILD_DIR}/.env"

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

    echo "Re-copying .env to ${COMPUTE_NODE}..."
    scp -o ProxyJump="$LOGIN_HOST" "$BUILD_DIR/.env" "${COMPUTE_NODE}:${REMOTE_BUILD_DIR}/.env"

    echo "Re-copying docker-dev.sh to ${COMPUTE_NODE}..."
    scp -o ProxyJump="$LOGIN_HOST" "$SCRIPT_DIR/docker-dev.sh" "${COMPUTE_NODE}:${REMOTE_SCRIPT}"

    echo "Rebuilding stack on ${COMPUTE_NODE}..."
    remote_exec "bash ${REMOTE_SCRIPT} rebuild"
    echo "Reopening SSH tunnels..."
    open_tunnels
}

cmd_status() {
    load_state
    remote_exec "bash ${REMOTE_SCRIPT} status"

    echo ""
    if lsof -ti:"${JUPYTER_LOCAL_PORT}" > /dev/null 2>&1; then
        echo "Tunnels: active (Jupyter localhost:${JUPYTER_LOCAL_PORT}, Grafana localhost:${GRAFANA_LOCAL_PORT})"
    else
        echo "Tunnels: inactive"
    fi
}

cmd_pull_changes() {
    load_state
    echo "Pulling changes from remote..."
    ssh -J "$LOGIN_HOST" "$COMPUTE_NODE" \
        "cd ${REMOTE_REPO_DIR} && tar -cf - \
            part-1-nim/*.ipynb part-1-nim/helpers/ \
            part-2-toolkit/*.ipynb part-2-toolkit/helpers/" \
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
