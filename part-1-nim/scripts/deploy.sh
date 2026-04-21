#!/usr/bin/env bash
set -euo pipefail

JUPYTER_LOCAL_PORT="${JUPYTER_LOCAL_PORT:-8888}"
GRAFANA_LOCAL_PORT="${GRAFANA_LOCAL_PORT:-3000}"
BGR_LOCAL_PORT="${BGR_LOCAL_PORT:-8000}"
REMOTE_REPO_DIR="/tmp/alchemi-playbook-part1"
STATE_FILE="/tmp/alchemi-playbook-part1-deploy.env"
REMOTE_SCRIPT="/tmp/docker-dev-part1.sh"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: deploy.sh <command> [args]

Commands:
  setup <login-host> <compute-node>  Cluster mode: bastion-jump to compute node,
                                     copy repo, build stack, open tunnels.
  setup-local <host>                 Direct mode (ws-loc, single Linux box):
                                     rsync repo, build stack, open tunnels over
                                     one SSH hop (no bastion).
  restart                            Push local edits, rebuild stack.
  pull-changes                       Sync remote Jupyter edits to the local repo.
  status                             Show Jupyter URL, BGR health, tunnel state.
  stop                               Close tunnels, stop stack, clear state.

NGC API key:
  The stack needs NGC_API_KEY for pulling the NIM image and for runtime. On the
  remote host, provide it via EITHER:
    - ~/.config/ngc/api_key  (one line, chmod 600 - recommended)
    - <repo>/.env            (NGC_API_KEY=<key>, gitignored; legacy convention)

Environment:
  JUPYTER_LOCAL_PORT  Local Jupyter port  (default: 8888)
  GRAFANA_LOCAL_PORT  Local Grafana port  (default: 3000)
  BGR_LOCAL_PORT      Local BGR port      (default: 8000)
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
        echo "No active deployment. Run 'deploy.sh setup ...' or 'deploy.sh setup-local ...' first."
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$STATE_FILE"
}

# -------- SSH helpers (use ProxyJump only when LOGIN_HOST is set) --------

ssh_args() {
    if [ -n "${LOGIN_HOST:-}" ]; then
        echo "-J $LOGIN_HOST"
    fi
}

remote_exec() {
    # shellcheck disable=SC2046
    ssh $(ssh_args) "$COMPUTE_NODE" "$@"
}

open_tunnels() {
    close_tunnels 2>/dev/null || true
    # shellcheck disable=SC2046
    ssh -f -N \
        -o ExitOnForwardFailure=yes \
        $(ssh_args) \
        -L "${JUPYTER_LOCAL_PORT}:localhost:8888" \
        -L "${GRAFANA_LOCAL_PORT}:localhost:3000" \
        -L "${BGR_LOCAL_PORT}:localhost:8000" \
        "$COMPUTE_NODE"
    sleep 1
    if lsof -ti:"${JUPYTER_LOCAL_PORT}" > /dev/null 2>&1; then
        echo "Tunnels open:"
        echo "  Jupyter:  localhost:${JUPYTER_LOCAL_PORT} -> ${COMPUTE_NODE}:8888"
        echo "  Grafana:  localhost:${GRAFANA_LOCAL_PORT} -> ${COMPUTE_NODE}:3000"
        echo "  BGR:      localhost:${BGR_LOCAL_PORT}     -> ${COMPUTE_NODE}:8000"
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

# -------- Repo transfer --------

copy_repo_tar() {
    # Used by 'setup' (cluster-mode) - tar-over-ssh, one-shot.
    echo "Copying repo to ${COMPUTE_NODE}:${REMOTE_REPO_DIR} (tar)..."
    COPYFILE_DISABLE=1 tar -C "$REPO_DIR" \
        --no-xattrs \
        --exclude='.git' --exclude='.env' --exclude='.claude' \
        --exclude='cached_responses' --exclude='outputs' \
        --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.ruff_cache' \
        --exclude='.ipynb_checkpoints' --exclude='.DS_Store' \
        --exclude='matplotlib-*' --exclude='tmp*' \
        --exclude='docker-compose.override.yml' \
        -cf - . \
        | ssh $(ssh_args) "$COMPUTE_NODE" \
            "mkdir -p ${REMOTE_REPO_DIR} && tar -C ${REMOTE_REPO_DIR} -xf -"
}

copy_repo_rsync() {
    # Used by 'setup-local' and 'restart' - rsync incremental, preserves .git
    # so git log / checkout works on the remote machine too.
    echo "Rsyncing repo to ${COMPUTE_NODE}:${REMOTE_REPO_DIR} ..."
    local ssh_cmd="ssh"
    if [ -n "${LOGIN_HOST:-}" ]; then
        ssh_cmd="ssh -J $LOGIN_HOST"
    fi
    rsync -az --delete \
        -e "$ssh_cmd" \
        --exclude='.env' --exclude='.claude/' \
        --exclude='outputs/' --exclude='__pycache__/' --exclude='.pytest_cache/' \
        --exclude='.ruff_cache/' --exclude='.ipynb_checkpoints/' --exclude='.DS_Store' \
        --exclude='matplotlib-*/' --exclude='tmp*/' \
        --exclude='docker-compose.override.yml' \
        "$REPO_DIR/" "$COMPUTE_NODE:$REMOTE_REPO_DIR/"
}

# -------- Commands --------

cmd_setup() {
    [ -z "${1:-}" ] || [ -z "${2:-}" ] && usage
    LOGIN_HOST="$1"
    COMPUTE_NODE="$2"
    save_state "$LOGIN_HOST" "$COMPUTE_NODE"

    if [ ! -f "$REPO_DIR/.env" ]; then
        echo "Error: .env not found locally. Copy .env.example to .env and set NGC_API_KEY,"
        echo "       or use 'setup-local' which reads ~/.config/ngc/api_key on the remote."
        exit 1
    fi

    copy_repo_tar

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
    echo "  BGR:        http://localhost:${BGR_LOCAL_PORT}"
}

cmd_setup_local() {
    [ -z "${1:-}" ] && usage
    LOGIN_HOST=""
    COMPUTE_NODE="$1"
    save_state "$LOGIN_HOST" "$COMPUTE_NODE"

    # Verify NGC key is findable on the remote machine; do NOT copy a local .env.
    echo "Checking NGC key on ${COMPUTE_NODE}..."
    if ! ssh "$COMPUTE_NODE" "test -f \$HOME/.config/ngc/api_key || test -f ${REMOTE_REPO_DIR}/.env" 2>/dev/null; then
        cat <<EOF >&2
NGC key not found on ${COMPUTE_NODE}. Provide one of:

  A. Recommended — place the key in a dot-config file (outside the repo):
       mkdir -p ~/.config/ngc
       printf '%s' '<your-ngc-api-key>' > ~/.config/ngc/api_key
       chmod 600 ~/.config/ngc/api_key

  B. Legacy — write a .env next to docker-compose.yml:
       echo 'NGC_API_KEY=<your-ngc-api-key>' > ${REMOTE_REPO_DIR}/.env
       chmod 600 ${REMOTE_REPO_DIR}/.env
EOF
        exit 1
    fi

    copy_repo_rsync
    scp "$SCRIPT_DIR/docker-dev.sh" "${COMPUTE_NODE}:${REMOTE_SCRIPT}"
    echo "Building and starting stack on ${COMPUTE_NODE}..."
    remote_exec "bash ${REMOTE_SCRIPT}"

    echo "Opening SSH tunnels (direct, no bastion)..."
    open_tunnels

    echo ""
    echo "Setup complete (local/direct mode)."
    echo "  JupyterLab: http://localhost:${JUPYTER_LOCAL_PORT}"
    echo "  Grafana:    http://localhost:${GRAFANA_LOCAL_PORT}"
    echo "  BGR:        http://localhost:${BGR_LOCAL_PORT}"
}

cmd_restart() {
    load_state
    echo "Syncing local changes to ${COMPUTE_NODE}..."
    copy_repo_rsync
    echo "Rebuilding stack on ${COMPUTE_NODE}..."
    remote_exec "bash ${REMOTE_SCRIPT} rebuild"
}

cmd_status() {
    load_state
    remote_exec "bash ${REMOTE_SCRIPT} status"
    echo ""
    if lsof -ti:"${JUPYTER_LOCAL_PORT}" > /dev/null 2>&1; then
        echo "Tunnel: active"
    else
        echo "Tunnel: inactive"
    fi
}

cmd_pull_changes() {
    load_state
    echo "Pulling changes from remote..."
    # shellcheck disable=SC2046
    ssh $(ssh_args) "$COMPUTE_NODE" \
        "cd ${REMOTE_REPO_DIR} && tar -cf - *.ipynb helpers/ scripts/build_notebook.py" \
        | tar -C "$REPO_DIR" -xf -
    echo "Remote edits applied locally. Review with 'git diff'."
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
    setup-local)  shift; cmd_setup_local "$@" ;;
    restart)      cmd_restart ;;
    pull-changes) cmd_pull_changes ;;
    status)       cmd_status ;;
    stop)         cmd_stop ;;
    *)            usage ;;
esac
