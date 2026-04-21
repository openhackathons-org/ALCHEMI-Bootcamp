#!/usr/bin/env bash
set -euo pipefail

LOCAL_PORT="${LOCAL_PORT:-8889}"
REMOTE_PORT="${REMOTE_PORT:-8889}"
STATE_FILE="/tmp/alchemi-playbook-part2-deploy.env"
REMOTE_SCRIPT="/tmp/docker-dev-part2.sh"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: deploy.sh <command> [args]

Commands:
  setup  <login-host> <compute-node>   First-time: copy files, build, start container, open tunnel
  restart                               Push local changes and rebuild container
  pull-changes                          Sync remote Jupyter edits to local working directory
  status                                Show Jupyter URL with token
  stop                                  Close tunnel and stop container

Environment variables:
  LOCAL_PORT    Local port for Jupyter (default: 8889)
  REMOTE_PORT   Remote port for Jupyter (default: 8889)
EOF
    exit 1
}

save_state() {
    cat > "$STATE_FILE" <<EOF
LOGIN_HOST=${LOGIN_HOST}
COMPUTE_NODE=${COMPUTE_NODE}
REMOTE_PORT=${REMOTE_PORT}
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

update_port_from_output() {
    local output="$1"
    local actual_port
    actual_port=$(echo "$output" | grep '^JUPYTER_PORT=' | tail -1 | cut -d= -f2) || true
    if [ -n "$actual_port" ] && [ "$actual_port" != "$REMOTE_PORT" ]; then
        echo "Note: Jupyter started on port ${actual_port} (requested ${REMOTE_PORT})"
        REMOTE_PORT="$actual_port"
    fi
}

open_tunnel() {
    close_tunnel 2>/dev/null || true
    ssh -f -N -o ExitOnForwardFailure=yes \
        -J "$LOGIN_HOST" \
        -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
        "$COMPUTE_NODE"
    sleep 1
    if lsof -ti:"${LOCAL_PORT}" > /dev/null 2>&1; then
        echo "Tunnel open: localhost:${LOCAL_PORT} -> ${COMPUTE_NODE}:${REMOTE_PORT} via ${LOGIN_HOST}"
    else
        echo "Failed to open SSH tunnel."
        return 1
    fi
}

close_tunnel() {
    local pid
    pid=$(lsof -ti:"${LOCAL_PORT}" 2>/dev/null) || true
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        echo "Tunnel closed."
    fi
}

copy_files() {
    local remote_dir="/tmp/alchemi-playbook-part2"
    echo "Copying files to ${COMPUTE_NODE}:${remote_dir}..."
    COPYFILE_DISABLE=1 tar -C "$REPO_DIR" \
        --no-xattrs \
        --exclude='.git' \
        --exclude='.claude' \
        --exclude='outputs' \
        --exclude='logs' \
        --exclude='__pycache__' \
        --exclude='.ipynb_checkpoints' \
        --exclude='.DS_Store' \
        --exclude='scripts' \
        -cf - . \
        | ssh -J "$LOGIN_HOST" "$COMPUTE_NODE" \
            "mkdir -p ${remote_dir} && tar -C ${remote_dir} -xf -"
}

cmd_setup() {
    [ -z "${1:-}" ] || [ -z "${2:-}" ] && usage
    LOGIN_HOST="$1"
    COMPUTE_NODE="$2"

    copy_files

    echo "Copying docker-dev.sh to ${COMPUTE_NODE}..."
    scp -o ProxyJump="$LOGIN_HOST" "$SCRIPT_DIR/docker-dev.sh" "${COMPUTE_NODE}:${REMOTE_SCRIPT}"

    echo "Building and starting container on ${COMPUTE_NODE}..."
    local output
    output=$(remote_exec "PORT=${REMOTE_PORT} bash ${REMOTE_SCRIPT}")
    echo "$output"
    update_port_from_output "$output"

    save_state

    echo "Opening SSH tunnel..."
    open_tunnel

    echo ""
    echo "Setup complete. Access JupyterLab at http://localhost:${LOCAL_PORT}"
}

cmd_restart() {
    load_state
    echo "Syncing local changes to ${COMPUTE_NODE}..."
    copy_files
    echo "Rebuilding container on ${COMPUTE_NODE}..."
    local output old_port
    old_port="$REMOTE_PORT"
    output=$(remote_exec "PORT=${REMOTE_PORT} bash ${REMOTE_SCRIPT} rebuild")
    echo "$output"
    update_port_from_output "$output"

    if [ "$REMOTE_PORT" != "$old_port" ]; then
        save_state
        echo "Reopening SSH tunnel for new port..."
        open_tunnel
    fi
}

cmd_status() {
    load_state
    remote_exec "bash ${REMOTE_SCRIPT} status"

    echo ""
    if lsof -ti:"${LOCAL_PORT}" > /dev/null 2>&1; then
        echo "Tunnel: active (localhost:${LOCAL_PORT})"
    else
        echo "Tunnel: inactive"
    fi
}

cmd_pull_changes() {
    load_state
    echo "Pulling changes from remote..."
    ssh -J "$LOGIN_HOST" "$COMPUTE_NODE" \
        "cd /tmp/alchemi-playbook-part2 && tar -cf - *.ipynb" \
        | tar -C "$REPO_DIR" -xf -
    echo "Remote notebook changes applied locally. Review with 'git diff'."
}

cmd_stop() {
    load_state
    close_tunnel
    echo "Stopping container on ${COMPUTE_NODE}..."
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
