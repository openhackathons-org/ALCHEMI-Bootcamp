#!/usr/bin/env bash
set -euo pipefail

TUTORIAL_DIR="/tmp/alchemi-playbook-part2"
IMAGE="alchemi-playbook-part2"
CONTAINER="alchemi-playbook-part2"
PORT="${PORT:-8890}"

cmd_start() {
    if [ ! -d "$TUTORIAL_DIR" ]; then
        echo "Error: Tutorial files not found at $TUTORIAL_DIR."
        echo "Run: deploy.sh setup <login-host> <compute-node>"
        exit 1
    fi

    docker build -t "$IMAGE" "$TUTORIAL_DIR"
    docker rm -f "$CONTAINER" 2>/dev/null || true
    docker run -d --name "$CONTAINER" \
        --gpus all \
        --network host \
        -v "$TUTORIAL_DIR":/workspace \
        "$IMAGE" \
        jupyter lab --ip=0.0.0.0 --port="$PORT" --no-browser --allow-root

    echo "Container started."
    sleep 3
    cmd_status
}

cmd_rebuild() {
    docker rm -f "$CONTAINER" 2>/dev/null || true
    docker build -t "$IMAGE" "$TUTORIAL_DIR"
    docker run -d --name "$CONTAINER" \
        --gpus all \
        --network host \
        -v "$TUTORIAL_DIR":/workspace \
        "$IMAGE" \
        jupyter lab --ip=0.0.0.0 --port="$PORT" --no-browser --allow-root

    echo "Container rebuilt and started."
    sleep 3
    cmd_status
}

cmd_restart() {
    docker restart "$CONTAINER"
    echo "Container restarted."
    sleep 3
    cmd_status
}

cmd_status() {
    echo "=== Container Status ==="
    docker ps --filter "name=$CONTAINER" --format "table {{.Names}}\t{{.Status}}"

    echo ""
    echo "=== Jupyter URL ==="
    docker logs "$CONTAINER" 2>&1 \
        | grep -oE 'http://127\.0\.0\.1:[0-9]+/lab\?token=[a-z0-9]+' \
        | tail -1 || echo "Jupyter not ready yet."
}

cmd_stop() {
    docker rm -f "$CONTAINER" 2>/dev/null || true
    echo "Container stopped and removed."
}

case "${1:-}" in
    restart) cmd_restart ;;
    rebuild) cmd_rebuild ;;
    status)  cmd_status ;;
    stop)    cmd_stop ;;
    *)       cmd_start ;;
esac
