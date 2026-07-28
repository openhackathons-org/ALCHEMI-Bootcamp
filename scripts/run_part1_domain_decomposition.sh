#!/usr/bin/env bash
set -euo pipefail

# Do not write import bytecode into the staged tutorial or Toolkit checkouts.
export PYTHONDONTWRITEBYTECODE=1

# Start one fresh torchrun process for a Part 1 domain-decomposition case.
#
# Slurm mode expects one task and one GPU per node. Local mode expects
# ALCHEMI_DOMAIN_GPUS to be the number of visible GPUs on one machine.

route_value() {
  local name="$1"
  awk -v name="$name" '
    {
      for (field = 1; field <= NF; field += 1) {
        if ($field == name && field < NF) {
          print $(field + 1)
          exit
        }
      }
    }
  '
}

interface_owning_address() {
  local address="$1"
  ip -4 -o addr show scope global \
    | awk -v address="$address" '
        {
          split($4, candidate, "/")
          if (candidate[1] == address) {
            print $2
            exit
          }
        }
      '
}

interface_global_address() {
  local interface="$1"
  ip -4 -o addr show dev "$interface" scope global \
    | awk '
        {
          split($4, candidate, "/")
          address = candidate[1]
          count += 1
        }
        END {
          if (count != 1) {
            exit 1
          }
          print address
        }
      '
}

require_interface_address() {
  local interface="$1"
  local address="$2"
  ip -4 -o addr show dev "$interface" scope global \
    | awk -v address="$address" '
        {
          split($4, candidate, "/")
          if (candidate[1] == address) {
            found = 1
          }
        }
        END { exit !found }
      '
}

resolve_master_address() {
  local route
  local interface
  local address

  if [[ -n "${ALCHEMI_DISTRIBUTED_IFACE:-}" ]]; then
    interface="$ALCHEMI_DISTRIBUTED_IFACE"
    ip link show dev "$interface" >/dev/null
    if ! address="$(interface_global_address "$interface")"; then
      echo "Interface $interface must own exactly one global IPv4 address" >&2
      return 2
    fi
  else
    route="$(ip -4 route get 1.1.1.1)"
    interface="$(printf '%s\n' "$route" | route_value dev)"
    address="$(printf '%s\n' "$route" | route_value src)"
  fi

  if [[ -z "$interface" || -z "$address" || "$interface" == "lo" ]]; then
    echo "Could not resolve a routed global IPv4 master endpoint" >&2
    return 2
  fi
  require_interface_address "$interface" "$address"
  printf '%s\n' "$address"
}

resolve_local_network() {
  : "${ALCHEMI_MASTER_ADDR:?set ALCHEMI_MASTER_ADDR on the allocation}"

  local route
  local interface
  local address
  local selection
  local owner

  owner="$(interface_owning_address "$ALCHEMI_MASTER_ADDR")"
  if [[ -n "${ALCHEMI_DISTRIBUTED_IFACE:-}" ]]; then
    interface="$ALCHEMI_DISTRIBUTED_IFACE"
    selection="explicit"
    ip link show dev "$interface" >/dev/null
    if [[ -n "$owner" ]]; then
      if [[ "$owner" != "$interface" ]]; then
        echo "Interface $interface does not own $ALCHEMI_MASTER_ADDR" >&2
        return 2
      fi
      address="$ALCHEMI_MASTER_ADDR"
    else
      route="$(ip -4 route get "$ALCHEMI_MASTER_ADDR" oif "$interface")"
      if [[ "$(printf '%s\n' "$route" | route_value dev)" != "$interface" ]]; then
        echo "Interface $interface cannot route to $ALCHEMI_MASTER_ADDR" >&2
        return 2
      fi
      address="$(printf '%s\n' "$route" | route_value src)"
    fi
  else
    selection="automatic"
    if [[ -n "$owner" && "$owner" != "lo" ]]; then
      interface="$owner"
      address="$ALCHEMI_MASTER_ADDR"
    else
      route="$(ip -4 route get "$ALCHEMI_MASTER_ADDR")"
      interface="$(printf '%s\n' "$route" | route_value dev)"
      address="$(printf '%s\n' "$route" | route_value src)"
    fi
  fi

  if [[ -z "$interface" || -z "$address" || "$interface" == "lo" ]]; then
    echo "Could not resolve a routed interface to $ALCHEMI_MASTER_ADDR" >&2
    return 2
  fi
  require_interface_address "$interface" "$address"

  export ALCHEMI_DISTRIBUTED_IFACE="$interface"
  export NCCL_SOCKET_IFNAME="=$interface"
  export GLOO_SOCKET_IFNAME="$interface"
  export ALCHEMI_DISTRIBUTED_ADDRESS="$address"
  export ALCHEMI_DISTRIBUTED_SELECTION="$selection"
}

if [[ "${1:-}" == "--print-master-address" ]]; then
  resolve_master_address
  exit
fi

if [[ "${1:-}" == "--print-network-record" ]]; then
  resolve_local_network
  printf "%s address=%s interface=%s nccl=%s gloo=%s selection=%s\n" \
    "$(hostname)" "$ALCHEMI_DISTRIBUTED_ADDRESS" \
    "$ALCHEMI_DISTRIBUTED_IFACE" "$NCCL_SOCKET_IFNAME" \
    "$GLOO_SOCKET_IFNAME" "$ALCHEMI_DISTRIBUTED_SELECTION"
  exit
fi

: "${ALCHEMI_MAIN_ENV:?set ALCHEMI_MAIN_ENV to the verified Conda base}"
: "${ALCHEMI_PYTHON_OVERLAY:?set ALCHEMI_PYTHON_OVERLAY to the verified Python package layer}"
: "${ALCHEMI_TOOLKIT_CORE_ROOT:?set ALCHEMI_TOOLKIT_CORE_ROOT to the exact Toolkit Core checkout}"
: "${ALCHEMI_TOOLKIT_OPS_ROOT:?set ALCHEMI_TOOLKIT_OPS_ROOT to the exact Toolkit-Ops checkout}"

export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
CORE_ROOT="$(realpath "$ALCHEMI_TOOLKIT_CORE_ROOT")"
OPS_ROOT="$(realpath "$ALCHEMI_TOOLKIT_OPS_ROOT")"
test -d "$CORE_ROOT"
test -d "$OPS_ROOT"
export ALCHEMI_TOOLKIT_CORE_ROOT="$CORE_ROOT"
export ALCHEMI_TOOLKIT_OPS_ROOT="$OPS_ROOT"
export PYTHONPATH="$CORE_ROOT:$OPS_ROOT${PYTHONPATH:+:$PYTHONPATH}"

READY_FILE="$ALCHEMI_PYTHON_OVERLAY/.part1-ready.json"
TORCHRUN="$ALCHEMI_PYTHON_OVERLAY/bin/torchrun"
test -s "$READY_FILE"
test -x "$TORCHRUN"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  : "${SLURM_NNODES:?Slurm did not provide SLURM_NNODES}"
  : "${SLURM_NODEID:?Slurm did not provide SLURM_NODEID}"
  : "${ALCHEMI_MASTER_ADDR:?set ALCHEMI_MASTER_ADDR on the allocation}"
  : "${ALCHEMI_MASTER_PORT:?set ALCHEMI_MASTER_PORT on the allocation}"

  resolve_local_network
  exec "$TORCHRUN" \
    --nnodes "$SLURM_NNODES" \
    --nproc-per-node 1 \
    --node-rank "$SLURM_NODEID" \
    --master-addr "$ALCHEMI_MASTER_ADDR" \
    --master-port "$ALCHEMI_MASTER_PORT" \
    "$@"
fi

: "${ALCHEMI_DOMAIN_GPUS:?set ALCHEMI_DOMAIN_GPUS for the local launch}"

exec "$TORCHRUN" \
  --standalone \
  --nnodes 1 \
  --nproc-per-node "$ALCHEMI_DOMAIN_GPUS" \
  "$@"
