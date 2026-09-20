#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.8-Flash-Next SFT via FSDPTurbo engine (torchrun direct launch).
#
# Launches FSDPTurbo's BaseTrainer directly via torchrun, bypassing Relax's
# Ray Serve orchestration. FSDPTurbo is used as a pip-installable library
# (pip install -e /path/to/FSDPTurbo).
#
# Default cluster: 4 nodes × 16 cards = 64 (Ascend910 NPU).
#
# Usage:
#   bash scripts/training/sft/run-qwen3.8-flash-next-fsdpturbo-sft-64gpu.sh
#
# Environment overrides:
#   NNODES=4               # number of nodes (default 4)
#   GPUS_PER_NODE=16       # cards per node (default 16)
#   MASTER_ADDR=...        # rank-0 node IP (multi-node required)
#   MASTER_PORT=29500      # torchrun rendezvous port
#   NUM_HIDDEN_LAYERS=2    # language layers (default 2 for smoke; "full" = all 48)
#   DATA_PATH=...          # override dataset_path in config.yaml
#   CONFIG_FILE=...        # override config.yaml path
#   FSDPTURBO_DIR=...      # FSDPTurbo repo root
#   WORK_DIR=...           # output & logs directory
#
# Smoke test (single node, 8 cards, 2 layers):
#   NNODES=1 GPUS_PER_NODE=8 NUM_HIDDEN_LAYERS=2 \
#     bash scripts/training/sft/run-qwen3.8-flash-next-fsdpturbo-sft-64gpu.sh

set -e

# --- NPU environment (auto-detected; skipped on CUDA) ---
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export HCCL_NPU_SOCKET_PORT_RANGE="22000,22999"
    export ASCEND_LAUNCH_BLOCKING=1
    export ACLNN_CACHE_LIMIT=100000
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_CONNECT_TIMEOUT=7200
fi

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
FSDPTURBO_DIR=${FSDPTURBO_DIR:-/path/to/FSDPTurbo}
TRAIN_PY="${FSDPTURBO_DIR}/examples/qwen3_8/train.py"
CONFIG_FILE=${CONFIG_FILE:-"${FSDPTURBO_DIR}/examples/qwen3_8/config.yaml"}

# --- Cluster Configuration (4 nodes × 16 cards = 64) ---
NNODES=${NNODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-16}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

# --- Layer reduction (smoke test default = 2 layers; "full" = all 48) ---
NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS:-2}

# --- Working directory (output & logs relative to here) ---
WORK_DIR=${WORK_DIR:-"${RELAX_ROOT}/exps/fsdp_turbo/qwen3_8"}
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# --- Data path override (creates a temporary config copy) ---
if [[ -n "${DATA_PATH:-}" ]]; then
    TMP_CONFIG="${WORK_DIR}/config_override.yaml"
    sed "s|^\([[:space:]]*dataset_path:\).*|\1 \"${DATA_PATH}\"|" "$CONFIG_FILE" > "$TMP_CONFIG"
    CONFIG_FILE="$TMP_CONFIG"
    echo "  [override] dataset_path -> $DATA_PATH"
    echo "  [override] config      -> $TMP_CONFIG"
fi

# --- Build extra args ---
EXTRA_ARGS=()
if [[ -n "$NUM_HIDDEN_LAYERS" && "$NUM_HIDDEN_LAYERS" != "full" ]]; then
    EXTRA_ARGS+=(--num-hidden-layers "$NUM_HIDDEN_LAYERS")
    LAYERS_DISPLAY="$NUM_HIDDEN_LAYERS"
else
    LAYERS_DISPLAY="full (48)"
fi
if [[ -n "${ULYSSES_SIZE:-}" ]]; then
    EXTRA_ARGS+=(--ulysses-size "$ULYSSES_SIZE")
fi

echo "============================================================"
echo "Qwen3.8-Flash-Next SFT via FSDPTurbo (torchrun)"
echo "  FSDPTurbo:  $FSDPTURBO_DIR"
echo "  train.py:   $TRAIN_PY"
echo "  config:     $CONFIG_FILE"
echo "  work_dir:   $WORK_DIR"
echo "  NNODES:     $NNODES"
echo "  GPUS:       $GPUS_PER_NODE"
echo "  layers:     $LAYERS_DISPLAY"
echo "============================================================"

LOG_DIR="${WORK_DIR}/logs"
LOG_FILE="qwen38_fsdp_$(date +%Y%m%d)_$(date +%H%M%S)"
mkdir -p "$LOG_DIR"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    "$TRAIN_PY" \
    --config "$CONFIG_FILE" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/${LOG_FILE}.log"
