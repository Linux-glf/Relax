#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.8-Flash-Next (qwen4_exp) 48-layer VL MTP SFT on pokemon-gpt4o-captions,
# 8xNPU single node (first 8 dies), ray-submit launch.
#
# Model: Qwen3.8-Flash-Next-48L-ple-1M-128E. The 48L model config declares the
# published 512-expert shape; this checkpoint carries 128 experts, so the
# script overrides --num-experts and enables --moe-grouped-gemm below.
#
# Topology on 8 dies:
#   dense  : TP 8 x PP 1 x CP 1 x DP 1 = 8
#   expert : ET 1 x EP 8 x PP 1 = 8, EDP = 1
# EP8 keeps 16 local experts per rank, well under the aclnnGroupedMatmulV5
# 128-weight ceiling. The default MegaMoE path uses FP32 main gradients.
# Adam optimizer state is offloaded to the host by default.
#
# Usage:
#   export MODEL_DIR=<dir holding Qwen3.8-Flash-Next-48L-ple-1M-128E> \
#          DATA_DIR=<dir holding sft/data/...> \
#          SAVE_DIR=<persistent checkpoint root>
#   bash scripts/training/sft/run-qwen38-flash-next-48L-pokemon-mtp-8xnpu.sh
#
# The default visible set is the first 8 dies on the node. Override NPU_DEVICES
# when a different 8-die slice is intended.

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d_%H:%M:%S")
echo "当前时间: $now"

# ── environment that has to be in place *before* the entrypoint starts Ray ──
export ASCEND_COREDUMP_SIGNAL=none
# 8 dies: world size = TP * PP * CP * EP * DP = 8 * 1 * 1 * 8 * 1.
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-63000-63150}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-64000-64150}"
export TMS_HOOK_MODE="preload"
export HYDRA_FULL_ERROR=1

# torch_npu caching allocator: one growable segment avoids the allocator
# fragmentation measured on the 8L recipes.
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF-expandable_segments:True}"

# QSA indexer auxiliary loss; 0 keeps the released frozen selector.
export RELAX_QSA_INDEXER_LOSS_COEFF="${RELAX_QSA_INDEXER_LOSS_COEFF:-0.001}"
export RELAX_QSA_INDEXER_LOSS_TOKENS="${RELAX_QSA_INDEXER_LOSS_TOKENS:-512}"

# Background polling for the one-step SFT prefetch lookahead.
export RELAX_SFT_PREFETCH_RETRY="${RELAX_SFT_PREFETCH_RETRY:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source the local entrypoint when not launched via an external one.
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-npu.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen38-flash-next-48L.sh"

# The released checkpoint is the 128-expert variant; the shared 48L config
# declares the published 512-expert shape.
MODEL_ARGS+=(
    --num-experts 128
)
MODEL_ARGS+=(
    --moe-grouped-gemm
)
MODEL_ARGS+=(
    --moe-token-dispatcher-type alltoall
)

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/pokemon}"
EXP_NAME="${EXP_NAME:-qwen38-flash-next-48L-mtp-sft-pokemon-npu8}"
CKPT_NAME="${CKPT_NAME:-Qwen3.8-Flash-Next-48L-ple-1M-128E}"

: "${MODEL_DIR:?export MODEL_DIR=<directory holding Qwen3.8-Flash-Next-48L-ple-1M-128E>}"
: "${DATA_DIR:?export DATA_DIR=<directory holding sft/data/...>}"
: "${SAVE_DIR:?export SAVE_DIR=<persistent checkpoint root>}"

POKEMON_DATA_DIR="${POKEMON_DATA_DIR:-${DATA_DIR}/sft/data/pokemon-gpt4o-captions}"
TRAIN_FILES=()
if [ -n "${TRAIN_PARQUET:-}" ]; then
    TRAIN_FILES+=("'${TRAIN_PARQUET}'")
else
    for lang in ${TRAIN_LANGS:-en zh}; do
        TRAIN_FILES+=("'${POKEMON_DATA_DIR}/pokemon_gpt4o_${lang}.parquet'")
    done
fi
PROMPT_DATA="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/${CKPT_NAME}"
    --ref-load "${MODEL_DIR}/${CKPT_NAME}"
    --megatron-to-hf-mode bridge
)
# SAVE_CHECKPOINT=0 is the smoke-test path: it avoids the final distributed
# checkpoint write (and its HCCL barrier) while keeping per-step loss logs.
if [ "${SAVE_CHECKPOINT:-1}" != "0" ]; then
    CKPT_ARGS+=(
        --save "${SAVE_DIR}/${EXP_NAME}"
        --save-interval "${SAVE_INTERVAL:-100}"
        --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP:-1}"
        --no-save-optim
    )
fi
CKPT_ARGS+=(
    --num-rollout "${NUM_ROLLOUT:-2}"
)
if [ -n "${LOAD_FROM:-}" ]; then
    CKPT_ARGS+=(
        --load "${LOAD_FROM}"
        --no-load-optim
    )
fi
if [ "${SAVE_HF:-0}" != "0" ]; then
    CKPT_ARGS+=(
        --save-hf "${SAVE_DIR}/${EXP_NAME}-hf"
    )
fi

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${PROMPT_DATA}"
    --input-key conversations
    --multimodal-keys '{"image":"images"}'
    --conversation-key-map '{"from":"role","value":"content","human":"user","gpt":"assistant"}'
    --image-max-token-num "${IMAGE_MAX_TOKEN_NUM:-128}"
    --sft-invalid-multimodal-strategy skip
    --sft-oversize-strategy "${SFT_OVERSIZE_STRATEGY:-keep}"
    --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-16384}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-8}"
    --micro-batch-size 1
    --seq-length 32768
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
    --data-pad-size-multiplier "${DATA_PAD_SIZE_MULTIPLIER:-1024}"
    --calculate-per-token-loss
    --per-rank-fetch
    --sft-prefetch-num-workers 16
    --sft-train-data-prefetch
    --sft-prefetch-buffer-size "${SFT_PREFETCH_BUFFER_SIZE:-512}"
)

MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-1}"
MTP_ARGS=(
    --mtp-num-layers "${MTP_NUM_LAYERS}"
)
if [ "${MTP_NUM_LAYERS}" != "0" ]; then
    MTP_ARGS+=(
        --enable-mtp-training
        --mtp-loss-scaling-factor 0.1
        --mtp-detach-paths none
    )
fi

EVAL_ARGS=()
PREDICT_ARGS=()

if [ "${PP_SIZE:-1}" = "1" ]; then
    PPL_LAYOUT='Et\*48mL'
else
    PPL_LAYOUT='Et\*24\|t\*24mL'
fi

PERF_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE:-8}"
    --pipeline-model-parallel-size "${PP_SIZE:-1}"
    --context-parallel-size "${CP_SIZE:-1}"
    --expert-model-parallel-size "${EP_SIZE:-8}"
    --expert-tensor-parallel-size 1
    --pipeline-model-parallel-layout "${PPL_LAYOUT}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --no-store-param-remainders
    --sft-chunked-logits
    --sft-logits-chunk-size "${SFT_LOGITS_CHUNK_SIZE:-2048}"
    --no-rope-fusion
    --no-offload-train
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-5
    --lr-decay-style cosine
    --min-lr 1e-6
    --lr-warmup-fraction 0.05
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
    --clip-grad 1.0
    --use-distributed-optimizer
)
if [ "${OPTIMIZER_CPU_OFFLOAD:-1}" != "0" ]; then
    OPTIMIZER_ARGS+=(
        --optimizer-cpu-offload
        --overlap-cpu-optimizer-d2h-h2d
        --use-precision-aware-optimizer
    )
fi
if [ "${OVERLAP_COMM:-0}" != "0" ]; then
    OPTIMIZER_ARGS+=(
        --overlap-grad-reduce
        --overlap-param-gather
    )
fi
if [ "${SEQUENCE_PARALLEL:-1}" != "0" ]; then
    PERF_ARGS+=(
        --sequence-parallel
    )
fi

TRAINABLE_ARGS=(
    --freeze-params-name-list visual ple.
)

WANDB_ARGS=(
    --use-clearml
    --use-metrics-service
    --use-tensorboard
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)
if [ "${LOGGING_OFF:-0}" != "0" ]; then
    WANDB_ARGS=(
        --no-use-metrics-service
        --no-use-tensorboard
    )
fi

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --attention-backend flash
    --no-check-for-nan-in-loss-and-grad
    --use-health-check
)
FP32_GRADS="${FP32_GRADS:-1}"
if [ "${FP32_GRADS}" != "0" ]; then
    MISC_ARGS+=(
        --accumulate-allreduce-grads-in-fp32
    )
else
    MISC_ARGS+=(
        --main-grads-dtype bf16
    )
fi

PROFILE_ARGS=()
if [ -n "${PROFILE_STEPS:-}" ]; then
    : "${TB_EXPERIMENT_NAME:?PROFILE_STEPS needs TB_EXPERIMENT_NAME set to an absolute dir for the traces}"
    PROFILE_ARGS+=(
        --use-pytorch-profiler
        --profile-target train_overall
        --profile-step-start "${PROFILE_STEPS%,*}"
        --profile-step-end "${PROFILE_STEPS#*,}"
    )
    if [ "${PROFILE_WITH_STACK:-0}" != "0" ]; then
        PROFILE_ARGS+=(
            --profile-with-stack
        )
    fi
fi

mkdir -p "${SCRIPT_DIR}/log"

NPU_COUNT="$(awk -F, '{print NF}' <<< "${ASCEND_RT_VISIBLE_DEVICES}")"

export RAY_NO_WAIT="${RAY_NO_WAIT-}"

# Set the Qwen3.8 training defaults and apply launch overrides.
TRAIN_ENV_JSON="$(FP32_GRADS="${FP32_GRADS}" python3 - "${RUNTIME_ENV_JSON:-}" <<'PY'
import json
import os
import sys

job_env = json.loads(sys.argv[1] or "{}").get("env_vars") or {}
defaults = {
    "QWEN38_MEGA_MOE": "1",
    "QWEN38_HC_IMPL": "npu",
    "QWEN38_QSA_GROUP": "1",
    "QWEN38_QSA_INDEXER_IMPL": "auto",
}
train_env = {name: os.environ.get(name) or job_env.get(name) or value for name, value in defaults.items()}
if os.environ["FP32_GRADS"] == "0" and train_env["QWEN38_MEGA_MOE"].strip().lower() in {"1", "true", "yes", "on"}:
    raise SystemExit("FP32_GRADS=0 requires QWEN38_MEGA_MOE=0. Packed MegaMoE needs FP32 main gradients.")
print(json.dumps(train_env))
PY
)"

ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="http://${MASTER_ADDR:-127.0.0.1}:8265" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON:-"{}"}" \
    -- python3 -m relax.entrypoints.train \
    --train-env-vars "${TRAIN_ENV_JSON}" \
    --resource "{\"sft\": [1, 0], \"actor\": [1, ${NPU_COUNT}]}" \
    --sft-max-in-flight-steps "${SFT_MAX_IN_FLIGHT_STEPS:-2}" \
    --num-data-storage-units "${NPU_COUNT}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${TRAINABLE_ARGS[@]}" \
    "${MTP_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${PREDICT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
