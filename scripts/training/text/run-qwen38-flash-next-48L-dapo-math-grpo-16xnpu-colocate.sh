#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.8-Flash-Next 48L (qwen4_exp) DAPO Math GRPO smoke/RL run on 16 NPUs,
# single-node colocate. The rollout engines and the Megatron actor share the
# same 16 dies.
#
# Topology:
#   dense  : TP 8 x PP 1 x CP 1 x DP 2 = 16
#   expert : ET 1 x EP 16 x PP 1 = 16, EDP = 1
#
# Usage:
#   export MODEL_DIR=<dir holding Qwen3.8-Flash-Next-48L-ple-1M-128E> \
#          DATA_DIR=<dir holding dapo-math-17k.jsonl and aime-2024/> \
#          SAVE_DIR=<persistent checkpoint root>
#   bash scripts/training/text/run-qwen38-flash-next-48L-dapo-math-grpo-16xnpu-colocate.sh
#
# Optional:
#   NPU_DEVICES       visible die list, default 0..15
#   SGLANG_SOURCE     SGLang source tree to put first on PYTHONPATH; leave unset
#                     in the built image, set to a checkout for local testing
#   SAVE_CHECKPOINT=0 skip checkpoint writes for a smoke run
#   NUM_ROLLOUT=2     keep the test short

set -ex
set -o pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

ulimit -n 65535

now=$(date "+%Y-%m-%d_%H:%M:%S")
echo "当前时间: $now"

export ASCEND_COREDUMP_SIGNAL=none
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-63000-63150}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-64000-64150}"
export TMS_HOOK_MODE="preload"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF-expandable_segments:True}"

# Qwen4Exp GDN backend on the NPU stack. The AscendC backend still routes
# through the Qwen3.5-VL flash_gated_delta_rule plugin, which needs torch.ops
# kernels this image does not register; triton is the backend used by the
# validated 48L SFT recipe.
export MINDSPEED_BRIDGE_GDN_USE_TORCH_CONV=1
export MINDSPEED_BRIDGE_GDN_BACKEND="${MINDSPEED_BRIDGE_GDN_BACKEND:-triton}"

# PLE must stay on the owning device ranks in this CP=1 colocate recipe.
export RELAX_QWEN38_PLE_STREAMED=0
export QWEN38_PLE_CPU_OFFLOAD=0
# The QSA indexer is frozen in this RL recipe.
export RELAX_QSA_INDEXER_LOSS_COEFF=0

# In a built image SGLang is installed at the pinned eef42710 revision. For a
# local source checkout, place it ahead of the editable install.
if [ -n "${SGLANG_SOURCE:-}" ]; then
    export PYTHONPATH="${SGLANG_SOURCE}:${PYTHONPATH:-}"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source the single-node NPU entrypoint when not started by an external one.
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

PROJECT_NAME="${PROJECT_NAME:-Relax/rl/dapo-math}"
EXP_NAME="${EXP_NAME:-qwen38-flash-next-48L-dapo-math-grpo-npu16}"
CKPT_NAME="${CKPT_NAME:-Qwen3.8-Flash-Next-48L-ple-1M-128E}"

: "${MODEL_DIR:?export MODEL_DIR=<directory holding Qwen3.8-Flash-Next-48L-ple-1M-128E>}"
: "${DATA_DIR:?export DATA_DIR=<directory holding dapo-math-17k.jsonl and aime-2024/>}"
: "${SAVE_DIR:?export SAVE_DIR=<persistent checkpoint root>}"

DAPO_DATA="${DAPO_DATA:-${DATA_DIR}/dapo-math-17k.jsonl}"
AIME_DATA="${AIME_DATA:-${DATA_DIR}/aime-2024/aime-2024.jsonl}"

NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/${CKPT_NAME}"
    --ref-load "${MODEL_DIR}/${CKPT_NAME}"
    --megatron-to-hf-mode bridge
)
# Keep the smoke-test path free of the final distributed checkpoint barrier.
if [ "${SAVE_CHECKPOINT:-1}" != "0" ]; then
    CKPT_ARGS+=(
        --save "${SAVE_DIR}/${EXP_NAME}"
        --save-interval "${SAVE_INTERVAL:-50}"
        --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP:-1}"
    )
fi
if [ -n "${LOAD_FROM:-}" ]; then
    CKPT_ARGS+=(
        --load "${LOAD_FROM}"
    )
fi

ROLLOUT_ARGS=(
    --prompt-data "${DAPO_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking":true}'
    --rollout-shuffle
    --rm-type dapo
    --reward-key score
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --rollout-max-prompt-len 2048
    --rollout-max-response-len 2048
    --rollout-max-context-len 4096
    --rollout-temperature 1.0
    --balance-data
)

EVAL_ARGS=(
    --log-passrate
    --skip-eval-before-train
    --eval-interval 20
    --eval-prompt-data aime "${AIME_DATA}"
    --eval-reward-key acc
    --n-samples-per-eval-prompt 4
    --eval-max-prompt-len 2048
    --eval-max-response-len 4096
    --eval-max-context-len 6144
    --eval-temperature 1.0
    --eval-top-p 0.7
)

PERF_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE:-8}"
    --sequence-parallel
    --pipeline-model-parallel-size "${PP_SIZE:-1}"
    --context-parallel-size "${CP_SIZE:-1}"
    --expert-model-parallel-size "${EP_SIZE:-16}"
    --expert-tensor-parallel-size 1
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --use-dynamic-batch-size
    --micro-batch-size 1
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
    --data-pad-size-multiplier "${DATA_PAD_SIZE_MULTIPLIER:-1024}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --no-rope-fusion
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --calculate-per-token-loss
    --use-tis
    --eps-clip 0.2
    --eps-clip-high 0.28
    --entropy-coef 0.0
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
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
if [ "${OVERLAP_COMM:-1}" != "0" ]; then
    OPTIMIZER_ARGS+=(
        --overlap-grad-reduce
        --overlap-param-gather
    )
fi

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
    --sglang-device npu
    --sglang-attention-backend ascend
    --sglang-pp-size 1
    --sglang-dp-size 1
    --sglang-ep-size 1
    --sglang-enable-dp-attention
    --sglang-enable-dp-lm-head
    --sglang-disable-radix-cache
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.45}"
    --sglang-max-running-requests 64
    --sglang-chunked-prefill-size 4096
    --sglang-max-prefill-tokens 4096
    --sglang-max-mamba-cache-size 128
    --sglang-mamba-ssm-dtype bfloat16
    --sglang-mm-feature-transport cpu
    --sglang-router-policy round_robin
)

WANDB_ARGS=(
    --use-clearml
    --use-metrics-service
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
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0.0
    --no-check-for-nan-in-loss-and-grad
)
if [ "${FP32_GRADS:-0}" != "0" ]; then
    MISC_ARGS+=(
        --accumulate-allreduce-grads-in-fp32
    )
else
    MISC_ARGS+=(
        --main-grads-dtype bf16
    )
fi

TRAINABLE_ARGS=(
    --freeze-params-name-list visual ple. indexer
)

mkdir -p "${SCRIPT_DIR}/log"

# Propagate the Qwen4/NPU knobs to Ray workers. They are intentionally not
# inherited from the submitting shell when an external Ray cluster is used.
export RUNTIME_ENV_JSON
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON") or "{}")
env = runtime.setdefault("env_vars", {})
for name, value in os.environ.items():
    if name.startswith(("CLEARML_", "HCCL_", "RELAX_QWEN38_", "RELAX_QSA_", "MINDSPEED_BRIDGE_", "SGLANG_", "TMS_")) or name in (
        "PYTHONPATH",
        "SGLANG_SOURCE",
        "QWEN38_PLE_CPU_OFFLOAD",
        "MASTER_ADDR",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "PYTORCH_NPU_ALLOC_CONF",
    ):
        env[name] = value
env["RAY_OVERRIDE_JOB_RUNTIME_ENV"] = "1"
env["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"
print(json.dumps(runtime))
PY
)

NPU_COUNT="$(awk -F, '{print NF}' <<< "${ASCEND_RT_VISIBLE_DEVICES}")"

export RAY_NO_WAIT="${RAY_NO_WAIT-}"

ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="http://${MASTER_ADDR:-127.0.0.1}:${RAY_DASHBOARD_PORT:-8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource "{\"actor\": [1, ${NPU_COUNT}], \"rollout\": [1, ${NPU_COUNT}]}" \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${TRAINABLE_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
