#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.8-Flash-Next (qwen4_exp) 8-layer VL MTP SFT on pokemon-gpt4o-captions,
# 16xNPU single node, ray-submit launch.
#
# This is the NPU counterpart of the reference 64xH800 full-48L script, and it
# is deliberately aligned to it in every knob that is *not* the model or the
# dataset -- same batch shape (gbs 32 / mbs 1), same MTP head and scaling
# factor, same optimizer hyper-parameters, same freeze set, same recompute
# policy. What differs, and why:
#
#   tensor-model-parallel-size   8 -> 2   --num-query-groups 2 must divide TP
#   pipeline-model-parallel-size 8 -> 2   16 dies, see the layout note below
#   expert-model-parallel-size   8 -> 4   MindSpeed's TEGroupedLinear feeds
#                                         torch_npu.npu_grouped_matmul, whose
#                                         aclnnGroupedMatmulV5 takes at most 128
#                                         weights, so 512 experts need EP >= 4
#   pipeline-model-parallel-layout        rewritten for PP=2 / 8 decoders
#   moe-token-dispatcher-type    flex -> alltoall
#   moe-flex-dispatcher-backend  deepep   dropped, needs NVSHMEM/DeepEP
#   the RELAX_QWEN38_QSA_* env block      dropped, no NPU counterpart exists
#                                         (see "not ported" below)
#   the dataset knobs                     this run reads a different corpus
#   max-tokens-per-gpu           32K->16K  the loss stage retains one fp32
#   sft-logits-chunk-size        2048->512 [tokens, V/TP] softmax per head
#                                         until that micro-batch's backward;
#                                         ~15 GiB/head at the 32K cap, which is
#                                         what OOM'd the dqa run (see SFT_ARGS)
#
# Usage:
#   export MODEL_DIR=<dir holding Qwen3.8-Flash-Next-8L-ple> \
#          DATA_DIR=<dir holding sft/data/...> EXP_DIR=<dir for checkpoints>
#   bash scripts/training/sft/run-qwen38-flash-next-8L-pokemon-mtp-16xnpu.sh
#
# CKPT_NAME selects the truncation, exactly as the 4L script does:
#   CKPT_NAME=Qwen3.8-Flash-Next-8L      layers only          (PLE off)
#   CKPT_NAME=Qwen3.8-Flash-Next-8L-ple  layers + n-gram table (PLE on)
#
# WARNING -- this script owns the box. entrypoint/local-npu.sh begins with
# `pkill -9 sglang` / `ray stop --force` / `pkill -9 ray` / `pkill -9 python`,
# so sourcing it (which the auto-source block below does whenever
# RELAX_ENTRYPOINT_MODE is unset) terminates every other Relax job on this
# node and restarts Ray from scratch. Under an external entrypoint
# (RELAX_ENTRYPOINT_MODE set by spmd-multinode.sh / ray-job-npu.sh) the
# auto-source is skipped and the run joins the existing cluster instead.

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d_%H:%M:%S")
echo "当前时间: $now"

# ── environment that has to be in place *before* the entrypoint starts Ray ──
# local-npu.sh sets neither of these, and its Ray head inherits whatever is
# exported here, so they must come before the source below.
export ASCEND_COREDUMP_SIGNAL=none
# 16 dies: world size = TP * PP * CP * EP * DP = 2 * 2 * 1 * 4 * 1.
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-63000-63150}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-64000-64150}"
export TMS_HOOK_MODE="preload"
export HYDRA_FULL_ERROR=1

# torch_npu's caching allocator, configured before torch is imported (hence
# here, not in the arg list). Nothing in the launch chain set it, so the
# allocator ran on defaults.
#
# Why it matters: the profiler shows 31,652 allocations per step per rank
# (`empty_tensor` 21,623 + `aten::empty_strided` 5,269 + `aten::empty` 3,554 +
# ...), costing 1.60 s/step -- 26% of the whole 6.14 s/step host budget -- and
# `empty_tensor` reports Device Self Duration = 0 everywhere, i.e. it is pure
# allocator host work with no kernel behind it. The average is 71 us per
# allocation where a warm-cache at::empty should be 2-5 us, and the worst
# single one (the wgrad buffer at tensor_parallel/layers.py:847) measured
# 8.2 ms. That signature -- cheap metadata ops fast, every allocation slow --
# is a block that cannot be reused while the device queue is deep, so each
# empty() falls through to a fresh physical allocation.
#
# `expandable_segments:True` is the allocator's answer to exactly that: one
# growable segment instead of a fixed pool of split blocks, so a free does not
# strand a right-sized hole. It is supported on NPU (torch_npu/npu/memory.py
# and NPUAllocatorConfig.h both carry it; `set_per_process_memory_fraction`'s
# docstring: "Only support expandable_segments:True or False").
#
# `${VAR-default}` not `${VAR:-default}` on purpose: an explicitly empty
# PYTORCH_NPU_ALLOC_CONF must disable this, not silently take the default.
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF-expandable_segments:True}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-npu.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen38-flash-next-8L.sh"

# The n-gram table's switches (`ple.py`: QWEN4_EXP_PLE_TABLE_MODE, the DCP gate
# QWEN4_EXP_PLE_TABLE_IN_DCP for the row-routed form, and the legacy offload switch
# QWEN4_EXP_PLE_CPU_OFFLOAD that has to agree with it) are read in the worker that
# builds the model, and the Ray workers do not inherit this shell. The MTP input
# mask switch (QWEN4_EXP_MTP_INPUT_MASK) is read there too: off by default, which
# keeps mtp_input_mask=None exactly as the GPU stack runs it; set it to 1 only to
# re-enable the NPU-only image/video-placeholder exclusion. They have to travel in
# RUNTIME_ENV_JSON's env_vars -- the same channel the 48L trainable recipe uses for
# QWEN4_EXP_PLE_CPU_OFFLOAD. The entrypoint above has already built that JSON; this
# merges into it rather than replacing it, so PYTHONPATH and the rest survive. With
# none of the variables set nothing is rewritten; the bridge's mask default is off.
if [ -n "${QWEN4_EXP_PLE_TABLE_MODE:-}" ] || [ -n "${QWEN4_EXP_PLE_TABLE_IN_DCP:-}" ] || [ -n "${QWEN4_EXP_PLE_CPU_OFFLOAD:-}" ] || [ -n "${QWEN4_EXP_MTP_INPUT_MASK:-}" ]; then
   RUNTIME_ENV_JSON="$(
      RELAX_RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON:-}" \
      RELAX_INJECT_TABLE_MODE="${QWEN4_EXP_PLE_TABLE_MODE:-}" \
      RELAX_INJECT_TABLE_IN_DCP="${QWEN4_EXP_PLE_TABLE_IN_DCP:-}" \
      RELAX_INJECT_OFFLOAD="${QWEN4_EXP_PLE_CPU_OFFLOAD:-}" \
      RELAX_INJECT_MTP_MASK="${QWEN4_EXP_MTP_INPUT_MASK:-}" \
      python3 - <<'PY'
import json
import os

raw = os.environ.get("RELAX_RUNTIME_ENV_JSON", "").strip()
config = json.loads(raw) if raw else {}
env_vars = config.get("env_vars", {})
table_mode = os.environ["RELAX_INJECT_TABLE_MODE"]
if table_mode:
    env_vars["QWEN4_EXP_PLE_TABLE_MODE"] = table_mode
table_in_dcp = os.environ["RELAX_INJECT_TABLE_IN_DCP"]
if table_in_dcp:
    env_vars["QWEN4_EXP_PLE_TABLE_IN_DCP"] = table_in_dcp
offload = os.environ["RELAX_INJECT_OFFLOAD"]
if offload:
    env_vars["QWEN4_EXP_PLE_CPU_OFFLOAD"] = offload
mtp_mask = os.environ["RELAX_INJECT_MTP_MASK"]
if mtp_mask:
    env_vars["QWEN4_EXP_MTP_INPUT_MASK"] = mtp_mask
config["env_vars"] = env_vars
print(json.dumps(config))
PY
   )"
   export RUNTIME_ENV_JSON
fi

PROJECT_NAME="${PROJECT_NAME:=Relax/sft/pokemon}"
EXP_NAME="${EXP_NAME:-qwen38-flash-next-8L-mtp-sft-pokemon-npu16}"
# Which truncation under test; the PLE variant adds the hashed n-gram table.
CKPT_NAME="${CKPT_NAME:-Qwen3.8-Flash-Next-8L-ple}"
# Where the checkpoint, the dataset and the runs live. Relax keeps no internal
# mount path in tracked files (.gitleaks.toml rejects them), so the three roots
# come from the environment -- export them once before running:
#   MODEL_DIR  holds Qwen3.8-Flash-Next-8L-ple (and any other checkpoint)
#   DATA_DIR   holds sft/data/...
#   EXP_DIR    receives <exp>/sft/<name> and the -hf export next to it
: "${MODEL_DIR:?export MODEL_DIR=<directory holding Qwen3.8-Flash-Next-8L-ple>}"
: "${DATA_DIR:?export DATA_DIR=<directory holding sft/data/...>}"
: "${EXP_DIR:?export EXP_DIR=<directory for checkpoints and the HF export>}"
# TRAIN_LANGS selects which language shards are trained on; it exists so the
# per-step time can be attributed to the data source (see ANALYSIS-8L.md 5.5 --
# the step time steps up exactly when the reader crosses from en into zh).
TRAIN_FILES=()
if [ -n "${TRAIN_PARQUET:-}" ]; then
    # A single parquet that replaces the pokemon shards (same schema). Used for
    # the synthetic corpus whose token profile mirrors the dqa/exp6.3 data.
    TRAIN_FILES+=("'${TRAIN_PARQUET}'")
else
    for lang in ${TRAIN_LANGS:-en zh}; do
        TRAIN_FILES+=("'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_${lang}.parquet'")
    done
fi
PROMPT_DATA="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/qwen38-flash-next-8L-sft}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/${CKPT_NAME}
   --ref-load ${MODEL_DIR}/${CKPT_NAME}
   --megatron-to-hf-mode bridge
   --save ${SAVE_DIR}/sft/${EXP_NAME}
   --save-interval 100
   --max-actor-ckpt-to-keep 1
   # NUM_EPOCH scales with gbs: the corpus is 1666 rows, so one epoch is 52
   # optimizer steps at gbs=32 but only 4 at gbs=384. Saturation runs raise it
   # to reach a steady state worth profiling; the default stays at the
   # reference's 1. NUM_ROLLOUT caps the total step count either way.
   --num-epoch ${NUM_EPOCH:-1}
   --num-rollout ${NUM_ROLLOUT:-20}
   # Model weights only: the full checkpoint (with the Adam moments) is huge for
   # this model, and the shared volume ${EXP_DIR} sits on runs at 100% -- the run
   # saves the weights and exports them to HF instead. Drop this flag to get a
   # resumable checkpoint back.
   --no-save-optim
)
# Resume from a distributed checkpoint (a save root holding
# latest_checkpointed_iteration.txt, which selects iter_<N>). It has to be a
# separate variable from EXP_NAME, not an uncommented --load above, when the
# resumed run is meant to keep the checkpoint it loads from: the save root is
# derived from EXP_NAME and the rotate step prunes it after every save.
# `--no-load-optim` pairs with `--no-save-optim`: these checkpoints carry the
# model and the RNG state but no optimizer section, and Megatron's loader
# otherwise insists on finding one. What a resume restores is therefore the
# weights -- the PLE table included when QWEN4_EXP_PLE_TABLE_IN_DCP=1 -- plus
# the iteration counter, which is what sets `start_rollout_id` and realigns the
# SFT data order (components/sft.py:806-813).
if [ -n "${LOAD_FROM:-}" ]; then
   CKPT_ARGS+=(--load "${LOAD_FROM}" --no-load-optim)
fi
# The HF export rewrites the whole 152 GB checkpoint, ~102 GB of which is the
# hashed n-gram table, so the baseline run skips it. SAVE_HF=1 turns it back on.
if [ "${SAVE_HF:-0}" != "0" ]; then
   CKPT_ARGS+=(--save-hf ${SAVE_DIR}/sft/${EXP_NAME}-hf)
fi

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   # Dataset mapping, deliberately NOT aligned to the reference (it reads a
   # `messages` JSONL and applies a chat template; this corpus is already
   # tokenised conversations in parquet).
   --input-key conversations
   --multimodal-keys '{"image":"images"}'
   --conversation-key-map '{"from":"role","value":"content","human":"user","gpt":"assistant"}'
   # -- from here down every knob is aligned to the reference script --
   # 128 is the reference value. IMPORTANT (verified 2026-10-18): this flag is
   # INERT on the SFT path. fetch_image -- the only consumer -- is called from
   # the rollout preprocessing and from ProcessorPool's resize hook, and the SFT
   # dataset builds its pool as ProcessorPool(hf_checkpoint, pool_size, trust_remote_code)
   # with multimodal_config=None (components/sft.py:355-359), which makes that
   # hook a pass-through. SFT images are therefore bounded only by the HF
   # processor's own config: shortest_edge 65536 px (64 tokens) and longest_edge
   # 16777216 px (16384 tokens) for this checkpoint. The synthetic exp6.3-profile
   # corpus sits inside that window by construction (per-image 64..14212 tokens),
   # so passing 16384 for it changes nothing -- length comes from the data.
   --image-max-token-num ${IMAGE_MAX_TOKEN_NUM:-128}
   --sft-invalid-multimodal-strategy skip
   # `keep` (the default) turns a sample longer than the microbatch capacity
   # (max_tokens_per_gpu * cp) into a single-document pack of its full length,
   # which for the exp6.3-profile corpus means packs up to 80k tokens and an
   # immediate OOM. `skip` drops those rows and is the only workable setting
   # for corpora whose tail exceeds --seq-length.
   --sft-oversize-strategy ${SFT_OVERSIZE_STRATEGY:-keep}
   # For a pure SFT run nothing in relax/engine/sft/ reads this -- it drives the
   # context_parallel_size derivation (backends/megatron/arguments.py:119) and
   # the validator at utils/arguments.py:4119-4131, and nothing else. That
   # validator requires max_tokens_per_gpu * cp >= rollout_max_context_len, so
   # this and --max-tokens-per-gpu must move together: change only one and you
   # either trip the validator or (under --dynamic-context-parallel, unused here)
   # silently change CP.
   #
   # 2026-10-16, the dqa corpus (`train_le32k_1008.jsonl`, avg 6171 / max 16356
   # tokens per sample, vs. pokemon's 9457 max): at the 32K cap the packer put
   # the two longest samples in ONE micro-batch (~30K tokens) and every
   # last-stage die OOM'd -- `[2048, V/TP]` fp32 buffers of 972 MiB against
   # 964 MiB free. The chunked SFT loss keeps one fp32 [tokens, V/TP] softmax
   # per head alive until that micro-batch's backward
   # (relax/utils/training/ppo_utils.py:583-596, ctx.save_for_backward(...)),
   # i.e. ~0.47 GiB per 1K tokens -- and --sft-logits-chunk-size does NOT bound
   # it, only the transient. Halving the cap halves it.
   #
   # 16384 is "just above the longest real sample", so nothing that packs today
   # stops packing. A longer sample still forms its own micro-batch (the sampler
   # always emits at least one sample, utils/arguments.py:4116-4118), so the
   # worst case is unchanged and every other case carries at most half the
   # tokens. Samples over the cap make the SFT dataset log an `oversize=keep`
   # warning (components/sft.py:366 binds capacity = max_tokens_per_gpu * cp) --
   # that is the data's tail speaking, not an error, and nothing is dropped.
   --rollout-max-context-len ${ROLLOUT_MAX_CONTEXT_LEN:-16384}
   --global-batch-size ${GLOBAL_BATCH_SIZE:-32}
   --micro-batch-size 1
   --seq-length 32768
   # SFT requires dynamic batching (relax/utils/arguments.py rejects
   # --loss-type sft without it), and dynamic batching is thd-only. thd is also
   # what the plugin's own GDN implements (modelling_qwen4_exp/gdn.py).
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-16384}
   # The pack is padded up to a multiple of tp_size * multiplier (8 * 4096 = 32768
   # here), and every token-proportional cost -- attention above all, which is
   # quadratic in the pack length -- is charged against the padded length. A
   # micro-batch of 48 pokemon samples carries ~14.7k real tokens, so the 32k
   # quantum more than doubles every forward. Lowering it trades a stable tensor
   # size for that gap; keep 4096 as the default for the long-sample corpora.
   --data-pad-size-multiplier ${DATA_PAD_SIZE_MULTIPLIER:-4096}
   --calculate-per-token-loss
   --per-rank-fetch
   --sft-prefetch-num-workers 16
   --sft-train-data-prefetch
   # --balance-data is NOT passed here even though the 4L recipe passed it: the
   # reference leaves it commented out, and relax/utils/arguments.py:4194-4198
   # auto-enables it for offline loss types anyway. Both scripts end up with
   # balance_data=True; the flag is inert either way.
   #
   # NOT aligned, absent from this tree: --sft-image-preprocess-on-rank, which
   # the reference uses to ship image file refs instead of pixel tensors.
)
# --sft-prefetch-buffer-size has no counterpart in the reference and is kept
# from the 4L recipe: dropping it changes only prefetch depth, not the data.
SFT_ARGS+=(--sft-prefetch-buffer-size ${SFT_PREFETCH_BUFFER_SIZE:-512})

# MTP aligned to the reference: 1 layer, scaling factor 0.1, detach path
# `none`. The CP == 1 text/bridge layout it needs is handled on the plugin
# side: `Qwen4ExpModel._prepare_mtp_loss_mask` right-rolls Relax's
# predictor-frame loss mask once, cancelling MCore's derived-label roll, so
# every depth's mask gates exactly the token it predicts. The MTP loss term is
# valid; keep MTP_NUM_LAYERS=1.
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-1}"
MTP_ARGS=(--mtp-num-layers "${MTP_NUM_LAYERS}")
if [ "${MTP_NUM_LAYERS}" != "0" ]; then
   MTP_ARGS+=(--enable-mtp-training --mtp-loss-scaling-factor 0.1 --mtp-detach-paths none)
fi

# SFT eval and the predict pass both drive a rollout service, which under
# --loss-type sft is a passive SGLang server (see components/rollout.py). This
# run has no rollout resource and SGLang cannot serve qwen4_exp yet, so the
# held-out split is only used to check the loss, not to generate.
EVAL_ARGS=(
    # --eval-size 32
    # --eval-interval 10
)

PREDICT_ARGS=(
    # --sft-predict-interval 10
    # --eval-temperature 0.0
    # --eval-max-response-len 512
    # --rollout-num-gpus-per-engine 2
    # --sglang-mem-fraction-static 0.6
)

# Pipeline layout, keyed off PP_SIZE. Two stages over 8 decoders split 4/4
# (`E+t*4 | t*4+m+L`); one stage puts all of them with the MTP layer and the
# loss head on the only rank. A single-stage layout gives num_stages=1, and
# `1 % PP == 0` holds, so the layout parser accepts it at PP=1. NUM_LAYERS must
# match the checkpoint's config.json -- the token count is validated against
# `num_layers` at build time, so a 16L checkpoint needs `Et*16mL`.
#
# The `*` and `|` are backslash-escaped on purpose -- see the note below.
if [ "${PP_SIZE:-2}" = "1" ]; then
   PPL_LAYOUT="Et\*${NUM_LAYERS:-8}mL"
else
   PPL_LAYOUT='Et\*4\|t\*4mL'
fi

PERF_ARGS=(
   # TP * PP * CP * EP = 2 * 2 * 1 * 4 = 16 dies, DP=1. EP is set by the expert
   # count, not by memory: MindSpeed runs the experts through
   # `torch_npu.npu_grouped_matmul` (mindspeed/te/pytorch/module/grouped_linear.py,
   # TEGroupedLinear.forward), which takes one weight tensor per *local* expert
   # and whose aclnnGroupedMatmulV5 kernel accepts at most 128 of them ("Length
   # of tensorList should not exceed 128"). 512 experts therefore need EP >= 4;
   # at EP=2 the run dies inside layer 0's MoE with error code 161002. TP drops
   # to 2 because --num-query-groups 2 has to divide it.
   #
   # NOTE on the shard-width law: expert optimizer state is sharded
   # world/(PP x ET) wide regardless of TP and EP, so PP is the only knob that
   # linearly inflates it. PP=2 here still fits because --optimizer-cpu-offload
   # keeps all of it on the host.
   --tensor-model-parallel-size ${TP_SIZE:-2}
   --pipeline-model-parallel-size ${PP_SIZE:-2}
   --context-parallel-size ${CP_SIZE:-1}
   --expert-model-parallel-size ${EP_SIZE:-4}
   --expert-tensor-parallel-size 1

   # Two pipeline stages over 8 decoder layers: E + t*4 | t*4 + m + L.
   # Sums to 8 decoders (== --num-layers), one embedding, one loss, and the
   # single MTP layer sits in the last stage -- never in PP rank 0, as the
   # validator requires.
   #
   # The `|` and `*` are backslash-escaped on purpose. `ray job submit` does not
   # quote: it space-joins the argv it is handed and gives the result to
   # `/bin/sh -c`, so a bare `|` becomes a pipe -- the first attempt died with
   # `/bin/sh: 1: t*4mL: not found`. dash collapses `Et\*4\|t\*4mL` back to
   # `Et*4|t*4mL`, which is what the layout parser receives (checked end to end
   # through a real job submission).
   --pipeline-model-parallel-layout "${PPL_LAYOUT}"

   # thd (the default) rather than bshd: upstream mcore's GatedDeltaNet refuses
   # packed sequences, but the plugin builds its own vendored Qwen4ExpGDN
   # (modelling_qwen4_exp/gdn.py), which implements the packed path. bshd is
   # not an option for SFT anyway -- --loss-type sft requires dynamic batching,
   # which requires thd.

   # 1 is already the maximum here, not a conservative starting point: under
   # --recompute-method uniform this is the layers-per-recompute-unit, so 1
   # checkpoints every layer individually (core/recompute.py:178-187). 8 layers
   # with num_layers 1 recomputes all 8 -- which is also why the MTP head
   # asserts on any other value (multi_token_prediction.py:1919). Do not "raise
   # this to buy memory": the block method would recompute the same 8 layers
   # while making the MTP head skip its own recompute entirely
   # (multi_token_prediction.py:1922), i.e. strictly more memory.
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1


   --no-store-param-remainders
   --no-gradient-accumulation-fusion

   --cross-entropy-loss-fusion
   --sft-chunked-logits
   # 2048 -> 512 (2026-10-16): the allocation that failed in the dqa OOM was one
   # [2048, V/TP] fp32 buffer (972 MiB, V=248320 / TP=2 -> 124160 per rank); the
   # four dies that died on 486 MiB are the same shape in bf16. This knob does
   # NOT touch the per-token retention above -- it bounds the transient (lm_head
   # output + fp32 upcast + normalised logits, ~2x the chunk in fp32), so 512
   # cuts that 4x at the cost of more, smaller launches on a sequence that is
   # chunked anyway. The MTP head shares this size
   # (megatron_patch/chunked_mtp_loss_patch.py:208).
   --sft-logits-chunk-size ${SFT_LOGITS_CHUNK_SIZE:-512}

   # The MTP head shifts per-token rotary frequencies by the prediction depth,
   # which needs the unfused rotary path.
   --no-rope-fusion
   # Load-bearing, and a silent divergence from the 4L recipe: --colocate
   # *implies* --offload-train (relax/utils/arguments.py:4324-4326 fills the
   # None default with True), which parks the actor's parameters and grads on
   # the host between rollouts. This run has no rollout service, so the actor
   # never yields the device and the offload is pure D2H/H2D traffic. The
   # reference disables it explicitly and so does this script.
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
# Keeping the Adam moments on the host costs a per-parameter host loop plus a
# D2H grads / H2D params copy every step. On the 8L truncation that loop
# measured 7.8 s of a 12.9 s step (`aten::_fused_adamw_`, 593 calls/step, all
# host self time), so it is worth being able to switch off; the reference recipe
# turns it on to save HBM for the full-depth model. The other two flags are
# meaningless without it -- `--overlap-cpu-optimizer-d2h-h2d` overlaps the very
# copies offload introduces, and `--use-precision-aware-optimizer` describes a
# host-resident state.
if [ "${OPTIMIZER_CPU_OFFLOAD:-1}" != "0" ]; then
   OPTIMIZER_ARGS+=(
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )
fi
# Communication overlap. Measured on the 8L truncation with the distributed
# optimizer on and both of these off, profiled step 8 summed over all 16 ranks:
# 63% of the device-busy time is communication -- hcom_allGather 1612 ms/rank,
# hcom_allReduce 887 ms, hcom_alltoallv 248 ms, hcom_reduceScatter 231 ms --
# while step_trace_time.csv reports Comm(Not Overlapped) 3185 ms against a
# 2979 ms total of communication kernels, i.e. essentially none of it is hidden
# behind compute. The collectives are latency-bound, not bandwidth-bound: the
# slowest single hcom_allGather in that step measured 2.8 s, which is a
# rendezvous wait for the slowest rank, not a data-movement cost.
#
# `--overlap-param-gather` all-gathers the next bucket's parameters while the
# optimizer is still working on the current one, and `--overlap-grad-reduce`
# overlaps the gradient reduce-scatter with the tail of the backward pass.
# Megatron requires the two together and requires a distributed optimizer, both
# of which hold here. `--overlap-param-gather-with-optimizer-step` is
# deliberately NOT passed: it needs interleaved pipeline parallelism and this
# recipe runs PP=1.
#
# The reasoning above was never A/B'd, and the first matched comparison came out
# the other way: summed over steps 2-11, step_time is 86.31 s with the pair ON
# and 82.15 s with it OFF. That is not a conclusion either -- 3.66 s of the
# 4.16 s difference is step 10 alone, the other nine steps differ by 0.5 s with
# five wins and five losses, and unlike the partition clear there is no mechanism
# evidence (no wait-column collapse, no log-order change). Re-run the pair before
# changing this default or trusting either number; see
# docs/draft/qwen38-flash-next-npu-optimization.md section 2.
if [ "${OVERLAP_COMM:-1}" != "0" ]; then
   OPTIMIZER_ARGS+=(
      --overlap-grad-reduce
      --overlap-param-gather
   )
fi
# Sequence parallelism. On, every row-parallel linear all-gathers its input in
# the forward and reduce-scatters the gradient in the backward; off, the forward
# is silent and only the backward's input-gradient all-reduce remains. The 8L
# profile of step 3-5 measures 60 of those gathers per step, of which 8 cost
# >=100 ms and own 96% of that operator's 27.2 s (see ANALYSIS-8L.md 2.2), so
# SEQUENCE_PARALLEL=0 is the one lever that removes collectives rather than
# reordering them. Activations are replicated across TP instead of sharded,
# which full recompute bounds; the arithmetic is unchanged either way.
if [ "${SEQUENCE_PARALLEL:-1}" != "0" ]; then
   PERF_ARGS+=(--sequence-parallel)
fi

# Freezing, aligned with the reference but NOT spelled the same way, because on
# the qwen4_exp path two of the reference's three mechanisms do nothing:
#
#   --freeze-vision-model / --freeze-vision-projection are declared (they are in
#   the Bridge `bridge_keys`) but implemented only by the glm_vl and qwen_vl
#   models. Nothing under models/qwen4_exp reads them, so passing them here
#   would be a silent no-op. The vision tower is instead named `visual.*`
#   (modelling_qwen4_exp/vl_model.py:114-123) and is frozen by the regex below,
#   which is what `freeze_vision_tower` (vision.py:1011) would have targeted.
#   Note the tower is built only on the stage that owns the embedding, so under
#   PP=2 it exists on PP rank 0 alone.
#
#   `per_layer_embedding` does not exist on this path at all: the PLE table is
#   host-mmap'd and explicitly "nothing here is a nn.Parameter"
#   (modelling_qwen4_exp/ple.py:18), so it is absent from the optimizer and the
#   checkpoint. The reference freezes it on the GPU tree, where it is a module.
#
# What remains aligned is the indexer, whose attribute path is identical
# (`self_attention.indexer`, modelling_qwen4_exp/layer.py:119).
# `--freeze-params-name-list` is nargs="*" (relax/utils/arguments.py:553-571), so it
# takes a *list* of independent re.search patterns. That is what lets this script
# dodge a real landmine: `ray job submit` does not quote -- it space-joins the argv
# and hands it to `/bin/sh -c`, so shell metacharacters inside a value are acted on
# by dash. The single-pattern form that expresses the same thing,
# `'(^|\.)visual\.|self_attention\.indexer\.'`, died with
# `/bin/sh: 1: Syntax error: "(" unexpected`; backslash-escaping is not a way out
# either, because dash eats the backslash in `\.` (verified through job submission).
# Two plain-substring patterns carry no metacharacters at all and select exactly
# the same tensors -- checked against the 8L checkpoint's own index: `visual` hits
# only model.visual.* (333 tensors) and `indexer` only *.self_attn.indexer.*
# (9 tensors), zero hits outside those components.
#
# The reference GPU script also freezes `per_layer_embedding`. On the GPU tree
# that name is the whole module, so it takes the six weights *around* the n-gram
# table with it -- `key_proj`, `value_proj`, the three grouped norms and the
# dilated depthwise conv. Those six are nn.Parameters here too (the bridge maps
# exactly them, qwen4_exp_bridge.py:315-325) and they currently train; only the
# 95 GiB table is really unfreezable, being host-mmap'd and never a Parameter
# (models/qwen4_exp/modelling_qwen4_exp/ple.py:18). FREEZE_PLE=1 completes the
# reference's freeze set; without it the baseline keeps the set this script has
# always had.
#
# Spelled as the six full weight names rather than a bare `ple` for two reasons:
# `ray job submit` space-joins the argv into `/bin/sh -c` (see above), so a
# pattern has to survive dash, and a substring that short also hits the
# `*_sampled_*` attributes of the inference side of the tree. Dots are not shell
# metacharacters, so the full names come through untouched.
TRAINABLE_ARGS=(
   --freeze-params-name-list visual indexer
)
if [ "${FREEZE_PLE:-0}" = "1" ]; then
   TRAINABLE_ARGS+=(
      ple.key_proj.weight ple.value_proj.weight
      ple.norm_key.weight ple.norm_query.weight ple.norm_conv.weight
      ple.conv1d.weight
   )
fi
# The reference GPU script freezes `per_layer_embedding`, i.e. the whole PLE
# module *including* the table.  On this tree the table is only a Parameter in
# the `device-ep` form (qwen4_exp_bridge.py:857 maps it to
# `decoder.layers.*.ple.ple_embedding.weight`); in every other mode it is a
# host-mmap'd buffer and cannot be frozen.  Freezing it removes its fp32
# master and Adam m/v from the CPU-offload optimizer -- 3.2B parameters per
# rank at EP16, i.e. 38.4 GB/rank of anonymous host state plus the matching
# pinned staging -- which is what makes a 16L checkpoint fit the host at all
# (the 8L shard fits with it trainable; 16L does not).
if [ "${FREEZE_TABLE:-0}" = "1" ]; then
   TRAINABLE_ARGS+=(ple.ple_embedding.weight)
fi

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --use-tensorboard
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name ${TB_EXPERIMENT_NAME:-${EXP_NAME}-${now}}
)

# Every logging backend sits on the actor's critical path: `log_perf_data` and
# `tracking_utils.flush_metrics` are blocking HTTP POSTs (one per rank per step
# for the flush) into a single metrics-service replica that then does its own
# TensorBoard/ClearML writes per metric. LOGGING_OFF=1 strips all of them so the
# cost can be measured; per-step loss and grad_norm still reach the log, because
# `log_rollout_data` prints them from inside `timer("train")` regardless.
if [ "${LOGGING_OFF:-0}" != "0" ]; then
   WANDB_ARGS=(--no-use-metrics-service --no-use-tensorboard)
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   --attention-backend flash
   # The argparse knob is negated: megatron/training/arguments.py registers
   # `--no-check-for-nan-in-loss-and-grad` with action="store_false", so
   # `check_for_nan_in_loss_and_grad` is **True** unless this is passed -- and
   # get_megatron_ddp_config copies it straight into DDP's
   # `check_for_nan_in_grad` (training.py:2715). That turns on
   # param_and_grad_buffer.check_grads, which per bucket computes
   # `grad_data.norm(p=2)` and then calls rerun_state_machine.validate_result
   # TWICE (torch.isnan, then torch.isinf). validate_result evaluates
   # `if rejection_func(result):` -- a __bool__ on a 0-dim tensor, i.e. a full
   # device sync -- and it does so inside start_grad_sync, *between* backward
   # and the DP gradient all-reduce, so it drains the pipeline at the worst
   # possible point. Nothing else uses the result.
   #
   # Measured on this recipe (16 dies, TP2/PP1/EP8, gbs 192): 4 calls per rank
   # per step, 98 ms each, 0.39 s/step per rank, worst single call 1.82 s.
   #
   # The loss-side half of the flag is `RerunStateMachineConfig
   # .check_for_nan_in_loss`, which only the glm_vl and kimi bridge models read
   # (glm53_flash_step.py:610, kimi_k3_step.py:136); nothing on the qwen4_exp
   # path does, so all we give up is the per-bucket grad NaN/Inf guard.
   # relax/backends/megatron/model.py:1293 reads the flag too, but only to
   # enable an fp16-overflow probe that is inert under bf16.
   --no-check-for-nan-in-loss-and-grad
   # --use-flash-attn is deliberately absent: it is a dead no-op (zero reads of
   # `use_flash_attn` in megatron/), the reference does not pass it either, and
   # `--attention-backend flash` is the knob that actually selects the kernel.
   --use-health-check
)
# fp32 DDP gradient accumulation doubles the model's grad buffers on device.
# It is the reference default and stays on unless FP32_GRADS=0, which is the
# escape hatch for the 16-die device-ep layout: TP1/PP1 replicates the dense
# parameters, so the bf16 grad buffer is what lets the full 95 GiB PLE table
# stay trainable on a 64 GiB die (see ANALYSIS-8L.md / the 48L device-ep
# recipe). The optimizer's master weights stay fp32 either way.
if [ "${FP32_GRADS:-1}" != "0" ]; then
   MISC_ARGS+=(--accumulate-allreduce-grads-in-fp32)
fi

# Off by default. PROFILE_STEPS="<first>,<last>" (inclusive, 0-based rollout ids)
# records the training loop with torch.profiler, one `<...>_ascend_pt` dir per
# rank under ${TB_EXPERIMENT_NAME}/train_trace/.
#
# The target is `train_overall`, not `train_actor`: `TrainProfiler` offers a
# `train_actor` scope, but nothing ever calls `iterate_train_actor`, so that
# choice silently profiles nothing. `train_overall` is the one wired to
# `prof.step()` in the actor's training loop.
#
# TB_EXPERIMENT_NAME doubles as the trace root -- `_get_trace_base_dir` does
# `os.path.join("traces", tb_experiment_name)`, and an absolute name discards
# the `traces` prefix, so naming it an absolute path is how the output leaves
# the Ray worker's throwaway cwd.
#
# torch_npu ignores --profile-dir-name/--profile-record-shapes and writes its
# raw data to $(cwd)/export_only_prof_dir, which it deletes on __del__; see the
# analysis notes for the post-hoc `analyse()` call.
#
# PROFILE_WITH_STACK=1 adds `--profile-with-stack`. Without it the exported
# operator table has a *completely empty* Call Stack column, so a host op can
# only be attributed by its Input Shapes -- and the two biggest host costs
# (`aten::_local_scalar_dense` 1.94 s/step, `empty_tensor` 1.54 s/step) both
# report empty shapes, i.e. they are unattributable. With the flag the stacks
# name the Python frame. Costs ~2x trace size, so point the run's cwd at a
# directory with room (`export_only_prof_dir` is ~12 GB for 16 ranks x 3 steps).
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
      PROFILE_ARGS+=(--profile-with-stack)
   fi
fi

mkdir -p log

# The placement group has to ask for exactly as many actor dies as the
# entrypoint was given, and the SFT batcher keeps one storage unit per die. The
# two are easy to let drift: a 16-die `NPU_DEVICES` with a hardcoded 8 trains a
# TP2/CP2/DP2 world (EP=4 then exceeds DP) while every flag still looks right --
# the only clue is `using world size: 8, data-parallel size: 2` in the log. One
# die count, read off the visible dies, is the way to keep them in step.
NPU_COUNT="$(awk -F, '{print NF}' <<< "${ASCEND_RT_VISIBLE_DEVICES}")"

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${MASTER_ADDR:-127.0.0.1}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "{\"sft\": [1, 0], \"actor\": [1, ${NPU_COUNT}]}" \
   --sft-max-in-flight-steps ${SFT_MAX_IN_FLIGHT_STEPS:-2} \
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
   "${PROFILE_ARGS[@]}"  2>&1 | tee log/qwen38-flash-next-8L-mtp-sft-pokemon-npu16-${now}.log
