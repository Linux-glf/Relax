#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.8-Flash-Next (qwen4_exp) FULL 48-layer VL MTP SFT on
# pokemon-gpt4o-captions, 64 dies = 4 nodes x 16 NPUs, ray-submit launch.
# **The n-gram table is the released one and stays frozen**: it is the model's
# PLE weight, host-pinned, never a parameter and never in the optimizer.
#
# This is the NPU counterpart of the reference 64xH800 full-48L script, and it
# is deliberately aligned to it in every knob that is *not* the model, the
# dataset or the parallel decomposition -- same batch shape (32K packed, mbs 1),
# same MTP head and scaling factor, same optimizer hyper-parameters, same freeze
# set, same recompute and logits-chunking policy. What differs, and why:
#
#   tensor-model-parallel-size   8 -> 4   --num-query-groups 2 has to divide TP;
#                                         TP=4 keeps 6 of the 12 q heads per rank
#                                         (attention.py:355-372). See "TP, and why 4"
#   pipeline-model-parallel-size 8 -> 1   64 dies over 4 nodes; PP=1 is the only
#                                         decomposition with no bubble and no
#                                         pipeline layout to keep in sync
#   expert-model-parallel-size   8 -> 32  world = TP x CP x PP x DP = 64 with
#                                         DP = 16, and EP x EDP tiles DP
#   moe-token-dispatcher-type    flex -> alltoall
#   moe-flex-dispatcher-backend  deepep   dropped, needs NVSHMEM/DeepEP
#   the RELAX_QWEN38_QSA_* env block      dropped, no NPU counterpart exists
#   the dataset knobs                     this run reads a different corpus
#
# ⚠️ Multi-node: this script owns ALL FOUR machines. entrypoint/
# local-npu-multinode.sh begins with `pkill -9 sglang` / `ray stop --force` /
# `pkill -9 ray` / `pkill -9 python`, so sourcing it (which the auto-source block
# below does whenever RELAX_ENTRYPOINT_MODE is unset) terminates every other
# Relax job on every node it is run on and restarts Ray from scratch. Run it on
# the four nodes at once, and only when they are yours:
#
#   export WORLD_SIZE=4 NUM_NPUS=16
#   node 1: export MASTER_ADDR=<ip1> POD_NAME=<ip1> HOST_IP=<ip1>
#   node 2: export MASTER_ADDR=<ip1> POD_NAME=<ip2> HOST_IP=<ip2>
#   node 3: export MASTER_ADDR=<ip1> POD_NAME=<ip3> HOST_IP=<ip3>
#   node 4: export MASTER_ADDR=<ip1> POD_NAME=<ip4> HOST_IP=<ip4>
#   (each node, all four in parallel:)
#   export MODEL_DIR=<dir holding Qwen3.8-Flash-Next> DATA_DIR=<dir with sft/data/...> \
#          EXP_DIR=<dir for checkpoints>
#   bash scripts/training/sft/run-qwen38-flash-next-48L-pokemon-mtp-64xnpu.sh
#
# Under an external entrypoint (RELAX_ENTRYPOINT_MODE set by spmd-multinode.sh /
# ray-job-npu.sh) the auto-source is skipped and the run joins the existing
# cluster instead.
#
# ── The decomposition, and the two budgets it is chosen against ─────────────
#
#   world 64 = TP 4 x CP 1 x PP 1 x DP 16,  DP 16 = EP 32 x EDP 2 (ET=1)
#
# Device memory per die (bf16 params, fp32 gradients -- see
# --accumulate-allreduce-grads-in-fp32 below; the optimizer state is on the
# host, see --optimizer-cpu-offload):
#
#   experts     120.796e9 / (EP x PP)  = 3.775e9 el  ->  7.03 GiB params
#                                                    -> 14.06 GiB grads @4 B
#   dense         8.004e9 / (TP x PP)  = 2.001e9 el  ->  3.73 GiB params
#   dense grads   7.535e9 trainable / TP             ->  7.02 GiB
#   activations  48 layers x 32K tokens x 2560 x 2 B under full recompute
#                                                    ->  8.05 GiB (167.8 MB per
#                                                       checkpointed layer)
#   workspace    MoE permute/a2a buffers, the chunked logits (2048 x 248320 x
#                4 B = 1.9 GiB) and the fixed communication buffers
#                                                    -> ~6 GiB, measured at 4L
#   ── 31.8 + 8.1 + 6 = ~46 GiB of the 61 GiB a die reports, ~15 GiB of headroom
#
# The parameter counts are the checkpoint's own safetensors headers, summed by
# name over its 131 files: 180.000e9 bf16 elements = 335.3 GiB, of which experts
# 120.796e9, table 51.200e9, and 8.004e9 of everything else (dense 4.895e9, the
# MTP layer 2.607e9, the vision tower 0.449e9, the PLE's own projections
# 0.033e9, the attention indexer 0.020e9). The grad figures are NOT
# `trainable / world`: `_ParamAndGradBuffer` allocates `grad_data` at the full
# local-parameter numel (param_and_grad_buffer.py:1230-1240) and
# `param.main_grad` is a view into it (distributed_data_parallel.py), so a rank
# pays 4 bytes per parameter it *holds*, not per parameter divided by the
# data-parallel group. The distributed optimizer shards the *optimizer state*,
# not the gradients.
#
# Host memory per node (~1.88 TiB usable):
#
#   pinned n-gram table  next_pow2(102.4e9 B / TP) x 16 ranks
#                      = next_pow2(25.6 GB) x 16 = 32 GiB x 16 = 512 GiB
#   optimizer state      12 B x 128.331e9 trainable / 64 ranks = 22.4 GiB/rank
#                      = 358 GiB per node at 16 ranks
#   ── ~0.85 TiB of 1.88 TiB
#
# Both figures are load-bearing and both have a measured basis: on Ascend a
# pinned allocation is charged as next_power_of_two(payload) and is invisible to
# /proc, RSS and the cgroup (measured on this node: a 95.37 GiB table -> 129.93
# GiB charged), and the 16-die 8L runs of this same model already ran with 1 TiB
# of pinned table plus ~0.5 TiB of host optimizer state on one node. TP=2 (the 8L
# setting) would double the table to 64 GiB/rank = 1 TiB/node here; TP=4 is what
# buys the headroom, and TP=8 would halve it again at 4x the redundant qkv work.
# TP, and why 4:
#
#   TP=4 with --num-query-groups 2 takes the `num_query_groups < world_size`
#   branch of attention.py:357-364: every rank computes 12 q heads and keeps
#   `num_attention_heads / tp_size` = 6 of them, the other 6 being computed and
#   thrown away. That is the price of TP=4, and it is bounded: only 12 of the 48
#   layers are full attention, so the redundant q-projection is
#   12 layers x 2560 x 6144 = 189M MACs per token against a per-token total
#   dominated by the MoE block's 48 x topk10 x 3 x 2560 x 640 = 2.6G -- a few
#   percent. In exchange the pinned table halves (1 TiB -> 512 GiB per node),
#   the dense params/grads halve, and each of the 16 DP ranks carries 4 samples
#   per step at gbs 64 instead of 2.

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d_%H:%M:%S")
echo "当前时间: $now"

# ── environment that has to be in place *before* the entrypoint starts Ray ──
# local-npu-multinode.sh sets none of these, and its Ray head inherits whatever
# is exported here, so they must come before the source below.
export ASCEND_COREDUMP_SIGNAL=none
# 64 dies over 4 nodes: world size = TP * CP * PP * DP = 4 * 1 * 1 * 16. Each
# node contributes its own 16 dies; this list is per-node, not per-cluster.
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
# Auto-source local environment when not launched via an external entrypoint.
# The multi-node flavour, not local-npu.sh: on the head node it starts the Ray
# head, waits for WORLD_SIZE nodes to join, and only then returns to this script
# (the other three nodes block in `sleep inf` inside it).
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-npu-multinode.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen38-flash-next-48L.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/sft/pokemon}"
EXP_NAME="${EXP_NAME:-qwen38-flash-next-48L-mtp-sft-pokemon-npu64}"
# The released checkpoint, table included: 180.0e9 bf16 elements, 336 GB on
# disk (131 safetensors files, 102.4 GB of them the n-gram table). The PLE
# layer is part of the model here (`ple_layer_ids: [2]`), so `ple_table_dir` is
# filled from this directory and the table's row shard is pinned at build time.
CKPT_NAME="${CKPT_NAME:-Qwen3.8-Flash-Next}"
# Where the checkpoint, the dataset and the runs live. Relax keeps no internal
# mount path in tracked files (.gitleaks.toml rejects them), so the three roots
# come from the environment -- export them once before running (on every node):
#   MODEL_DIR  holds Qwen3.8-Flash-Next (and any other checkpoint)
#   DATA_DIR   holds sft/data/...
#   EXP_DIR    receives <exp>/sft/<name> and the -hf export next to it
: "${MODEL_DIR:?export MODEL_DIR=<directory holding Qwen3.8-Flash-Next>}"
: "${DATA_DIR:?export DATA_DIR=<directory holding sft/data/...>}"
: "${EXP_DIR:?export EXP_DIR=<directory for checkpoints and the HF export>}"

# Fail in the first second, not after the 336 GB checkpoint has been read: the
# bridge resolves `--hf-checkpoint` on every rank, and a path that does not
# exist there fails as a KeyError inside the index lookup.
if [ ! -d "${MODEL_DIR}/${CKPT_NAME}" ]; then
   echo "FATAL: ${MODEL_DIR}/${CKPT_NAME} is not a directory (MODEL_DIR=${MODEL_DIR})" >&2
   exit 2
fi

# TRAIN_LANGS selects which language shards are trained on; it exists so the
# per-step time can be attributed to the data source (the 8L step time steps up
# exactly when the reader crosses from en into zh).
TRAIN_FILES=()
for lang in ${TRAIN_LANGS:-en zh}; do
    TRAIN_FILES+=("'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_${lang}.parquet'")
done
PROMPT_DATA="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/qwen38-flash-next-48L-sft}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/${CKPT_NAME}
   --ref-load ${MODEL_DIR}/${CKPT_NAME}
   --megatron-to-hf-mode bridge
   --save ${SAVE_DIR}/sft/${EXP_NAME}
   # --load ${SAVE_DIR}/sft/${EXP_NAME}
   --save-interval 100
   --max-actor-ckpt-to-keep 1
   # NUM_EPOCH scales with gbs: the corpus is 1666 rows, so one epoch is 26
   # optimizer steps at gbs=64 and 104 at gbs=16. The default keeps NUM_ROLLOUT
   # (20) inside a single pass.
   --num-epoch ${NUM_EPOCH:-1}
   --num-rollout ${NUM_ROLLOUT:-20}
   # Model weights only: the full checkpoint (with the Adam moments) is huge for
   # this model, and the shared volume ${EXP_DIR} sits on runs at 100% -- the run
   # saves the weights and exports them to HF instead. Drop this flag to get a
   # resumable checkpoint back.
   --no-save-optim
)
# The HF export rewrites the whole 336 GB checkpoint, ~102 GB of which is the
# hashed n-gram table, so the baseline run skips it. SAVE_HF=1 turns it back on
# (and needs ~240 GB free wherever ${SAVE_DIR} lives).
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
   --image-max-token-num 128
   --sft-invalid-multimodal-strategy skip
   # For a pure SFT run nothing in relax/engine/sft/ reads this -- it drives the
   # context_parallel_size derivation (backends/megatron/arguments.py:119) and
   # the validator at utils/arguments.py:4126, and nothing else. Since that
   # validator requires max_tokens_per_gpu * cp >= rollout_max_context_len, a
   # 32K value here forces --max-tokens-per-gpu to 32K as well, which forces the
   # whole per-rank batch into ONE micro-batch. Lowering it (to just above the
   # longest real sample, 9457 tokens) lets the token-budget packer split the
   # batch into gradient-accumulation micro-batches again.
   --rollout-max-context-len ${ROLLOUT_MAX_CONTEXT_LEN:-32768}
   # gbs 64 over DP 16 = 4 samples per rank per step, ~38K tokens, which the
   # packer splits into two ~19K-token micro-batches at --max-tokens-per-gpu
   # 32768. The reference's gbs 32 would be 2 samples per rank here: enough to
   # be correct, too little to amortize a step's collectives over (the 8L
   # profile is 63% communication).
   --global-batch-size ${GLOBAL_BATCH_SIZE:-64}
   --micro-batch-size 1
   --seq-length 32768
   # SFT requires dynamic batching (relax/utils/arguments.py rejects
   # --loss-type sft without it), and dynamic batching is thd-only. thd is also
   # what the plugin's own GDN implements (modelling_qwen4_exp/gdn.py).
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-32768}
   --data-pad-size-multiplier 4096
   --calculate-per-token-loss
   --per-rank-fetch
   --sft-prefetch-num-workers 16
   --sft-train-data-prefetch
   # --balance-data is NOT passed here even though the 4L recipe passed it: the
   # reference leaves it commented out, and relax/utils/arguments.py:4194-4198
   # auto-enables it for offline loss types anyway. Both scripts end up with
   # balance_data=True; the flag is inert either way.
)
# --sft-prefetch-buffer-size has no counterpart in the reference and is kept
# from the 4L recipe: dropping it changes only prefetch depth, not the data.
SFT_ARGS+=(--sft-prefetch-buffer-size ${SFT_PREFETCH_BUFFER_SIZE:-512})

# MTP aligned to the reference: 1 layer, scaling factor 0.1, detach path
# `none`. The CP == 1 text/bridge layout it needs is handled on the plugin
# side (`Qwen4ExpModel._prepare_mtp_loss_mask`); the CP > 1 right-roll exists
# (`_roll_loss_mask_right_cp`) but is unverified, and the PLE table's row shard
# is not widened for CP -- so this recipe stays at CP == 1.
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

# Pipeline layout, keyed off PP_SIZE. One stage puts all 48 decoders with the
# MTP layer and the loss head on the only rank; two stages split 24/24 as
# `E+t*24 | t*24+m+L`, so the single MTP layer sits in the last stage and never
# in PP rank 0, as the validator requires.
#
# The `*` and `|` are backslash-escaped on purpose -- see the note below.
if [ "${PP_SIZE:-1}" = "1" ]; then
   PPL_LAYOUT='Et\*48mL'
else
   PPL_LAYOUT='Et\*24\|t\*24mL'
fi

PERF_ARGS=(
   # TP * CP * PP * DP = 4 * 1 * 1 * 16 = 64 dies, with DP = EP * EDP = 32 * 2.
   # The expert grid tiles the world on its own axis (--expert-tensor-parallel-
   # size 1), so EP=32 is 16 local experts per rank and the a2a spans the four
   # nodes; the 128-weight ceiling of aclnnGroupedMatmulV5 (which is what caps
   # EP at 4 with 512 experts) is nowhere near binding here.
   --tensor-model-parallel-size ${TP_SIZE:-4}
   --pipeline-model-parallel-size ${PP_SIZE:-1}
   --context-parallel-size ${CP_SIZE:-1}
   --expert-model-parallel-size ${EP_SIZE:-32}
   --expert-tensor-parallel-size 1

   # One pipeline stage over 48 decoder layers: E + t*48 + m + L. The `|` and
   # `*` are backslash-escaped on purpose. `ray job submit` does not quote: it
   # space-joins the argv it is handed and gives the result to `/bin/sh -c`, so
   # a bare `|` becomes a pipe -- the first 8L attempt died with
   # `/bin/sh: 1: t*4mL: not found`. dash collapses `Et\*48mL` back to
   # `Et*48mL`, which is what the layout parser receives.
   --pipeline-model-parallel-layout "${PPL_LAYOUT}"

   # thd (the default) rather than bshd: upstream mcore's GatedDeltaNet refuses
   # packed sequences, but the plugin builds its own vendored Qwen4ExpGDN
   # (modelling_qwen4_exp/gdn.py), which implements the packed path. bshd is
   # not an option for SFT anyway -- --loss-type sft requires dynamic batching,
   # which requires thd.

   # 1 is already the maximum here, not a conservative starting point: under
   # --recompute-method uniform this is the layers-per-recompute-unit, so 1
   # checkpoints every layer individually (core/recompute.py:178-187). 48 layers
   # with num_layers 1 recomputes all 48 -- which is also why the MTP head
   # asserts on any other value (multi_token_prediction.py:1919). Do not "raise
   # this to buy memory": the block method would recompute the same layers
   # while making the MTP head skip its own recompute entirely
   # (multi_token_prediction.py:1922), i.e. strictly more memory.
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1


   --no-store-param-remainders
   --no-gradient-accumulation-fusion

   --cross-entropy-loss-fusion
   --sft-chunked-logits
   --sft-logits-chunk-size ${SFT_LOGITS_CHUNK_SIZE:-2048}

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
# D2H grads / H2D params copy every step. Measured on the 4L run's trace (rank 0,
# two steps): `aten::_fused_adamw_` 673 calls/step at 4.72 s of host self time
# with zero device time, plus 5.20 s of `aten::copy_` -- together they are most
# of the Free 11.4 s that the same trace's step_trace_time.csv reports (the
# device is idle 68% of that step). At full depth this loop is NOT 12x the 4L
# one: the Adam call granularity follows the bucket size, and the per-rank
# optimizer state only grows 5.336e9/4 -> 128.331e9/64 = 1.33e9 -> 2.01e9
# elements (1.5x), so 5-12 s per step is the range to check against, not minutes.
#
# What is not optional is the offload itself: 128.33e9 trainable parameters x
# 12 B = 1.4 TiB of fp32 master/m/v, which no 64-die layout can hold on device
# (12 B x 128.33e9 / 64 ranks = 24 GB per rank, on top of the ~32 GiB of params
# and grads). OPTIMIZER_CPU_OFFLOAD=0 is kept as a knob for diagnosing the host
# loop, not as a configuration that fits.
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
# The reasoning above was never A/B'd, and the first matched comparison on 16
# dies came out the other way (steps 2-11: 86.31 s with the pair ON, 82.15 s
# with it OFF, of which 3.66 s is step 10 alone). Re-run the pair here before
# trusting either number; at 64 dies across four nodes the latency term is
# larger, not smaller.
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
# >=100 ms and own 96% of that operator's 27.2 s, so SEQUENCE_PARALLEL=0 is the
# one lever that removes collectives rather than reordering them. Per rank, the
# all-gather volume is T x (TP-1)/TP x hidden x 2 B, so TP=4 pays 1.5x what TP=2
# pays for the same token count -- the first A/B this script deserves.
# Activations are replicated across TP instead of sharded, which full recompute
# bounds; the arithmetic is unchanged either way.
if [ "${SEQUENCE_PARALLEL:-1}" != "0" ]; then
   PERF_ARGS+=(--sequence-parallel)
fi

# The 48L run has one more OOM-adjacent knob than the 8L one: the order the
# rows of the table are pinned versus the model's own buffers. It is not a knob.
#
# Freezing, aligned with the reference but NOT spelled the same way, because on
# the qwen4_exp path two of the reference's three mechanisms do nothing:
#
#   --freeze-vision-model / --freeze-vision-projection are declared (they are in
#   the Bridge `bridge_keys`) but implemented only by the glm_vl and qwen_vl
#   models. Nothing under models/qwen4_exp reads them, so passing them here
#   would be a silent no-op. The vision tower is instead named `visual.*`
#   (modelling_qwen4_exp/vl_model.py:114-123) and is frozen by the regex below.
#   At PP=1 the tower (0.449e9 parameters) is built on every rank.
#
#   `per_layer_embedding` does not exist on this path at all: the PLE table is
#   host-pinned and never a nn.Parameter (modelling_qwen4_exp/ple.py), so it is
#   absent from the optimizer and from the DCP checkpoint. That is exactly what
#   this script wants -- the released 95.37 GiB table is not trainable at any
#   legal TP: at TP=4 its row shard is 25.6 GB of parameters and 4 bytes per
#   element of grad on top, which is a die's entire budget.
#
# `--freeze-params-name-list` is nargs="*" (relax/utils/arguments.py:553-571), so it
# takes a *list* of independent re.search patterns. That is what lets this script
# dodge a real landmine: `ray job submit` does not quote -- it space-joins the argv
# and hands it to `/bin/sh -c`, so shell metacharacters inside a value are acted on
# by dash. The single-pattern form that expresses the same thing,
# `'(^|\.)visual\.|self_attention\.indexer\.'`, died with
# `/bin/sh: 1: Syntax error: "(" unexpected`; backslash-escaping is not a way out
# either, because dash eats the backslash in `\.` (verified through job submission).
# Two plain-substring patterns carry no metacharacters at all and select exactly
# the same tensors.
TRAINABLE_ARGS=(
   --freeze-params-name-list visual indexer
)

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
# TensorBoard/ClearML writes per metric. At 64 ranks that replica is fed 4x what
# the 8L run fed it; LOGGING_OFF=1 strips all of it so the cost can be measured.
# Per-step loss and grad_norm still reach the log either way.
if [ "${LOGGING_OFF:-0}" != "0" ]; then
   WANDB_ARGS=(--no-use-metrics-service --no-use-tensorboard)
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   # This is what makes the grads 4 B per parameter rather than 2. The reference
   # passes it and the 8L runs pass it; dropping it halves the expert and dense
   # grad buffers (~9.5 GiB of the ~42 GiB above) and is the first thing to try
   # if a die comes up short.
   #
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
   # Measured on the 8L recipe (16 dies, TP2/PP1/EP8, gbs 192): 4 calls per rank
   # per step, 98 ms each, 0.39 s/step per rank, worst single call 1.82 s.
   --no-check-for-nan-in-loss-and-grad
   # --use-flash-attn is deliberately absent: it is a dead no-op (zero reads of
   # `use_flash_attn` in megatron/), the reference does not pass it either, and
   # `--attention-backend flash` is the knob that actually selects the kernel.
   --use-health-check
)

# Off by default. PROFILE_STEPS="<first>,<last>" (inclusive, 0-based rollout ids)
# records the training loop with torch.profiler, one `<...>_ascend_pt` dir per
# rank under ${TB_EXPERIMENT_NAME}/train_trace/. At 64 ranks the raw data is
# ~4x the 8L runs' 12 GB -- point the run's cwd at a directory with room.
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

# The placement group has to ask for exactly as many actor dies as there are
# visible dies on the four nodes, and the SFT batcher keeps one storage unit per
# die. The two are easy to let drift: a 64-die cluster with a hardcoded 16 trains
# a TP4/EP32/DP4 world while every flag still looks right -- the only clue is
# `using world size: 16, data-parallel size: 4` in the log. The node count is
# read from the same place the entrypoint read it, WORLD_SIZE.
NPU_COUNT="$(awk -F, '{print NF}' <<< "${ASCEND_RT_VISIBLE_DEVICES}")"
NODE_COUNT="${WORLD_SIZE:-1}"
ACTOR_DIES=$(( NPU_COUNT * NODE_COUNT ))
if [ "${CP_SIZE:-1}" != "1" ]; then
   echo "FATAL: this script is written for CP=1 (CP=${CP_SIZE}); the CP>1 MTP mask" >&2
   echo "       right-roll is unverified and the PLE table's row shard is not widened to include CP." >&2
   exit 2
fi

ray job submit ${RAY_NO_WAIT:+--no-wait} \
   --address="http://${HOST_IP:-${MASTER_ADDR:-127.0.0.1}}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "{\"sft\": [1, 0], \"actor\": [${NODE_COUNT}, ${ACTOR_DIES}]}" \
   --sft-max-in-flight-steps ${SFT_MAX_IN_FLIGHT_STEPS:-6} \
   --num-data-storage-units "${ACTOR_DIES}" \
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
   "${PROFILE_ARGS[@]}"  2>&1 | tee log/qwen38-flash-next-48L-mtp-sft-pokemon-npu64-${now}.log
