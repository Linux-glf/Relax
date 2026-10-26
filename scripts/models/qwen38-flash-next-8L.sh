# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.8-Flash-Next, 8-layer truncation (model_type qwen4_exp,
# Qwen4ExpForConditionalGeneration). Architecturally identical to the full
# checkpoint, with decoder layers 8..47 dropped by
# scripts/tools/truncate_qwen4_exp_ckpt.py -- so the hybrid stack is
# linear_attention x3, full_attention, linear_attention x3, full_attention, and
# the 1-layer MTP head and the 27-layer vision tower are kept.
#
# Values below are the ones the checkpoint's config.json declares; keep them in
# sync with it. Note that under --megatron-to-hf-mode bridge the model config is
# built by the Bridge provider from that same config.json, and only the keys in
# relax/backends/megatron/model_provider.py's `bridge_keys` are overridden by
# the CLI (num-layers, moe-layer-freq, moe-ffn-hidden-size, moe-router-topk,
# moe-shared-expert-intermediate-size, rotary-base, moe-grouped-gemm, ...), so
# a wrong value here does change the model.

# NUM_LAYERS overrides the truncation depth so the same architecture block
# serves deeper slices (e.g. NUM_LAYERS=16 for Qwen3.8-Flash-Next-16L-ple).
# It must match the checkpoint's config.json num_hidden_layers, and the
# pipeline layout in the training script must use the same value.
NLAYERS="${NUM_LAYERS:-8}"

arr=()
for ((i=0; i<NLAYERS; i++)); do
  arr+=(1)
done

printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"


MODEL_ARGS=(
   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 24
   --num-query-groups 2
   --kv-channels 256
   --num-layers $NLAYERS
   --hidden-size 2560
   --ffn-hidden-size 640
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   # `rope` is a placeholder: the model really runs mrope, which the provider
   # declares (and Relax keeps `position_embedding_type` out of `bridge_keys`,
   # so the CLI cannot change it). The CLI value still has to be a legal one
   # for Megatron's validators, and `mrope` cannot be spelled here at all --
   # it demands a `--mrope-section` flag that this parser does not define.
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 248320

   --rotary-base 10000000

   # moe
   --moe-ffn-hidden-size 640
   --moe-shared-expert-intermediate-size 640
   --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall
   --moe-router-topk 10
   --moe-layer-freq $MOE_LAYER_FREQ
   --num-experts 512
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   --moe-permute-fusion
   --moe-aux-loss-coeff 0

   # qwen4_exp specific
   --attention-output-gate
   --moe-shared-expert-gate
)
