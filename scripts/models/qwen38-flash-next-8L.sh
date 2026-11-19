# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.8-Flash-Next, 8-layer truncation (model_type qwen4_exp).
# The model architecture is declared by the checkpoint's config.json; the CLI
# values below stay inside the bridge_keys surface of the target 26.2.0 stack.
# NUM_LAYERS overrides the truncation depth and must match config.json.

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

   --normalization RMSNorm
   --apply-layernorm-1p
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
)
