# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Patch ``process_mtp_loss`` (in ``multi_token_prediction`` and its copy in
# ``gpt_model``) to compute each MTP depth's head + cross-entropy in SEQUENCE
# CHUNKS, so the full ``[S, V/TP]`` per-depth logits never materialize.
#
# WHY: the native branch runs each MTP head on the full sequence -> ``[S, V/TP]``
# logits (7.6 GiB bf16 per head at S=65536); with 3 MTP depths this dominates the
# last PP stage and OOMs (worse under CP=2). ``--sft-chunked-logits`` chunks the
# MAIN head but not the MTP heads, and the fused-linear CE that avoids logits is
# Blackwell-only.
#
# EQUIVALENCE: CE reduces per token, so head+CE per sequence slice and
# concatenating the ``[b, chunk]`` losses rebuilds the ``[b, s]`` loss
# bit-for-bit. Everything after the loss is copied verbatim from upstream.
#
# _passthrough (load-bearing): the chunked forward installs ``_passthrough`` on the
# instance ``output_layer.forward`` and counts MTP head calls. To avoid corrupting
# that counter, the head is called via the CLASS-level forward (invisible to the
# instance override), so MTP consumes ZERO intercepted calls. The paired
# ``mtp_output_layer_calls=0`` in relax model.py uses the SAME gate — keep them in
# sync.
#
# GATE: chunks only when ``enable_mtp_training`` AND ``sft_chunked_logits``;
# otherwise delegates to the captured original. Chunk size = ``--sft-logits-chunk-size``.

import inspect
import sys

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

try:
    import torch
    from megatron.core import parallel_state
    from megatron.core.transformer import multi_token_prediction as _mtp_mod
    from megatron.core.transformer.multi_token_prediction import (
        MTPLossAutoScaler,
        MTPLossLoggingHelper,
        roll_tensor,
    )

    # Capture the genuine upstream implementation BEFORE we overwrite the bindings,
    # so the delegate path (fuse-linear / gate-off) runs the original verbatim.
    _ORIG_PROCESS_MTP_LOSS = _mtp_mod.process_mtp_loss
    _ORIG_PROCESS_MTP_LOSS_PARAMETERS = set(inspect.signature(_ORIG_PROCESS_MTP_LOSS).parameters)
    # Megatron renamed ``save_loss_to_tracker`` to ``save_metrics_to_tracker`` when
    # the MTP acceptance counts moved next to the loss. Accept either spelling.
    if hasattr(MTPLossLoggingHelper, "save_metrics_to_tracker"):
        _MTP_TRACKER_API = "metrics"
        _MTP_TRACKER_PARAMETERS = set(inspect.signature(MTPLossLoggingHelper.save_metrics_to_tracker).parameters)
    elif hasattr(MTPLossLoggingHelper, "save_loss_to_tracker"):
        _MTP_TRACKER_API = "loss"
        _MTP_TRACKER_PARAMETERS = set(inspect.signature(MTPLossLoggingHelper.save_loss_to_tracker).parameters)
    else:
        _MTP_TRACKER_API = "none"
        _MTP_TRACKER_PARAMETERS = set()
    _COMPUTE_MTP_ACCEPTANCE_COUNTS = getattr(_mtp_mod, "_compute_mtp_acceptance_counts", None)

    def _call_original_process_mtp_loss(*args, **kwargs):
        """Delegate across Megatron versions that added TP group/input IDs/masks."""
        return _ORIG_PROCESS_MTP_LOSS(
            *args,
            **{name: value for name, value in kwargs.items() if name in _ORIG_PROCESS_MTP_LOSS_PARAMETERS},
        )

    @torch._dynamo.disable
    def _chunked_head_ce(
        mtp_hidden,
        mtp_labels,
        output_layer,
        output_weight,
        runtime_gather_output,
        compute_language_model_loss,
        scale_logits_fn,
        chunk_size,
        tp_group=None,
        loss_mask=None,
    ):
        """Per-depth head+CE in sequence chunks -> per-token loss [b, s].

        SP alignment (load-bearing): with ``--sequence-parallel`` the hidden
        arrives SP-scattered ``[S/TP, b, H]`` while labels are full-sequence.
        Chunking the scattered input and letting the head all-gather each chunk
        misaligns with the label slice (the ``[2048] vs [1024]`` IndexError), so
        we gather the hidden to full sequence ONCE (cheap — only the logits are
        large and stay chunked), then run the head with ``sequence_parallel=False``
        per chunk, mapping ``[chunk, b, H] -> [chunk, b, V/TP]`` 1:1 with the
        label slice.

        Runs inside ``torch._dynamo.config.patch(disable=True)``, not just the
        decorator: the MTP loss runs nested inside a live torch.compile region
        (the router's ``@torch.compile`` on every MoE layer), and dynamo still
        fake-traces the call boundary and chokes on the logits getitem unless the
        outer trace is globally disabled. Pure memory orchestration; no numeric
        effect.
        """
        head_forward = type(output_layer).forward  # bypass _passthrough (instance-level)
        sp_enabled = bool(getattr(output_layer, "sequence_parallel", False))
        with torch._dynamo.config.patch(disable=True):
            # Gather SP-scattered hidden to full sequence once; head runs with SP
            # disabled per chunk. When SP is off it is already full-sequence.
            if sp_enabled:
                from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

                tp_group = (
                    tp_group
                    or getattr(output_layer, "tp_group", None)
                    or parallel_state.get_tensor_model_parallel_group()
                )
                mtp_hidden = gather_from_sequence_parallel_region(
                    mtp_hidden, tensor_parallel_output_grad=False, group=tp_group
                )  # [S/TP, b, H] -> [S, b, H]
                output_layer.sequence_parallel = False

            seq_len = mtp_hidden.shape[0]
            per_chunk_losses = []
            correct = None
            total = None
            try:
                for start in range(0, seq_len, chunk_size):
                    end = min(start + chunk_size, seq_len)
                    logits_slice, _ = head_forward(
                        output_layer,
                        mtp_hidden[start:end],  # [chunk, b, H]
                        weight=output_weight,
                        runtime_gather_output=runtime_gather_output,
                    )  # [chunk, b, V/TP] (no re-gather: SP disabled)
                    if scale_logits_fn is not None:
                        logits_slice = scale_logits_fn(logits_slice)
                    # compute_language_model_loss(labels[b,s], logits[s,b,V]) -> [b, s];
                    # slice labels on the seq axis (dim=1) to match the logits slice.
                    loss_slice = compute_language_model_loss(mtp_labels[:, start:end], logits_slice)
                    per_chunk_losses.append(loss_slice)
                    if _COMPUTE_MTP_ACCEPTANCE_COUNTS is not None and loss_mask is not None:
                        chunk_correct, chunk_total = _COMPUTE_MTP_ACCEPTANCE_COUNTS(
                            logits_slice,
                            mtp_labels[:, start:end],
                            loss_mask[:, start:end],
                            output_layer,
                            runtime_gather_output,
                            tp_group,
                        )
                        correct = chunk_correct if correct is None else correct + chunk_correct
                        total = chunk_total if total is None else total + chunk_total
                    del logits_slice
            finally:
                if sp_enabled:
                    output_layer.sequence_parallel = True
        return torch.cat(per_chunk_losses, dim=1), correct, total  # [b, s], rebuilt exactly

    def _save_mtp_loss_to_tracker(
        mtp_loss, num_tokens, layer_number, config, correct=None, total=None, metric_avg_group=None
    ):
        """Log MTP loss across the legacy and token-aware tracker APIs."""
        avg_group = metric_avg_group or parallel_state.get_data_parallel_group(with_context_parallel=True)
        if _MTP_TRACKER_API == "metrics":
            # Megatron >= core_r0.21: the tracker consumes the already-normalized loss
            # together with the acceptance counts, mirroring upstream process_mtp_loss.
            zero = torch.zeros((), device=mtp_loss.device, dtype=torch.float32)
            mtp_loss_for_log = (torch.sum(mtp_loss) * (num_tokens > 0).to(mtp_loss.dtype)) / num_tokens.clamp(min=1)
            MTPLossLoggingHelper.save_metrics_to_tracker(
                mtp_loss_for_log,
                correct if correct is not None else zero,
                total if total is not None else zero,
                layer_number,
                config.mtp_num_layers,
                avg_group=avg_group,
            )
            # The metrics tracker stores this microbatch's already-normalized
            # mean, and the trainer multiplies the sum by 1/num_microbatches:
            # that is a mean of microbatch means, whose value depends on how the
            # dynamic batcher partitioned the global batch. The GPU stack's
            # per-token tracker logs sum(CE)/sum(tokens), which is invariant to
            # that partition. Keep the raw numerator and denominator here; the
            # ``track_mtp_metrics`` wrapper installed below substitutes the
            # ratio before the original tracker reduces and logs it.
            tracker = MTPLossLoggingHelper.tracker
            if "_relax_mtp_loss_sum" not in tracker:
                tracker["_relax_mtp_loss_sum"] = torch.zeros(
                    config.mtp_num_layers, device=mtp_loss.device, dtype=torch.float32
                )
                tracker["_relax_mtp_token_count"] = torch.zeros(
                    config.mtp_num_layers, device=mtp_loss.device, dtype=torch.float32
                )
            tracker["_relax_mtp_loss_sum"][layer_number] += torch.sum(mtp_loss).detach().float()
            tracker["_relax_mtp_token_count"][layer_number] += num_tokens.detach().float()
            return
        if _MTP_TRACKER_API == "none":
            return
        if "num_tokens" in _MTP_TRACKER_PARAMETERS:
            mtp_loss_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers
            kwargs = {
                "avg_group": avg_group,
                "calculate_per_token_loss": config.calculate_per_token_loss,
            }
            if "correct" in _MTP_TRACKER_PARAMETERS:
                kwargs.update(correct=correct, total=total)
            MTPLossLoggingHelper.save_loss_to_tracker(
                mtp_loss_scale * torch.sum(mtp_loss),
                num_tokens,
                layer_number,
                config.mtp_num_layers,
                **kwargs,
            )
            return

        mtp_loss_for_log = (torch.sum(mtp_loss) * (num_tokens > 0).to(mtp_loss.dtype)) / num_tokens.clamp(min=1)
        MTPLossLoggingHelper.save_loss_to_tracker(
            mtp_loss_for_log,
            layer_number,
            config.mtp_num_layers,
            avg_group=avg_group,
        )

    def _resolve_chunk_size() -> int:
        """MTP head chunk size = --sft-logits-chunk-size (default 1024)."""
        try:
            from megatron.training.global_vars import get_args

            args = get_args()
        except Exception:
            args = None
        cs = int(getattr(args, "sft_logits_chunk_size", 1024) or 1024) if args is not None else 1024
        return cs if cs > 0 else 1024

    def _chunked_mtp_enabled() -> bool:
        """Gate: chunk MTP heads only for the MTP + sft-chunked-logits SFT run.

        MUST match the mtp_output_layer_calls==0 gate in relax/backends/megatron/
        model.py (single source of truth for the _passthrough accounting).
        """
        try:
            from megatron.training.global_vars import get_args

            args = get_args()
        except Exception:
            return False
        if args is None:
            return False
        return bool(getattr(args, "enable_mtp_training", False)) and bool(getattr(args, "sft_chunked_logits", False))

    @torch._dynamo.disable
    def process_mtp_loss(
        hidden_states,
        labels,
        loss_mask,
        output_layer,
        output_weight,
        runtime_gather_output,
        is_training,
        compute_language_model_loss,
        config,
        cp_group=None,
        tp_group=None,
        packed_seq_params=None,
        scale_logits_fn=None,
        input_ids=None,
        mtp_input_mask=None,
        metric_avg_group=None,
        main_hidden_states=None,
    ):
        """Chunked drop-in for ``process_mtp_loss`` (native CE branch only).

        Numerically identical to upstream; only the head+CE is chunked.
        Delegates to the original when the gate is off or the fused-linear CE
        path is used.
        """
        fuse_linear_cross_entropy = config.cross_entropy_loss_fusion and config.cross_entropy_fusion_impl == "linear"
        # Caller-supplied main hidden states change what the MTP block returns and are
        # only used by hybrid models; this tree's qwen4_exp/SFT path never sets them.
        # `mtp_input_mask` (VL conditioning mask) and the per-path detach flags are
        # modeled below -- the GPU `feat/qwen4exp-cp-multimodal` branch runs exactly
        # that combination, so delegating here would silently change what the two
        # stacks supervise.
        unsupported = main_hidden_states is not None
        # Delegate verbatim when: fused-linear CE (Blackwell path -- don't chunk), or
        # the chunked-MTP gate is off (preserve exact upstream behavior), or there is
        # no label source at all (upstream returns the hidden states untouched).
        #
        # `labels is None` alone is NOT a reason to delegate: Relax's SFT forward_step
        # always passes `labels=None` and relies on upstream deriving them from
        # `input_ids`, so treating it as a delegate condition sent every SFT+MTP step
        # down the unchunked path. That is not merely a memory regression -- the
        # unchunked path calls `output_layer(...)`, i.e. the INSTANCE forward, which
        # `_bypass_output_layer` has replaced with its `_passthrough`. Relax sizes that
        # bypass with `mtp_output_layer_calls=0` precisely because this patch promises
        # to reach the head through the class-level forward, so the delegate made the
        # MTP head's logits call take the main head's slot: the head received gathered
        # hidden states where it expected logits, and the real main-head call then
        # raised "called more than once after all MTP head calls". The derivation below
        # restores the promise; `_chunked_mtp_enabled()` stays the single gate.
        if (
            fuse_linear_cross_entropy
            or unsupported
            or not _chunked_mtp_enabled()
            or (labels is None and input_ids is None)
        ):
            return _call_original_process_mtp_loss(
                hidden_states,
                labels,
                loss_mask,
                output_layer,
                output_weight,
                runtime_gather_output,
                is_training,
                compute_language_model_loss,
                config,
                cp_group=cp_group,
                tp_group=tp_group,
                packed_seq_params=packed_seq_params,
                scale_logits_fn=scale_logits_fn,
                input_ids=input_ids,
                mtp_input_mask=mtp_input_mask,
                metric_avg_group=metric_avg_group,
                main_hidden_states=main_hidden_states,
            )

        # ---- chunked native path (mirrors upstream lines 650-723 exactly) ----
        hidden_states_list = torch.chunk(hidden_states, 1 + config.mtp_num_layers, dim=0)
        hidden_states = hidden_states_list[0]

        # Mirrors upstream's labels-is-None branch, which this patch now owns: derive
        # the SFT-format labels by rolling the input ids left, and roll the loss mask
        # once in lockstep so the position the roll fabricated (there is no token past
        # the window) stays masked. Both rolls land BEFORE `original_num_tokens`,
        # matching upstream's ordering -- that count feeds the per-token-loss
        # normalization and must see the post-roll mask.
        derived_labels_from_input_ids = labels is None
        if derived_labels_from_input_ids:
            labels, _ = roll_tensor(
                input_ids,
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
                return_sum=False,
            )

        # lm-head detach, mirroring upstream's `config.mtp_detach_heads` block but
        # keyed on the per-path flag Relax writes (`configure_mtp_detach_paths`).
        # The embedding/backbone paths are detached by the MTP layer itself; keeping
        # this here honors `--mtp-detach-paths lm-head` even on partial selections.
        if getattr(config, "mtp_detach_lm_head", getattr(config, "mtp_detach_heads", False)):
            if output_weight is not None:
                output_weight = output_weight.detach()
            else:
                output_weight = output_layer.weight.detach()

        mtp_labels = labels.clone()
        if loss_mask is None:
            loss_mask = torch.ones_like(mtp_labels)
        if derived_labels_from_input_ids:
            loss_mask, _ = roll_tensor(
                loss_mask,
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
                return_sum=False,
            )

        # per-token count BEFORE rolling (used by the per-token-loss normalization).
        original_num_tokens = loss_mask.sum()

        # VL conditioning mask: image/video placeholder tokens cannot condition an MTP
        # depth, so every depth that consumes one is excluded from the loss (and the
        # MTP layer selectively detaches their embedding rows). The mask arrives in
        # the input-id frame, not the predictor frame the loss mask is rolled from,
        # so it must NOT get the bridge's right-roll compensation.
        cumulative_mtp_input_mask = None
        if mtp_input_mask is not None:
            assert mtp_input_mask.shape == loss_mask.shape, (
                f"mtp_input_mask shape {tuple(mtp_input_mask.shape)} must match "
                f"loss_mask shape {tuple(loss_mask.shape)}"
            )
            mtp_input_mask = mtp_input_mask.to(dtype=torch.bool)

        chunk_size = _resolve_chunk_size()

        for mtp_layer_number in range(config.mtp_num_layers):
            mtp_labels, _ = roll_tensor(
                mtp_labels, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
            )
            if mtp_input_mask is not None:
                # Roll the loss mask and the conditioning mask together so a depth
                # whose conditioning token is an image/video placeholder contributes
                # nothing. `cumulative` makes one invalid conditioning token mask
                # every later depth on that path (upstream semantics).
                mask_metadata = torch.cat((loss_mask, mtp_input_mask.to(dtype=loss_mask.dtype)), dim=0)
                mask_metadata, _ = roll_tensor(
                    mask_metadata, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
                )
                loss_mask, mtp_input_mask = mask_metadata.chunk(2, dim=0)
                mtp_input_mask = mtp_input_mask.to(dtype=torch.bool)
                if cumulative_mtp_input_mask is None:
                    cumulative_mtp_input_mask = mtp_input_mask
                else:
                    cumulative_mtp_input_mask = cumulative_mtp_input_mask & mtp_input_mask
                layer_loss_mask = loss_mask * cumulative_mtp_input_mask
                num_tokens = layer_loss_mask.sum()
            else:
                loss_mask, num_tokens = roll_tensor(
                    loss_mask, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
                )
                layer_loss_mask = loss_mask

            # Chunked head+CE (dynamo-disabled) -> per-token loss [b, s].
            mtp_loss, correct, total = _chunked_head_ce(
                hidden_states_list[mtp_layer_number + 1],
                mtp_labels,
                output_layer,
                output_weight,
                runtime_gather_output,
                compute_language_model_loss,
                scale_logits_fn,
                chunk_size,
                tp_group=tp_group,
                loss_mask=layer_loss_mask if is_training else None,
            )

            # ---- identical to upstream from here (operates on [b, s], no vocab dim) ----
            mtp_loss = layer_loss_mask * mtp_loss
            # Upstream logs the UNSCALED per-depth loss (before mtp_loss_scale) and
            # only folds the scale into the gradient path — mirror that exactly.
            if is_training:
                _save_mtp_loss_to_tracker(
                    mtp_loss, num_tokens, mtp_layer_number, config, correct, total, metric_avg_group
                )
            mtp_loss_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers
            if config.calculate_per_token_loss:
                num_tokens_safe = torch.clamp(num_tokens, min=1)
                mtp_loss_normalized = mtp_loss_scale * mtp_loss * (original_num_tokens / num_tokens_safe)
                hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_normalized)
            else:
                safe_num_tokens = num_tokens.clamp(min=1)
                hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss / safe_num_tokens)

        return hidden_states

    # Force gpt_model into sys.modules so we can replace its (separately-bound) copy.
    import megatron.core.models.gpt.gpt_model as _gpt_mod  # noqa: E402

    _mtp_mod.process_mtp_loss = process_mtp_loss
    if hasattr(_gpt_mod, "process_mtp_loss"):
        _gpt_mod.process_mtp_loss = process_mtp_loss
    _mamba_mod = sys.modules.get("megatron.core.models.mamba.mamba_model")
    if _mamba_mod is not None and hasattr(_mamba_mod, "process_mtp_loss"):
        _mamba_mod.process_mtp_loss = process_mtp_loss

    # Token-weighted MTP-loss metric. The npu-port tracker logs a mean of
    # per-microbatch means, so `train/mtp_loss` changes when the micro-batch/DP
    # partition changes even if the per-token loss is identical. Wrap the
    # tracker's entry point: reduce the raw sums/token counts collected above
    # with SUM, replace `loss_values` with their ratio, and let the original
    # implementation log it (its DP AVG then averages identical values).
    _ORIG_TRACK_MTP_METRICS = getattr(MTPLossLoggingHelper, "track_mtp_metrics", None)
    if _ORIG_TRACK_MTP_METRICS is not None:

        def _track_mtp_metrics_token_weighted(loss_scale, *args, **kwargs):
            tracker = MTPLossLoggingHelper.tracker
            loss_sums = tracker.pop("_relax_mtp_loss_sum", None)
            token_counts = tracker.pop("_relax_mtp_token_count", None)
            if loss_sums is not None and token_counts is not None and "loss_values" in tracker:
                packed = torch.cat((loss_sums, token_counts))
                for group_key in ("reduce_group", "avg_group"):
                    group = tracker.get(group_key)
                    if group is not None:
                        torch.distributed.all_reduce(packed, group=group)
                loss_sums, token_counts = packed.chunk(2)
                values = loss_sums / token_counts.clamp(min=1)
                tracker["loss_values"] = values.to(dtype=tracker["loss_values"].dtype)
                tracker["reduce_group"] = None
                loss_scale = 1.0
            return _ORIG_TRACK_MTP_METRICS(loss_scale, *args, **kwargs)

        MTPLossLoggingHelper.track_mtp_metrics = staticmethod(_track_mtp_metrics_token_weighted)

    logger.info(
        "Relax chunked MTP loss patch applied to process_mtp_loss "
        "(multi_token_prediction + gpt_model bindings; native CE branch chunked by "
        "--sft-logits-chunk-size, gated on enable_mtp_training AND sft_chunked_logits; "
        "MTP loss logged token-weighted)."
    )

except ImportError as exc:  # pragma: no cover - Megatron not importable in some envs
    logger.warning("Relax chunked MTP loss patch not applied — Megatron import failed (%r).", exc)
