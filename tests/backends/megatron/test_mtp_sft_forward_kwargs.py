# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""MTP training forwards must keep labels and loss masks aligned."""

from argparse import Namespace

import pytest
import torch
from megatron.core.packed_seq_params import PackedSeqParams


try:
    from relax.backends.megatron.model import _attach_mtp_forward_kwargs
except (ImportError, AssertionError) as _exc:
    pytest.skip(f"relax.backends.megatron.model unavailable: {_exc}", allow_module_level=True)


def _mk_args(enable_mtp_training: bool) -> Namespace:
    return Namespace(enable_mtp_training=enable_mtp_training)


def test_attach_mtp_forward_kwargs_noop_when_disabled():
    forward_kwargs = {"loss_mask": None}
    original = forward_kwargs.copy()

    _attach_mtp_forward_kwargs(_mk_args(enable_mtp_training=False), {}, forward_kwargs)

    assert forward_kwargs == original


def test_attach_mtp_forward_kwargs_preserves_existing_loss_mask():
    tokens = torch.tensor([[1, 2, 3, 4]])
    existing_loss_mask = object()
    full_loss_masks = object()
    batch = {"tokens": tokens, "full_loss_masks": full_loss_masks}
    forward_kwargs = {"loss_mask": existing_loss_mask}

    _attach_mtp_forward_kwargs(_mk_args(enable_mtp_training=True), batch, forward_kwargs)

    torch.testing.assert_close(forward_kwargs["mtp_kwargs"]["mtp_labels"], torch.tensor([[2, 3, 4, 0]]))
    assert forward_kwargs["loss_mask"] is existing_loss_mask


def test_attach_mtp_forward_kwargs_restores_bridge_unsplit_loss_mask():
    tokens = torch.tensor([[1, 2, 3, 4]])
    full_loss_masks = object()
    batch = {"tokens": tokens, "full_loss_masks": full_loss_masks}
    forward_kwargs = {"loss_mask": None}

    _attach_mtp_forward_kwargs(_mk_args(enable_mtp_training=True), batch, forward_kwargs)

    torch.testing.assert_close(forward_kwargs["mtp_kwargs"]["mtp_labels"], torch.tensor([[2, 3, 4, 0]]))
    assert forward_kwargs["loss_mask"] is full_loss_masks


def test_attach_mtp_forward_kwargs_preserves_pre_shifted_vl_cp_labels():
    local_labels = torch.tensor([[2, 3, 4, 0]])
    local_mask = torch.tensor([[1, 1, 0, 0]])
    batch = {
        "tokens": torch.tensor([[1, 2, 3, 4]]),
        "full_loss_masks": torch.ones(1, 4),
        "unsplit_mtp_labels": local_labels,
        "unsplit_mtp_loss_mask": local_mask,
    }
    forward_kwargs = {"loss_mask": None}

    _attach_mtp_forward_kwargs(_mk_args(enable_mtp_training=True), batch, forward_kwargs)

    assert forward_kwargs["mtp_kwargs"]["mtp_labels"] is local_labels
    assert forward_kwargs["loss_mask"] is local_mask


def test_attach_mtp_forward_kwargs_stops_shift_at_packed_document_boundary():
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=3,
        max_seqlen_kv=3,
    )
    batch = {"tokens": tokens, "packed_seq_params": packed, "full_loss_masks": torch.ones_like(tokens)}
    forward_kwargs = {"loss_mask": None}

    _attach_mtp_forward_kwargs(_mk_args(enable_mtp_training=True), batch, forward_kwargs)

    torch.testing.assert_close(forward_kwargs["mtp_kwargs"]["mtp_labels"], torch.tensor([[2, 3, 0, 5, 6, 0]]))
