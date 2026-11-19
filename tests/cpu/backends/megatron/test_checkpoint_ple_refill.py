# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The native DCP resume path must refill a frozen PLE table."""

import torch

from relax.backends.megatron.checkpoint import _refill_frozen_ple_tables


class _TableModule(torch.nn.Module):
    def __init__(self, *, trainable: bool, loaded: bool = False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=trainable)
        self.table_path = "/tmp/ple-table"
        self._table_loaded = loaded
        self.calls = 0

    def load_table(self):
        self.calls += 1
        self._table_loaded = True


def test_refill_frozen_ple_table_after_native_resume():
    frozen = _TableModule(trainable=False)
    chunk = torch.nn.Module()
    chunk.add_module("ple", frozen)

    _refill_frozen_ple_tables([chunk])
    assert frozen.calls == 1
    assert frozen._table_loaded is True


def test_refill_skips_trainable_and_already_loaded_tables():
    trainable = _TableModule(trainable=True)
    loaded = _TableModule(trainable=False, loaded=True)
    chunk = torch.nn.Module()
    chunk.add_module("trainable", trainable)
    chunk.add_module("loaded", loaded)

    _refill_frozen_ple_tables([chunk])
    assert trainable.calls == 0
    assert loaded.calls == 0
