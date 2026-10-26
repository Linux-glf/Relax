# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the Qwen4Exp PLE branch of the weight sync.

The n-gram table and the three hash-index buffers of a PLE layer are module *buffers*,
and the sync's enumerator lets exactly one buffer through (``expert_bias``), so they are
not in the param buckets and ``_iter_hf_params`` would never see them: a rollout would be
handed a model whose n-gram table is its construction-time zeros, with nothing in the log
to say so. ``BridgeConverter.ple_entries`` finds them in the bridge task map instead and
``iter_ple_shards`` ships them one shard at a time, because the table is 47.7 GiB per rank
on the host and neither the TP all-gather nor ``convert`` (which materialises everything a
mapping returns -- 128 shards) may touch it.

The four properties worth pinning down offline, each of them a silent failure in
production: the entry list is exactly the PLE state and is stable across ranks (it is the
order the collective shard assembly runs in); an entry the task map has lost raises
instead of shipping less; the table is assembled shard by shard, lazily; and the table
never reaches ``all_gather_param``. The mappings are fakes -- the real ones need a
checkpoint -- but they are the real code path: same ``BridgeConverter``, same
``_iter_hf_params``.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from types import ModuleType, SimpleNamespace

import pytest
import torch


# ---------------------------------------------------------------------------
# Module-level mocking: stub out megatron so we can import without a GPU/checkpoint.
# ---------------------------------------------------------------------------


class _FakeBaseMapping:
    """Stands in for ``MegatronParamMapping``: the four process groups, nothing else.

    ``collect_all_mappings`` only walks mappings of that type, so the fakes have to be
    instances of whatever ``param_mapping`` exports for the group-nulling path to run.
    """

    pp_group = "pp-group"
    _tp_group = "tp-group"
    _etp_group = "etp-group"
    ep_group = "ep-group"

    def megatron_to_hf(self, megatron_weights, megatron_module):
        raise NotImplementedError


_PP_STATE = {"size": 1}

_MEGATRON_MODULES = [
    "megatron",
    "megatron.core",
    "megatron.core.mpu",
    "megatron.core.transformer",
    "megatron.core.transformer.transformer_layer",
    "megatron.bridge",
    "megatron.bridge.models",
    "megatron.bridge.models.conversion",
    "megatron.bridge.models.conversion.param_mapping",
    "megatron.bridge.peft",
    "megatron.bridge.peft.lora",
]

_MISSING = object()
_saved: dict[str, object] = {}


def _install_megatron_stubs() -> None:
    for name in _MEGATRON_MODULES:
        _saved.setdefault(name, sys.modules.get(name, _MISSING))
        sys.modules[name] = ModuleType(name)
        sys.modules[name].__path__ = []
    sys.modules["megatron.core"].mpu = sys.modules["megatron.core.mpu"]
    sys.modules["megatron.core.mpu"].get_pipeline_model_parallel_world_size = lambda: _PP_STATE["size"]
    sys.modules["megatron.core.transformer.transformer_layer"].get_transformer_layer_offset = lambda: 0
    sys.modules["megatron.bridge.models.conversion.param_mapping"].MegatronParamMapping = _FakeBaseMapping
    sys.modules["megatron.bridge.peft.lora"].LoRAMerge = _FakeBaseMapping


def _restore_megatron_stubs() -> None:
    for name, original in _saved.items():
        if original is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


_install_megatron_stubs()
try:
    pytest.importorskip("triton")

    import torch.distributed as dist  # noqa: E402

    from relax.backends.megatron.weight_update import bridge_converter as bridge_converter_module  # noqa: E402
    from relax.backends.megatron.weight_update import hf_weight_iterator_bridge as iterator_module  # noqa: E402
    from relax.backends.megatron.weight_update.bridge_converter import BridgeConverter  # noqa: E402
finally:
    _restore_megatron_stubs()


@pytest.fixture(autouse=True)
def _megatron_stubs():
    """The fakes are only ``MegatronParamMapping`` while the stubs are in place.

    ``collect_all_mappings`` imports the class at call time, so the stubs have to be
    installed for the duration of the test, not just for the import of the module.
    """
    _install_megatron_stubs()
    _PP_STATE["size"] = 1
    # ``sys.modules`` is not enough for `mpu`: the modules under test may have been imported
    # first by another test module (with its own stubs, or with the real megatron), which is
    # the object they call. Patch what they hold.
    patched: list[tuple[object, object]] = []
    for module in (bridge_converter_module, iterator_module):
        if any(module.mpu is mpu for mpu, _ in patched):
            continue
        original = getattr(module.mpu, "get_pipeline_model_parallel_world_size", _MISSING)
        patched.append((module.mpu, original))
        module.mpu.get_pipeline_model_parallel_world_size = lambda: _PP_STATE["size"]
    yield
    for mpu, original in patched:
        if original is _MISSING:
            del mpu.get_pipeline_model_parallel_world_size
        else:
            mpu.get_pipeline_model_parallel_world_size = original
    _restore_megatron_stubs()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

_LAYER = "decoder.layers.0"
_TABLE = f"{_LAYER}.ple.ple_embedding.weight"
_INDEX_BUFFERS = [
    f"{_LAYER}.ple.ple_embedding.layer_multipliers",
    f"{_LAYER}.ple.ple_embedding.ngram_heads_offsets",
    f"{_LAYER}.ple.ple_embedding.ngram_heads_vocab_sizes",
]
# What `ple_entries()` must return for one PLE layer, in the order it returns it: the
# alphabetical one, which is what makes the shard-assembly collectives line up. Spelled
# out rather than derived, so that the test pins the order instead of restating `sorted`.
_PLE_ORDER = [
    f"{_LAYER}.ple.ple_embedding.layer_multipliers",
    f"{_LAYER}.ple.ple_embedding.ngram_heads_offsets",
    f"{_LAYER}.ple.ple_embedding.ngram_heads_vocab_sizes",
    _TABLE,
]


class FakePLEMapping(_FakeBaseMapping, Mapping):
    """The table's mapping: a lazy ``{shard_name: tensor}`` view that assembles on demand."""

    def __init__(self, num_shards: int = 5, width: int = 3, short_by: int = 0):
        self.num_shards = num_shards
        self._names = [f"{_LAYER}.ple.ple_embedding.ngram_embedding.shard_{i}.weight" for i in range(num_shards)]
        self._width = width
        self._short_by = short_by
        self.assembled: list[int] = []
        self.calls: list[tuple] = []

    def megatron_to_hf(self, megatron_weights, megatron_module):
        self.calls.append((megatron_weights, megatron_module))
        return self

    def __iter__(self):
        return iter(self._names[: self.num_shards - self._short_by])

    def __len__(self):
        return self.num_shards - self._short_by

    def __getitem__(self, name):
        index = self._names.index(name)
        self.assembled.append(index)
        return torch.full((4, self._width), float(index), dtype=torch.bfloat16)

    def assemble_shard(self, *args, **kwargs):  # the duck-typed marker that selects this route
        raise AssertionError("the fake assembles through __getitem__")


class FakeReplicatedMapping(_FakeBaseMapping):
    """A hash-index buffer's mapping: replicated, returned whole, no ``assemble_shard``."""

    def __init__(self, hf_name: str):
        self._hf_name = hf_name
        self.groups_seen = None

    def megatron_to_hf(self, megatron_weights, megatron_module):
        self.groups_seen = (self.pp_group, self._tp_group, self._etp_group, self.ep_group)
        return {self._hf_name: torch.arange(3)}


def _make_model(with_ple: bool = True) -> torch.nn.Module:
    """A model tree with the four PLE buffer names, and nothing else that matters."""
    model = torch.nn.Module()
    parent = model
    for name in ("decoder", "layers", "0", "ple", "ple_embedding"):
        child = torch.nn.Module()
        parent.add_module(name, child)
        parent = child
    ple_embedding = parent
    if with_ple:
        ple_embedding.register_buffer("weight", torch.zeros(8, 2, dtype=torch.bfloat16))
        for leaf in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
            ple_embedding.register_buffer(leaf, torch.zeros(3, dtype=torch.long))
    else:
        # An ordinary persistent buffer elsewhere in the model, and a non-persistent one
        # inside the PLE module: neither belongs to the PLE state.
        ple_embedding.register_buffer("weight_cache", torch.zeros(1), persistent=False)
        model.get_submodule("decoder.layers.0").register_buffer("not_the_table", torch.zeros(1))
    return model


def _task(mapping, param_weight=None, megatron_module=None):
    return SimpleNamespace(
        mapping=mapping,
        param_weight=param_weight if param_weight is not None else torch.zeros(8, 2, dtype=torch.bfloat16),
        megatron_module=megatron_module if megatron_module is not None else SimpleNamespace(config=None),
    )


def _ple_tasks(table_mapping=None, table_buffer=None, **overrides) -> dict[str, object]:
    """The task map a PLE model produces: the table plus its three index buffers.

    A model missing one of them is denied a task map in production (``ple_entries``
    refuses), so the fakes keep all four and drop them explicitly where that is the point.
    """
    tasks = {
        _TABLE: _task(table_mapping if table_mapping is not None else FakePLEMapping(), param_weight=table_buffer)
    }
    for name in _INDEX_BUFFERS:
        tasks[name] = _task(FakeReplicatedMapping(name))
    tasks.update(overrides)
    return tasks


def _converter(model: torch.nn.Module, tasks: dict[str, object]) -> BridgeConverter:
    """A real ``BridgeConverter`` whose task map is given, so ``init_tasks`` never runs."""
    converter = BridgeConverter(args=SimpleNamespace(vocab_size=100), model=[model], quantization_config=None)
    converter._bridge_task_map = tasks
    converter._config_map = {}
    return converter


# ---------------------------------------------------------------------------
# ple_entries: the discovered set, and the guard against losing one
# ---------------------------------------------------------------------------


def test_ple_entries_returns_the_table_and_index_buffers_sorted():
    model = _make_model()
    tasks = _ple_tasks()
    # A map is a dict: shuffled on the way in, sorted on the way out.
    tasks = dict(reversed(list(tasks.items())))
    tasks["decoder.layers.0.mlp.router.weight"] = _task(FakeReplicatedMapping("unrelated"))
    converter = _converter(model, tasks)
    assert _PLE_ORDER == sorted(_PLE_ORDER)  # the expected order is the alphabetical one
    assert converter.ple_entries() == _PLE_ORDER
    assert sorted(converter.ple_entries()) == sorted([_TABLE, *_INDEX_BUFFERS])  # and nothing else


def test_ple_entries_is_empty_for_a_model_without_ple():
    # The model here carries a *non-persistent* buffer inside the PLE module; the bridge
    # enumerates persistent buffers only, so it has no task by construction and must not be
    # demanded -- and with the four real ones absent the entry list is empty.
    model = _make_model(with_ple=False)
    converter = _converter(model, {"decoder.layers.0.mlp.router.weight": _task(FakeReplicatedMapping("router"))})
    assert converter.ple_entries() == []


def test_ple_entries_rejects_a_local_ple_buffer_the_task_map_lost():
    model = _make_model()
    tasks = _ple_tasks()
    del tasks[_TABLE]
    converter = _converter(model, tasks)
    with pytest.raises(ValueError, match="holds the Qwen4Exp PLE buffer"):
        converter.ple_entries()


def test_ple_entries_rejects_pipeline_parallelism():
    _PP_STATE["size"] = 2
    model = _make_model()
    converter = _converter(model, _ple_tasks())
    with pytest.raises(NotImplementedError, match="pipeline-model-parallel-size=2"):
        converter.ple_entries()


def test_ple_entries_does_not_rebuild_the_task_map():
    # The entry point runs once per sync and must not re-run the bridge discovery.
    model = _make_model()
    converter = _converter(model, _ple_tasks())
    assert converter._bridge_mapping_registry is None
    assert converter.ple_entries() == _PLE_ORDER


# ---------------------------------------------------------------------------
# iter_ple_shards: the table's shard-at-a-time route
# ---------------------------------------------------------------------------


def test_iter_ple_shards_assembles_the_table_one_shard_at_a_time():
    model = _make_model()
    mapping = FakePLEMapping(num_shards=5)
    buffer = torch.zeros(8, 2, dtype=torch.bfloat16)
    module = SimpleNamespace(config=None)
    converter = _converter(model, {_TABLE: _task(mapping, param_weight=buffer, megatron_module=module)})

    shards = converter.iter_ple_shards(_TABLE)
    assert mapping.assembled == []  # nothing is built before it is asked for
    first_name, first = next(shards)
    assert first_name.endswith("shard_0.weight")
    assert mapping.assembled == [0]
    assert mapping.calls == [(buffer, module)]  # the buffer and the module come from the task

    rest = list(shards)
    assert mapping.assembled == [0, 1, 2, 3, 4]
    assert [tensor[0, 0].item() for _name, tensor in rest] == [1.0, 2.0, 3.0, 4.0]
    assert first.shape == (4, 3)


def test_iter_ple_shards_refuses_a_table_mapping_that_cannot_assemble():
    model = _make_model()
    mapping = FakeReplicatedMapping(_TABLE)  # e.g. the plugin's mapping was renamed or reshaped
    converter = _converter(model, {_TABLE: _task(mapping)})
    with pytest.raises(ValueError, match="has no `assemble_shard`"):
        list(converter.iter_ple_shards(_TABLE))


def test_iter_ple_shards_refuses_a_partial_shard_series():
    model = _make_model()
    mapping = FakePLEMapping(num_shards=3, short_by=1)
    converter = _converter(model, {_TABLE: _task(mapping)})
    with pytest.raises(RuntimeError, match="handed over 2 of its 3 HF shards"):
        list(converter.iter_ple_shards(_TABLE))


def test_iter_ple_shards_refuses_an_entry_without_a_task():
    model = _make_model()
    converter = _converter(model, {})
    with pytest.raises(ValueError, match="no bridge conversion task"):
        list(converter.iter_ple_shards(_TABLE))


# ---------------------------------------------------------------------------
# iter_ple_shards: the index buffers' ordinary route
# ---------------------------------------------------------------------------


def test_iter_ple_shards_converts_an_index_buffer_with_the_groups_nulled():
    model = _make_model()
    name = _INDEX_BUFFERS[1]
    mapping = FakeReplicatedMapping(name)
    converter = _converter(model, {name: _task(mapping)})

    converted = list(converter.iter_ple_shards(name))
    assert [hf_name for hf_name, _tensor in converted] == [name]
    assert torch.equal(converted[0][1], torch.arange(3))
    # `convert`'s contract: the mapping sees no process groups (it must not gather on its
    # own here) and gets them back afterwards.
    assert mapping.groups_seen == (None, None, None, None)
    assert (mapping.pp_group, mapping._tp_group, mapping._etp_group, mapping.ep_group) == (
        "pp-group",
        "tp-group",
        "etp-group",
        "ep-group",
    )


def test_iter_ple_shards_yields_every_shard_under_its_checkpoint_name():
    # The names are the checkpoint's `shard_<i>.weight` spelling: the rollout keys the
    # table by them, and one rank's ordering is every rank's ordering.
    model = _make_model()
    mapping = FakePLEMapping(num_shards=2)
    converter = _converter(model, {_TABLE: _task(mapping)})
    assert [name for name, _tensor in converter.iter_ple_shards(_TABLE)] == [
        f"{_LAYER}.ple.ple_embedding.ngram_embedding.shard_{index}.weight" for index in range(2)
    ]


# ---------------------------------------------------------------------------
# _iter_hf_params: the two streams meet without a gather
# ---------------------------------------------------------------------------


def _bare_iterator(converter):
    """An ``HfWeightIteratorBridge`` with no buckets, so only the PLE loop is left."""
    iterator = object.__new__(iterator_module.HfWeightIteratorBridge)
    iterator.args = SimpleNamespace(update_weight_buffer_size=1024)
    iterator._bridge_converter = converter
    iterator._expert_buckets = []
    iterator._non_expert_buckets = []
    iterator._vanilla_key_map = {}
    iterator.lora_merge_mode = False
    iterator.lora_adapter_mode = False
    return iterator


def test_iter_hf_params_yields_the_ple_shards_without_gathering(monkeypatch):
    model = _make_model()
    mapping = FakePLEMapping(num_shards=3)
    buffer = torch.zeros(8, 2, dtype=torch.bfloat16)
    converter = _converter(model, _ple_tasks(table_mapping=mapping, table_buffer=buffer))
    index_mapping = converter._bridge_task_map[_INDEX_BUFFERS[0]].mapping

    gathered: list[str] = []
    monkeypatch.setattr(iterator_module, "all_gather_param", lambda *_a, **_k: gathered.append("called"))
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(iterator_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))

    iterator = _bare_iterator(converter)
    produced = list(iterator._iter_hf_params({}))

    assert gathered == []  # the table's 47.7 GiB never goes through a TP all-gather
    # Every entry of the group, in `ple_entries()` order -- the three index buffers, then the
    # table's shards under their checkpoint names.
    assert [name for name, _tensor in produced] == [
        *_INDEX_BUFFERS,
        *[f"{_LAYER}.ple.ple_embedding.ngram_embedding.shard_{index}.weight" for index in range(3)],
    ]
    assert mapping.calls[0][0] is buffer  # the task's own buffer, not a gathered copy
    assert index_mapping.groups_seen == (None, None, None, None)  # the index buffer took `convert`


def test_iter_hf_params_is_unchanged_without_ple(monkeypatch):
    model = _make_model(with_ple=False)
    converter = _converter(model, {})
    gathered: list[str] = []
    monkeypatch.setattr(iterator_module, "all_gather_param", lambda *_a, **_k: gathered.append("called"))
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(iterator_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))

    assert list(_bare_iterator(converter)._iter_hf_params({})) == []
    assert gathered == []


def test_bridge_converter_never_imports_the_gatherer():
    # The route exists precisely so that the table is never all-gathered; a future edit that
    # reaches for `all_gather_param` inside the converter would be a 47.7 GiB surprise.
    assert not hasattr(bridge_converter_module, "all_gather_param")
