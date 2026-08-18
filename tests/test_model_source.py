# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from relax.utils import model_source as m


@pytest.fixture(autouse=True)
def reset_provider_registry(monkeypatch):
    monkeypatch.setattr(m, "_PROVIDERS", {})
    monkeypatch.setattr(m, "_FROZEN", False)


@pytest.fixture()
def arguments_module(monkeypatch):
    router_pkg = ModuleType("sglang_router")
    launch_router = ModuleType("sglang_router.launch_router")
    launch_router.RouterArgs = object
    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch_router)

    sglang_arguments = ModuleType("relax.backends.sglang.arguments")
    sglang_arguments.sglang_parse_args = lambda: None
    sglang_arguments.validate_args = lambda args: args
    monkeypatch.setitem(sys.modules, "relax.backends.sglang.arguments", sglang_arguments)

    module_name = "relax.utils.arguments"
    original_module = sys.modules.get(module_name)
    module_was_loaded = module_name in sys.modules
    utils_module = sys.modules.get("relax.utils")
    arguments_attr_was_set = utils_module is not None and hasattr(utils_module, "arguments")
    original_arguments_attr = getattr(utils_module, "arguments", None) if arguments_attr_was_set else None

    sys.modules.pop(module_name, None)
    if utils_module is not None and arguments_attr_was_set:
        delattr(utils_module, "arguments")
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if module_was_loaded:
            sys.modules[module_name] = original_module
        if utils_module is not None:
            if arguments_attr_was_set:
                setattr(utils_module, "arguments", original_arguments_attr)
            elif hasattr(utils_module, "arguments"):
                delattr(utils_module, "arguments")


def test_resolve_model_source_rejects_multiple_matches_and_freezes_registry():
    m.register_model_source_provider("a", lambda argv: m.ModelSource("s3://a/model/"))
    m.register_model_source_provider("b", lambda argv: m.ModelSource("s3://b/model/"))

    with pytest.raises(RuntimeError, match="Multiple model source providers matched: a, b"):
        m.resolve_model_source(["train.py"])
    with pytest.raises(RuntimeError, match="registry is frozen"):
        m.register_model_source_provider("c", lambda argv: None)


def test_resolve_model_source_reports_invalid_provider_result():
    m.register_model_source_provider("broken", lambda argv: "s3://bucket/model/")

    with pytest.raises(RuntimeError, match="provider 'broken' failed: expected ModelSource or None"):
        m.resolve_model_source(["train.py"])


def test_model_source_aliases_normalize_local_paths_and_ignore_empty_values():
    args = SimpleNamespace(
        model_source=m.ModelSource("s3://bucket/model/"),
        _model_source_original_hf_checkpoint="/models/original/",
    )

    assert m.is_model_source_alias(args, "/models/original")
    assert not m.is_model_source_alias(args, "")
    assert "" not in m.model_source_path_aliases(args, args.model_source.uri)
    assert "." not in m.model_source_path_aliases(args, args.model_source.uri)


def test_parse_args_passes_provider_source_without_rewriting_argv(monkeypatch, arguments_module):
    source = m.ModelSource("s3://bucket/model/", provider_name="test")
    m.register_model_source_provider("test", lambda argv: source)
    original_argv = ["train.py", "--hf-checkpoint", "/models/original", "--foo", "bar"]
    captured = {}

    def fake_parse(add_custom_arguments=None, *, model_source=None):
        captured["argv"] = list(arguments_module.sys.argv)
        captured["source"] = model_source
        return "parsed"

    monkeypatch.setattr(arguments_module.sys, "argv", original_argv)
    monkeypatch.setattr(arguments_module, "_parse_args_impl", fake_parse)
    monkeypatch.setattr(arguments_module, "_pre_parse_cli_model_source", lambda: None)

    assert arguments_module.parse_args() == "parsed"
    assert captured == {
        "argv": original_argv,
        "source": source,
    }
    assert arguments_module.sys.argv is original_argv


def test_disable_s3_model_download_skips_registered_provider(monkeypatch, arguments_module):
    called = False

    def provider(_argv):
        nonlocal called
        called = True
        return m.ModelSource("s3://bucket/model/")

    m.register_model_source_provider("test", provider)
    monkeypatch.setattr(arguments_module.sys, "argv", ["train.py", "--disable-s3-model-download"])
    monkeypatch.setattr(arguments_module, "_parse_args_impl", lambda _custom=None, *, model_source=None: model_source)

    assert arguments_module.parse_args() is None
    assert not called


def test_megatron_parse_applies_model_source_after_cli_parse(monkeypatch):
    pytest.importorskip("megatron.training.arguments")
    from relax.backends.megatron import arguments

    parsed = SimpleNamespace(
        hf_checkpoint="/models/from-cli",
        critic_train_only=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
    )
    source = m.ModelSource("s3://bucket/model/")
    monkeypatch.setattr(arguments, "_megatron_parse_args", lambda **_kwargs: parsed)
    monkeypatch.setattr(arguments, "_derive_cluster_args_from_resource", lambda _args: None)
    monkeypatch.setattr(arguments, "_set_default_megatron_args", lambda args: args)

    result = arguments.megatron_parse_args(None, skip_hf_validate=True, model_source=source)

    assert result._model_source_original_hf_checkpoint == "/models/from-cli"
    assert result.hf_checkpoint == source.uri


def test_megatron_parse_without_model_source_does_not_add_provenance(monkeypatch):
    pytest.importorskip("megatron.training.arguments")
    from relax.backends.megatron import arguments

    parsed = SimpleNamespace(
        hf_checkpoint="/models/from-cli",
        critic_train_only=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
    )
    monkeypatch.setattr(arguments, "_megatron_parse_args", lambda **_kwargs: parsed)
    monkeypatch.setattr(arguments, "_derive_cluster_args_from_resource", lambda _args: None)
    monkeypatch.setattr(arguments, "_set_default_megatron_args", lambda args: args)

    result = arguments.megatron_parse_args(None, skip_hf_validate=True)

    assert not hasattr(result, "_model_source_original_hf_checkpoint")
    assert result.hf_checkpoint == "/models/from-cli"
