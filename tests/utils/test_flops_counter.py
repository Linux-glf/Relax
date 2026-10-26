# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math
import sys
import types

import pytest

from relax.utils.training.flops_counter import (
    _DEVICE_FLOPS,
    FlopsCounter,
    _count_hybrid_layer_types,
    _qwen4_exp_gdn_layer_n,
    _qwen4_exp_hyper_connection_n,
    _qwen4_exp_moe_layer_n,
    _qwen4_exp_ple_layer_n,
    _qwen4_exp_qsa_layer_n,
    get_device_peak_flops,
)


# ---------------------------------------------------------------------------
# Helper: lightweight config object (mirrors verl test pattern)
# ---------------------------------------------------------------------------
class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


# ---------------------------------------------------------------------------
# GPU peak FLOPS tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "device_name, expected_raw_flops",
    [
        ("NVIDIA H100 80GB HBM3", 989e12),
        ("NVIDIA H800", 989e12),
        ("NVIDIA A100-SXM4-80GB", 312e12),
        ("NVIDIA A800-SXM4-80GB", 312e12),
        ("NVIDIA L40S", 362.05e12),
        ("NVIDIA H20", 148e12),
        ("Ascend910B3", 354e12),
    ],
)
def test_get_device_peak_flops_known_gpus(device_name, expected_raw_flops):
    result = get_device_peak_flops(unit="T", device_name=device_name)
    expected_tflops = expected_raw_flops / 1e12
    assert math.isclose(result, expected_tflops, rel_tol=1e-6), (
        f"Expected {expected_tflops} TFLOPS for {device_name}, got {result}"
    )


def test_get_device_peak_flops_unknown_gpu():
    result = get_device_peak_flops(unit="T", device_name="SomeUnknownGPU-XYZ")
    assert result == float("inf")


def test_get_device_peak_flops_defaults_to_cpu_when_device_properties_unavailable(monkeypatch):
    device_module = types.ModuleType("relax.utils.device")

    def get_device_properties():
        raise AttributeError("module 'torch.cpu' has no attribute 'get_device_properties'")

    device_module.device_module = types.SimpleNamespace(get_device_properties=get_device_properties)
    monkeypatch.setitem(sys.modules, "relax.utils.device", device_module)

    result = get_device_peak_flops(unit="T")

    assert math.isclose(result, _DEVICE_FLOPS["CPU"] / 1e12, rel_tol=1e-6)


def test_get_device_peak_flops_unit_conversion():
    tflops = get_device_peak_flops(unit="T", device_name="NVIDIA H100")
    gflops = get_device_peak_flops(unit="G", device_name="NVIDIA H100")
    assert math.isclose(tflops * 1000, gflops, rel_tol=1e-6)


def test_device_flops_table_has_common_gpus():
    common_keys = ["H100", "H800", "A100", "A800", "H20"]
    for key in common_keys:
        assert key in _DEVICE_FLOPS, f"Missing common GPU {key} in _DEVICE_FLOPS"


# ---------------------------------------------------------------------------
# FlopsCounter model FLOPS tests (expected values from verl test suite)
# ---------------------------------------------------------------------------
FLOPS_TEST_CONFIGS = {
    "qwen3_dense": {
        "config": {
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [180997438046208 / 1e12, 648394032807936 / 1e12],
    },
    "qwen3_moe": {
        "config": {
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "vocab_size": 151936,
            "num_hidden_layers": 48,
            "num_key_value_heads": 4,
            "num_attention_heads": 32,
            "head_dim": 128,
            "moe_intermediate_size": 768,
            "num_experts_per_tok": 8,
            "num_experts": 128,
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [78593069678592 / 1e12, 306570470621184 / 1e12],
    },
    "deepseek_v3": {
        "config": {
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "vocab_size": 129280,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 61,
            "first_k_dense_replace": 3,
            "num_attention_heads": 128,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "intermediate_size": 18432,
            "qk_nope_head_dim": 128,
            "q_lora_rank": 1536,
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [848766538088448 / 1e12, 3145850406567936 / 1e12],
    },
}


FLOPS_TEST_VL_CONFIGS = {
    "qwen3_5": {
        "config": {
            "model_type": "qwen3_5",
            "text_config": {
                "vocab_size": 248320,
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 32,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "layer_types": ["linear_attention" if bool((i + 1) % 4) else "full_attention" for i in range(32)],
            },
            "vision_config": {
                "num_heads": 16,
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "out_hidden_size": 4096,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "patch_size": 16,
            },
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "images_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [206090394402816 / 1e12, 724521757704192 / 1e12],
    },
    "qwen3_5_moe": {
        "config": {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "vocab_size": 248320,
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
                "num_experts_per_tok": 8,
                "num_experts": 256,
                "layer_types": ["linear_attention" if bool((i + 1) % 4) else "full_attention" for i in range(40)],
            },
            "vision_config": {
                "num_heads": 16,
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "out_hidden_size": 2048,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "patch_size": 16,
            },
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "images_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [88082762170368 / 1e12, 321470349705216 / 1e12],
    },
}

# Qwen3.8-Flash-Next. The reference 4-layer checkpoint (the small adaptation
# target) is the smallest config that exercises every term of the estimator at
# once: one QSA layer + three GDN layers, one PLE layer, hyper-connections, the
# MTP head. Values are the reference's own `config.json`; the expected TFLOPS are
# the arithmetic done longhand from the checkpoint's tensor shapes (see
# `test_qwen4_exp_layer_terms_match_reference_checkpoint_shapes` below), not by
# reusing the estimator.
FLOPS_TEST_QWEN4_EXP_CONFIGS = {
    "qwen4_exp_4l": {
        "config": {
            "model_type": "qwen4_exp",
            "vocab_size": 248320,
            "hidden_size": 2560,
            "num_hidden_layers": 4,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "num_experts_per_tok": 10,
            "num_experts": 512,
            "hc_count": 4,
            "hc_lowrank": 320,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_compress_ratio": 4,
            "indexer_budget": 2048,
            "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 4,
            "ple_layer_ids": [2],
            "mtp_num_hidden_layers": 1,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        },
        "batch_seqlens_list": [[512, 1024, 2048], [4096, 4096, 4096]],
        "expected_tflops_list": [55933565140992 / 1e12, 192479533989888 / 1e12],
    },
}

FLOPS_TEST_OMNI_CONFIGS = {
    "qwen3_omni_moe": {
        "config": {
            "model_type": "qwen3_omni_moe",
            "thinker_config": {
                "text_config": {
                    "hidden_size": 2048,
                    "vocab_size": 3584,
                    "num_hidden_layers": 28,
                    "num_key_value_heads": 4,
                    "num_attention_heads": 28,
                    "head_dim": 128,
                    "moe_intermediate_size": 768,
                    "num_experts_per_tok": 8,
                    "num_experts": 128,
                },
                "vision_config": {
                    "num_heads": 16,
                    "depth": 27,
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "out_hidden_size": 3584,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "in_channels": 3,
                    "patch_size": 16,
                    "deepstack_visual_indexes": [8, 16, 24],
                },
                "audio_config": {
                    "d_model": 1280,
                    "num_hidden_layers": 32,
                    "encoder_attention_heads": 20,
                    "encoder_ffn_dim": 5120,
                    "num_mel_bins": 128,
                    "output_dim": 3584,
                    "n_window": 100,
                },
            },
        },
        "batch_seqlens": [512, 1024, 2048],
        "images_seqlens": [512, 1024],
        "audio_seqlens": [500, 1000],
        "expected_tflops": 43060677083136 / 1e12,
    },
}


@pytest.mark.parametrize("config_name", list(FLOPS_TEST_CONFIGS.keys()))
def test_flops_counter_model_estimation(config_name):
    test_data = FLOPS_TEST_CONFIGS[config_name]
    config = Config(test_data["config"])
    counter = FlopsCounter(config)

    for batch_seqlens, expected_tflops in zip(
        test_data["batch_seqlens_list"], test_data["expected_tflops_list"], strict=True
    ):
        # delta_time=1 so returned value = raw TFLOPS
        estimated_tflops, _ = counter.estimate(batch_seqlens, delta_time=1.0)
        assert math.isclose(estimated_tflops, expected_tflops, rel_tol=1e-6), (
            f"{config_name}: expected {expected_tflops:.2f} TFLOPS, got {estimated_tflops:.2f}"
        )


@pytest.mark.parametrize("config_name", list(FLOPS_TEST_VL_CONFIGS.keys()))
def test_flops_counter_vl_estimation(config_name):
    test_data = FLOPS_TEST_VL_CONFIGS[config_name]
    config = Config(test_data["config"])
    counter = FlopsCounter(config)

    for batch_seqlens, images_seqlens, expected_tflops in zip(
        test_data["batch_seqlens_list"],
        test_data["images_seqlens_list"],
        test_data["expected_tflops_list"],
        strict=True,
    ):
        estimated_tflops, _ = counter.estimate(batch_seqlens, delta_time=1.0, images_seqlens=images_seqlens)
        assert math.isclose(estimated_tflops, expected_tflops, rel_tol=1e-6), (
            f"{config_name}: expected {expected_tflops:.2f} TFLOPS, got {estimated_tflops:.2f}"
        )


@pytest.mark.parametrize("config_name", list(FLOPS_TEST_QWEN4_EXP_CONFIGS.keys()))
def test_flops_counter_qwen4_exp_estimation(config_name):
    test_data = FLOPS_TEST_QWEN4_EXP_CONFIGS[config_name]
    config = Config(test_data["config"])
    counter = FlopsCounter(config)

    for batch_seqlens, expected_tflops in zip(
        test_data["batch_seqlens_list"], test_data["expected_tflops_list"], strict=True
    ):
        estimated_tflops, peak_tflops = counter.estimate(batch_seqlens, delta_time=1.0)
        assert math.isclose(estimated_tflops, expected_tflops, rel_tol=1e-6), (
            f"{config_name}: expected {expected_tflops:.6f} TFLOPS, got {estimated_tflops:.6f}"
        )
        # An unsupported model_type falls back to the dense formula (or to zero)
        # with a warning instead of raising, so pin the dispatch itself: a peak of
        # 0.0 here means the estimate never reached a real handler.
        assert peak_tflops > 0


@pytest.mark.parametrize("config_name", list(FLOPS_TEST_QWEN4_EXP_CONFIGS.keys()))
def test_flops_counter_qwen4_exp_layer_terms_match_reference_checkpoint_shapes(config_name):
    """Every per-layer term equals the sum of that layer's weight shapes.

    Frozen from ``Qwen3.8-Flash-Next-4L-ple``'s safetensors headers, so this
    fails if a term drifts from what the checkpoint actually stores -- the
    end-to-end TFLOPS above would keep passing if two drifted terms cancelled.
    """
    config = Config(FLOPS_TEST_QWEN4_EXP_CONFIGS[config_name]["config"])
    expected = {
        "gdn": 57958400,  # in_proj + conv1d + out_proj
        "qsa": 51445760,  # q/k/v/o_proj + indexer q/k/v_proj
        "moe": 55380480,  # 10 routed experts + router + shared expert
        "hyper_connection": 13189120,  # 2 mixers x (down/lowrank + up/lowrank + gate)
        "ple": 32808960,  # key_proj + ple(-> hidden) + conv
    }
    actual = {
        "gdn": _qwen4_exp_gdn_layer_n(config),
        "qsa": _qwen4_exp_qsa_layer_n(config),
        "moe": _qwen4_exp_moe_layer_n(config),
        "hyper_connection": _qwen4_exp_hyper_connection_n(config),
        "ple": _qwen4_exp_ple_layer_n(config),
    }
    assert actual == expected


def test_flops_counter_mtp_num_layers_overrides_the_config():
    """The MTP depth counted is the one trained, not the one the config declares.

    The 4-layer checkpoint ships an MTP head, so a run with ``--mtp-num-layers 0``
    would otherwise be credited for a head it never builds -- ~30% of this
    model's FLOPs, and the same factor on its MFU.
    """
    config = Config(FLOPS_TEST_QWEN4_EXP_CONFIGS["qwen4_exp_4l"]["config"])
    batch_seqlens = [512, 1024, 2048]

    with_mtp, _ = FlopsCounter(config, mtp_num_layers=1).estimate(batch_seqlens, delta_time=1.0)
    without_mtp, _ = FlopsCounter(config, mtp_num_layers=0).estimate(batch_seqlens, delta_time=1.0)
    from_config, _ = FlopsCounter(config).estimate(batch_seqlens, delta_time=1.0)

    # The config's own depth (one layer) is the default when the caller has no opinion.
    assert from_config == with_mtp
    assert without_mtp < with_mtp
    # The difference is exactly the MTP term: 6 * N_mtp * tokens.
    assert math.isclose(with_mtp - without_mtp, 6 * 775375360 * sum(batch_seqlens) / 1e12, rel_tol=1e-9)


def test_count_hybrid_layer_types_treats_renamed_sparse_attention_as_full():
    """A non-linear layer counts as full attention whatever it is called.

    transformers' ``Qwen4ExpConfig`` rewrites the reference's
    ``"full_attention"`` into ``"qwen_sparse_attention"``. Counting by exact
    match silently reports zero full-attention layers and drops them from the
    estimate -- 12 of 48 layers for the full model, and the MFU of a whole run
    with it.
    """
    base = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=2,
    )
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    renamed = ["linear_attention", "linear_attention", "linear_attention", "qwen_sparse_attention"]

    assert _count_hybrid_layer_types(Config({**base, "layer_types": layer_types})) == (1, 3)
    assert _count_hybrid_layer_types(Config({**base, "layer_types": renamed})) == (1, 3)
    # No layer_types at all: fall back to the 1-in-N interval, counted from the
    # config's num_hidden_layers.
    assert _count_hybrid_layer_types(Config({**base, "num_hidden_layers": 8})) == (2, 6)


@pytest.mark.parametrize("config_name", list(FLOPS_TEST_OMNI_CONFIGS.keys()))
def test_flops_counter_omni_estimation(config_name):
    test_data = FLOPS_TEST_OMNI_CONFIGS[config_name]
    config = Config(test_data["config"])
    counter = FlopsCounter(config)

    estimated_tflops, _ = counter.estimate(
        test_data["batch_seqlens"],
        delta_time=1.0,
        images_seqlens=test_data["images_seqlens"],
        audio_seqlens=test_data["audio_seqlens"],
    )
    assert math.isclose(estimated_tflops, test_data["expected_tflops"], rel_tol=1e-6), (
        f"{config_name}: expected {test_data['expected_tflops']:.2f} TFLOPS, got {estimated_tflops:.2f}"
    )


def test_mfu_calculation():
    peak_tflops = get_device_peak_flops(unit="T", device_name="NVIDIA H100 80GB HBM3")
    achieved_tflops = 200.0
    mfu = achieved_tflops / peak_tflops
    assert 0 < mfu < 1
    assert math.isclose(mfu, 200.0 / 989.0, rel_tol=1e-6)


def test_unknown_model_type_fallback():
    config = Config(
        {
            "model_type": "unknown_model_xyz",
            "hidden_size": 4096,
            "vocab_size": 151936,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }
    )
    counter = FlopsCounter(config)
    estimated, _ = counter.estimate([1024, 2048], delta_time=1.0)
    assert estimated > 0


def test_unknown_model_type_missing_fields_returns_zero():
    config = Config({"model_type": "unknown_model_xyz"})
    counter = FlopsCounter(config)
    estimated, _ = counter.estimate([1024, 2048], delta_time=1.0)
    assert estimated == 0
