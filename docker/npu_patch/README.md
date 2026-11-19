# NPU patches (26.2.0 / core_r0.18.0)

Target versions and commit pins are the `ARG *_COMMIT` values in `Dockerfile.npu`.

| Patch | Target repository | Applied at image build |
|---|---|---|
| `megatron.patch` | Megatron-LM core_v0.18.0 | 4 required items: torch_memory_saver, Adam-step restore, local l2norm, training args fallback; plus `nvrx.py` compatibility for nvidia-resiliency-ext 0.4.1 (no `__version__`) |
| `megatron-bridge.patch` | Megatron-Bridge v0.5.0 | OmegaConf/remove_non_pickleables fix + qwen3_asr registration compatibility with transformers 5.16.1 |
| `megatron-adaptor.patch` | MegatronAdaptor 26.2.0_core_r0.18.0 | profile argparse duplicate-registration guard |
| `transformer-engine-npu.patch` | TransformerEngineNPU 26.2.0_release_v2.17 | ring-attention write-back fast path |
| `mindspeed-ops.patch` | MindSpeed-Ops 26.2.0 | GDN chunk-index cache and NT de-specialization |
| `sglang-npu.patch` | sglang v0.5.15 | 920 patch + `qwen3_asr` AutoConfig registration compatibility with transformers 5.16.1 |
| `sgl-kernel-npu.patch` | sgl-kernel-npu 2026.7.2 | unchanged from the 920 image |
| `hyper-multicore-ready-init.patch` | hyper-multicore 6236275 snapshot | ready-state initialization race fix |

MindSpeed-Bridge is not patched from this directory. Its Qwen4Exp feature
series lives on the `qwen4_exp_npu` branch of
https://gitcode.com/ascend/MindSpeed-Bridge (push the local branch before
building the image) and `Dockerfile.npu` pins the branch head through
`MINDSPEED_BRIDGE_COMMIT`.

`mindspeed-bridge.patch`, `mindspeed-bridge-qwen4-exp.patch`, `mindspeed.patch` and
`torch-memory-saver.patch` are intentionally not present in this NPU stack.

Qwen4 custom operators are built from `ops-transformer feature/qwen4_exp`
(`380a837b`): QSA (`qwen38_qsa`) and mHC (`mhc_tgr`) are separate vendor run
packages; see `Dockerfile.npu`.

The final image pins `transformers==5.16.1` and `tokenizers==0.23.3`
(transformers 5.16.1 requires `tokenizers>=0.23.1,<0.24.0`).
