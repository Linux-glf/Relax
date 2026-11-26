# NPU patches (26.2.0 / core_r0.18.0)

Target versions and commit pins are the `ARG *_COMMIT` values in `Dockerfile.npu`.

| Patch                              | Target repository                                  | Applied at image build                                                                                                                                                                                                                      |
| ---------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `megatron.patch`                   | Megatron-LM core_v0.18.0                           | 5 required items: torch_memory_saver, Adam-step restore, local l2norm, training args fallback, MTP `mtp_kwargs` routing into core_v0.18 `process_mtp_loss`; plus `nvrx.py` compatibility for nvidia-resiliency-ext 0.4.1 (no `__version__`) |
| `megatron-bridge.patch`            | Megatron-Bridge v0.5.0                             | OmegaConf/remove_non_pickleables fix + qwen3_asr registration compatibility with transformers 5.16.1 + Qwen3VL `mtp_kwargs` pass-through to `_postprocess`                                                                                  |
| `megatron-adaptor.patch`           | MegatronAdaptor 26.2.0_core_r0.18.0                | profile argparse duplicate-registration guard                                                                                                                                                                                               |
| `transformer-engine-npu.patch`     | TransformerEngineNPU 26.2.0_release_v2.17          | ring-attention write-back fast path                                                                                                                                                                                                         |
| `mindspeed-ops.patch`              | MindSpeed-Ops 26.2.0                               | GDN chunk-index cache and NT de-specialization                                                                                                                                                                                              |
| `sglang-npu.patch`                 | sglang `eef42710c4507a506b`                        | Relax runtime compatibility only: NPU availability probe, `torch_memory_saver` `cuda_graph` API compatibility, health-check sampling parameters                                                                                             |
| `sgl-kernel-npu.patch`             | sgl-kernel-npu 2026.11.0.post4                     | unchanged from the 920 image                                                                                                                                                                                                                |
| `hyper-multicore-ready-init.patch` | hyper-multicore 530be72 (`moonep` branch snapshot) | ready-state initialization race fix                                                                                                                                                                                                         |

The Qwen3.5 9B/35B SGLang optimization series from the v0.5.15 image is not
carried to SGLang `eef42710c4507a506b`: that release includes native
Qwen3.8-Flash-Next NPU support, while the old Qwen3.5 model code was
refactored upstream. `sglang-npu.patch` keeps only model-independent Relax
runtime compatibility changes.

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
