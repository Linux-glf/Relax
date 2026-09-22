# FSDPTurbo SFT 训练

本页介绍 FSDPTurbo SFT 训练组件——一个使用 `torchrun` 直接驱动 FSDPTurbo 的 `BaseTrainer` 的启动脚本，绕过 Relax 的 Ray Serve 编排。它专为从 FSDP2 + 专家并行中受益的大规模多模态 MoE 模型（如 Qwen3.8-Flash-Next）设计。

### 适用场景

| 场景 | 使用本脚本 | 使用标准 Relax SFT |
| --- | --- | --- |
| FSDP2 后端（非 Megatron） | 是 | 否 |
| Megatron-LM TP/PP/CP 后端 | 否 | 是 |
| 需要 Ray Serve 编排 | 否 | 是 |
| MoE + FSDP2 + EP 组合 | 是 | 否 |
| PLE n-gram embedding CPU 卸载 | 是 | 否 |

## 启动脚本

脚本 `run-qwen3.8-flash-next-fsdpturbo-sft-64xgpu.sh` 按以下顺序执行：

1. **Source NPU 环境**——若存在 `/usr/local/Ascend/ascend-toolkit/set_env.sh`（Ascend NPU），则设置 `HCCL_CONNECT_TIMEOUT`、`ACLNN_CACHE_LIMIT`、`PYTORCH_NPU_ALLOC_CONF` 等。CUDA 环境下跳过。
2. **解析路径**——定位 `FSDPTURBO_DIR`（默认 `/data2/m00659926/FSDPTurbo`）、`train.py` 和 `config.yaml`。
3. **读取集群配置**——从环境变量读取 `NNODES`、`GPUS_PER_NODE`、`MASTER_ADDR`、`MASTER_PORT`。
4. **层裁剪**——`NUM_HIDDEN_LAYERS=2`（冒烟测试）或 `full`（全部 48 层）。
5. **数据路径覆盖**——若设置了 `DATA_PATH`，用 `sed` 创建临时配置副本。
6. **通过 `torchrun` 启动**——以 `--config` 和额外参数启动 `train.py`，输出同时写入带时间戳的日志文件。

### 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `NNODES` | `4` | 节点数 |
| `GPUS_PER_NODE` | `16` | 每节点卡数（GPU/NPU） |
| `MASTER_ADDR` | `127.0.0.1` | rank-0 节点 IP（多节点必填） |
| `MASTER_PORT` | `29500` | torchrun 会合端口 |
| `NUM_HIDDEN_LAYERS` | `2` | 语言模型层数；`2` 冒烟测试，`full` 全部 48 层 |
| `DATA_PATH` | _(未设置)_ | 覆盖 config.yaml 中的 `dataset_path` |
| `CONFIG_FILE` | `${FSDPTURBO_DIR}/examples/qwen3_8/config.yaml` | 覆盖配置文件路径 |
| `FSDPTURBO_DIR` | `/data2/m00659926/FSDPTurbo` | FSDPTurbo 仓库根目录 |
| `WORK_DIR` | `${RELAX_ROOT}/exps/fsdp_turbo/qwen3_8` | 输出与日志目录 |
| `ULYSSES_SIZE` | _(未设置)_ | 覆盖 Ulysses 上下文并行大小 |

### 冒烟测试

单节点、8 卡、2 层：

```bash
NNODES=1 GPUS_PER_NODE=8 NUM_HIDDEN_LAYERS=2 \
  bash scripts/training/sft/run-qwen3.8-flash-next-fsdpturbo-sft-64xgpu.sh
```

### 全量规模

默认集群：4 节点 × 16 卡 = 64（Ascend910 NPU）：

```bash
NNODES=4 GPUS_PER_NODE=16 NUM_HIDDEN_LAYERS=full \
  bash scripts/training/sft/run-qwen3.8-flash-next-fsdpturbo-sft-64xgpu.sh
```

## 配置（`config.yaml`）

### 模型

| 键 | 默认值 | 说明 |
| --- | --- | --- |
| `model.model_name_or_path` | `/path/to/Qwen3.8-Flash-Next` | HuggingFace 模型路径 |
| `model.torch_dtype` | `bf16` | 模型精度 |
| `model.num_hidden_layers` | `null` | `null` = 全部 48 层；设置 2–4 用于调试 |
| `model.mtp_num_layers` | `1` | MTP 层数；`0` = 禁用 |

### 分布式

| 键 | 默认值 | 说明 |
| --- | --- | --- |
| `distributed.fully_shard_parallel_size` | `16` | FSDP2 full-shard 组大小 |
| `distributed.tensor_parallel_size` | `1` | TP（默认关闭） |
| `distributed.ulysses_parallel_size` | `1` | Ulysses CP（默认关闭） |
| `distributed.expert_parallel_size` | `16` | EP 组大小（512 专家 / 8 = 每卡 64） |
| `distributed.vocab_parallel_size` | _(注释)_ | PLE embedding 的 VP；取消注释以保留在 GPU 上 |

### 内存

| 键 | 默认值 | 说明 |
| --- | --- | --- |
| `memory.recompute` | `true` | 梯度检查点 |
| `memory.recompute_plan` | visual blocks + language layers | 需要重计算的模块 |
| `memory.chunk_batch` | `false` | 分块 loss 计算 |
| `memory.swap_activation` | `false` | 激活值卸载到 CPU |

## 模型架构（Qwen3.8-Flash-Next）

| 属性 | 值 |
| --- | --- |
| 总层数 | 48 |
| 注意力类型 | 12 层 full-attn（每 4 层）+ 36 层 linear-attn（GDN） |
| 专家数 | 512（top-10 路由） |
| 隐藏层大小 | 2560 |
| PLE n-gram embedding | ~51B 参数（128 分片，默认 CPU 卸载） |
| 权重大小 | 336 GB（bf16，131 个 safetensors 分片） |
| MTP | 1 层（Multi-Token Prediction） |

## 输出与日志

所有输出写入 `WORK_DIR`（默认 `${RELAX_ROOT}/exps/fsdp_turbo/qwen3_8`）：

```
exps/fsdp_turbo/qwen3_8/
├── logs/
│   └── qwen38_fsdp_20260922_143000.log   # 带时间戳的训练日志
├── output/qwen3_8_flash_next/             # 检查点（按 save_steps 间隔）
└── config_override.yaml                    # 临时配置（若设置了 DATA_PATH）
```
