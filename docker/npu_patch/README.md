# NPU patches

Each `*.patch` here is applied by `docker/Dockerfile.npu` to a checkout pinned at
the matching `*_COMMIT` build arg, with `patch -p1` (then committed as "base
line" so the rest of the image build sees a clean tree):

| Patch | Applied to | Build arg |
| --- | --- | --- |
| `megatron-bridge.patch` | `/root/Megatron-Bridge` | `MEGATRON_BRIDGE_COMMIT` |
| `megatron.patch` | `/root/Megatron-LM` | `MEGATRON_COMMIT` |
| `mindspeed-bridge.patch` | `/root/MindSpeed-Bridge` | `MINDSPEED_BRIDGE_COMMIT` |
| `mindspeed-bridge-qwen4-exp.patch` | `/root/MindSpeed-Bridge` | `MINDSPEED_BRIDGE_COMMIT` |
| `mindspeed-ops.patch` | `/root/MindSpeed-Ops` | `MINDSPEED_OPS_COMMIT` |
| `megatron-adaptor.patch` | `/root/MegatronAdaptor` | `MEGATRON_ADAPTOR_COMMIT` |
| `transformer-engine-npu.patch` | `/root/TransformerEngineNPU` | `TRANSFORMER_ENGINE_NPU_COMMIT` |
| `sglang-npu.patch` | `/root/sglang` | `git am` after the pinned cherry-pick |
| `sgl-kernel-npu.patch` | `/root/sgl-kernel-npu` | `git am` after the submodule checkout |

`docker/Dockerfile.npu` and `.github/workflows/integration-test.yml` both hash
`docker/npu_patch/**`, so any edit here invalidates the cached image layers.

## How these are generated

Each patch is a diff of the **live** worktree of its target repo against that
repo's pinned commit, so the set as a whole is a byte-for-byte description of the
environment the runs were measured in: applying all nine to pristine checkouts
at the pins reproduces the live trees exactly. Keep it that way -- a patch that
is merely *applicable* can still be stale, and then the image stops reproducing
what was actually run.

## MindSpeed-Bridge is split in two

`mindspeed-bridge.patch` and `mindspeed-bridge-qwen4-exp.patch` are diffs of the
same live tree, split by path so the Qwen3.8 adaptation is reviewable on its own.

The qwen4_exp patch carries 37 files: the `mindspeed_bridge/models/qwen4_exp/**`
tree (34 files, including the two `tests/__init__.py` and
`tests/make_ple_table_variant.py`, the offline builder for the shrunken-table
checkpoint) plus the three feature-wiring files through which the trainable-table
feature is registered and the multi-tensor-scale fix is measured against the
table's gradient shard:

```
mindspeed_bridge/models/qwen4_exp/**                       (34 files)
mindspeed_bridge/features/feature_list.py
mindspeed_bridge/features/megatron/common/fast_multi_tensor_scale.py
mindspeed_bridge/features/megatron/common/sparse_table_adam.py
```

The base patch carries the remaining 10 files, which belong to the concurrent
GDN/CP workstream that shares the same worktree:

```
mindspeed_bridge/features/data/vlm_datasets/shuffle_control.py
mindspeed_bridge/features/megatron/common/fla_npu_causal_conv1d.py
mindspeed_bridge/features/megatron/common/gdn_cp_a2a_fast_perm.py
mindspeed_bridge/features/megatron/common/gdn_packed_seq_host_cache.py
mindspeed_bridge/features/megatron/moe_permute_masked_select.py
mindspeed_bridge/features/megatron/qwen_vl/gdn.py
mindspeed_bridge/models/glm_vl/glm53_flash_bridge.py
mindspeed_bridge/models/kimi/kimi_k3_bridge.py
mindspeed_bridge/models/qwen_vl/modelling_qwen3_vl/gated_delta_net_old.py
mindspeed_bridge/models/qwen_vl/qwen35_vl_provider.py
```

None of them is imported by qwen4_exp -- that tree vendors its own GDN and CP
helpers under `modelling_qwen4_exp/` -- so they stay out of the Qwen3.8
adaptation. Applying the base patch first and the qwen4_exp patch second
reproduces the live tree.

`megatron-bridge.patch` is still required: `remove_non_pickleables` is called
when a mapping broadcasts a module config (qwen4_exp uses `QKVMapping`), and the
pinned bridge rejects OmegaConf containers with "Unions of containers are not
supported" without this patch.

It used to carry a second file, `src/megatron/bridge/peft/utils.py`
(`relax.utils.device.is_npu_available` gating for `MixedFusedLayerNorm` /
`HAVE_APEX`), which is **not** present in the live tree and so has never run.
Regenerating from the live tree dropped it; it remains recoverable from this
file's git history.

`mindspeed-ops.patch` raises the `prepare_chunk_indices` LRU cache from 3 to 16
entries. It is a GDN-side tuning independent of the triton-ascend version, so it
stays with the 3.6.0 upgrade.

## Checking consistency

```bash
bash scripts/tools/verify_npu_patches.sh            # all nine patches, ~1 min
bash scripts/tools/verify_npu_patches.sh sglang     # one repo
```

For each patch it makes a detached worktree at that repo's pin, applies the patch
there (`patch -p1`, or `git am` for the two mbox ones), and `cmp`s every file the
patch touches against its live counterpart -- plus the reverse direction, that no
live modification vs the pin was left out of the patch. The worktrees live under
`$WT_ROOT` (default `/data/lazypig/patchverify`, not `/tmp`: the root filesystem
runs close to full) and are removed afterwards, on failure too.

The one-liner version, per patch:

```bash
git -C <target-repo> apply --reverse --check docker/npu_patch/<patch>   # necessary, not sufficient
git -C <target-repo> worktree add --detach /tmp/wt <pinned-commit>
(cd /tmp/wt && patch -p1 < /root/Relax/docker/npu_patch/<patch>)   # MindSpeed-Bridge: apply base, then qwen4_exp
diff -r --exclude=.git /tmp/wt <target-repo>          # sglang: use `git am`
git -C <target-repo> worktree remove --force /tmp/wt
```

The reverse check alone is not sufficient -- a patch can be applied live *and*
stale, which is exactly how the MindSpeed-Bridge patches drifted once (six
qwen4_exp files were older than the tree, so the image would have built a model
that predates the `mtp_model_layer` rename) and again on 2026-10-16 (the
trainable-table work touched six of the same files and added
`make_ple_table_variant.py`; regenerating from the live tree picked up all
seven). The worktree check is what catches that.

`sgl-kernel-npu.patch` is the **exception**: the live checkout is pristine at its
tag (clean tree, HEAD == `2026.7.2`), so there is no live post-image to compare
against -- the patch exists only inside the image build. For it the verifier
checks provenance instead: all 15 pre-image blob hashes in the diff equal the
blobs at the tag, the 18 files it creates are absent there, and it applies as one
commit. That is a weaker statement than for the other seven, and it is worth
knowing when reading the table above: nothing on this machine has the
sgl-kernel-npu changes checked out.

All eight currently pass. The seven live-mirroring patches are byte-identical to
the live tree on every file they touch; the two `git am` patches apply as 2 and 1
commits respectively.
