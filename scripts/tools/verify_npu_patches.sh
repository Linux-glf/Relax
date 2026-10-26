#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# The gate for `docker/npu_patch/`: a patch is only as good as "pristine
# checkout at the pin + patch == the live tree that was measured". The reverse
# check (git apply --reverse --check) is necessary but NOT sufficient -- a patch
# can apply live *and* be older than the tree, which is exactly how
# mindspeed-bridge.patch drifted once (six qwen4_exp files older than the tree,
# so an image built from it would have carried a model predating the
# `mtp_model_layer` rename).
#
# So this script re-does the whole thing: a detached worktree at each repo's pin,
# `patch -p1` (or `git am` for the two that are applied that way), then a
# byte-wise `cmp` of every file the patch touches against its live counterpart,
# and a check that no live file the patch *should* have covered was left out.
#
# One repo is the exception and says so: `sgl-kernel-npu.patch` is not part of
# the live tree at all (the live checkout is pristine at its tag, and the patch
# enters only inside the image build), so there is no live post-image to compare
# against. For it the check is provenance instead -- every pre-image blob hash in
# the diff must equal the blob at HEAD -- plus "applies as one commit at the tag".
#
# Read-only with respect to the source trees: the worktrees live outside them and
# are removed at the end, including on failure.
#
# Usage: bash scripts/tools/verify_npu_patches.sh [repo ...]

set -uo pipefail

PATCH_DIR=/root/Relax/docker/npu_patch
WT_ROOT=${WT_ROOT:-/data/lazypig/patchverify}
# Repo -> "patch-name[,patch-name]:pin"; two patches on one repo are applied in
# order and checked together (MindSpeed-Bridge: base, then qwen4_exp). sglang's
# pin is the commit the patch is applied *to* (v0.5.15 + the local install commit
# + the pinned cherry-pick), not a tag.
declare -A SPEC=(
  [Megatron-LM]="megatron:b26f370df24169e64164aede07507bdda64ef4ec"
  [Megatron-Bridge]="megatron-bridge:8ec5c9cdc2bf1af3247a2dbe73696534d6055814"
  [MindSpeed-Bridge]="mindspeed-bridge,mindspeed-bridge-qwen4-exp:3b06ac9d42a0b66e12027ca3607fde702e63958e"
  [MindSpeed-Ops]="mindspeed-ops:ca9543d28230494b661cea80629819d465b8d7db"
  [MegatronAdaptor]="megatron-adaptor:34fe86cdbf7c2f6b16cbea9f43fb3ed860f18812"
  [TransformerEngineNPU]="transformer-engine-npu:0ae3862f41f551dadf499b474899a1c7a1dcbb4c"
  [sglang]="sglang-npu:cfbf5d70a8355c8435e4dae09d9bf84b31bca306"
  [sgl-kernel-npu]="sgl-kernel-npu:7a396de"
)
# Repos whose live tree is the reference the patch has to reproduce. 0 = the
# patch is an external artifact: no post-image exists live, so compare blobs.
declare -A LIVE_REF=( [sgl-kernel-npu]=0 )

REPOS=("$@")
if [ ${#REPOS[@]} -eq 0 ]; then
    REPOS=("${!SPEC[@]}")
fi

mkdir -p "${WT_ROOT}"
failed=0
for repo in "${REPOS[@]}"; do
    spec="${SPEC[$repo]:-}"
    if [ -z "${spec}" ]; then
        echo "FAIL ${repo}: unknown repo (add it to SPEC)"; failed=1; continue
    fi
    patch_list="${spec%%:*}"; pin="${spec##*:}"
    IFS=',' read -ra patches <<< "${patch_list}"
    live="/root/${repo}"; wt="${WT_ROOT}/${repo}"

    echo "=== ${repo}  <-  ${patch_list}   (pin ${pin:0:12})"
    git -C "${live}" worktree remove --force "${wt}" 2>/dev/null
    rm -rf "${wt}"
    if ! git -C "${live}" worktree add --detach "${wt}" "${pin}" >/dev/null 2>&1; then
        echo "FAIL ${repo}: cannot create a worktree at ${pin}"; failed=1; continue
    fi

    # The two `git am` patches are mboxes; the rest are plain diffs. A repo may
    # need more than one patch; apply them in the order listed.
    applied=1
    for patch in "${patches[@]}"; do
        case "${repo}" in
            sglang|sgl-kernel-npu)
                if git -C "${wt}" am "${PATCH_DIR}/${patch}.patch" >/tmp/am.$$ 2>&1; then
                    echo "  am:  ${patch}.patch ($(git -C "${wt}" log --oneline "${pin}"..HEAD | wc -l) commit(s))"
                else
                    echo "  FAIL ${patch}.patch did not apply as git am:"; sed 's/^/    /' /tmp/am.$$ | head -8
                    applied=0; break
                fi
                ;;
            *)
                if patch -d "${wt}" -p1 --no-backup-if-mismatch < "${PATCH_DIR}/${patch}.patch" >/tmp/patch.$$ 2>&1; then
                    echo "  ok:  ${patch}.patch"
                else
                    echo "  FAIL ${patch}.patch did not apply:"; sed 's/^/    /' /tmp/patch.$$ | head -8
                    applied=0; break
                fi
                ;;
        esac
    done
    if [ "${applied}" -eq 0 ]; then
        failed=1; git -C "${live}" worktree remove --force "${wt}" >/dev/null 2>&1; continue
    fi

    # Byte-wise comparison of every file the patches claim to touch, plus the
    # files they should have covered: the live tree's own delta against the pin.
    mapfile -t touched < <(
        for patch in "${patches[@]}"; do
            grep '^diff --git' "${PATCH_DIR}/${patch}.patch" | sed 's|diff --git a/||; s| b/.*||'
        done | sort -u
    )
    bad=0
    if [ "${LIVE_REF[$repo]:-1}" = "0" ]; then
        # No live post-image: the strongest statement available is that the patch
        # was cut against exactly this base -- every pre-image blob hash in the
        # diff must be the blob at HEAD, and the files it creates must be absent
        # there.
        if RELAX_LIVE="${live}" RELAX_PATCH="${PATCH_DIR}/${patches[0]}.patch" python3 -I - <<'PY'
import os
import re
import subprocess

live = os.environ["RELAX_LIVE"]
text = open(os.environ["RELAX_PATCH"]).read()
same = created = 0
for block in re.split(r"(?m)^(?=diff --git )", text):
    if not block.strip():
        continue
    path = block.split("\n", 1)[0].split(" b/")[-1]
    match = re.search(r"(?m)^index ([0-9a-f]+)\.\.([0-9a-f]+)", block)
    if not match:
        continue
    at_head = subprocess.run(["git", "rev-parse", f"HEAD:{path}"], capture_output=True, text=True, cwd=live)
    if not at_head.returncode:
        if at_head.stdout.strip().startswith(match.group(1)):
            same += 1
        else:
            print(f"    pre-image blob differs from HEAD: {path}")
            raise SystemExit(1)
    elif block.lstrip().startswith("diff") and "new file mode" in block:
        created += 1
    else:
        print(f"    path absent at HEAD but the diff modifies it: {path}")
        raise SystemExit(1)
print(f"    base provenance: {same} modified files match the pinned blob, {created} created files absent")
PY
        then
            echo "  OK   live tree is the pristine base; patch applies as a commit on top"
        else
            echo "  FAIL provenance check"; bad=1
        fi
    else
        for f in "${touched[@]}"; do
            if [ ! -e "${live}/${f}" ]; then
                echo "  MISSING in live: ${f}"; bad=1; continue
            fi
            cmp -s "${wt}/${f}" "${live}/${f}" || { echo "  DIFFERS: ${f}"; bad=1; }
        done

        # The reverse direction: every live modification vs the pin must be
        # covered, or the patch is missing part of the environment it claims to
        # describe.
        mapfile -t live_delta < <(git -C "${live}" status --porcelain | awk '$1 ~ /^M|^A|^D$/ {print $2}' | sort)
        for f in "${live_delta[@]}"; do
            grep -qxF "diff --git a/${f} b/${f}" <(
                for patch in "${patches[@]}"; do grep '^diff --git' "${PATCH_DIR}/${patch}.patch"; done
            ) || { echo "  NOT COVERED by the patches (live has it modified): ${f}"; bad=1; }
        done
    fi

    if [ "${bad}" -eq 0 ]; then
        if [ "${LIVE_REF[$repo]:-1}" != "0" ]; then
            echo "  OK   ${#touched[@]} files byte-identical to the live tree"
        fi
    else
        failed=1
    fi
    git -C "${live}" worktree remove --force "${wt}" >/dev/null 2>&1
done

rm -f /tmp/am.$$ /tmp/patch.$$
if [ "${failed}" -eq 0 ]; then
    echo "ALL PATCHES VERIFIED"
else
    echo "VERIFICATION FAILED"
fi
exit "${failed}"
