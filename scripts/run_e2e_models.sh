#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# tt-emule end-to-end MODEL regression (single-device).
#
# Runs tt-metal model demos end-to-end under software emulation, driving the
# emule-vendored copy of simple_text_demo.py at
#   tt-metal/tests/emule/models/test_tt_transformers_text_demo.py
# (see that file's header — the only deltas from upstream are host-sampling +
# trace-off). The curated entry list below is the source of truth for what runs;
# add a model by adding a run_model line.
#
# Arch is selected by TT_EMULE_ARCH (wormhole|blackhole); the per-arch wrappers
# run_e2e_models_<arch>.sh set it for you.
#
# Required env:
#   TT_METAL_DIR  — path to tt-metal source tree
#
# Optional env:
#   TT_EMULE_ARCH — wormhole (default) | blackhole
#   BUILD_DIR     — default $TT_METAL_DIR/build_emule
#   PYTEST_BIN    — default /opt/ttmlir-toolchain/venv/bin/pytest
#   GTEST_XML_DIR — write junit XML to this dir (CI artifact path)
#   E2E_TIMEOUT   — per-entry timeout in seconds (default 7200)
#   SHARD_INDEX/SHARD_COUNT — 1-based round-robin sharding (default 1/1)

set -o pipefail

TT_METAL_DIR="${TT_METAL_DIR:?TT_METAL_DIR must be set}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
PYTEST_BIN="${PYTEST_BIN:-/opt/ttmlir-toolchain/venv/bin/pytest}"
TT_EMULE_ARCH="${TT_EMULE_ARCH:-wormhole}"
E2E_TIMEOUT="${E2E_TIMEOUT:-7200}"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"
E2E_TEST="$TT_METAL_DIR/tests/emule/models/test_tt_transformers_text_demo.py::test_demo_text"

# Per-arch single-device config: (mesh device key recognised by the demo's
# mesh_device map → (1,1); cluster descriptor for the emulated topology).
case "$TT_EMULE_ARCH" in
    wormhole)
        MESH_DEVICE="N150"
        CLUSTER_DESC="$CLUSTER_EXAMPLES/wormhole_N150.yaml"
        ;;
    blackhole)
        MESH_DEVICE="P150"
        CLUSTER_DESC="$CLUSTER_EXAMPLES/blackhole_P150.yaml"
        ;;
    *)
        echo "ERROR: TT_EMULE_ARCH must be wormhole|blackhole (got '$TT_EMULE_ARCH')" >&2
        exit 2
        ;;
esac

GTEST_XML_DIR="${GTEST_XML_DIR:-}"
[ -n "$GTEST_XML_DIR" ] && mkdir -p "$GTEST_XML_DIR"

# Sharding — round-robin over run_model invocations. SHARD_INDEX is 1-based.
SHARD_INDEX="${SHARD_INDEX:-1}"
SHARD_COUNT="${SHARD_COUNT:-1}"
if ! [[ "$SHARD_COUNT" =~ ^[0-9]+$ ]] || [ "$SHARD_COUNT" -lt 1 ]; then
    echo "ERROR: SHARD_COUNT must be a positive integer (got '$SHARD_COUNT')" >&2
    exit 2
fi
if ! [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]] || [ "$SHARD_INDEX" -lt 1 ] || [ "$SHARD_INDEX" -gt "$SHARD_COUNT" ]; then
    echo "ERROR: SHARD_INDEX must be in [1, $SHARD_COUNT] (got '$SHARD_INDEX')" >&2
    exit 2
fi
ENTRY_NUM=0

PASS=0
FAIL=0

# run_model <name> <hf_model> <pytest args...>
# Runs the vendored demo end-to-end for one model/scenario. Remaining args are
# passed straight to pytest and MUST include a -k selector plus any --overrides.
# Each entry runs in its own subshell with a fresh HF_MODEL + full emule env.
run_model() {
    local name="$1"; shift
    local hf_model="$1"; shift
    # Optional per-entry arch restriction: set ONLY_ARCH before the call to pin
    # an entry to one arch. Consumed (reset) here so it never leaks to the next
    # entry. Counted for sharding BEFORE the restriction check so the shard→entry
    # mapping stays identical across archs.
    local only_arch="$ONLY_ARCH"; ONLY_ARCH=""
    ENTRY_NUM=$((ENTRY_NUM + 1))
    # Skip entries not assigned to this shard.
    if [ $(( (ENTRY_NUM - 1) % SHARD_COUNT + 1 )) -ne "$SHARD_INDEX" ]; then
        return
    fi
    if [ -n "$only_arch" ] && [ "$only_arch" != "$TT_EMULE_ARCH" ]; then
        echo "--- $name ---"
        echo "  SKIP ($TT_EMULE_ARCH: entry restricted to $only_arch)"
        return
    fi
    # Require an explicit -k selector. $E2E_TEST points at a single, heavily
    # parametrized test (test_demo_text has dozens of variants: long-context,
    # multi-device DP, TG, stress, ...). Without -k, pytest would collect and run
    # the ENTIRE matrix — a slow, mostly-unsupported run. Guard on -k presence,
    # not merely non-empty args, so an entry passing only an --override (e.g.
    # --max_generated_tokens) can't silently trigger the full sweep.
    local has_k=0
    for _arg in "$@"; do
        case "$_arg" in -k|-k*) has_k=1; break ;; esac
    done
    if [ "$has_k" -eq 0 ]; then
        echo "--- $name ---"; echo "  FAIL (run_model requires a -k selector to pick a test_demo_text variant)"
        FAIL=$((FAIL + 1)); return
    fi
    echo "--- $name (HF_MODEL=$hf_model) ---"
    if (
        export PYTHONPATH="$TT_METAL_DIR/ttnn:$TT_METAL_DIR/tools:$BUILD_DIR/lib:$TT_METAL_DIR:${PYTHONPATH:-}"
        export LD_LIBRARY_PATH="$BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
        export TT_METAL_HOME="$TT_METAL_DIR"
        export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
        export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_DESC"
        export TT_METAL_EMULE_MODE=1
        export TT_METAL_SLOW_DISPATCH_MODE=1
        export MESH_DEVICE="$MESH_DEVICE"
        export HF_MODEL="$hf_model"
        # Neutralize the demo's is_ci_env gate (conftest: is_ci_env == CI=="true").
        # GitHub Actions sets CI=true, which would skip non-ci_only entries like
        # batch-1 (simple_text_demo.py: `if is_ci_env: if not ci_only: skip`). Our
        # -k selection already picks exactly what runs, so force it off for
        # deterministic behavior identical locally and in CI. Weight download uses
        # is_ci_v2_env (TT_GH_CI_INFRA) and the accuracy gate uses token_accuracy —
        # both independent of CI, so this is safe.
        export CI=false
        local junit_args=()
        if [ -n "$GTEST_XML_DIR" ]; then
            junit_args=(--junitxml="$GTEST_XML_DIR/${name}.xml")
        fi
        # Run from TT_METAL_DIR so the demo's relative paths (sample prompts,
        # reference_outputs/*.refpt) resolve.
        cd "$TT_METAL_DIR" || exit 1
        timeout "$E2E_TIMEOUT" "$PYTEST_BIN" -v --tb=short "${junit_args[@]}" "$E2E_TEST" "$@" 2>&1
    ); then
        echo "  PASS"; PASS=$((PASS + 1))
    else
        echo "  FAIL"; FAIL=$((FAIL + 1))
    fi
}

echo "========================================"
echo " E2E models — $TT_EMULE_ARCH ($MESH_DEVICE single-device)"
echo "========================================"
echo "  TT_METAL_DIR: $TT_METAL_DIR"
echo "  BUILD_DIR:    $BUILD_DIR"
echo "  CLUSTER_DESC: $CLUSTER_DESC"
echo "  SHARD:        $SHARD_INDEX of $SHARD_COUNT"
echo ""

# ---- Llama-3.2-1B-Instruct (ungated unsloth mirror — no HF token needed) ----
# Teacher-forced top1/top5 token accuracy vs the checked-in reference .refpt.
# The primary correctness gate; validates deep-position SDPA-decode. Trace is
# already off in the vendored file, and ci-token-matching sets token_accuracy=True.
# The ci-token-matching row defaults to 500 generated tokens; at emule's
# ~20s/token that overruns the per-entry timeout, so cap it to 48. Prefill is the
# reference's first half (~512 tokens) and decode starts at pos 512, so even 48
# tokens stay on the multi-chunk SDPA-decode path — the deep-position coverage is
# kept, just over a shorter (bounded-runtime) span.
#
# Wormhole only: the emule blackhole compute is correct here (identical
# 89.58%/97.92% top1/top5 to wormhole), but upstream's centralized accuracy
# thresholds have no P150 (blackhole) entry, so the gate raises "Could not find
# centralized accuracy targets ... on P150". Restrict the accuracy gate to the
# arch upstream provides a target for; blackhole still runs the batch-1 smoke.
ONLY_ARCH=wormhole \
run_model "llama1b_token_matching" "unsloth/Llama-3.2-1B-Instruct" \
    -k "performance-ci-token-matching" --max_generated_tokens 48

# Full 16-layer prefill+decode generative smoke (host sampling). Bounded token
# count keeps nightly runtime reasonable; asserts nothing beyond crash-free E2E.
run_model "llama1b_batch1" "unsloth/Llama-3.2-1B-Instruct" \
    -k "performance and batch-1" --max_generated_tokens 32

echo ""
echo "========================================"
echo " Results: $PASS passed, $FAIL failed"
echo "========================================"

[ "$FAIL" -eq 0 ]
