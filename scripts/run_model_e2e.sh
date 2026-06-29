#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# tt-emule model end-to-end (e2e) test suite.
#
# Runs the emule-only model tests under tt-metal/tests/emule/ — full
# Llama-class prefill+decode on emulated HW, asserting coherent generation past
# the multi-chunk SDPA-decode boundary (sequence position 128). These are heavy
# (real model weights, ~15-20 min each), so they live in a NIGHTLY job and are
# NOT part of the per-PR regression (scripts/run_ttnn_pytests_*.sh).
#
# Arch-parameterized (one script, both archs) since the only difference is the
# mock cluster descriptor + MESH_DEVICE.
#
# Required env:
#   TT_METAL_DIR   — path to tt-metal source tree
#   TT_EMULE_ARCH  — wormhole | blackhole
#
# Optional env:
#   BUILD_DIR      — default $TT_METAL_DIR/build_emule
#   PYTEST_BIN     — default /opt/ttmlir-toolchain/venv/bin/pytest
#   GTEST_XML_DIR  — write junit XML to this dir (CI artifact path)
#   SHARD_INDEX/SHARD_COUNT — round-robin sharding over the entries (default 1/1)
#   HF_MODEL/HF_HOME/HF_HUB_OFFLINE — model weights (inherited from the environment)

set -o pipefail

TT_METAL_DIR="${TT_METAL_DIR:?TT_METAL_DIR must be set}"
TT_EMULE_ARCH="${TT_EMULE_ARCH:?TT_EMULE_ARCH must be set (wormhole|blackhole)}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
PYTEST_BIN="${PYTEST_BIN:-/opt/ttmlir-toolchain/venv/bin/pytest}"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"
EMULE_TEST_DIR="$TT_METAL_DIR/tests/emule"

case "$TT_EMULE_ARCH" in
    wormhole)  CLUSTER_DESC="$CLUSTER_EXAMPLES/wormhole_N150.yaml"; MESH="N150" ;;
    blackhole) CLUSTER_DESC="$CLUSTER_EXAMPLES/blackhole_P100.yaml"; MESH="P100" ;;
    *) echo "ERROR: TT_EMULE_ARCH must be wormhole|blackhole, got '$TT_EMULE_ARCH'" >&2; exit 2 ;;
esac

GTEST_XML_DIR="${GTEST_XML_DIR:-}"
[ -n "$GTEST_XML_DIR" ] && mkdir -p "$GTEST_XML_DIR"

# Sharding — round-robin over run_pytest invocations (1-based SHARD_INDEX), so as
# the e2e test list grows the slow entries spread across shards / parallel runners.
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

run_pytest() {
    local name="$1"; shift
    ENTRY_NUM=$((ENTRY_NUM + 1))
    # Skip entries not assigned to this shard.
    if [ $(( (ENTRY_NUM - 1) % SHARD_COUNT + 1 )) -ne "$SHARD_INDEX" ]; then
        return
    fi
    if [ "$#" -eq 0 ]; then
        echo "--- $name ---"; echo "  FAIL (no test target supplied to run_pytest)"
        FAIL=$((FAIL + 1)); return
    fi
    echo "--- $name ---"
    if (
        export PYTHONPATH="$TT_METAL_DIR/ttnn:$TT_METAL_DIR/tools:$BUILD_DIR/lib:$TT_METAL_DIR:${PYTHONPATH:-}"
        export LD_LIBRARY_PATH="$BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
        export TT_METAL_HOME="$TT_METAL_DIR"
        export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
        export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_DESC"
        export TT_METAL_EMULE_MODE=1
        export TT_METAL_SLOW_DISPATCH_MODE=1
        export MESH_DEVICE="$MESH"
        # The demo opens input_prompts + the token-accuracy reference via paths
        # RELATIVE to CWD (it assumes CWD == tt-metal root). Run from there.
        cd "$TT_METAL_DIR" || exit 1
        # HF_MODEL / HF_HOME / HF_HUB_OFFLINE are inherited from the environment
        # (set by the CI workflow's weights-cache step) — the subshell only adds
        # vars, it does not scrub them.
        local junit_args=()
        if [ -n "$GTEST_XML_DIR" ]; then
            junit_args=(--junitxml="$GTEST_XML_DIR/${name}.xml")
        fi
        # e2e model runs are slow under emule slow-dispatch (~15-20 min); give a
        # generous per-entry timeout (vs the 900s used by the op-level suites).
        timeout 3600 "$PYTEST_BIN" -v -s --tb=short "${junit_args[@]}" "$@" 2>&1
    ); then
        echo "  PASS"; PASS=$((PASS + 1))
    else
        echo "  FAIL"; FAIL=$((FAIL + 1))
    fi
}

echo "========================================"
echo " Model e2e — ${TT_EMULE_ARCH} (${MESH})"
echo "========================================"
echo "  TT_METAL_DIR:  $TT_METAL_DIR"
echo "  BUILD_DIR:     $BUILD_DIR"
echo "  PYTEST_BIN:    $PYTEST_BIN"
echo "  CLUSTER_DESC:  $CLUSTER_DESC"
echo "  HF_MODEL:      ${HF_MODEL:-<unset>}"
echo "  HF_HOME:       ${HF_HOME:-<unset>}"
echo "  GTEST_XML_DIR: ${GTEST_XML_DIR:-<unset>}"
echo "  SHARD:         $SHARD_INDEX of $SHARD_COUNT"
echo ""

# --- e2e model tests: the tt-metal text-generation demo, copied into tests/emule/
# and modified for emule (trace off + host sampling). Run the demo's own rows via -k. ---
DEMO="$EMULE_TEST_DIR/test_demo_emule.py"

# ci-token-matching: teacher-forced top-1/top-5 vs the bundled reference (Tale of Two
# Cities). Decodes from position ~512 — exercises the multi-chunk SDPA-decode path.
run_pytest "llama_ci_token_matching" "$DEMO" -k "ci-token-matching and performance" --max_generated_tokens 32

# condiment batch-1: free-running generation; no built-in metric, so this is a SMOKE
# test — it passes iff the full prefill+decode demo runs to completion without error.
run_pytest "llama_condiment_smoke"   "$DEMO" -k "batch-1 and accuracy"              --max_generated_tokens 48

echo ""
echo "========================================"
echo " Results: $PASS passed, $FAIL failed"
echo "========================================"
[ "$FAIL" -eq 0 ]
