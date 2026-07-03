#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# tt-emule TTNN pytest regression — Wormhole n300 (2-device mesh).
#
# Runs the subset of tt-metal/tests/ttnn/unit_tests/ pytest entries that pass
# cleanly under emule on a 2-chip n300 mesh. The list is curated: each entry is
# either a whole file or a function/-k subset verified all-PASS in standalone
# runs. Today's coverage is the fabric CCL suite (multi-chip collectives over the
# teleport path); other multi-device domains can extend the entry list later.
#
# Required env:
#   TT_METAL_DIR  — path to tt-metal source tree
#
# Optional env:
#   BUILD_DIR     — default $TT_METAL_DIR/build_emule
#   PYTEST_BIN    — default /opt/ttmlir-toolchain/venv/bin/pytest
#   GTEST_XML_DIR — write junit XML to this dir (CI artifact path)

set -o pipefail

TT_METAL_DIR="${TT_METAL_DIR:?TT_METAL_DIR must be set}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
PYTEST_BIN="${PYTEST_BIN:-/opt/ttmlir-toolchain/venv/bin/pytest}"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"
CCL_TEST_DIR="$TT_METAL_DIR/tests/ttnn/unit_tests/operations/ccl"

GTEST_XML_DIR="${GTEST_XML_DIR:-}"
[ -n "$GTEST_XML_DIR" ] && mkdir -p "$GTEST_XML_DIR"

# Sharding — round-robin over run_pytest invocations. SHARD_INDEX is 1-based.
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
    # Remaining args are passed straight to pytest: one or more test targets
    # (file or file::node) plus any flags (-k, --deselect). Passing multiple
    # node targets in one invocation amortizes the per-process `import ttnn` +
    # collection + device/cluster init.
    ENTRY_NUM=$((ENTRY_NUM + 1))
    # Skip entries not assigned to this shard.
    if [ $(( (ENTRY_NUM - 1) % SHARD_COUNT + 1 )) -ne "$SHARD_INDEX" ]; then
        return
    fi
    # Guard: with no target, pytest would collect/run the entire CWD suite — a
    # very expensive CI failure mode. Treat a target-less entry as an authoring
    # error and fail it loudly.
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
        # 2-chip n300 descriptor (vs the single-device N150 used by the wormhole
        # script). The cross-chip CCL teleport path needs the 2-device cluster.
        export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N300.yaml"
        export TT_METAL_EMULE_MODE=1
        export TT_METAL_SLOW_DISPATCH_MODE=1
        local junit_args=()
        if [ -n "$GTEST_XML_DIR" ]; then
            junit_args=(--junitxml="$GTEST_XML_DIR/${name}.xml")
        fi
        local forked_args=()
        [ "${FORKED:-0}" = "1" ] && forked_args=(--forked)
        timeout 1200 "$PYTEST_BIN" -v --tb=short "${forked_args[@]}" "${junit_args[@]}" "$@" 2>&1
    ); then
        echo "  PASS"; PASS=$((PASS + 1))
    else
        echo "  FAIL"; FAIL=$((FAIL + 1))
    fi
}

echo "========================================"
echo " TTNN pytest — Wormhole n300 (2-device mesh)"
echo "========================================"
echo "  TT_METAL_DIR: $TT_METAL_DIR"
echo "  BUILD_DIR:    $BUILD_DIR"
echo "  PYTEST_BIN:   $PYTEST_BIN"
echo "  GTEST_XML_DIR: ${GTEST_XML_DIR:-<unset>}"
echo "  SHARD:        $SHARD_INDEX of $SHARD_COUNT"
echo ""

# ---- Fabric CCL (multi-chip collectives over the teleport path) ----
# Each entry was verified all-PASS (xfail allowed) in standalone n300 runs.

# Canonical high-level CCL: all_gather / reduce_scatter / all_broadcast +
# composite variants × l1_default/l1_small (exercises the fabric mux path).
run_pytest "ccl_l1_small_semaphores" "$CCL_TEST_DIR/test_ccl_l1_small_semaphores.py"

echo ""
echo "========================================"
echo " Results: $PASS passed, $FAIL failed"
echo "========================================"

[ "$FAIL" -eq 0 ]
