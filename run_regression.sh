#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_METAL_DIR="${TT_METAL_DIR:-/localdev/xchin/project_emule/tt-metal}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule_clang}"
TEST_DIR="$BUILD_DIR/test/tt_emule"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"

PASS=0; FAIL=0; SKIP=0

run_test() {
    local name="$1"; shift
    if [ ! -f "$1" ]; then
        echo "  SKIP: $name (binary not found)"
        SKIP=$((SKIP + 1))
        return
    fi
    echo "--- $name ---"
    if "$@" 2>&1 | tail -5; then
        echo "  PASS"
        PASS=$((PASS + 1))
    else
        echo "  FAIL"
        FAIL=$((FAIL + 1))
    fi
}

echo "========================================"
echo " tt-emule Regression Tests"
echo "========================================"
echo "tt-metal: $TT_METAL_DIR"
echo "build:    $BUILD_DIR"

# Tier 1: Host-only (no env vars needed)
echo ""
echo "== Tier 1: Host-only =="
unset TT_METAL_MOCK_CLUSTER_DESC_PATH TT_METAL_EMULATED_MODE TT_METAL_SLOW_DISPATCH_MODE 2>/dev/null || true

run_test "bit_utils"          "$TEST_DIR/test_bit_utils"
run_test "host_buffer"        "$TEST_DIR/test_host_buffer"
run_test "tilize_untilize"    "$TEST_DIR/test_tilize_untilize"
run_test "blockfloat_common"  "$TEST_DIR/test_blockfloat_common"

for t in adjacent contains intersects merge iterator; do
    run_test "CoreRange_$t" "$TEST_DIR/test_CoreRange_$t"
done

for t in construct contains intersects merge; do
    run_test "CoreRangeSet_$t" "$TEST_DIR/test_CoreRangeSet_$t"
done

# Tier 2+3: Buffer I/O + JIT (wormhole for buffer tests)
echo ""
echo "== Tier 2: Buffer I/O =="

export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N150.yaml"
export TT_METAL_EMULATED_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1

run_test "SimpleL1Buffer"   "$TEST_DIR/test_simple_l1_buffer" --gtest_filter="MeshDeviceFixture.TestSimpleL1Buffer*"
run_test "SimpleDramBuffer" "$TEST_DIR/test_simple_dram_buffer"

echo ""
echo "== Tier 3: JIT Kernel Execution =="

run_test "TensixL1Tile"     "$TEST_DIR/test_simple_l1_buffer" --gtest_filter="*Tensix*"

# Tier 4: TTNN (blackhole — larger worker grid, no wormhole mmap exhaustion for ttnn)
echo ""
echo "== Tier 4: TTNN Relational INT32 =="

export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/blackhole_P100.yaml"

run_test "ttnn_relational" "$TEST_DIR/test_ttnn_relational_int"

# Tier 5: TTNN Matmul Sweep (wormhole — multi-core matmul with semaphore sync)
echo ""
echo "== Tier 5: TTNN Matmul Sweep =="

export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N150.yaml"

run_test "ttnn_matmul_sweep" "$TEST_DIR/test_ttnn_matmul_sweep"

echo ""
echo "========================================"
echo " Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "========================================"

[ "$FAIL" -eq 0 ]
