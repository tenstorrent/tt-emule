#!/bin/bash
set -o pipefail

# Quick regression: standalone C++ tests + a handful of D2M matmul tests.
# Runs in ~6-8 minutes vs ~45 min for full D2M.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_METAL_DIR="${TT_METAL_DIR:-/localdev/arminale/tt-metal}"
TT_MLIR_DIR="${TT_MLIR_DIR:-/localdev/arminale/tt-mlir}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule_clang}"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"

PASS=0; FAIL=0; SKIP=0

echo "========================================"
echo " Quick Regression (standalone + D2M matmul spot-check)"
echo "========================================"

# Clear JIT cache before starting
echo "Clearing JIT cache..."
rm -rf /tmp/tt_emule_jit_* /tmp/tt_emule_src_*

# ── Part 1: Standalone regression ──────────────────────────────────────────────
echo ""
echo "== Part 1: Standalone C++ regression =="
echo ""

bash "$SCRIPT_DIR/run_regression.sh"
STANDALONE_RC=$?

# ── Part 2: D2M matmul spot-check ─────────────────────────────────────────────
echo ""
echo "== Part 2: D2M matmul spot-check =="
echo ""

# Environment setup (mirrors run_d2m_regression.sh)
cd "$TT_MLIR_DIR"

if [ -f "$TT_MLIR_DIR/env/activate" ]; then
    source "$TT_MLIR_DIR/env/activate"
fi

export PYTHONPATH="$TT_METAL_DIR/ttnn:$BUILD_DIR/lib:$TT_MLIR_DIR/build/python_packages:$TT_MLIR_DIR/build/runtime/python:${PYTHONPATH:-}"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
export TT_MLIR_HOME="$TT_MLIR_DIR"
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N150.yaml"
export TT_METAL_EMULATED_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1

# Clear JIT cache
rm -rf /tmp/tt_emule_jit_* /tmp/tt_emule_src_*

D2M_LOG_DIR="/tmp/tt_emule_d2m_quick_$$"
mkdir -p "$D2M_LOG_DIR"

TEST_DIR="$TT_MLIR_DIR/test/python/golden"

# 5 representative D2M matmul tests:
#   1. matmul_tile, bf16, small shape (baseline)
#   2. matmul_block, bf16, rectangular shape
#   3. matmul_tile, f32, small shape (float32 path)
#   4. matmul_tile, bf16, l1_acc (L1 accumulation path)
#   5. higher-rank batched matmul (3D)
D2M_TESTS=(
    "matmul_tile_bf16_512x512:test_metal_matmul.py::test_matmul_ttnn_shapes_single_buffered[ttmetal-no_l1_acc-matmul_tile-bf16-512x512x512]"
    "matmul_block_bf16_512x1024:test_metal_matmul.py::test_matmul_ttnn_shapes_single_buffered[ttmetal-no_l1_acc-matmul_block-bf16-512x1024x1024]"
    "matmul_tile_f32_512x512:test_metal_matmul.py::test_matmul_ttnn_shapes_single_buffered[ttmetal-no_l1_acc-matmul_tile-f32-512x512x512]"
    "matmul_tile_bf16_l1acc:test_metal_matmul.py::test_matmul_ttnn_shapes_single_buffered[ttmetal-l1_acc-matmul_tile-bf16-512x512x512]"
    "matmul_3d_batch4:test_metal_matmul_higher_rank.py::test_matmul_3d_single_core[ttmetal-batch4_small]"
)

for entry in "${D2M_TESTS[@]}"; do
    name="${entry%%:*}"
    test_id="${entry#*:}"
    file="${test_id%%::*}"
    test_path="$TEST_DIR/$file"

    if [ ! -f "$test_path" ]; then
        echo "  SKIP: $name ($file not found)"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo "--- $name ---"
    log_file="$D2M_LOG_DIR/${name}.log"
    START=$(date +%s)
    timeout 300 pytest "$test_path::${test_id#*::}" -v --tb=short -p no:cacheprovider --forked > "$log_file" 2>&1
    RC=$?
    END=$(date +%s)
    ELAPSED=$((END - START))
    summary=$(grep -E "^=+" "$log_file" | tail -1)
    if [ "$RC" -eq 0 ]; then
        echo "  PASS in ${ELAPSED}s: $summary"
        PASS=$((PASS + 1))
    elif [ "$RC" -eq 124 ]; then
        echo "  HUNG/TIMEOUT after ${ELAPSED}s (log: $log_file)"
        FAIL=$((FAIL + 1))
    else
        echo "  FAIL in ${ELAPSED}s: $summary"
        FAIL=$((FAIL + 1))
    fi

    # Clear JIT cache between tests
    rm -rf /tmp/tt_emule_jit_*
done

echo ""
echo "========================================"
echo " D2M spot-check: $PASS passed, $FAIL failed, $SKIP skipped"
echo " Logs: $D2M_LOG_DIR"
echo "========================================"

# Exit with failure if standalone or D2M had failures
if [ "$STANDALONE_RC" -ne 0 ] || [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
