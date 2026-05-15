#!/bin/bash
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_METAL_DIR="${TT_METAL_DIR:-/localdev/arminale/tt-metal}"
TT_MLIR_DIR="${TT_MLIR_DIR:-/localdev/arminale/tt-mlir}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule_clang}"
TEST_DIR="$TT_MLIR_DIR/test/python/golden/d2m"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"
LOG_DIR="/tmp/tt_emule_d2m_logs_$$"
TIMEOUT="${TIMEOUT:-1800}"

SERIAL=0
PYTEST_EXTRA_ARGS=()

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --serial) SERIAL=1 ;;
        *) PYTEST_EXTRA_ARGS+=("$arg") ;;
    esac
done

# D2M test files
# tt-mlir PR #8091 moved tests from test/python/golden/test_metal_*.py to
# test/python/golden/d2m/test_*.py. test_metal_matmul_higher_rank.py and
# test_metal_tensor_collapsing.py were deleted in that refactor.
TEST_FILES=(
    test_matmul.py
    test_tilize.py
    test_dma.py
    test_layout.py
    test_allocate.py
    test_masking.py
    test_reductions.py
    test_bfp8_typecast.py
    test_tms.py
    test_virtual_grid_rowmajor.py
    test_virtual_grids.py
)

echo "========================================"
echo " D2M Golden Test Regression"
echo "========================================"
echo "tt-metal: $TT_METAL_DIR"
echo "tt-mlir:  $TT_MLIR_DIR"
echo "build:    $BUILD_DIR"

# Must cd to tt-mlir root so conftest can find ttrt-artifacts/system_desc.ttsys
cd "$TT_MLIR_DIR"

# Activate tt-mlir environment
if [ -f "$TT_MLIR_DIR/env/activate" ]; then
    echo "Activating tt-mlir environment..."
    source "$TT_MLIR_DIR/env/activate"
fi

# Set PYTHONPATH for ttnn (from tt-metal build) and ttmlir_runtime (from tt-mlir build).
# $TT_METAL_DIR/tools is required because ttnn's __init__.py imports the tracy python
# module from tt-metal-main/tools/tracy/.
export PYTHONPATH="$TT_METAL_DIR/ttnn:$TT_METAL_DIR/tools:$BUILD_DIR/lib:$TT_MLIR_DIR/build/python_packages:$TT_MLIR_DIR/build/runtime/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
export TT_MLIR_HOME="$TT_MLIR_DIR"

# Ensure the emule-built _ttnn.so is used (source tree may have stale copy from non-emule build)
if [ -f "$BUILD_DIR/lib/_ttnn.so" ]; then
    ln -sf "$BUILD_DIR/lib/_ttnn.so" "$TT_METAL_DIR/ttnn/ttnn/_ttnn.so"
fi

# Set emulation env vars
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N150.yaml"
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1

# Clear stale JIT cache
echo "Clearing JIT cache..."
rm -rf /tmp/tt_emule_jit_* /tmp/tt_emule_src_*

# Create log directory
mkdir -p "$LOG_DIR"

echo ""
echo "Running ${#TEST_FILES[@]} test files (mode: $([ $SERIAL -eq 1 ] && echo serial || echo parallel))"
echo "Timeout per file: ${TIMEOUT}s"
echo "Logs: $LOG_DIR"
echo ""

# Launch tests
declare -A PIDS
PASS=0
FAIL=0
HUNG=0

for tf in "${TEST_FILES[@]}"; do
    test_path="$TEST_DIR/$tf"
    if [ ! -f "$test_path" ]; then
        echo "  SKIP: $tf (not found)"
        continue
    fi

    log_file="$LOG_DIR/${tf%.py}.log"

    if [ $SERIAL -eq 1 ]; then
        echo "--- $tf ---"
        START=$(date +%s)
        timeout "$TIMEOUT" pytest "$test_path" -v --tb=short -p no:cacheprovider --forked "${PYTEST_EXTRA_ARGS[@]}" > "$log_file" 2>&1
        RC=$?
        END=$(date +%s)
        ELAPSED=$((END - START))
        if [ "$RC" -eq 124 ]; then
            echo "  HUNG/TIMEOUT after ${ELAPSED}s (log: $log_file)"
            HUNG=$((HUNG + 1))
        elif [ "$RC" -ne 0 ]; then
            summary=$(grep -E "^=+" "$log_file" | tail -1)
            echo "  FAIL in ${ELAPSED}s: $summary"
            FAIL=$((FAIL + 1))
        else
            summary=$(grep -E "^=+" "$log_file" | tail -1)
            echo "  PASS in ${ELAPSED}s: $summary"
            PASS=$((PASS + 1))
        fi
        # Clear JIT cache between test files
        rm -rf /tmp/tt_emule_jit_*
    else
        timeout "$TIMEOUT" pytest "$test_path" -v --tb=short -p no:cacheprovider --forked "${PYTEST_EXTRA_ARGS[@]}" > "$log_file" 2>&1 &
        PIDS[$tf]=$!
    fi
done

# Wait for parallel processes
if [ $SERIAL -eq 0 ]; then
    for tf in "${!PIDS[@]}"; do
        pid=${PIDS[$tf]}
        log_file="$LOG_DIR/${tf%.py}.log"
        if wait "$pid"; then
            summary=$(grep -E "^=+" "$log_file" | tail -1)
            echo "  PASS: $tf  $summary"
            PASS=$((PASS + 1))
        else
            RC=$?
            if [ "$RC" -eq 124 ]; then
                echo "  HUNG: $tf (timeout after ${TIMEOUT}s, log: $log_file)"
                HUNG=$((HUNG + 1))
            else
                summary=$(grep -E "^=+" "$log_file" | tail -1)
                echo "  FAIL: $tf  $summary"
                FAIL=$((FAIL + 1))
            fi
        fi
    done
fi

echo ""
echo "========================================"
echo " D2M Results: $PASS passed, $FAIL failed, $HUNG hung"
echo " Logs: $LOG_DIR"
echo "========================================"

[ "$FAIL" -eq 0 ] && [ "$HUNG" -eq 0 ]
