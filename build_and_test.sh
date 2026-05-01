#!/bin/bash
# build_and_test.sh — Build tt-emule + tt-metal and run the full regression suite.
#
# Assumes:
#   - tt-emule and tt-metal are sibling directories under the same parent
#   - tt-metal is on the arminale/emule-metal-base branch (or compatible)
#   - tt-metal submodules (umd, tracy, tt_llk) are initialized
#   - clang-20, cmake, ninja are installed
#
# Usage:
#   cd /path/to/tt-emule
#   ./build_and_test.sh
#
# Override paths:
#   TT_METAL_DIR=/path/to/tt-metal ./build_and_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_EMULE_DIR="$SCRIPT_DIR"
TT_METAL_DIR="${TT_METAL_DIR:-$(dirname "$SCRIPT_DIR")/tt-metal}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule_clang}"

echo "========================================"
echo " tt-emule Build & Test"
echo "========================================"
echo "tt-emule:  $TT_EMULE_DIR"
echo "tt-metal:  $TT_METAL_DIR"
echo "build dir: $BUILD_DIR"
echo ""

# --- Validate directory layout ---
if [ ! -d "$TT_EMULE_DIR/src" ] && [ ! -f "$TT_EMULE_DIR/CMakeLists.txt" ] && [ ! -f "$TT_EMULE_DIR/run_regression.sh" ]; then
    echo "ERROR: tt-emule directory not found or invalid at $TT_EMULE_DIR"
    exit 1
fi

if [ ! -d "$TT_METAL_DIR/tt_metal" ]; then
    echo "ERROR: tt-metal directory not found at $TT_METAL_DIR"
    echo "  Expected tt-metal and tt-emule as sibling directories."
    echo "  Set TT_METAL_DIR=/path/to/tt-metal to override."
    exit 1
fi

# --- Validate prerequisites ---
echo "== Checking prerequisites =="
MISSING=""
for cmd in clang-20 clang++-20 cmake ninja; do
    if ! command -v "$cmd" &>/dev/null; then
        MISSING="$MISSING $cmd"
    fi
done
if [ -n "$MISSING" ]; then
    echo "ERROR: Missing required tools:$MISSING"
    exit 1
fi
echo "  clang-20:  $(clang-20 --version | head -1)"
echo "  cmake:     $(cmake --version | head -1)"
echo "  ninja:     $(ninja --version)"
echo ""

# --- Validate submodules ---
echo "== Checking tt-metal submodules =="
UMD_YAML="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml"
if [ ! -f "$UMD_YAML" ]; then
    echo "ERROR: UMD submodule not initialized."
    echo "  Run: cd $TT_METAL_DIR && git submodule update --init tt_metal/third_party/umd"
    exit 1
fi

TRACY_CMAKE="$TT_METAL_DIR/tt_metal/third_party/tracy/CMakeLists.txt"
if [ ! -f "$TRACY_CMAKE" ]; then
    echo "ERROR: tracy submodule not initialized."
    echo "  Run: cd $TT_METAL_DIR && git submodule update --init tt_metal/third_party/tracy"
    exit 1
fi

TT_LLK_DIR="$TT_METAL_DIR/tt_metal/third_party/tt_llk"
if [ ! -d "$TT_LLK_DIR" ] || [ -z "$(ls -A "$TT_LLK_DIR" 2>/dev/null)" ]; then
    echo "ERROR: tt_llk submodule not initialized."
    echo "  Run: cd $TT_METAL_DIR && git submodule update --init tt_metal/third_party/tt_llk"
    exit 1
fi
echo "  UMD:    OK"
echo "  tracy:  OK"
echo "  tt_llk: OK"
echo ""

# --- Configure tt-metal ---
echo "== Configuring tt-metal with emulation =="
cmake -B "$BUILD_DIR" \
    -S "$TT_METAL_DIR" \
    -G Ninja \
    -DCMAKE_C_COMPILER=clang-20 \
    -DCMAKE_CXX_COMPILER=clang++-20 \
    -DCMAKE_AR=/usr/bin/llvm-ar-20 \
    -DCMAKE_RANLIB=/usr/bin/llvm-ranlib-20 \
    -DCMAKE_BUILD_TYPE=Release \
    -DTT_METAL_USE_TT_EMULE=ON \
    -DTT_METAL_EMULATION=ON \
    -DTT_EMULE_PATH="$TT_EMULE_DIR" \
    -DWITH_PYTHON_BINDINGS=ON \
    -DENABLE_TRACY=OFF \
    -DTT_INSTALL=OFF
echo ""

# --- Build ---
echo "== Building tt-metal (this may take a while) =="
cmake --build "$BUILD_DIR" -j"$(nproc)"
echo ""

# --- Verify build outputs ---
echo "== Verifying build outputs =="
TEST_BIN_DIR="$BUILD_DIR/test/tt_emule"
EXPECTED_BINS=(test_matmul_X_tile test_matmul_block test_dataflow_buffer test_simple_l1_buffer)
ALL_FOUND=true
for bin in "${EXPECTED_BINS[@]}"; do
    if [ -f "$TEST_BIN_DIR/$bin" ]; then
        echo "  $bin: OK"
    else
        echo "  $bin: MISSING"
        ALL_FOUND=false
    fi
done
if [ "$ALL_FOUND" = false ]; then
    echo "WARNING: Some test binaries are missing. The regression may report skips."
fi
echo ""

# --- Run regression ---
echo "== Running regression suite =="
cd "$TT_EMULE_DIR"
export TT_METAL_DIR
export BUILD_DIR
exec ./run_regression.sh
