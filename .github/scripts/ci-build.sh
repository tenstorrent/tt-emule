#!/usr/bin/env bash
# Configure and build tt-metal against the workspace's tt-emule source.
#
# Required env:
#   TT_EMULE_DIR    path to tt-emule workspace (the PR source)
#   TT_METAL_DIR    path to tt-metal workspace (checked out at pinned SHA)
#   BUILD_DIR       absolute path for the build tree (default: $TT_METAL_DIR/build_emule_clang)
#
# Mirrors the configure + build phase of build_and_test.sh so that local and CI
# runs share the same flags.

set -euo pipefail

: "${TT_EMULE_DIR:?TT_EMULE_DIR must be set}"
: "${TT_METAL_DIR:?TT_METAL_DIR must be set}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule_clang}"

export CCACHE_DIR="${CCACHE_DIR:-$HOME/.ccache}"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

echo "== ci-build.sh =="
echo "  TT_EMULE_DIR: $TT_EMULE_DIR"
echo "  TT_METAL_DIR: $TT_METAL_DIR"
echo "  BUILD_DIR:    $BUILD_DIR"
echo "  CCACHE_DIR:   $CCACHE_DIR"
echo ""

echo "== Configuring tt-metal =="
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
    -DWITH_PYTHON_BINDINGS=OFF \
    -DENABLE_TRACY=OFF \
    -DENABLE_DISTRIBUTED=OFF \
    -DTT_INSTALL=OFF

echo ""
echo "== Building tt-metal =="
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo ""
echo "== ccache stats =="
ccache --show-stats || true
