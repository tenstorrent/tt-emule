#!/usr/bin/env bash
# Configure and build tt-mlir (with embedded tt-metal build) against tt-emule.
#
# Required env:
#   TT_EMULE_DIR   path to tt-emule workspace (the PR source)
#   TT_MLIR_DIR    path to tt-mlir workspace (checked out at pinned SHA)
#
# Optional env:
#   BUILD_DIR      tt-mlir build dir (default: $TT_MLIR_DIR/build)
#
# Designed to run inside the tt-mlir CI Docker image which has
# /opt/ttmlir-toolchain pre-populated. Outside that image, env/activate
# will warn and the cmake configure will fail — that's expected.

set -euo pipefail

: "${TT_EMULE_DIR:?TT_EMULE_DIR must be set}"
: "${TT_MLIR_DIR:?TT_MLIR_DIR must be set}"
BUILD_DIR="${BUILD_DIR:-$TT_MLIR_DIR/build}"

export CCACHE_DIR="${CCACHE_DIR:-$HOME/.ccache}"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

mkdir -p "$CCACHE_DIR"
ccache --set-config max_size=10G 2>/dev/null || true
ccache --set-config compression=true 2>/dev/null || true
ccache --zero-stats >/dev/null 2>&1 || true

echo "== ci-build-mlir.sh =="
echo "  TT_EMULE_DIR: $TT_EMULE_DIR"
echo "  TT_MLIR_DIR:  $TT_MLIR_DIR"
echo "  BUILD_DIR:    $BUILD_DIR"
echo "  CCACHE_DIR:   $CCACHE_DIR"
echo ""

echo "== Injecting tt-emule cmake args into tt-mlir's tt-metal ExternalProject =="
python3 "$TT_EMULE_DIR/.github/scripts/inject_ttemule_into_mlir.py" \
    "$TT_MLIR_DIR/third_party/CMakeLists.txt" \
    "$TT_EMULE_DIR"
echo ""

echo "== Sourcing tt-mlir env/activate =="
# Must cd into TT_MLIR_DIR — activate uses $(pwd) for path setup.
cd "$TT_MLIR_DIR"
# env/activate references unbound vars (e.g. _ACTIVATE_ECHO_TOOLCHAIN_DIR_AND_EXIT);
# disable `set -u` around the source.
set +u
# shellcheck disable=SC1091
source env/activate
set -u
echo "  TTMLIR_TOOLCHAIN_DIR=${TTMLIR_TOOLCHAIN_DIR:-unset}"
echo "  TTMLIR_VENV_DIR=${TTMLIR_VENV_DIR:-unset}"
which python
python --version
echo ""

echo "== Configuring tt-mlir =="
cmake -B "$BUILD_DIR" -S "$TT_MLIR_DIR" -G Ninja \
    -DCMAKE_C_COMPILER=clang-17 \
    -DCMAKE_CXX_COMPILER=clang++-17 \
    -DCMAKE_BUILD_TYPE=Release \
    -DTTMLIR_ENABLE_RUNTIME=ON \
    -DTT_RUNTIME_ENABLE_TTMETAL=ON \
    -DTTMLIR_ENABLE_STABLEHLO=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

echo ""
echo "== Building tt-mlir (and tt-metal-against-tt-emule transitively) =="
# The tt-metal ExternalProject is part of tt-mlir's default build. A plain
# all-target build covers both, plus python_packages and runtime libs needed
# by D2M pytest.
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo ""
echo "== ccache stats =="
ccache --show-stats || true

echo ""
echo "== Build artifacts =="
ls -la "$BUILD_DIR/lib/" 2>/dev/null | head -20 || true
echo "..."
ls -la "$TT_MLIR_DIR/third_party/tt-metal/src/tt-metal/build/lib/" 2>/dev/null | head -20 || true
