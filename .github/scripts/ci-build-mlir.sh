#!/usr/bin/env bash
# Build tt-metal independently (with tt-emule integration) and then build
# tt-mlir against that pre-built tree via TTMLIR_TTMETAL_SOURCE_DIR.
#
# Both projects compile with clang-20. tt-mlir's ExternalProject_Add steps for
# tt-metal are pre-stamped so they don't re-configure / rebuild what we just
# built.
#
# Required env:
#   TT_EMULE_DIR    path to tt-emule workspace (the PR source)
#   TT_MLIR_DIR     path to tt-mlir workspace (checked out at pinned SHA)
#   TT_METAL_DIR    path to tt-metal workspace at the SHA tt-mlir pins
#
# Optional env:
#   BUILD_DIR       tt-mlir build dir (default: $TT_MLIR_DIR/build)

set -euo pipefail

: "${TT_EMULE_DIR:?TT_EMULE_DIR must be set}"
: "${TT_MLIR_DIR:?TT_MLIR_DIR must be set}"
: "${TT_METAL_DIR:?TT_METAL_DIR must be set}"
BUILD_DIR="${BUILD_DIR:-$TT_MLIR_DIR/build}"
TT_METAL_BUILD="$TT_METAL_DIR/build_Release"

export CCACHE_DIR="${CCACHE_DIR:-$HOME/.ccache}"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

mkdir -p "$CCACHE_DIR"
ccache --set-config max_size=10G 2>/dev/null || true
ccache --set-config compression=true 2>/dev/null || true
ccache --zero-stats >/dev/null 2>&1 || true

echo "== ci-build-mlir.sh =="
echo "  TT_EMULE_DIR:    $TT_EMULE_DIR"
echo "  TT_MLIR_DIR:     $TT_MLIR_DIR"
echo "  TT_METAL_DIR:    $TT_METAL_DIR"
echo "  TT_METAL_BUILD:  $TT_METAL_BUILD"
echo "  BUILD_DIR:       $BUILD_DIR"
echo "  CCACHE_DIR:      $CCACHE_DIR"
echo ""

# NOTE: we deliberately do NOT source tt-mlir's env/activate before the
# tt-metal build. env/activate sets TT_METAL_HOME / TT_METAL_RUNTIME_ROOT
# to $TT_MLIR_DIR/third_party/tt-metal/src/tt-metal, which tt-metal's
# precompile_fw tool then bakes in (it loads core_descriptors/*.yaml from
# that path at runtime). We want tt-metal to use its own real source dir
# instead. env/activate is sourced later, just before the tt-mlir build.

# ---------------------------------------------------------------------------
# 1. Build tt-metal out-of-tree with tt-emule integration.
# ---------------------------------------------------------------------------
# Workaround for missing #include in tt-metal/umd at currently-pinned SHAs:
# cluster.cpp references SWEmuleChip (under #ifdef TT_UMD_BUILD_EMULE) but
# never includes umd/device/chip/sw_emule_chip.hpp. Idempotent in-place patch.
UMD_CLUSTER="$TT_METAL_DIR/tt_metal/third_party/umd/device/cluster.cpp"
if [ -f "$UMD_CLUSTER" ] && ! grep -q 'sw_emule_chip.hpp' "$UMD_CLUSTER"; then
    echo "== Patching $UMD_CLUSTER to include sw_emule_chip.hpp =="
    sed -i '/^#include "umd\/device\/chip\/mock_chip.hpp"$/a #include "umd/device/chip/sw_emule_chip.hpp"' \
        "$UMD_CLUSTER"
    grep -n 'sw_emule_chip.hpp\|mock_chip.hpp' "$UMD_CLUSTER" | head -3
fi

echo "== Configuring tt-metal (out-of-tree, with -DTT_METAL_USE_EMULE=ON) =="
cmake -B "$TT_METAL_BUILD" -S "$TT_METAL_DIR" -G Ninja \
    -DCMAKE_C_COMPILER=clang-20 \
    -DCMAKE_CXX_COMPILER=clang++-20 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$TT_METAL_BUILD" \
    -DCPM_SOURCE_CACHE="$TT_METAL_DIR/.cpmcache" \
    -DENABLE_CCACHE=ON \
    -DWITH_PYTHON_BINDINGS=ON \
    -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON \
    -DENABLE_TRACY=OFF \
    -DENABLE_LIBCXX=OFF \
    -DENABLE_DISTRIBUTED=ON \
    -DTT_USE_SYSTEM_SFPI=OFF \
    -DTT_METAL_USE_EMULE=ON \
    -DTT_EMULE_PATH="$TT_EMULE_DIR"

echo ""
echo "== Building tt-metal =="
cmake --build "$TT_METAL_BUILD" -j"$(nproc)"

# Mirror the build->build_Release symlink tt-mlir's ExternalProject's
# create_build_symlink step usually makes; downstream paths reference it.
ln -sfn "$TT_METAL_BUILD" "$TT_METAL_DIR/build"

# ---------------------------------------------------------------------------
# 2. Pre-stamp tt-mlir's tt-metal ExternalProject_Add so cmake --build doesn't
#    re-configure or rebuild our pre-built tree.
# ---------------------------------------------------------------------------
STAMP_DIR="$TT_MLIR_DIR/third_party/tt-metal/src/tt-metal-stamp"
mkdir -p "$STAMP_DIR"
for step in mkdir download update patch configure build install create_build_symlink; do
    touch "$STAMP_DIR/tt-metal-$step"
done
echo ""
echo "== Pre-stamped tt-metal ExternalProject steps under $STAMP_DIR =="
ls -la "$STAMP_DIR"

# ---------------------------------------------------------------------------
# 3. Source tt-mlir env/activate now (deferred from earlier) — needed for the
#    venv-installed python deps the tt-mlir build pulls in (StableHLO bindings
#    etc.). Its TT_METAL_HOME override is harmless here.
# ---------------------------------------------------------------------------
echo ""
echo "== Sourcing tt-mlir env/activate =="
cd "$TT_MLIR_DIR"
set +u
# shellcheck disable=SC1091
source env/activate
set -u
which python && python --version

# ---------------------------------------------------------------------------
# 4. Configure and build tt-mlir against the pre-built tt-metal source tree.
# ---------------------------------------------------------------------------
echo ""
echo "== Configuring tt-mlir (TTMLIR_TTMETAL_SOURCE_DIR=$TT_METAL_DIR) =="
cmake -B "$BUILD_DIR" -S "$TT_MLIR_DIR" -G Ninja \
    -DCMAKE_C_COMPILER=clang-20 \
    -DCMAKE_CXX_COMPILER=clang++-20 \
    -DCMAKE_BUILD_TYPE=Release \
    -DTTMLIR_ENABLE_RUNTIME=ON \
    -DTT_RUNTIME_ENABLE_TTMETAL=ON \
    -DTTMLIR_ENABLE_STABLEHLO=ON \
    -DTTMLIR_TTMETAL_SOURCE_DIR="$TT_METAL_DIR" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

echo ""
echo "== Building tt-mlir =="
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo ""
echo "== ccache stats =="
ccache --show-stats || true

echo ""
echo "== Build artifacts =="
ls -la "$BUILD_DIR/lib/" 2>/dev/null | head -20 || true
echo "..."
ls -la "$TT_METAL_BUILD/lib/" 2>/dev/null | head -20 || true
