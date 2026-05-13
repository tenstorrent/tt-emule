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
# Use tt-metal's libc++ toolchain file. ENABLE_LIBCXX=ON alone only verifies
# libc++ is installed (cmake/compilers.cmake:5) — it does NOT pass
# -stdlib=libc++ to the compiler. The toolchain file at
# cmake/x86_64-linux-clang-20-libcpp-toolchain.cmake sets
# CMAKE_CXX_FLAGS_INIT="-stdlib=libc++" and is what actually switches the
# stdlib. Without it, tt-metal's C++20 ranges code fails to compile against
# Ubuntu 22.04's libstdc++ (from gcc-11).
cmake -B "$BUILD_DIR" \
    -S "$TT_METAL_DIR" \
    -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$TT_METAL_DIR/cmake/x86_64-linux-clang-20-libcpp-toolchain.cmake" \
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
