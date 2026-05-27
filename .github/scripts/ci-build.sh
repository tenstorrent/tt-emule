#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Configure and build tt-metal against the workspace's tt-emule source.
#
# Required env:
#   TT_EMULE_DIR    path to tt-emule workspace (the PR source)
#   TT_METAL_DIR    path to tt-metal workspace (checked out at pinned SHA)
#   BUILD_DIR       absolute path for the build tree (default: $TT_METAL_DIR/build_emule)
#
# Mirrors the configure + build phase of BUILD_GUIDE.md's Phase 3 so that local
# and CI runs share the same toolchain (clang-20 + libstdc++) and build dir
# (build_emule). The CI build is a subset of the guide's flags — Python
# bindings and distributed support are OFF here because the metal-regression
# CI only exercises the C++ gtest binaries.

set -euo pipefail

: "${TT_EMULE_DIR:?TT_EMULE_DIR must be set}"
: "${TT_METAL_DIR:?TT_METAL_DIR must be set}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"

export CCACHE_DIR="${CCACHE_DIR:-$HOME/.ccache}"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

echo "== ci-build.sh =="
echo "  TT_EMULE_DIR: $TT_EMULE_DIR"
echo "  TT_METAL_DIR: $TT_METAL_DIR"
echo "  BUILD_DIR:    $BUILD_DIR"
echo "  CCACHE_DIR:   $CCACHE_DIR"
echo ""

# Workaround for missing #include in tt-metal/umd at currently-pinned SHAs:
# cluster.cpp references SWEmuleChip (under #ifdef TT_UMD_BUILD_EMULE) but
# never includes umd/device/chip/sw_emule_chip.hpp. Idempotent in-place patch.
# Tracked upstream in https://github.com/tenstorrent/tt-umd/pull/2645.
UMD_CLUSTER="$TT_METAL_DIR/tt_metal/third_party/umd/device/cluster.cpp"
if [ -f "$UMD_CLUSTER" ] && ! grep -q 'sw_emule_chip.hpp' "$UMD_CLUSTER"; then
    echo "== Patching $UMD_CLUSTER to include sw_emule_chip.hpp =="
    sed -i '/^#include "umd\/device\/chip\/mock_chip.hpp"$/a #include "umd/device/chip/sw_emule_chip.hpp"' \
        "$UMD_CLUSTER"
fi

echo "== Configuring tt-metal =="
# Use tt-metal's libstdc++ toolchain file. This matches BUILD_GUIDE.md's Phase 3
# and the stdlib tt-mlir's runtime is built against, so a single toolchain
# choice covers both C++ regression (this script) and D2M (ci-build-mlir.sh).
# libstdc++ is also tt-metal upstream's default (build_metal.sh:89).
#
# Point clang at gcc-13's libstdc++ headers. Ubuntu 22.04's system libstdc++
# (gcc-11) does not implement enough of C++20 ranges to compile tt-metal —
# levelized_graph.cpp's std::ranges::for_each over a views::filter trips
# constraint failures in gcc-11's ranges_algo.h. ci-build-mlir.sh uses the
# same flag on the same runner. Local dev boxes typically default to
# gcc-12+ and don't need this; the tt-ubuntu-2204 CI runner does.
#
# The tt-emule integration is selected with -DTT_METAL_USE_EMULE=ON (no extra
# TT_ prefix). Earlier revisions of this script passed -DTT_METAL_USE_TT_EMULE
# and -DTT_METAL_EMULATION which tt-metal didn't recognize, so the build
# silently defaulted to TT_METAL_USE_EMULE=OFF and built stock tt-metal.
GCC13_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
cmake -B "$BUILD_DIR" \
    -S "$TT_METAL_DIR" \
    -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$TT_METAL_DIR/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake" \
    -DCMAKE_C_FLAGS="$GCC13_FLAGS" \
    -DCMAKE_CXX_FLAGS="$GCC13_FLAGS" \
    -DCMAKE_AR=/usr/bin/llvm-ar-20 \
    -DCMAKE_RANLIB=/usr/bin/llvm-ranlib-20 \
    -DCMAKE_BUILD_TYPE=Release \
    -DTT_METAL_USE_EMULE=ON \
    -DTT_EMULE_PATH="$TT_EMULE_DIR" \
    -DWITH_PYTHON_BINDINGS=OFF \
    -DENABLE_TRACY=OFF \
    -DENABLE_DISTRIBUTED=OFF \
    -DTT_METAL_BUILD_TESTS=ON \
    -DTTNN_BUILD_TESTS=ON \
    -DTT_INSTALL=OFF \
    -DTT_USE_SYSTEM_SFPI=OFF

echo ""
echo "== Building tt-metal =="
# Build only the specific gtest binaries that run_regression.sh exercises.
# Building "all" pulls in tests like tests/ttnn/tracy/cpp/test_get_programs_perf_data
# which links against the python-bindings ttnn target — that target doesn't exist
# under WITH_PYTHON_BINDINGS=OFF, so the link step fails.
cmake --build "$BUILD_DIR" -j"$(nproc)" --target \
    unit_tests_api \
    unit_tests_integration \
    unit_tests_legacy \
    unit_tests_data_movement \
    unit_tests_per_core_allocation \
    unit_tests_ttnn

echo ""
echo "== ccache stats =="
ccache --show-stats || true
