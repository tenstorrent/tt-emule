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
# (build_emule). Distributed support is OFF here — only the D2M pipeline needs
# it (see ci-build-mlir.sh) — but Python bindings are ON because the ttnn-
# pytest job downstream needs to `import ttnn._ttnn`.

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
# The tt-emule integration is selected with -DTT_METAL_USE_EMULE=ON (no extra
# TT_ prefix). Earlier revisions of this script passed -DTT_METAL_USE_TT_EMULE
# and -DTT_METAL_EMULATION which tt-metal didn't recognize, so the build
# silently defaulted to TT_METAL_USE_EMULE=OFF and built stock tt-metal.
cmake -B "$BUILD_DIR" \
    -S "$TT_METAL_DIR" \
    -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$TT_METAL_DIR/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTT_METAL_USE_EMULE=ON \
    -DTT_EMULE_PATH="$TT_EMULE_DIR" \
    -DWITH_PYTHON_BINDINGS=ON \
    -DENABLE_TRACY=OFF \
    -DENABLE_DISTRIBUTED=OFF \
    -DTT_METAL_BUILD_TESTS=ON \
    -DTTNN_BUILD_TESTS=ON \
    -DTT_INSTALL=OFF \
    -DTT_USE_SYSTEM_SFPI=OFF

echo ""
echo "== Building tt-metal =="
# Build only the specific binaries downstream jobs need: the gtest binaries
# that run_regression.sh exercises, plus the `ttnn` cmake target that produces
# the _ttnn nanobind extension the ttnn-pytest job imports (`_ttnn.so` is the
# output file; `ttnn` is the cmake target name). cmake transitively builds
# their deps (libtt_metal.so, _ttnncpp.so, libtt_stl.so).
cmake --build "$BUILD_DIR" -j"$(nproc)" --target \
    unit_tests_api \
    unit_tests_integration \
    unit_tests_legacy \
    unit_tests_data_movement \
    unit_tests_per_core_allocation \
    unit_tests_ttnn \
    ttnn

echo ""
echo "== Creating post-build symlinks =="
# Mirror BUILD_GUIDE.md Phase 3's "Post-build symlinks" subsection so the
# uploaded artifact is self-contained for downstream jobs (no source-tree
# regeneration needed). actions/upload-artifact@v4 resolves symlinks during
# upload, so each symlink ends up as a file copy at the symlink path — the
# .so RPATH (`$ORIGIN/../lib`) then resolves correctly inside the artifact.
cd "$BUILD_DIR/lib"
ln -sfn ../tt_metal/libtt_metal.so libtt_metal.so
ln -sfn ../tt_stl/libtt_stl.so     libtt_stl.so
ln -sfn ../ttnn/_ttnncpp.so        _ttnncpp.so
ln -sfn ../ttnn/_ttnn.so           _ttnn.so

echo ""
echo "== ccache stats =="
ccache --show-stats || true
