#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# emule env wrapper for perf/bench_eltwise_unary.py.
#
# Sets the same environment the ttnn pytest regression uses (slow dispatch,
# mock cluster, emule mode) and runs the portable benchmark with --backend
# emule. All extra args are forwarded to the python script, e.g.:
#
#   perf/run_bench_emule.sh --op exp --side-tiles 1 2 4 8 16 32 --out /tmp/emule.csv
#
# Required env:
#   TT_METAL_DIR  — path to tt-metal source tree (with build_emule/)
# Optional env:
#   BUILD_DIR     — default $TT_METAL_DIR/build_emule
#   PYTHON_BIN    — default /opt/ttmlir-toolchain/venv/bin/python
#   ARCH          — wormhole | blackhole (selects mock cluster desc); default wormhole

set -euo pipefail

TT_METAL_DIR="${TT_METAL_DIR:?TT_METAL_DIR must be set}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
PYTHON_BIN="${PYTHON_BIN:-/opt/ttmlir-toolchain/venv/bin/python}"
ARCH="${ARCH:-wormhole}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"

case "$ARCH" in
    wormhole)  CLUSTER_DESC="$CLUSTER_EXAMPLES/wormhole_N150.yaml"; MESH=N150 ;;
    blackhole) CLUSTER_DESC="$CLUSTER_EXAMPLES/blackhole_P150.yaml"; MESH=P150 ;;
    *) echo "ERROR: ARCH must be wormhole|blackhole, got '$ARCH'" >&2; exit 1 ;;
esac

# Post-build symlink so `import ttnn` resolves to the emule-built _ttnn.so.
if [ ! -e "$TT_METAL_DIR/ttnn/ttnn/_ttnn.so" ]; then
    ln -sfn "$BUILD_DIR/ttnn/_ttnn.so" "$TT_METAL_DIR/ttnn/ttnn/_ttnn.so"
fi

export PYTHONPATH="$TT_METAL_DIR/ttnn:$TT_METAL_DIR/tools:$BUILD_DIR/lib:$TT_METAL_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_HOME="$TT_METAL_DIR"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_DESC"
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export MESH_DEVICE="$MESH"

exec "$PYTHON_BIN" "$SCRIPT_DIR/bench_eltwise_unary.py" --backend "emule-$ARCH" "$@"
