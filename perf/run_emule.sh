#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Run perf/bench.py against tt-emule (host software emulation, slow dispatch,
# wormhole N150). Mirrors the env block from scripts/run_ttnn_pytests_wormhole.sh.
# Any args are forwarded to bench.py.
#
#   TT_METAL_DIR=/path/to/tt-metal perf/run_emule.sh --graph all
#
set -euo pipefail

TT_METAL_DIR="${TT_METAL_DIR:?TT_METAL_DIR must be set}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
PYTHON_BIN="${PYTHON_BIN:-/opt/ttmlir-toolchain/venv/bin/python}"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$TT_METAL_DIR/ttnn:$TT_METAL_DIR/tools:$BUILD_DIR/lib:$TT_METAL_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_HOME="$TT_METAL_DIR"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N150.yaml"
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export MESH_DEVICE=N150

exec "$PYTHON_BIN" "$HERE/bench.py" --backend emule "$@"
