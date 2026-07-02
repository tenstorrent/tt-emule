#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Drive tt-metal's sweep_framework on emule to collect per-op e2e_perf.
#
# Runs the two-stage upstream flow unchanged — sweeps_parameter_generator.py
# (vectors_export/) then sweeps_runner.py --perf (results_export/*.json) — under
# the emule environment, plus the few accommodations emule needs vs a silicon
# box (see below). On silicon you run the SAME two upstream commands with your
# normal tt-metal env and none of these accommodations.
#
#   perf/run_sweep_emule.sh [MODULE] [SUITE]
#   perf/run_sweep_emule.sh eltwise.unary.exp.exp nightly     # default
#
# Emule-specific accommodations applied here (NOT needed on silicon):
#   ARCH_NAME=wormhole_b0            the runner validates ARCH_NAME; the box's
#                                    default ('compute') is rejected.
#   TT_SMI_RESET_COMMAND=true        there is no device to reset and no tt-smi;
#   TT_SMI_RESET_FALLBACK_COMMAND=none  point the reset hook at a no-op.
#
# Prerequisite (one-time): the runner imports `enlighten` (progress bars):
#   /opt/ttmlir-toolchain/venv/bin/python -m pip install enlighten
#
# Required env: TT_METAL_DIR. Optional: BUILD_DIR, PYTHON_BIN, ARCH (wormhole|blackhole).

set -euo pipefail

TT_METAL_DIR="${TT_METAL_DIR:?TT_METAL_DIR must be set}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
PYTHON_BIN="${PYTHON_BIN:-/opt/ttmlir-toolchain/venv/bin/python}"
ARCH="${ARCH:-wormhole}"
MODULE="${1:-eltwise.unary.exp.exp}"
SUITE="${2:-nightly}"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"

case "$ARCH" in
    wormhole)  CLUSTER_DESC="$CLUSTER_EXAMPLES/wormhole_N150.yaml"; MESH=N150; ARCH_NAME=wormhole_b0 ;;
    blackhole) CLUSTER_DESC="$CLUSTER_EXAMPLES/blackhole_P150.yaml"; MESH=P150; ARCH_NAME=blackhole ;;
    *) echo "ERROR: ARCH must be wormhole|blackhole, got '$ARCH'" >&2; exit 1 ;;
esac

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
# emule accommodations (see header):
export ARCH_NAME
export TT_SMI_RESET_COMMAND="${TT_SMI_RESET_COMMAND:-true}"
export TT_SMI_RESET_FALLBACK_COMMAND="${TT_SMI_RESET_FALLBACK_COMMAND:-none}"

cd "$TT_METAL_DIR/tests/sweep_framework"

echo "== generating vectors: module=$MODULE suite=$SUITE =="
"$PYTHON_BIN" sweeps_parameter_generator.py --module-name "$MODULE" --suite-name "$SUITE" --tag perfrun

echo "== running sweep with --perf =="
"$PYTHON_BIN" sweeps_runner.py \
    --module-name "$MODULE" --suite-name "$SUITE" \
    --vector-source vectors_export --result-dest results_export \
    --perf --main-proc-verbose --tag perfrun

echo "== results in: $TT_METAL_DIR/tests/sweep_framework/results_export/ =="
ls -t "$TT_METAL_DIR/tests/sweep_framework/results_export/" | grep -v '^oprun' | head -1
