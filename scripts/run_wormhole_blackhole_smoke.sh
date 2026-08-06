#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Focused latest-main smoke test for the paths needed to validate tt-emule:
#   1. Wormhole JIT L1 reader/writer transfers
#   2. Wormhole matmul across fidelity, destination, and synchronization modes
#   3. Blackhole JIT L1 reader/writer transfers
#
# Required env:
#   TT_METAL_DIR  — path to the matching tt-metal source tree
# Optional env:
#   BUILD_DIR     — override $TT_METAL_DIR/build_emule

: "${TT_METAL_DIR:?Set TT_METAL_DIR to the tt-metal source tree}"

BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
API_BIN="$BUILD_DIR/test/tt_metal/unit_tests_api"
INTEGRATION_BIN="$BUILD_DIR/test/tt_metal/unit_tests_integration"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"

for required_file in \
    "$API_BIN" \
    "$INTEGRATION_BIN" \
    "$CLUSTER_EXAMPLES/wormhole_N150.yaml" \
    "$CLUSTER_EXAMPLES/blackhole_P100.yaml"; do
    if [ ! -e "$required_file" ]; then
        echo "ERROR: required file not found: $required_file" >&2
        exit 2
    fi
done

export TT_METAL_HOME="$TT_METAL_DIR"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export LD_LIBRARY_PATH="$BUILD_DIR/tt_metal:${LD_LIBRARY_PATH:-}"

SMOKE_CACHE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/tt_emule_smoke.XXXXXX")"
trap 'rm -rf "$SMOKE_CACHE_ROOT"' EXIT

run_l1_smoke() {
    local arch_name="$1"
    local descriptor="$2"

    echo "== $arch_name: JIT L1 round-trip (2/4/6 KiB) =="
    export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/$descriptor"
    export TT_EMULE_JIT_CACHE_DIR="$SMOKE_CACHE_ROOT/${arch_name}_l1"
    "$API_BIN" --gtest_filter=MeshDeviceFixture.TensixTestSimpleL1ReadWriteTileLo
}

run_l1_smoke wormhole wormhole_N150.yaml

echo "== Wormhole: single-tile matmul fidelity matrix =="
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N150.yaml"
export TT_EMULE_JIT_CACHE_DIR="$SMOKE_CACHE_ROOT/wormhole_matmul"
"$INTEGRATION_BIN" --gtest_filter=MeshDispatchFixture.TensixMatmulSingleTile

run_l1_smoke blackhole blackhole_P100.yaml

echo "All Wormhole and Blackhole smoke tests passed."
