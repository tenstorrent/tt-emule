#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Emulation Toggle Verification
#
# Proves that TT_METAL_EMULATED_MODE env var toggles between silicon and
# emulated execution from the SAME binary. Requires:
#   - tt-metal built with TT_METAL_EMULATION=ON
#   - A Tenstorrent device on this machine (for silicon phase)
#
# Usage:
#   ./run_toggle_test.sh
#   TT_METAL_DIR=/path/to/tt-metal BUILD_DIR=/path/to/build ./run_toggle_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_METAL_DIR="${TT_METAL_DIR:-/localdev/arminale/tt-metal}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule_clang}"
TEST_DIR="$BUILD_DIR/test/tt_emule"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"

export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
PASS=0; FAIL=0; SKIP=0

run_test() {
    local name="$1"; shift
    if [ ! -f "$1" ]; then
        echo "  SKIP: $name (binary not found)"
        SKIP=$((SKIP + 1))
        return
    fi
    echo "--- $name ---"
    if "$@" 2>&1 | tail -20; then
        echo "  PASS"
        PASS=$((PASS + 1))
    else
        echo "  FAIL"
        FAIL=$((FAIL + 1))
    fi
}

echo "========================================"
echo " Emulation Toggle Verification"
echo "========================================"
echo "tt-metal: $TT_METAL_DIR"
echo "build:    $BUILD_DIR"
echo ""

# ==========================================================
# Phase 1: Silicon (no emulation env vars)
# SiliconActive tests activate and confirm TargetDevice::Silicon.
# EmulationActive tests auto-skip.
# ==========================================================
echo "== Phase 1: Silicon =="
echo "(TT_METAL_EMULATED_MODE unset -- real hardware)"
echo ""
unset TT_METAL_MOCK_CLUSTER_DESC_PATH TT_METAL_EMULATED_MODE TT_METAL_SLOW_DISPATCH_MODE 2>/dev/null || true

run_test "silicon" \
    "$TEST_DIR/test_emulation_toggle"

# ==========================================================
# Phase 2: Emulated (TT_METAL_EMULATED_MODE=1)
# EmulationActive tests activate and confirm TargetDevice::Emulated.
# SiliconActive tests auto-skip.
# ==========================================================
echo ""
echo "== Phase 2: Emulated =="
echo "(TT_METAL_EMULATED_MODE=1 -- software emulation)"
echo ""

export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N150.yaml"
export TT_METAL_EMULATED_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1

run_test "emulated" \
    "$TEST_DIR/test_emulation_toggle"

# ==========================================================
# Summary
# ==========================================================
echo ""
echo "========================================"
echo " Toggle Verification Results"
echo "========================================"
echo " Silicon:  SiliconActive.* activated, EmulationActive.* skipped"
echo " Emulated: EmulationActive.* activated, SiliconActive.* skipped"
echo ""
echo " $PASS passed, $FAIL failed, $SKIP skipped"
echo "========================================"

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "VERDICT: Toggle verification FAILED"
    exit 1
fi

if [ "$SKIP" -ne 0 ]; then
    echo ""
    echo "VERDICT: Some tests skipped (binaries not found?)"
    exit 1
fi

echo ""
echo "VERDICT: Same binaries produce correct results on both silicon and"
echo "         emulated paths. The env var toggle works."
exit 0
