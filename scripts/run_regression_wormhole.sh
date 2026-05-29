#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Wormhole regression runner.
#
# Tiers:
#   1      — host-only (no device, no env vars)
#   2      — Buffer I/O (wormhole_N150.yaml)
#   3      — JIT Kernel Execution (wormhole_N150.yaml)
#   3k     — Data Movement Phase 8 (wormhole_N150.yaml)
#   3l     — DM Direct Write & DRAM Unary (wormhole_N150.yaml)
#   5      — TTNN Matmul Sweep (wormhole_N150.yaml)
#   5b     — TTNN Reduction (wormhole_N150.yaml)
#
# Required env:
#   TT_METAL_DIR  — path to tt-metal source tree (with build_emule/ inside)
# Optional env:
#   BUILD_DIR     — override default $TT_METAL_DIR/build_emule
#   GTEST_XML_DIR — write per-test XML to this directory (for CI classification)

BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
TEST_DIR="$BUILD_DIR/test/tt_metal"
TTNN_TEST_DIR="$BUILD_DIR/test/ttnn"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"

API_BIN="$TEST_DIR/unit_tests_api"
INTEGRATION_BIN="$TEST_DIR/unit_tests_integration"
LEGACY_BIN="$TEST_DIR/unit_tests_legacy"
DM_BIN="$TEST_DIR/unit_tests_data_movement"
PER_CORE_ALLOC_BIN="$TEST_DIR/unit_tests_per_core_allocation"
TTNN_BIN="$TTNN_TEST_DIR/unit_tests_ttnn"

PASS=0; FAIL=0; SKIP=0

GTEST_XML_DIR="${GTEST_XML_DIR:-}"
[ -n "$GTEST_XML_DIR" ] && mkdir -p "$GTEST_XML_DIR"

_gtest_xml_args() {
    local name="$1"
    if [ -n "$GTEST_XML_DIR" ]; then
        local safe
        safe=$(echo "$name" | tr -c '[:alnum:]._-' '_')
        echo "--gtest_output=xml:$GTEST_XML_DIR/${safe}.xml"
    fi
}

run_test() {
    local name="$1"; shift
    if [ ! -f "$1" ]; then
        echo "  FAIL: $name (binary not found: $1)"
        FAIL=$((FAIL + 1))
        return
    fi
    echo "--- $name ---"
    local xml_arg
    xml_arg="$(_gtest_xml_args "$name")"
    if [ -n "$xml_arg" ]; then
        if "$@" "$xml_arg" 2>&1 | tail -5; then
            echo "  PASS"; PASS=$((PASS + 1))
        else
            echo "  FAIL"; FAIL=$((FAIL + 1))
        fi
    else
        if "$@" 2>&1 | tail -5; then
            echo "  PASS"; PASS=$((PASS + 1))
        else
            echo "  FAIL"; FAIL=$((FAIL + 1))
        fi
    fi
}

run_test_verbose() {
    local name="$1"; shift
    if [ ! -f "$1" ]; then
        echo "  FAIL: $name (binary not found: $1)"
        FAIL=$((FAIL + 1))
        return
    fi
    echo "--- $name ---"
    local xml_arg
    xml_arg="$(_gtest_xml_args "$name")"
    if [ -n "$xml_arg" ]; then
        if "$@" "$xml_arg" 2>&1; then
            echo "  PASS"; PASS=$((PASS + 1))
        else
            echo "  FAIL"; FAIL=$((FAIL + 1))
        fi
    else
        if "$@" 2>&1; then
            echo "  PASS"; PASS=$((PASS + 1))
        else
            echo "  FAIL"; FAIL=$((FAIL + 1))
        fi
    fi
}

echo "========================================"
echo " tt-emule Regression Tests — Wormhole"
echo "========================================"
echo "tt-metal: $TT_METAL_DIR"
echo "build:    $BUILD_DIR"

JIT_CACHE_DIR="/tmp/tt_emule_jit_cache_$(id -u)"
if [ -d "$JIT_CACHE_DIR" ]; then
    echo "Clearing JIT cache: $JIT_CACHE_DIR"
    rm -rf "$JIT_CACHE_DIR"
fi

# ===========================================================================
# Tier 1: Host-only (no device, no env vars needed)
# ===========================================================================
echo ""
echo "== Tier 1: Host-only =="
unset TT_METAL_MOCK_CLUSTER_DESC_PATH TT_METAL_EMULE_MODE TT_METAL_SLOW_DISPATCH_MODE 2>/dev/null || true

run_test "bit_utils"          "$API_BIN" \
    --gtest_filter="Host.ExtractBitArray:Host.PackBitArray:Host.PackExtractBitArray:Host.ExtractPackBitArray"
run_test "host_buffer"        "$API_BIN" --gtest_filter="HostBufferTest.*"
run_test "tilize_untilize"    "$API_BIN" --gtest_filter="TilizeUntilizeTests/*"
run_test "blockfloat_common"  "$API_BIN" --gtest_filter="BlockfloatCommonTests/*"

for t in adjacent contains intersects merge iterator; do
    tcap="$(echo "$t" | sed 's/.*/\u&/')"
    run_test "CoreRange_$t" "$API_BIN" \
        --gtest_filter="CoreCoordFixture.TestCoreRange${tcap}*:CoreCoordFixture.TestCoreRangeNot${tcap}*"
done

for t in construct contains intersects merge; do
    tcap="$(echo "$t" | sed 's/.*/\u&/')"
    if [ "$t" = "construct" ]; then
        run_test "CoreRangeSet_$t" "$API_BIN" \
            --gtest_filter="CoreCoordFixture.TestCoreRangeSetValidConstruct*:CoreCoordFixture.TestCoreRangeSetInvalidConstruct*"
    else
        run_test "CoreRangeSet_$t" "$API_BIN" \
            --gtest_filter="CoreCoordFixture.TestCoreRangeSet${tcap}*:CoreCoordFixture.TestCoreRangeSetNot${tcap}*:CoreCoordFixture.TestCoreRangeSet${tcap}CoreCoord:CoreCoordFixture.TestCoreRangeSet${tcap}CoreRange:CoreCoordFixture.TestCoreRangeSet${tcap}NoSolution"
    fi
done

run_test "dst_capacity_bf16"  "$API_BIN" --gtest_filter="DstStandalone.*:DstJitBF16.*"
run_test "dst_capacity_fp32"  "$API_BIN" --gtest_filter="DstJitFP32.*"

# ===========================================================================
# Tier 2: Buffer I/O
# ===========================================================================
echo ""
echo "== Tier 2: Buffer I/O =="

export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N150.yaml"
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"

run_test "SimpleL1Buffer"   "$API_BIN" --gtest_filter="MeshDeviceFixture.TestSimpleL1Buffer*"
run_test "SimpleDramBuffer" "$API_BIN" --gtest_filter="MeshDeviceFixture.TestSimpleDram*"

# ===========================================================================
# Tier 3: JIT Kernel Execution
# ===========================================================================
echo ""
echo "== Tier 3: JIT Kernel Execution =="

run_test "TensixL1Tile"     "$API_BIN" --gtest_filter="MeshDeviceFixture.TensixTestSimpleL1ReadWrite*"

# ===========================================================================
# Tier 3k: Data Movement Tests (Phase 8)
# ===========================================================================
echo ""
echo "== Tier 3k: Data Movement Tests (Phase 8) =="

run_test "DmLoopbackPacketSizes" "$DM_BIN" \
    --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementLoopbackPacketSizes"
run_test "DmLoopbackDirectedIdeal" "$DM_BIN" \
    --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementLoopbackDirectedIdeal"
run_test "DmOneFromOnePacketSizes" "$DM_BIN" \
    --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementOneFromOnePacketSizes" \
    --gtest_also_run_disabled_tests
run_test "DmOneFromOneDirectedIdeal" "$DM_BIN" \
    --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementOneFromOneDirectedIdeal"

# ===========================================================================
# Tier 3l: DM Direct Write & DRAM Unary
# ===========================================================================
echo ""
echo "== Tier 3l: DM Direct Write & DRAM Unary =="

run_test "DmDirectWritePerformance" "$DM_BIN" \
    --gtest_filter="GenericMeshDeviceFixture.TensixDirectWritePerformanceComparison"
run_test "DmDirectWriteAddressPatterns" "$DM_BIN" \
    --gtest_filter="GenericMeshDeviceFixture.TensixDirectWriteAddressPatterns"
run_test "DmDirectWriteMulticast" "$DM_BIN" \
    --gtest_filter="GenericMeshDeviceFixture.TensixDirectWriteMulticast"

run_test "DramUnaryPacketSizes" "$DM_BIN" \
    --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementDRAMPacketSizes" \
    --gtest_also_run_disabled_tests
run_test "DramUnaryCoreLocations" "$DM_BIN" \
    --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementDRAMCoreLocations"
run_test "DramUnaryDRAMChannels" "$DM_BIN" \
    --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementDRAMChannels"
run_test "DramUnaryDirectedIdeal" "$DM_BIN" \
    --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementDRAMDirectedIdeal"

# ===========================================================================
# Tier 5: TTNN Matmul Sweep
# ===========================================================================
echo ""
echo "== Tier 5: TTNN Matmul Sweep =="

run_test_verbose "ttnn_matmul_sweep" "$TTNN_BIN" --gtest_filter="MatmulSweep/MatmulSweepFixture.*"

# ===========================================================================
# Tier 5b: TTNN Reduction
# ===========================================================================
echo ""
echo "== Tier 5b: TTNN Reduction =="

run_test "ttnn_sum_last_dim_unaligned" "$TTNN_BIN" \
    --gtest_filter="SumTensorLastDimTests/SumTensorLastDimFixture.SumTensorCorrectly/0"
run_test "ttnn_sum_last_dim_aligned" "$TTNN_BIN" \
    --gtest_filter="SumTensorLastDimTests/SumTensorLastDimFixture.SumTensorCorrectly/1"
run_test "ttnn_sum_first_dim_unaligned" "$TTNN_BIN" \
    --gtest_filter="SumTensorFirstDimTests/SumTensorFirstDimFixture.SumTensorCorrectly/0"
run_test "ttnn_sum_first_dim_aligned" "$TTNN_BIN" \
    --gtest_filter="SumTensorFirstDimTests/SumTensorFirstDimFixture.SumTensorCorrectly/1"
run_test "ttnn_sum_both_dims_unaligned" "$TTNN_BIN" \
    --gtest_filter="SumTensorBothDimsTests/SumTensorBothDimsFixture.SumTensorCorrectly/0"
run_test "ttnn_sum_both_dims_aligned" "$TTNN_BIN" \
    --gtest_filter="SumTensorBothDimsTests/SumTensorBothDimsFixture.SumTensorCorrectly/1"
run_test "ttnn_minmax_last_dim" "$TTNN_BIN" \
    --gtest_filter="MinMaxTensorLastDimTests/MinMaxTensorLastDimFixture.MinMaxTensorCorrectly/*"
run_test "ttnn_minmax_first_dim" "$TTNN_BIN" \
    --gtest_filter="MinMaxTensorFirstDimTests/MinMaxTensorFirstDimFixture.MinMaxTensorCorrectly/*"
run_test "ttnn_minmax_both_dims" "$TTNN_BIN" \
    --gtest_filter="MinMaxTensorBothDimsTests/MinMaxTensorBothDimsFixture.MinMaxTensorCorrectly/*"

echo ""
echo "========================================"
echo " Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "========================================"

[ "$FAIL" -eq 0 ]
