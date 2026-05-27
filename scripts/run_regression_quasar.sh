#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Quasar regression runner.
#
# Tiers:
#   1   — host-only (no device, no env vars)
#   3b  — DFB Multi-P/C STRIDED (quasar_Q1.yaml)
#   3c  — DFB Compute Bridge Tests — DM→Tensix and Tensix→DM (quasar_Q1.yaml)
#   3d  — DFB Multi-DFB Pipeline (quasar_Q1.yaml)
#   3e  — DFB BLOCKED Consumer (quasar_Q1.yaml)
#   3f  — DFB Multi-Core (quasar_Q1.yaml)
#   3g  — DFB Config Validation (quasar_Q1.yaml)
#   3h  — Quasar Compute Kernel Tests (quasar_Q1.yaml)
#   3i  — Quasar Semaphore Tests (quasar_Q1.yaml)
#   3j  — Simple DM + RISC-V Atomics (quasar_Q1.yaml)
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
echo " tt-emule Regression Tests — Quasar"
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
# Tiers 3b–3j: Quasar DFB / Compute / Semaphores / Atomics
# ===========================================================================

export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/quasar_Q1.yaml"
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
export ARCH_NAME=QUASAR

# ===========================================================================
# Tier 3b: DFB Multi-P/C STRIDED
# ===========================================================================
echo ""
echo "== Tier 3b: DFB Multi-P/C STRIDED =="

# Group A: DM-DM multi-P/C explicit/implicit sync
run_test "DMTest1xDFB1Sx1S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB1Sx1S/ImplicitSyncFalse"
run_test "DMTest1xDFB1Sx1S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB1Sx1S/ImplicitSyncTrue"
run_test "DMTest1xDFB1Sx4S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB1Sx4S/ImplicitSyncFalse"
run_test "DMTest1xDFB1Sx4S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB1Sx4S/ImplicitSyncTrue"
run_test "DMTest1xDFB4Sx1S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB4Sx1S/ImplicitSyncFalse"
run_test "DMTest1xDFB4Sx1S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB4Sx1S/ImplicitSyncTrue"
run_test "DMTest1xDFB4Sx4S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB4Sx4S/ImplicitSyncFalse"
run_test "DMTest1xDFB4Sx4S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB4Sx4S/ImplicitSyncTrue"
run_test "DMTest1xDFB2Sx4S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB2Sx4S/ImplicitSyncFalse"
run_test "DMTest1xDFB2Sx4S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB2Sx4S/ImplicitSyncTrue"
run_test "DMTest1xDFB4Sx2S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB4Sx2S/ImplicitSyncFalse"
run_test "DMTest1xDFB4Sx2S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB4Sx2S/ImplicitSyncTrue"

# ===========================================================================
# Tier 3c: DFB Compute Bridge Tests (Groups B/C)
# ===========================================================================
echo ""
echo "== Tier 3c: DFB Compute Bridge Tests (Groups B/C) =="

# Group B: DM producer → Tensix consumer
run_test "DMTensixTest1xDFB1Sx1S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB1Sx1S/ImplicitSyncFalse"
run_test "DMTensixTest1xDFB1Sx1S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB1Sx1S/ImplicitSyncTrue"
run_test "DMTensixTest1xDFB1Sx4S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB1Sx4S/ImplicitSyncFalse"
run_test "DMTensixTest1xDFB1Sx4S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB1Sx4S/ImplicitSyncTrue"
run_test "DMTensixTest1xDFB4Sx1S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB4Sx1S/ImplicitSyncFalse"
run_test "DMTensixTest1xDFB4Sx1S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB4Sx1S/ImplicitSyncTrue"
run_test "DMTensixTest1xDFB4Sx4S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB4Sx4S/ImplicitSyncFalse"
run_test "DMTensixTest1xDFB4Sx4S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB4Sx4S/ImplicitSyncTrue"
run_test "DMTensixTest1xDFB2Sx4S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB2Sx4S/ImplicitSyncFalse"
run_test "DMTensixTest1xDFB2Sx4S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB2Sx4S/ImplicitSyncTrue"
run_test "DMTensixTest1xDFB4Sx2S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB4Sx2S/ImplicitSyncFalse"
run_test "DMTensixTest1xDFB4Sx2S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB4Sx2S/ImplicitSyncTrue"

# Group C: Tensix producer → DM consumer
run_test "TensixDMTest1xDFB1Sx1S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB1Sx1S/ImplicitSyncFalse"
run_test "TensixDMTest1xDFB1Sx1S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB1Sx1S/ImplicitSyncTrue"
run_test "TensixDMTest1xDFB1Sx4S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB1Sx4S/ImplicitSyncFalse"
run_test "TensixDMTest1xDFB1Sx4S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB1Sx4S/ImplicitSyncTrue"
run_test "TensixDMTest1xDFB4Sx1S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB4Sx1S/ImplicitSyncFalse"
run_test "TensixDMTest1xDFB4Sx1S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB4Sx1S/ImplicitSyncTrue"
run_test "TensixDMTest1xDFB4Sx4S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB4Sx4S/ImplicitSyncFalse"
run_test "TensixDMTest1xDFB4Sx4S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB4Sx4S/ImplicitSyncTrue"
run_test "TensixDMTest1xDFB2Sx4S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB2Sx4S/ImplicitSyncFalse"
run_test "TensixDMTest1xDFB2Sx4S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB2Sx4S/ImplicitSyncTrue"
run_test "TensixDMTest1xDFB4Sx2S" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB4Sx2S/ImplicitSyncFalse"
run_test "TensixDMTest1xDFB4Sx2S_IS" "$API_BIN" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB4Sx2S/ImplicitSyncTrue"

# ===========================================================================
# Tier 3d: DFB Multi-DFB Pipeline (Group D)
# ===========================================================================
echo ""
echo "== Tier 3d: DFB Multi-DFB Pipeline (Group D) =="

run_test "DMTensixDMTest2xDFB1Sx1S" "$API_BIN" \
    --gtest_filter="MeshDeviceFixture.DMTensixDMTest2xDFB1Sx1S"
run_test "DMTensixDMTest1xDFB2Sx1S1xDFB1Sx2S" "$API_BIN" \
    --gtest_filter="MeshDeviceFixture.DMTensixDMTest1xDFB2Sx1S1xDFB1Sx2S"
run_test "DMTensixDMTest1xDFB4Sx1S1xDFB1Sx4S" "$API_BIN" \
    --gtest_filter="MeshDeviceFixture.DMTensixDMTest1xDFB4Sx1S1xDFB1Sx4S"

# ===========================================================================
# Tier 3e: DFB BLOCKED Consumer
# ===========================================================================
echo ""
echo "== Tier 3e: DFB BLOCKED Consumer =="

for test in DMTest1xDFB1Sx4B DMTest1xDFB4Sx1B DMTest1xDFB4Sx4B DMTest1xDFB4Sx2B DMTest1xDFB2Sx4B; do
    run_test "${test}" "$API_BIN" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncFalse"
    run_test "${test}_IS" "$API_BIN" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncTrue"
done

for test in DMTensixTest1xDFB1Sx4B DMTensixTest1xDFB4Sx1B DMTensixTest1xDFB4Sx4B DMTensixTest1xDFB4Sx2B DMTensixTest1xDFB2Sx4B; do
    run_test "${test}" "$API_BIN" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncFalse"
    run_test "${test}_IS" "$API_BIN" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncTrue"
done

for test in TensixDMTest1xDFB1Sx4B TensixDMTest1xDFB4Sx1B TensixDMTest1xDFB4Sx4B TensixDMTest1xDFB4Sx2B TensixDMTest1xDFB2Sx4B; do
    run_test "${test}" "$API_BIN" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncFalse"
    run_test "${test}_IS" "$API_BIN" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncTrue"
done

# ===========================================================================
# Tier 3f: DFB Multi-Core
# ===========================================================================
echo ""
echo "== Tier 3f: DFB Multi-Core =="

for test in MultiCoreDMTest2Core_1Sx1S MultiCoreDMTest2Core_2Sx2S; do
    run_test "${test}" "$API_BIN" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncFalse"
    run_test "${test}_IS" "$API_BIN" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncTrue"
done

for test in MultiCoreDMTest2Core_1Sx4B; do
    run_test "${test}" "$API_BIN" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncFalse"
    run_test "${test}_IS" "$API_BIN" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncTrue"
done

# ===========================================================================
# Tier 3g: DFB Config Validation
# ===========================================================================
echo ""
echo "== Tier 3g: DFB Config Validation =="

for test in DMTest1xDFB1Sx4SConfig DMTensixTest1xDFB4Sx1SConfig DMTest1xDFB4Sx1SConfig \
            DMTest1xDFB4Sx4SConfig DMTest1xDFB2Sx4SConfig DMTest1xDFB4Sx2SConfig \
            DMTest1xDFB1Sx1BConfig; do
    run_test "${test}" "$API_BIN" \
        --gtest_filter="MeshDeviceFixture.${test}"
done

for test in MultiCoreDFB_1P1C_Strided_NoImplicitSync MultiCoreDFB_1P1C_Strided_ImplicitSync \
            MultiCoreDFB_HomogeneousGrid_SingleGroup; do
    run_test "${test}" "$API_BIN" \
        --gtest_filter="MeshDeviceFixture.${test}"
done

# ===========================================================================
# Tier 3h: Quasar Compute Kernel Tests
# ===========================================================================
echo ""
echo "== Tier 3h: Quasar Compute Kernel Tests =="

run_test "QuasarComputeKernelMultipleThreads" "$LEGACY_BIN" \
    --gtest_filter="MeshDeviceSingleCardFixture.QuasarComputeKernelMultipleThreads"
run_test "QuasarComputeKernelSingleThread" "$LEGACY_BIN" \
    --gtest_filter="MeshDeviceSingleCardFixture.QuasarComputeKernelSingleThread"
run_test "QuasarCreateMultipleComputeKernelsSingleCluster" "$LEGACY_BIN" \
    --gtest_filter="MeshDeviceSingleCardFixture.QuasarCreateMultipleComputeKernelsSingleCluster"

run_test "QuasarComputeKernelTLS" "$LEGACY_BIN" \
    --gtest_filter="MeshDeviceSingleCardFixture.QuasarComputeKernelTLS"

run_test "QuasarMatmulBlock" "$INTEGRATION_BIN" \
    --gtest_filter="MeshDispatchFixture.TensixMatmulBlock"
run_test "QuasarMatmulBlockInitShort" "$INTEGRATION_BIN" \
    --gtest_filter="MeshDispatchFixture.TensixMatmulBlockInitShort" --gtest_also_run_disabled_tests

run_test "QuasarMatmulBlockDemo" "$INTEGRATION_BIN" \
    --gtest_filter="MeshDispatchFixture.QuasarMatmulBlock"

# ===========================================================================
# Tier 3i: Quasar Semaphore Tests
# ===========================================================================
echo ""
echo "== Tier 3i: Quasar Semaphore Tests =="

run_test "QuasarComputeKernelSemaphores" "$LEGACY_BIN" \
    --gtest_filter="MeshDeviceSingleCardFixture.QuasarComputeKernelSemaphores"
run_test "QuasarDmAndComputeKernelSemaphores" "$LEGACY_BIN" \
    --gtest_filter="MeshDeviceSingleCardFixture.QuasarDmAndComputeKernelSemaphores"
run_test "DmLoopback" "$LEGACY_BIN" \
    --gtest_filter="MeshDeviceSingleCardFixture.DmLoopback"

# ===========================================================================
# Tier 3j: Simple DM + RISC-V Atomics
# ===========================================================================
echo ""
echo "== Tier 3j: Simple DM + RISCV Atomics =="

run_test "SingleDmL1Write" "$LEGACY_BIN" \
    --gtest_filter="MeshDeviceSingleCardFixture.SingleDmL1Write"
run_test "TestAtomicLoadStoreRISCV" "$LEGACY_BIN" \
    --gtest_filter="RISCVAtomicsFixture.TestAtomicLoadStoreRISCV"
run_test "TestAtomicAddFetchRISCV" "$LEGACY_BIN" \
    --gtest_filter="RISCVAtomicsFixture.TestAtomicAddFetchRISCV"
run_test "TestAtomicCASRISCV" "$LEGACY_BIN" \
    --gtest_filter="RISCVAtomicsFixture.TestAtomicCASRISCV"

unset ARCH_NAME

# ===========================================================================

echo ""
echo "========================================"
echo " Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "========================================"

[ "$FAIL" -eq 0 ]
