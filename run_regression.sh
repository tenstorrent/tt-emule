#!/bin/bash
set -euo pipefail

# Regression runner for tt-emule against the upstream-rebased tt-metal.
#
# Upstream tt-metal consolidated per-test binaries into four gtest binaries:
#   unit_tests_api             — core api/DFB/simple buffer/dst tests
#   unit_tests_integration     — matmul / sfpu integration tests
#   unit_tests_legacy          — private Quasar compute / DM tests
#   unit_tests_data_movement   — phase-8 DM (loopback, one_from_one, direct_write, dram_unary)
#   unit_tests_per_core_allocation — per-core allocator mode (separate because
#                                    TT_METAL_ALLOCATOR_MODE_HYBRID must be set
#                                    before RunTimeOptions caches it)
#   unit_tests_ttnn            — ttnn op tests (add/sub/matmul/relational/sweep)
#
# Each Tier-N run_test invocation below points at one of those binaries with
# a --gtest_filter that isolates the intended test.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
TEST_DIR="$BUILD_DIR/test/tt_metal"
TTNN_TEST_DIR="$BUILD_DIR/test/ttnn"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"

# Consolidated gtest binaries
API_BIN="$TEST_DIR/unit_tests_api"
INTEGRATION_BIN="$TEST_DIR/unit_tests_integration"
LEGACY_BIN="$TEST_DIR/unit_tests_legacy"
DM_BIN="$TEST_DIR/unit_tests_data_movement"
PER_CORE_ALLOC_BIN="$TEST_DIR/unit_tests_per_core_allocation"
TTNN_BIN="$TTNN_TEST_DIR/unit_tests_ttnn"

PASS=0; FAIL=0; SKIP=0

run_test() {
    local name="$1"; shift
    if [ ! -f "$1" ]; then
        echo "  SKIP: $name (binary not found: $1)"
        SKIP=$((SKIP + 1))
        return
    fi
    echo "--- $name ---"
    if "$@" 2>&1 | tail -5; then
        echo "  PASS"
        PASS=$((PASS + 1))
    else
        echo "  FAIL"
        FAIL=$((FAIL + 1))
    fi
}

run_test_verbose() {
    local name="$1"; shift
    if [ ! -f "$1" ]; then
        echo "  SKIP: $name (binary not found: $1)"
        SKIP=$((SKIP + 1))
        return
    fi
    echo "--- $name ---"
    if "$@" 2>&1; then
        echo "  PASS"
        PASS=$((PASS + 1))
    else
        echo "  FAIL"
        FAIL=$((FAIL + 1))
    fi
}

# Negative test: command is expected to fail (non-zero exit) AND emit a marker
# string proving the failure mode is the intended one (e.g. ASan abort).
run_negative_test() {
    local name="$1"; shift
    local marker="$1"; shift
    if [ ! -f "$1" ]; then
        echo "  SKIP: $name (binary not found: $1)"
        SKIP=$((SKIP + 1))
        return
    fi
    echo "--- $name (negative) ---"
    local out rc
    out=$("$@" 2>&1) && rc=0 || rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "  FAIL ($name exited 0; expected non-zero)"
        FAIL=$((FAIL + 1))
        return
    fi
    if echo "$out" | grep -q -- "$marker"; then
        echo "  PASS (exit=$rc, marker '$marker' present)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL ($name exited $rc but did not emit '$marker')"
        FAIL=$((FAIL + 1))
    fi
}

echo "========================================"
echo " tt-emule Regression Tests"
echo "========================================"
echo "tt-metal: $TT_METAL_DIR"
echo "build:    $BUILD_DIR"

# Always wipe the JIT kernel cache so tests compile from source each run.
JIT_CACHE_DIR="/tmp/tt_emule_jit_cache_$(id -u)"
if [ -d "$JIT_CACHE_DIR" ]; then
    echo "Clearing JIT cache: $JIT_CACHE_DIR"
    rm -rf "$JIT_CACHE_DIR"
fi

# Tier 0: Standalone tt-emule tests (built in standalone CMake)
echo ""
echo "== Tier 0: Standalone =="

STANDALONE_BUILD="${STANDALONE_BUILD:-$SCRIPT_DIR/build}"
STANDALONE_TEST="$STANDALONE_BUILD/tests"

if [ -d "$STANDALONE_BUILD" ]; then
    run_test "dfb_passthrough"    "$STANDALONE_TEST/dfb_passthrough/test_dfb_passthrough"
    run_test "dfb_multi_consumer" "$STANDALONE_TEST/dfb_multi_consumer/test_dfb_mc"
    run_test "eltwise_add"        "$STANDALONE_TEST/eltwise_add/test_eltwise_add"
else
    echo "  SKIP: standalone build not found at $STANDALONE_BUILD"
fi

# Tier 1: Host-only (no env vars needed)
echo ""
echo "== Tier 1: Host-only =="
unset TT_METAL_MOCK_CLUSTER_DESC_PATH TT_METAL_EMULE_MODE TT_METAL_SLOW_DISPATCH_MODE 2>/dev/null || true

run_test "bit_utils"          "$API_BIN" \
    --gtest_filter="Host.ExtractBitArray:Host.PackBitArray:Host.PackExtractBitArray:Host.ExtractPackBitArray"
run_test "host_buffer"        "$API_BIN" --gtest_filter="HostBufferTest.*"
run_test "tilize_untilize"    "$API_BIN" --gtest_filter="TilizeUntilizeTests/*"
run_test "blockfloat_common"  "$API_BIN" --gtest_filter="BlockfloatCommonTests/*"

for t in adjacent contains intersects merge iterator; do
    # CoreCoordFixture.TestCoreRange{Adjacent,NotAdjacent,Contains,NotContains,...}
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

# test_dst_capacity (bf16 + fp32) → unit_tests_api has DstStandalone.* (host-side)
# and DstJitBF16.* (JIT-compiled BF16 variant). DstJitFP32 is not in the binary:
# upstream only compiles the BF16 jit variant. Keep the original two entries so
# the fp32 case surfaces as FAIL (missing) until the fp32 variant is wired up.
run_test "dst_capacity_bf16"  "$API_BIN" --gtest_filter="DstStandalone.*:DstJitBF16.*"
run_test "dst_capacity_fp32"  "$API_BIN" --gtest_filter="DstJitFP32.*"

# Tier 2+3: Buffer I/O + JIT (wormhole for buffer tests)
echo ""
echo "== Tier 2: Buffer I/O =="

export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N150.yaml"
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"

run_test "SimpleL1Buffer"   "$API_BIN" --gtest_filter="MeshDeviceFixture.TestSimpleL1Buffer*"
run_test "SimpleDramBuffer" "$API_BIN" --gtest_filter="MeshDeviceFixture.TestSimpleDram*"

echo ""
echo "== Tier 3: JIT Kernel Execution =="

run_test "TensixL1Tile"     "$API_BIN" --gtest_filter="MeshDeviceFixture.TensixTestSimpleL1ReadWrite*"

echo ""
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/quasar_Q1.yaml"
export ARCH_NAME=QUASAR

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

echo ""
echo "== Tier 3d: DFB Multi-DFB Pipeline (Group D) =="

run_test "DMTensixDMTest2xDFB1Sx1S" "$API_BIN" \
    --gtest_filter="MeshDeviceFixture.DMTensixDMTest2xDFB1Sx1S"
run_test "DMTensixDMTest1xDFB2Sx1S1xDFB1Sx2S" "$API_BIN" \
    --gtest_filter="MeshDeviceFixture.DMTensixDMTest1xDFB2Sx1S1xDFB1Sx2S"
run_test "DMTensixDMTest1xDFB4Sx1S1xDFB1Sx4S" "$API_BIN" \
    --gtest_filter="MeshDeviceFixture.DMTensixDMTest1xDFB4Sx1S1xDFB1Sx4S"

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

echo ""
echo "== Tier 3i: Quasar Semaphore Tests =="

run_test "QuasarComputeKernelSemaphores" "$LEGACY_BIN" \
    --gtest_filter="MeshDeviceSingleCardFixture.QuasarComputeKernelSemaphores"
run_test "QuasarDmAndComputeKernelSemaphores" "$LEGACY_BIN" \
    --gtest_filter="MeshDeviceSingleCardFixture.QuasarDmAndComputeKernelSemaphores"
run_test "DmLoopback" "$LEGACY_BIN" \
    --gtest_filter="MeshDeviceSingleCardFixture.DmLoopback"

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

echo ""
echo "== Tier 3k: Data Movement Tests (Phase 8) =="

unset ARCH_NAME
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N150.yaml"

run_test "DmLoopbackPacketSizes" "$DM_BIN" \
    --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementLoopbackPacketSizes"
run_test "DmLoopbackDirectedIdeal" "$DM_BIN" \
    --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementLoopbackDirectedIdeal"
run_test "DmOneFromOnePacketSizes" "$DM_BIN" \
    --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementOneFromOnePacketSizes" \
    --gtest_also_run_disabled_tests
run_test "DmOneFromOneDirectedIdeal" "$DM_BIN" \
    --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementOneFromOneDirectedIdeal"

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

# Tier 4: TTNN (blackhole — larger worker grid)
echo ""
echo "== Tier 4: TTNN Relational INT32 =="

export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/blackhole_P100.yaml"

run_test "ttnn_relational"        "$TTNN_BIN" --gtest_filter="RelationalUnaryTests/*"
run_test "ttnn_add_int_emulated"  "$TTNN_BIN" --gtest_filter="AddUnaryTests/*"
run_test "ttnn_sub_int"           "$TTNN_BIN" --gtest_filter="SubUnaryTests/*"
run_test "ttnn_matmul"            "$TTNN_BIN" \
    --gtest_filter="SingleTileMatmulFixture.*:MultiTileMatmulFixture.*"

# Tier 5: TTNN Matmul Sweep (wormhole)
echo ""
echo "== Tier 5: TTNN Matmul Sweep =="

export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N150.yaml"

run_test_verbose "ttnn_matmul_sweep" "$TTNN_BIN" --gtest_filter="MatmulSweep/MatmulSweepFixture.*"

# Tier 5b: TTNN Reduction (wormhole) — sum on last dim, tile-aligned 3200×64 BF16
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

# Note: a Quasar variant of this test is not exercised because the upstream
# W-reduce host factory uses CreateKernel (non-Quasar DataMovementKernel),
# which Metal rejects on Quasar with "DataMovementKernel is not supported on
# Quasar. Use QuasarDataMovementKernel instead."  This is an upstream factory
# limitation, not an emulator stub gap; revisit when the W-reduce factory
# gains a Quasar code path.

# Tier 6: Silicon toggle proof (requires real hardware)
echo ""
echo "== Tier 6: Silicon (toggle proof) =="

unset TT_METAL_MOCK_CLUSTER_DESC_PATH TT_METAL_EMULE_MODE TT_METAL_SLOW_DISPATCH_MODE 2>/dev/null || true

run_test "ttnn_add_int_silicon" "$TTNN_BIN" --gtest_filter="AddUnaryTests/*"

# Tier 7: ASan negative tests
# Each entry below is expected to fail with a non-zero exit code AND emit a
# specific marker string. The wrapper inverts both into a PASS. Tests are
# automatically picked up when their binaries exist (standalone build_asan
# for the tt-emule libemule tests; build_emule_asan/unit_tests_integration
# for the JIT-path tests that exercise AllocatorImpl per-buffer poisoning).
echo ""
echo "== Tier 7: ASan Negative =="

STANDALONE_ASAN_BUILD="${STANDALONE_ASAN_BUILD:-$SCRIPT_DIR/build_asan}"
if [ -d "$STANDALONE_ASAN_BUILD/tests/asan" ]; then
    # Positive control: writing inside an l1_alloc'd region must NOT fire
    # ASan. If this test FAILS, the alloc-side hook regressed and the
    # negative tests below would be passing on the initial blanket poison
    # alone, not real per-buffer poisoning.
    run_test "asan_inbounds_l1_alloc" \
        "$STANDALONE_ASAN_BUILD/tests/asan/test_asan_inbounds_l1_alloc"
    run_negative_test "asan_oob_slot_tail" "[EMULE]" \
        "$STANDALONE_ASAN_BUILD/tests/asan/test_asan_oob_slot_tail"
    run_negative_test "asan_oob_l1_alloc"  "[EMULE]" \
        "$STANDALONE_ASAN_BUILD/tests/asan/test_asan_oob_l1_alloc"
    run_negative_test "asan_oob_dram"      "[EMULE]" \
        "$STANDALONE_ASAN_BUILD/tests/asan/test_asan_oob_dram"
    run_negative_test "asan_oob_noc_read"  "[EMULE]" \
        "$STANDALONE_ASAN_BUILD/tests/asan/test_asan_oob_noc_read"
else
    echo "  SKIP: standalone ASan tests (build dir not found: $STANDALONE_ASAN_BUILD)"
    SKIP=$((SKIP + 5))
fi

# JIT-path tests live in unit_tests_integration; they require the wormhole
# SOC env (Tier 6 unset some of it).
if [ -f "$INTEGRATION_BIN" ]; then
    export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N150.yaml"
    export TT_METAL_EMULE_MODE=1
    export TT_METAL_SLOW_DISPATCH_MODE=1
    export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
    export TT_METAL_HOME="$TT_METAL_DIR"
    export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0}:halt_on_error=1:abort_on_error=0"
    unset ARCH_NAME 2>/dev/null || true
    rm -rf "/tmp/tt_emule_jit_cache_$(id -u)" 2>/dev/null || true
    # Positive control: alloc-hook MUST unpoison the buffer's region. If this
    # fails, the OOB negative below would be passing on the initial blanket
    # poison alone — meaning per-buffer poisoning isn't actually working.
    run_test "AsanL1BufferInBoundsWrite" "$INTEGRATION_BIN" \
        --gtest_filter="MeshDispatchFixture.AsanL1BufferInBoundsWrite"
    run_negative_test "AsanL1BufferOverflow" "AddressSanitizer:" "$INTEGRATION_BIN" \
        --gtest_filter="MeshDispatchFixture.AsanL1BufferOverflow"
    run_negative_test "AsanL1BufferUseAfterFree" "AddressSanitizer:" "$INTEGRATION_BIN" \
        --gtest_filter="MeshDispatchFixture.AsanL1BufferUseAfterFree"
    run_negative_test "AsanDramBufferUseAfterFree" "AddressSanitizer:" "$INTEGRATION_BIN" \
        --gtest_filter="MeshDispatchFixture.AsanDramBufferUseAfterFree"
else
    echo "  SKIP: JIT ASan tests (binary not found: $INTEGRATION_BIN)"
    SKIP=$((SKIP + 1))
fi

echo ""
echo "========================================"
echo " Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "========================================"

[ "$FAIL" -eq 0 ]
