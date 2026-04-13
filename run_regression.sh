#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_METAL_DIR="${TT_METAL_DIR:-/localdev/arminale/tt-metal}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule_clang}"
TEST_DIR="$BUILD_DIR/test/tt_emule"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"

LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/regression_$(date +%Y%m%d_%H%M%S).log}"
PASS=0; FAIL=0; SKIP=0

run_test() {
    local name="$1"; shift
    if [ ! -f "$1" ]; then
        echo "  SKIP: $name (binary not found)"
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
        echo "  SKIP: $name (binary not found)"
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

echo "========================================"
echo " tt-emule Regression Tests"
echo "========================================"
echo "tt-metal: $TT_METAL_DIR"
echo "build:    $BUILD_DIR"

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
unset TT_METAL_MOCK_CLUSTER_DESC_PATH TT_METAL_EMULATED_MODE TT_METAL_SLOW_DISPATCH_MODE 2>/dev/null || true

run_test "bit_utils"          "$TEST_DIR/test_bit_utils"
run_test "host_buffer"        "$TEST_DIR/test_host_buffer"
run_test "tilize_untilize"    "$TEST_DIR/test_tilize_untilize"
run_test "blockfloat_common"  "$TEST_DIR/test_blockfloat_common"
run_test "emulation_toggle_default" "$TEST_DIR/test_emulation_toggle" --gtest_filter="EmulationToggle.*"

for t in adjacent contains intersects merge iterator; do
    run_test "CoreRange_$t" "$TEST_DIR/test_CoreRange_$t"
done

for t in construct contains intersects merge; do
    run_test "CoreRangeSet_$t" "$TEST_DIR/test_CoreRangeSet_$t"
done

run_test "dst_capacity_bf16"  "$TEST_DIR/test_dst_capacity_bf16"
run_test "dst_capacity_fp32"  "$TEST_DIR/test_dst_capacity_fp32"

# Tier 2+3: Buffer I/O + JIT (wormhole for buffer tests)
echo ""
echo "== Tier 2: Buffer I/O =="

export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N150.yaml"
export TT_METAL_EMULATED_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"

run_test "SimpleL1Buffer"   "$TEST_DIR/test_simple_l1_buffer" --gtest_filter="MeshDeviceFixture.TestSimpleL1Buffer*"
run_test "SimpleDramBuffer" "$TEST_DIR/test_simple_dram_buffer"
run_test "emulation_toggle_active" "$TEST_DIR/test_emulation_toggle"

echo ""
echo "== Tier 3: JIT Kernel Execution =="

run_test "TensixL1Tile"     "$TEST_DIR/test_simple_l1_buffer" --gtest_filter="*Tensix*"

echo ""
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/quasar_1chip.yaml"
export ARCH_NAME=QUASAR

echo ""
echo "== Tier 3b: DFB Multi-P/C STRIDED (test_dataflow_buffer) =="

DFB_TEST="$TEST_DIR/test_dataflow_buffer"

# Group A: DM-DM multi-P/C explicit sync (exact filters avoid matching Tensix variants)
run_test "DMTest1xDFB1Sx1S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB1Sx1S/ImplicitSyncFalse"
run_test "DMTest1xDFB1Sx1S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB1Sx1S/ImplicitSyncTrue"
run_test "DMTest1xDFB1Sx4S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB1Sx4S/ImplicitSyncFalse"
run_test "DMTest1xDFB1Sx4S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB1Sx4S/ImplicitSyncTrue"
run_test "DMTest1xDFB4Sx1S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB4Sx1S/ImplicitSyncFalse"
run_test "DMTest1xDFB4Sx1S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB4Sx1S/ImplicitSyncTrue"
run_test "DMTest1xDFB4Sx4S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB4Sx4S/ImplicitSyncFalse"
run_test "DMTest1xDFB4Sx4S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB4Sx4S/ImplicitSyncTrue"
run_test "DMTest1xDFB2Sx4S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB2Sx4S/ImplicitSyncFalse"
run_test "DMTest1xDFB2Sx4S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB2Sx4S/ImplicitSyncTrue"
run_test "DMTest1xDFB4Sx2S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB4Sx2S/ImplicitSyncFalse"
run_test "DMTest1xDFB4Sx2S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTest1xDFB4Sx2S/ImplicitSyncTrue"

echo ""
echo "== Tier 3c: DFB Compute Bridge Tests (Groups B/C) =="

# Group B: DM producer → Tensix consumer
run_test "DMTensixTest1xDFB1Sx1S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB1Sx1S/ImplicitSyncFalse"
run_test "DMTensixTest1xDFB1Sx1S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB1Sx1S/ImplicitSyncTrue"
run_test "DMTensixTest1xDFB1Sx4S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB1Sx4S/ImplicitSyncFalse"
run_test "DMTensixTest1xDFB1Sx4S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB1Sx4S/ImplicitSyncTrue"
run_test "DMTensixTest1xDFB4Sx1S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB4Sx1S/ImplicitSyncFalse"
run_test "DMTensixTest1xDFB4Sx1S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB4Sx1S/ImplicitSyncTrue"
run_test "DMTensixTest1xDFB4Sx4S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB4Sx4S/ImplicitSyncFalse"
run_test "DMTensixTest1xDFB4Sx4S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB4Sx4S/ImplicitSyncTrue"
run_test "DMTensixTest1xDFB2Sx4S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB2Sx4S/ImplicitSyncFalse"
run_test "DMTensixTest1xDFB2Sx4S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB2Sx4S/ImplicitSyncTrue"
run_test "DMTensixTest1xDFB4Sx2S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB4Sx2S/ImplicitSyncFalse"
run_test "DMTensixTest1xDFB4Sx2S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.DMTensixTest1xDFB4Sx2S/ImplicitSyncTrue"

# Group C: Tensix producer → DM consumer
run_test "TensixDMTest1xDFB1Sx1S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB1Sx1S/ImplicitSyncFalse"
run_test "TensixDMTest1xDFB1Sx1S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB1Sx1S/ImplicitSyncTrue"
run_test "TensixDMTest1xDFB1Sx4S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB1Sx4S/ImplicitSyncFalse"
run_test "TensixDMTest1xDFB1Sx4S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB1Sx4S/ImplicitSyncTrue"
run_test "TensixDMTest1xDFB4Sx1S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB4Sx1S/ImplicitSyncFalse"
run_test "TensixDMTest1xDFB4Sx1S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB4Sx1S/ImplicitSyncTrue"
run_test "TensixDMTest1xDFB4Sx4S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB4Sx4S/ImplicitSyncFalse"
run_test "TensixDMTest1xDFB4Sx4S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB4Sx4S/ImplicitSyncTrue"
run_test "TensixDMTest1xDFB2Sx4S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB2Sx4S/ImplicitSyncFalse"
run_test "TensixDMTest1xDFB2Sx4S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB2Sx4S/ImplicitSyncTrue"
run_test "TensixDMTest1xDFB4Sx2S" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB4Sx2S/ImplicitSyncFalse"
run_test "TensixDMTest1xDFB4Sx2S_IS" "$DFB_TEST" \
    --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.TensixDMTest1xDFB4Sx2S/ImplicitSyncTrue"

echo ""
echo "== Tier 3d: DFB Multi-DFB Pipeline (Group D) =="

# Group D: DM→Tensix→DM multi-DFB pipeline
run_test "DMTensixDMTest2xDFB1Sx1S" "$DFB_TEST" \
    --gtest_filter="MeshDeviceFixture.DMTensixDMTest2xDFB1Sx1S"
run_test "DMTensixDMTest1xDFB2Sx1S1xDFB1Sx2S" "$DFB_TEST" \
    --gtest_filter="MeshDeviceFixture.DMTensixDMTest1xDFB2Sx1S1xDFB1Sx2S"
run_test "DMTensixDMTest1xDFB4Sx1S1xDFB1Sx4S" "$DFB_TEST" \
    --gtest_filter="MeshDeviceFixture.DMTensixDMTest1xDFB4Sx1S1xDFB1Sx4S"

echo ""
echo "== Tier 3e: DFB BLOCKED Consumer =="

# BLOCKED DM-DM
for test in DMTest1xDFB1Sx4B DMTest1xDFB4Sx1B DMTest1xDFB4Sx4B DMTest1xDFB4Sx2B DMTest1xDFB2Sx4B; do
    run_test "${test}" "$DFB_TEST" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncFalse"
    run_test "${test}_IS" "$DFB_TEST" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncTrue"
done

# BLOCKED DM→Tensix
for test in DMTensixTest1xDFB1Sx4B DMTensixTest1xDFB4Sx1B DMTensixTest1xDFB4Sx4B DMTensixTest1xDFB4Sx2B DMTensixTest1xDFB2Sx4B; do
    run_test "${test}" "$DFB_TEST" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncFalse"
    run_test "${test}_IS" "$DFB_TEST" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncTrue"
done

# BLOCKED Tensix→DM
for test in TensixDMTest1xDFB1Sx4B TensixDMTest1xDFB4Sx1B TensixDMTest1xDFB4Sx4B TensixDMTest1xDFB4Sx2B TensixDMTest1xDFB2Sx4B; do
    run_test "${test}" "$DFB_TEST" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncFalse"
    run_test "${test}_IS" "$DFB_TEST" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncTrue"
done

echo ""
echo "== Tier 3f: DFB Multi-Core =="

for test in MultiCoreDMTest2Core_1Sx1S MultiCoreDMTest2Core_2Sx2S; do
    run_test "${test}" "$DFB_TEST" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncFalse"
    run_test "${test}_IS" "$DFB_TEST" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncTrue"
done

# BLOCKED multi-core
for test in MultiCoreDMTest2Core_1Sx4B; do
    run_test "${test}" "$DFB_TEST" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncFalse"
    run_test "${test}_IS" "$DFB_TEST" \
        --gtest_filter="ImplicitSync/DFBImplicitSyncParamFixture.${test}/ImplicitSyncTrue"
done

echo ""
echo "== Tier 3g: DFB Config Validation =="

DFB_CFG_TEST="$TEST_DIR/test_dataflow_buffer_configs"

# STRIDED config tests (7 passing)
for test in DMTest1xDFB1Sx4SConfig DMTensixTest1xDFB4Sx1SConfig DMTest1xDFB4Sx1SConfig \
            DMTest1xDFB4Sx4SConfig DMTest1xDFB2Sx4SConfig DMTest1xDFB4Sx2SConfig \
            DMTest1xDFB1Sx1BConfig; do
    run_test "${test}" "$DFB_CFG_TEST" \
        --gtest_filter="MeshDeviceFixture.${test}"
done

# Multi-core config tests (3 passing)
for test in MultiCoreDFB_1P1C_Strided_NoImplicitSync MultiCoreDFB_1P1C_Strided_ImplicitSync \
            MultiCoreDFB_HomogeneousGrid_SingleGroup; do
    run_test "${test}" "$DFB_CFG_TEST" \
        --gtest_filter="MeshDeviceFixture.${test}"
done

echo ""
echo "== Tier 3h: Quasar Compute Kernel Tests =="

# Create a fake simulator directory with a valid soc_descriptor.yaml so that
# get_simulator_enabled() returns true (the env var is set and the path is non-
# empty) while the actual device is still emulated (ChipType::EMULATED).
EMULE_SIM_DIR="$(mktemp -d /tmp/tt_emule_sim.XXXXXX)"
ln -sf "$TT_METAL_DIR/tt_metal/soc_descriptors/quasar_32_arch.yaml" "$EMULE_SIM_DIR/soc_descriptor.yaml"
export TT_METAL_SIMULATOR="$EMULE_SIM_DIR"

run_test "QuasarComputeKernelMultipleThreads" "$TEST_DIR/test_quasar_compute_kernels" \
    --gtest_filter="*QuasarComputeKernelMultipleThreads*"
run_test "QuasarComputeKernelSingleThread" "$TEST_DIR/test_quasar_compute_kernels" \
    --gtest_filter="*QuasarComputeKernelSingleThread*"
run_test "QuasarCreateMultipleComputeKernelsSingleCluster" "$TEST_DIR/test_quasar_compute_kernels" \
    --gtest_filter="*QuasarCreateMultipleComputeKernelsSingleCluster*"

run_test "QuasarComputeKernelTLS" "$TEST_DIR/test_globals_tls" \
    --gtest_filter="*QuasarComputeKernelTLS*"

run_test "QuasarMatmulBlock" "$TEST_DIR/test_matmul_X_tile" \
    --gtest_filter="*TensixMatmulBlock"
run_test "QuasarMatmulBlockInitShort" "$TEST_DIR/test_matmul_X_tile" \
    --gtest_filter="*TensixMatmulBlockInitShort" --gtest_also_run_disabled_tests

echo ""
echo "== Tier 3i: Quasar Semaphore Tests =="

run_test "QuasarComputeKernelSemaphores" "$TEST_DIR/test_quasar_semaphores" \
    --gtest_filter="*QuasarComputeKernelSemaphores*"
run_test "QuasarDmAndComputeKernelSemaphores" "$TEST_DIR/test_quasar_semaphores" \
    --gtest_filter="*QuasarDmAndComputeKernelSemaphores*"
run_test "DmLoopback" "$TEST_DIR/test_dm_loopback" \
    --gtest_filter="*DmLoopback*"

echo ""
echo "== Tier 3j: Simple DM + RISCV Atomics =="

run_test "SingleDmL1Write" "$TEST_DIR/test_single_dm_l1_write" \
    --gtest_filter="*SingleDmL1Write*"
run_test "TestAtomicLoadStoreRISCV" "$TEST_DIR/test_riscv_atomics" \
    --gtest_filter="*TestAtomicLoadStoreRISCV*"
run_test "TestAtomicAddFetchRISCV" "$TEST_DIR/test_riscv_atomics" \
    --gtest_filter="*TestAtomicAddFetchRISCV*"
run_test "TestAtomicCASRISCV" "$TEST_DIR/test_riscv_atomics" \
    --gtest_filter="*TestAtomicCASRISCV*"

unset TT_METAL_SIMULATOR
rm -rf "$EMULE_SIM_DIR"

echo ""
echo "== Tier 3k: Data Movement Tests (Phase 8) =="

unset ARCH_NAME
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N150.yaml"

run_test "DmLoopbackPacketSizes" "$TEST_DIR/test_dm_loopback_noc" \
    --gtest_filter="*LoopbackPacketSizes*"
run_test "DmLoopbackDirectedIdeal" "$TEST_DIR/test_dm_loopback_noc" \
    --gtest_filter="*LoopbackDirectedIdeal*"
run_test "DmOneFromOnePacketSizes" "$TEST_DIR/test_dm_one_from_one" \
    --gtest_filter="*OneFromOnePacketSizes" --gtest_also_run_disabled_tests
run_test "DmOneFromOneDirectedIdeal" "$TEST_DIR/test_dm_one_from_one" \
    --gtest_filter="*OneFromOneDirectedIdeal*"

# Tier 4: TTNN (blackhole — larger worker grid, no wormhole mmap exhaustion for ttnn)
echo ""
echo "== Tier 4: TTNN Relational INT32 =="

export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/blackhole_P100.yaml"

run_test "ttnn_relational" "$TEST_DIR/test_ttnn_relational_int"
run_test "ttnn_add_int_emulated" "$TEST_DIR/test_ttnn_add_int"

# Tier 5: TTNN Matmul Sweep (wormhole — multi-core matmul with semaphore sync)
echo ""
echo "== Tier 5: TTNN Matmul Sweep =="

export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N150.yaml"

run_test_verbose "ttnn_matmul_sweep" "$TEST_DIR/test_ttnn_matmul_sweep"

# Tier 6: Silicon toggle proof (requires real hardware)
echo ""
echo "== Tier 6: Silicon (toggle proof) =="

unset TT_METAL_MOCK_CLUSTER_DESC_PATH TT_METAL_EMULATED_MODE TT_METAL_SLOW_DISPATCH_MODE 2>/dev/null || true

run_test "ttnn_add_int_silicon" "$TEST_DIR/test_ttnn_add_int"

echo ""
echo "========================================"
echo " Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "========================================"

[ "$FAIL" -eq 0 ]
