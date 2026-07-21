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
FIBER_ASAN_BIN="$TEST_DIR/unit_tests_fiber_asan"
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

# CI_TIER selects which entries run: full (all, default) | pr (only PR_TIER) |
# deferred (the rest); see ci-regression-all.sh. PR_TIER is the wormhole C++
# smoke set for the PR gate — skewed to WH-specific divergence from blackhole
# (the full arch): DRAM bank topology, tilize/format, basic L1/JIT/NOC/reduce.
CI_TIER="${CI_TIER:-full}"
PR_TIER=(
    fiber_asan_isolation
    tilize_untilize
    SimpleL1Buffer
    SimpleDramBuffer
    TensixL1Tile
    # DramUnaryDRAMChannels / DmLoopbackPacketSizes removed from PR tier — the 8 DM
    # tests are disabled below pending the emule fix for tt-metal #49437 brisc-link
    # regression (see tracking issue #280).
    ttnn_sum_last_dim_aligned
)
_pr_tier_has() { local n="$1" e; for e in "${PR_TIER[@]}"; do [ "$e" = "$n" ] && return 0; done; return 1; }
# Return 0 (skip this entry) when CI_TIER excludes it.
_tier_skip() {
    case "$CI_TIER" in
        full)     return 1 ;;
        pr)       _pr_tier_has "$1" && return 1 || return 0 ;;
        deferred) _pr_tier_has "$1" && return 0 || return 1 ;;
        *) echo "ERROR: bad CI_TIER='$CI_TIER' (full|pr|deferred)" >&2; exit 2 ;;
    esac
}

run_test() {
    local name="$1"; shift
    if _tier_skip "$name"; then SKIP=$((SKIP + 1)); return; fi
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
    if _tier_skip "$name"; then SKIP=$((SKIP + 1)); return; fi
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

start_time=$(date +%s)

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

# Per-fiber ASAN sanitizer-state isolation (pure unit test, no device). Regression
# fence for the fiber-engine per-fiber-state fix — see EmuleSanitizerState.
run_test "fiber_asan_isolation" "$FIBER_ASAN_BIN"

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

run_test "dst_capacity_bf16"  "$API_BIN" --gtest_filter="DstJitBF16.*"
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
# Tier 3a: ASan Checks
# ===========================================================================
echo ""
echo "== Tier 3a: ASan Checks =="

# Filters use globs so positive controls + future additions are picked up
# automatically — any change to an emule check should be validated by re-running
# this block (see SANITIZER_CHECKS.md).
# The DRAM-read alignment death/control tests are arch-specific (WH = 32 B, BH =
# 64 B): exclude the _BH variants here — under wormhole_N150.yaml they'd fail the
# 32 B rule / expect the 64 B message. blackhole runs the mirror set (excludes _WH).
run_test "alignment_writes"       "$API_BIN" --gtest_filter="MeshDeviceFixture.Noc*:-MeshDeviceFixture.*_BH"
run_test "cb_leak"                "$API_BIN" --gtest_filter="MeshDeviceFixture.Dirty_CB_*"
# CB_Reservation: the *Overflow* death-tests are grouped ALONE (no in-parent
# LaunchProgram before them), and the non-death ExactCapacity control runs in a
# separate invocation. Under the fiber runtime a prior non-death LaunchProgram
# leaves the parent multi-threaded, so a later EXPECT_DEATH fork inherits the
# worker pool and the child deadlocks — split so each death-test forks clean.
run_test "cb_pages"               "$API_BIN" --gtest_filter="MeshDeviceFixture.CB_Reservation_Overflow_*"
run_test "cb_reserve_exact"       "$API_BIN" --gtest_filter="MeshDeviceFixture.CB_Reservation_ExactCapacity_NoViolation"
run_test "host_alignment"         "$API_BIN" --gtest_filter="MeshDeviceFixture.Host_Alignment_*"
run_test "metadata_size"          "$API_BIN" --gtest_filter="MeshDeviceFixture.Metadata_*"
run_test "noc_without_barrier"    "$API_BIN" --gtest_filter="MeshDeviceFixture.NoC_Barrier_*"
run_test "padded_write"           "$API_BIN" --gtest_filter="MeshDeviceFixture.Tensor_Padding_*"
run_test "semaphore_write"        "$API_BIN" --gtest_filter="MeshDeviceFixture.Semaphore_*"
run_test "tensor_bad_access"      "$API_BIN" --gtest_filter="MeshDeviceFixture.Host_UAF_*"
# Object_Intent: the fiber scheduler now restores the per-fiber resolved-range log on
# swap-in and launch_cores re-runs the pre/post Object-Intent snapshot+verify around each
# single-kernel core (tt-emule #241), so the *Violation* death-tests fire again. Grouped
# apart from the non-death controls so each EXPECT_DEATH forks clean (same fork-after-parent-
# threads split as CB_Reservation / CB_Boundary above).
run_test "object_intent_violation" "$API_BIN" --gtest_filter="MeshDeviceFixture.Object_Intent_*Violation*:-MeshDeviceFixture.*NoViolation*"
run_test "object_intent_controls"  "$API_BIN" --gtest_filter="MeshDeviceFixture.Object_Intent_*NoViolation*"
# CB_Boundary checks use fiber-local CB reserved-pages state (not the range thread_locals),
# so they fire correctly — they only need the death-tests grouped apart from the non-death
# controls (same fork-after-parent-threads split as CB_Reservation above).
run_test "write_beyond_res_pages" "$API_BIN" --gtest_filter="MeshDeviceFixture.CB_Boundary_*Violation*:-MeshDeviceFixture.*NoViolation*"
run_test "cb_boundary_controls"   "$API_BIN" --gtest_filter="MeshDeviceFixture.CB_Boundary_*NoViolation*"
run_test "write_outside_tensor"   "$API_BIN" --gtest_filter="MeshDeviceFixture.OOB_Tensor_*"

# ===========================================================================
# Tier 3k: Data Movement Tests (Phase 8)
# ===========================================================================
echo ""
echo "== Tier 3k: Data Movement Tests (Phase 8) =="

# WORKAROUND: see .claude/skills/workarounds (WA-2)
# DISABLED pending emule fix for tt-metal #49437 (Metal 2.0 MakeMeshWorkloadFromSpecs)
# brisc-link regression: kernel_brisc.ld:26 non-constant addr → sender kernel fails to
# link on the a00b003 pin. tt-metal CI is green on these; emule-build-specific. Re-enable
# once fixed (see tracking issue #280).
# run_test "DmLoopbackPacketSizes" "$DM_BIN" \
#     --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementLoopbackPacketSizes"
# run_test "DmLoopbackDirectedIdeal" "$DM_BIN" \
#     --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementLoopbackDirectedIdeal"
# run_test "DmOneFromOnePacketSizes" "$DM_BIN" \
#     --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementOneFromOnePacketSizes" \
#     --gtest_also_run_disabled_tests
# run_test "DmOneFromOneDirectedIdeal" "$DM_BIN" \
#     --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementOneFromOneDirectedIdeal"

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

# DISABLED pending emule fix for tt-metal #49437 brisc-link regression (see Tier 3k note
# above / tracking issue #280). Re-enable once fixed.
# run_test "DramUnaryPacketSizes" "$DM_BIN" \
#     --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementDRAMPacketSizes" \
#     --gtest_also_run_disabled_tests
# run_test "DramUnaryCoreLocations" "$DM_BIN" \
#     --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementDRAMCoreLocations"
# run_test "DramUnaryDRAMChannels" "$DM_BIN" \
#     --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementDRAMChannels"
# run_test "DramUnaryDirectedIdeal" "$DM_BIN" \
#     --gtest_filter="GenericMeshDeviceFixture.TensixDataMovementDRAMDirectedIdeal"

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

# ===========================================================================

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

echo ""
echo "================================================================================"
echo " Results: $PASS passed, $FAIL failed, $SKIP skipped, ran in $elapsed seconds"
echo "================================================================================"

[ "$FAIL" -eq 0 ]
