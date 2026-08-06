#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Blackhole regression runner.
#
# Tiers:
#   1  — host-only (no device, no env vars)
#   4  — TTNN INT32 Relational / Add / Sub / Matmul (blackhole_P100.yaml)
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
echo " tt-emule Regression Tests — Blackhole"
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
# Tier 3a: ASan Checks
# ===========================================================================
echo ""
echo "== Tier 3a: ASan Checks =="

export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/blackhole_P100.yaml"
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"

# Mirror of wormhole's Tier 3a (same 12 MeshDeviceFixture.* sanitizer suites,
# same per-process/fork-safe splits). Filters use globs so positive controls +
# future additions are picked up automatically — any change to an emule check
# should be validated by re-running this block (see SANITIZER_CHECKS.md).
# The DRAM-read alignment death/control tests are arch-specific (WH = 32 B, BH =
# 64 B): exclude the _WH variants here — under blackhole_P100.yaml they'd fail
# the 64 B rule / expect the 32 B message. wormhole runs the mirror set (excludes _BH).
run_test "alignment_writes"       "$API_BIN" --gtest_filter="MeshDeviceFixture.Noc*:-MeshDeviceFixture.*_WH"
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
# Per-check selection (TT_METAL_EMULE_ASAN_CHECKS): a named check must still fire
# and an unnamed one must not. Pure env/predicate tests plus one behavioural
# control; no death tests (see the file header for why).
run_test "asan_check_selection" "$API_BIN" --gtest_filter="*Asan_Checks_*"

# ===========================================================================
# Tier 4: TTNN INT32 (blackhole_P100.yaml)
# ===========================================================================
echo ""
echo "== Tier 4: TTNN Relational INT32 =="

export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/blackhole_P100.yaml"
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"

run_test "ttnn_relational"        "$TTNN_BIN" --gtest_filter="RelationalUnaryTests/*"
run_test "ttnn_add_int_emulated"  "$TTNN_BIN" --gtest_filter="AddUnaryTests/*"
run_test "ttnn_sub_int"           "$TTNN_BIN" --gtest_filter="SubUnaryTests/*"
run_test "ttnn_matmul"            "$TTNN_BIN" \
    --gtest_filter="SingleTileMatmulFixture.*:MultiTileMatmulFixture.*"

# ===========================================================================

echo ""
echo "========================================"
echo " Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "========================================"

[ "$FAIL" -eq 0 ]
