#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# tt-emule data_movement pytest regression — single-device N150.
#
# Runs the subset of tt-metal/tests/ttnn/unit_tests/operations/data_movement/
# pytest entries that pass cleanly under emule. The list is curated: each
# entry is either a whole file or a function/-k subset that has been
# verified all-PASS in standalone runs.
#
# Required env:
#   TT_METAL_DIR  — path to tt-metal source tree
#
# Optional env:
#   BUILD_DIR     — default $TT_METAL_DIR/build_emule
#   PYTEST_BIN    — default /opt/ttmlir-toolchain/venv/bin/pytest
#   GTEST_XML_DIR — write junit XML to this dir (CI artifact path)

set -o pipefail

TT_METAL_DIR="${TT_METAL_DIR:?TT_METAL_DIR must be set}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
PYTEST_BIN="${PYTEST_BIN:-/opt/ttmlir-toolchain/venv/bin/pytest}"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"
DM_TEST_DIR="$TT_METAL_DIR/tests/ttnn/unit_tests/operations/data_movement"

GTEST_XML_DIR="${GTEST_XML_DIR:-}"
[ -n "$GTEST_XML_DIR" ] && mkdir -p "$GTEST_XML_DIR"

PASS=0
FAIL=0

run_pytest() {
    local name="$1"; shift
    local test_path="$1"; shift
    echo "--- $name ---"
    if (
        export PYTHONPATH="$TT_METAL_DIR/ttnn:$TT_METAL_DIR/tools:$BUILD_DIR/lib:$TT_METAL_DIR:${PYTHONPATH:-}"
        export LD_LIBRARY_PATH="$BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
        export TT_METAL_HOME="$TT_METAL_DIR"
        export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
        export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/wormhole_N150.yaml"
        export TT_METAL_EMULE_MODE=1
        export TT_METAL_SLOW_DISPATCH_MODE=1
        export MESH_DEVICE=N150
        local junit_args=()
        if [ -n "$GTEST_XML_DIR" ]; then
            junit_args=(--junitxml="$GTEST_XML_DIR/${name}.xml")
        fi
        timeout 900 "$PYTEST_BIN" "$test_path" -v --tb=short --forked "${junit_args[@]}" "$@" 2>&1
    ); then
        echo "  PASS"; PASS=$((PASS + 1))
    else
        echo "  FAIL"; FAIL=$((FAIL + 1))
    fi
}

echo "========================================"
echo " data_movement pytest (N150 single-device)"
echo "========================================"
echo "  TT_METAL_DIR: $TT_METAL_DIR"
echo "  BUILD_DIR:    $BUILD_DIR"
echo "  PYTEST_BIN:   $PYTEST_BIN"
echo "  GTEST_XML_DIR: ${GTEST_XML_DIR:-<unset>}"
echo ""

# Each entry below was verified all-PASS in standalone runs during bring-up.

# Whole files (no -k filter needed)
run_pytest "dm_test_non_zero_indices"  "$DM_TEST_DIR/test_non_zero_indices.py"
run_pytest "dm_test_full"              "$DM_TEST_DIR/test_full.py"
run_pytest "dm_test_repeat_interleave" "$DM_TEST_DIR/test_repeat_interleave.py"
run_pytest "dm_test_concat_iterative"  "$DM_TEST_DIR/test_concat_iterative.py"

# Single-function entries (full pass within the function)
run_pytest "dm_test_clone_shape"                "$DM_TEST_DIR/test_clone.py::test_clone_shape"
run_pytest "dm_test_clone_callback"             "$DM_TEST_DIR/test_clone.py::test_clone_callback"
run_pytest "dm_test_creation_ones"              "$DM_TEST_DIR/test_creation.py::test_ones"
run_pytest "dm_test_creation_zeros"             "$DM_TEST_DIR/test_creation.py::test_zeros"
run_pytest "dm_test_creation_full"              "$DM_TEST_DIR/test_creation.py::test_full"
run_pytest "dm_test_creation_arange_defaults"   "$DM_TEST_DIR/test_creation.py::test_arange_defaults"
run_pytest "dm_test_creation_arange_tile"       "$DM_TEST_DIR/test_creation.py::test_arange_tile_layout"
run_pytest "dm_test_creation_empty"             "$DM_TEST_DIR/test_creation.py::test_empty"

# -k filter entries (capture passing subsets within partial-pass files)
run_pytest "dm_test_repeat"                  "$DM_TEST_DIR/test_repeat.py" -k 'not BFLOAT8_B and not test_pc_with_different'
run_pytest "dm_test_gather"                  "$DM_TEST_DIR/test_gather.py" -k 'not test_gather_general'
run_pytest "dm_test_concat_5d"               "$DM_TEST_DIR/test_concat.py" -k 'test_concat_5d'
run_pytest "dm_test_concat_many_inputs"      "$DM_TEST_DIR/test_concat.py" -k 'test_concat_many_inputs'
run_pytest "dm_test_fill_pad_float"          "$DM_TEST_DIR/test_fill_pad.py" -k 'test_fill_pad_float'
run_pytest "dm_test_fill_pad_int"            "$DM_TEST_DIR/test_fill_pad.py" -k 'test_fill_pad_int'
run_pytest "dm_test_embedding_tiled_input"   "$DM_TEST_DIR/test_embedding.py" -k 'test_embedding_tiled_input'
run_pytest "dm_test_embedding_tiled"         "$DM_TEST_DIR/test_embedding.py" -k 'test_tiled and not test_embedding_tiled'
run_pytest "dm_test_moe_embedding"           "$DM_TEST_DIR/test_embedding.py" -k 'test_moe_embedding'
run_pytest "dm_test_embedding_base_case"     "$DM_TEST_DIR/test_embedding.py" -k 'test_base_case'

echo ""
echo "========================================"
echo " Results: $PASS passed, $FAIL failed"
echo "========================================"

[ "$FAIL" -eq 0 ]
