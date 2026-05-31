#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# tt-emule TTNN pytest regression — single-device N150.
#
# Runs the subset of tt-metal/tests/ttnn/unit_tests/ pytest entries that pass
# cleanly under emule. The list is curated: each entry is either a whole file
# or a function/-k subset that has been verified all-PASS in standalone runs.
# Today's coverage is the data_movement suite; new domains can extend the
# entry list as they come online.
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
BF_TEST_DIR="$TT_METAL_DIR/tests/ttnn/unit_tests/base_functionality"
REDUCE_TEST_DIR="$TT_METAL_DIR/tests/ttnn/unit_tests/operations/reduce"
MATMUL_TEST_DIR="$TT_METAL_DIR/tests/ttnn/unit_tests/operations/matmul"

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
echo " TTNN pytest (N150 single-device)"
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
run_pytest "dm_test_full_like"               "$DM_TEST_DIR/test_full_like.py" -k 'not sharded'

# tests/ttnn/unit_tests/base_functionality/ — whole-file all-pass entries.
run_pytest "bf_test_as_tensor"                              "$BF_TEST_DIR/test_as_tensor.py"
run_pytest "bf_test_cluster"                                "$BF_TEST_DIR/test_cluster.py"
run_pytest "bf_test_database"                               "$BF_TEST_DIR/test_database.py"
run_pytest "bf_test_device"                                 "$BF_TEST_DIR/test_device.py"
run_pytest "bf_test_device_synchronize"                     "$BF_TEST_DIR/test_device_synchronize.py"
run_pytest "bf_test_dump_and_load"                          "$BF_TEST_DIR/test_dump_and_load.py"
run_pytest "bf_test_expand"                                 "$BF_TEST_DIR/test_expand.py"
run_pytest "bf_test_get_optimal_worker_cores_for_sharded"   "$BF_TEST_DIR/test_get_optimal_worker_cores_for_sharded_tensor.py"
run_pytest "bf_test_getitem"                                "$BF_TEST_DIR/test_getitem.py"
run_pytest "bf_test_global_circular_buffer"                 "$BF_TEST_DIR/test_global_circular_buffer.py"
run_pytest "bf_test_global_semaphore"                       "$BF_TEST_DIR/test_global_semaphore.py"
run_pytest "bf_test_grid_to_cores"                          "$BF_TEST_DIR/test_grid_to_cores.py"
run_pytest "bf_test_item"                                   "$BF_TEST_DIR/test_item.py"
run_pytest "bf_test_narrow"                                 "$BF_TEST_DIR/test_narrow.py"
run_pytest "bf_test_print_tensor"                           "$BF_TEST_DIR/test_print_tensor.py"
run_pytest "bf_test_shape"                                  "$BF_TEST_DIR/test_shape.py"
run_pytest "bf_test_squeeze"                                "$BF_TEST_DIR/test_squeeze.py"
run_pytest "bf_test_to_and_from_torch"                      "$BF_TEST_DIR/test_to_and_from_torch.py"
run_pytest "bf_test_to_dtype"                               "$BF_TEST_DIR/test_to_dtype.py"
run_pytest "bf_test_unsqueeze"                              "$BF_TEST_DIR/test_unsqueeze.py"
run_pytest "bf_test_view"                                   "$BF_TEST_DIR/test_view.py"
run_pytest "bf_test_torch_conversion"     "$BF_TEST_DIR/test_torch_conversion.py"
run_pytest "bf_test_untilize_bfloat8_b"   "$BF_TEST_DIR/test_untilize_bfloat8_b.py"
run_pytest "bf_test_comparison_mode"      "$BF_TEST_DIR/test_comparison_mode.py"

# Partial-pass entries: -k filters exclude known-failing variants (test_chunk
# and test_multi_device intentionally omitted).

run_pytest "bf_test_to_and_from_device"    "$BF_TEST_DIR/test_to_and_from_device.py" \
    -k 'not test_to_and_from_multiple_times'

run_pytest "bf_test_graph_trace_utils"     "$BF_TEST_DIR/test_graph_trace_utils.py" \
    -k 'not test_no_dispatch_vs_normal_mode_comparison and not test_normal_mode_shows_real_addresses'

# pytest -k can't match `=`/`.` in parametrize names, so the SIGFPE variant
# is dropped via --deselect (rootdir-relative nodeid — absolute path doesn't
# match, verified locally).
run_pytest "bf_test_graph_capture"         "$BF_TEST_DIR/test_graph_capture.py" \
    -k 'not test_program_cache_invalidation_across_dispatch_modes' \
    --deselect 'tests/ttnn/unit_tests/base_functionality/test_graph_capture.py::test_graph_capture[mode=RunMode.NORMAL-size=64-scalar=3]'

run_pytest "bf_test_graph_report"          "$BF_TEST_DIR/test_graph_report.py" \
    -k 'not TestDurationExtraction and not TestFastOperationGraphTracking and not test_resnet50_e2e_graph_capture'

run_pytest "bf_test_reshape"               "$BF_TEST_DIR/test_reshape.py" \
    -k 'not test_reshape_block_shard and not test_reshape_cw_div2_rm and not test_reshape_cw_mul2_rm and not test_reshape_height_shard and not test_reshape_hw_div2_rm and not test_reshape_hw_mul2_rm and not test_reshape_hw_rm_with_program_cache and not test_reshape_sharded_permute_rm and not test_reshape_sharded_rm and not test_reshape_width_shard and not test_reshape_tile'

run_pytest "bf_test_roll"                  "$BF_TEST_DIR/test_roll.py" \
    -k 'test_roll_tile_padding'

run_pytest "bf_test_tilize_untilize_2D"    "$BF_TEST_DIR/test_tilize_untilize_2D.py" \
    -k 'test_untilize_with_unpadding_2D'

run_pytest "bf_test_to_layout"             "$BF_TEST_DIR/test_to_layout.py" \
    -k 'test_int_untilize or test_tensor_to_tile_layout_shape_verification or test_to_from_01d or test_to_layout_low_perf or test_to_layout_pad_value_on_host or test_to_layout_page_error or test_to_layout_wide_tensor or test_untilize_w1 or test_untilize_w2 or test_untilize_w3 or test_untilize_w4 or test_wan22_failure'

run_pytest "bf_test_to_memory_config"      "$BF_TEST_DIR/test_to_memory_config.py" \
    -k '(test_to_memory_config_block_sharded or test_to_memory_config_rm_interleaved_to_legacy_2D_sharded_large_row or test_to_memory_config_uint16) or (test_to_memory_config and not test_to_memory_config_)'

# `test_copy_uint16` substring also matches the failing
# `test_copy_uint16_to_memory_config`; explicit exclusion handles it.
run_pytest "bf_test_copy"                  "$BF_TEST_DIR/test_copy.py" \
    -k '((test_copy_rm_interleaved_to_legacy_2D_sharded_large_row or test_copy_uint16) and not test_copy_uint16_to_memory_config) or (test_copy and not test_copy_)'

# indexed_fill: HEIGHT_SHARDED L1 tests that pass with current emule.
# B4/B2 excluded: program factory missing mode CTA arg causes TensorAccessorArgs static_assert.
# tile_layout/dim tests excluded: compute_uniform.cpp needs pack_tile 3-arg (fixed in common.h).
# block_sharded excluded: cb_reserve_back overflow due to same program factory regression.
run_pytest "dm_test_indexed_fill_sharded"   "$DM_TEST_DIR/test_indexed_fill.py::test_indexed_fill_sharded" -k 'B8-b3-D64 or B6-b4-D128'

# reduce/test_sum: test_sum function only (192 parametrizations all pass).
# test_sum_global excluded: BF16 multi-tile accumulation (batch_size=16)
# has a numeric gap unrelated to the reduce LLK shims.
# test_sum_4d/nd_shard/subcores excluded: untested.
run_pytest "reduce_test_sum" "$REDUCE_TEST_DIR/test_sum.py" -k 'test_sum and not test_sum_global and not test_sum_4d and not test_sum_nd_shard and not test_sum_subcores'

# test_tilize: only test_tilize_fp32_truncation passes cleanly today (4/4 PASS).
# Other variants fail on sharded TensorAccessor (40+), bfp4 conversion, or large-row PCC —
# all out of scope for routine LLK bring-up.
run_pytest "dm_test_tilize_fp32_truncation" "$DM_TEST_DIR/test_tilize.py" -k 'test_tilize_fp32_truncation'

# test_tilizer: single bfloat16 -> bfloat8_b device tilizer test; 1/1 PASS.
run_pytest "dm_test_tilizer" "$DM_TEST_DIR/test_tilizer.py"

# test_tosa_gather: 6/10 small-C shapes pass. C >= 96 (multi-tile C-dim)
# gathers diverge — likely a multi-tile gather kernel issue, deferred.
run_pytest "dm_test_tosa_gather" "$DM_TEST_DIR/test_tosa_gather.py" \
    --deselect "$DM_TEST_DIR/test_tosa_gather.py::test_tosa_gather_general[N=128-K=64-C=128-W=32]" \
    --deselect "$DM_TEST_DIR/test_tosa_gather.py::test_tosa_gather_general[N=2-K=32-C=96-W=32]" \
    --deselect "$DM_TEST_DIR/test_tosa_gather.py::test_tosa_gather_general[N=64-K=128-C=256-W=128]" \
    --deselect "$DM_TEST_DIR/test_tosa_gather.py::test_tosa_gather_general[N=128-K=128-C=128-W=64]"

# reduce/test_max: all 228 parametrizations across test_max, test_max_4d,
# test_max_2d, test_max_global, test_max_dim pass cleanly with no filter.
run_pytest "reduce_test_max" "$REDUCE_TEST_DIR/test_max.py"

# matmul/test_linear: only the two non-sharded, non-broadcast-bias functions
# pass uniformly today (17/17). The other 19 functions fail on dram-sharded,
# width-sharded, bias-broadcast/batched, or core-grid configs that emule
# doesn't model — separate from routine LLK bring-up.
run_pytest "matmul_test_linear" "$MATMUL_TEST_DIR/test_linear.py" -k 'test_linear_fp32_acc or test_vector_linear'

# matmul/test_addmm: only the four input-validation/negative tests pass
# uniformly (4/4). The parametric tests (square_matrices, alpha_beta,
# rectangular_matrices, etc.) have ~4.5% pass rate today — primarily small
# matrix-size PCC sensitivity and dtype coverage gaps that need a separate
# investigation, not routine LLK bring-up.
run_pytest "matmul_test_addmm" "$MATMUL_TEST_DIR/test_addmm.py" \
    -k 'test_alpha_zero_should_throw_error or test_input_tensor_with_invalid_shape or test_unsupported_dtype_should_throw_error'

echo ""
echo "========================================"
echo " Results: $PASS passed, $FAIL failed"
echo "========================================"

[ "$FAIL" -eq 0 ]
