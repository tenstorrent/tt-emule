#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# tt-emule TTNN pytest regression — Blackhole (single-device P100).
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
ELT_TEST_DIR="$TT_METAL_DIR/tests/ttnn/unit_tests/operations/eltwise"
FUSED_TEST_DIR="$TT_METAL_DIR/tests/ttnn/unit_tests/operations/fused"

GTEST_XML_DIR="${GTEST_XML_DIR:-}"
[ -n "$GTEST_XML_DIR" ] && mkdir -p "$GTEST_XML_DIR"

# Sharding — round-robin over run_pytest invocations. SHARD_INDEX is 1-based.
# Round-robin (not contiguous block) so that slow entries like reduce_test_max
# and reduce_test_sum distribute across shards rather than bunching together.
SHARD_INDEX="${SHARD_INDEX:-1}"
SHARD_COUNT="${SHARD_COUNT:-1}"
if ! [[ "$SHARD_COUNT" =~ ^[0-9]+$ ]] || [ "$SHARD_COUNT" -lt 1 ]; then
    echo "ERROR: SHARD_COUNT must be a positive integer (got '$SHARD_COUNT')" >&2
    exit 2
fi
if ! [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]] || [ "$SHARD_INDEX" -lt 1 ] || [ "$SHARD_INDEX" -gt "$SHARD_COUNT" ]; then
    echo "ERROR: SHARD_INDEX must be in [1, $SHARD_COUNT] (got '$SHARD_INDEX')" >&2
    exit 2
fi
ENTRY_NUM=0

PASS=0
FAIL=0

run_pytest() {
    local name="$1"; shift
    # Remaining args are passed straight to pytest: one or more test targets
    # (file or file::node) plus any flags (-k, --deselect). Passing multiple
    # node targets in a single invocation lets one file's curated subset run in
    # ONE process — amortizing the per-process `import ttnn` + collection +
    # device/cluster init (~tens of seconds) that --forked would otherwise pay
    # for every entry. Group a file's targets by their -k flags before merging.
    ENTRY_NUM=$((ENTRY_NUM + 1))
    # Skip entries not assigned to this shard.
    if [ $(( (ENTRY_NUM - 1) % SHARD_COUNT + 1 )) -ne "$SHARD_INDEX" ]; then
        return
    fi
    # Guard: with no target, pytest would collect/run the entire CWD suite — a
    # very expensive CI failure mode. Treat a target-less entry as an authoring
    # error and fail it loudly.
    if [ "$#" -eq 0 ]; then
        echo "--- $name ---"; echo "  FAIL (no test target supplied to run_pytest)"
        FAIL=$((FAIL + 1)); return
    fi
    echo "--- $name ---"
    if (
        export PYTHONPATH="$TT_METAL_DIR/ttnn:$TT_METAL_DIR/tools:$BUILD_DIR/lib:$TT_METAL_DIR:${PYTHONPATH:-}"
        export LD_LIBRARY_PATH="$BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
        export TT_METAL_HOME="$TT_METAL_DIR"
        export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
        export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/blackhole_P100.yaml"
        export TT_METAL_EMULE_MODE=1
        export TT_METAL_SLOW_DISPATCH_MODE=1
        export MESH_DEVICE=P100
        local junit_args=()
        if [ -n "$GTEST_XML_DIR" ]; then
            junit_args=(--junitxml="$GTEST_XML_DIR/${name}.xml")
        fi
        # --forked isolates each test item in its own process, but forking the
        # heavy ttnn process per item costs ~0.3s/item and dominates wall-clock.
        # Default is OFF: a file's items run in one process, relying on the
        # function-scoped device fixture for per-test reset (validated suite-wide
        # at 82/82 entries, 8176 tests, 0 failures). Set FORKED=1 to restore
        # per-item process isolation when debugging a crash or state-bleed.
        local forked_args=()
        [ "${FORKED:-0}" = "1" ] && forked_args=(--forked)
        timeout 900 "$PYTEST_BIN" -v --tb=short "${forked_args[@]}" "${junit_args[@]}" "$@" 2>&1
    ); then
        echo "  PASS"; PASS=$((PASS + 1))
    else
        echo "  FAIL"; FAIL=$((FAIL + 1))
    fi
}

echo "========================================"
echo " TTNN pytest — Blackhole (P100 single-device)"
echo "========================================"
echo "  TT_METAL_DIR: $TT_METAL_DIR"
echo "  BUILD_DIR:    $BUILD_DIR"
echo "  PYTEST_BIN:   $PYTEST_BIN"
echo "  GTEST_XML_DIR: ${GTEST_XML_DIR:-<unset>}"
echo "  SHARD:        $SHARD_INDEX of $SHARD_COUNT"
echo ""

# Each entry below was verified all-PASS in standalone runs during bring-up.

# Whole files (no -k filter needed)
run_pytest "dm_test_non_zero_indices"  "$DM_TEST_DIR/test_non_zero_indices.py"
run_pytest "dm_test_full"              "$DM_TEST_DIR/test_full.py"
run_pytest "dm_test_repeat_interleave" "$DM_TEST_DIR/test_repeat_interleave.py"
# Deselect the 128-input dim=-1 case: its tilize step over-subscribes L1 after upstream
# tt-metal #44307 added an unconditional staging CB (not an emule bug; over-budget on HW too).
run_pytest "dm_test_concat_iterative"  "$DM_TEST_DIR/test_concat_iterative.py" \
    --deselect "tests/ttnn/unit_tests/operations/data_movement/test_concat_iterative.py::test_concat_lg_tensor_1[dim=-1-input_shapes=((1, 1, 1, 5000), (1, 1, 1, 33))-tensor_layout=Layout.TILE-num_inputs=128]"

# Single-function entries (full pass within the function)
run_pytest "dm_test_clone" "$DM_TEST_DIR/test_clone.py::test_clone_shape" "$DM_TEST_DIR/test_clone.py::test_clone_callback" "$DM_TEST_DIR/test_clone.py::test_clone_dtype_conversion"
run_pytest "dm_test_creation" "$DM_TEST_DIR/test_creation.py::test_ones" "$DM_TEST_DIR/test_creation.py::test_zeros" "$DM_TEST_DIR/test_creation.py::test_full" "$DM_TEST_DIR/test_creation.py::test_arange_defaults" "$DM_TEST_DIR/test_creation.py::test_arange_tile_layout" "$DM_TEST_DIR/test_creation.py::test_empty" "$DM_TEST_DIR/test_creation.py::test_arange" "$DM_TEST_DIR/test_creation.py::test_full_with_opt_tensor" "$DM_TEST_DIR/test_creation.py::test_full_like" "$DM_TEST_DIR/test_creation.py::test_full_like_bf8b" "$DM_TEST_DIR/test_creation.py::test_empty_like" "$DM_TEST_DIR/test_creation.py::test_zeros_like" "$DM_TEST_DIR/test_creation.py::test_ones_like" "$DM_TEST_DIR/test_creation.py::test_zeros_bfp8" "$DM_TEST_DIR/test_creation.py::test_zeros_bfp4"

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
run_pytest "dm_test_full_like"               "$DM_TEST_DIR/test_full_like.py"
run_pytest "dm_test_sharded_to_interleaved_oob" "$DM_TEST_DIR/test_sharded_to_interleaved_oob.py"

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

# pytest -k can't match `=`/`.` in parametrize names
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

run_pytest "bf_test_copy"                  "$BF_TEST_DIR/test_copy.py" \
    -k '((test_copy_rm_interleaved_to_legacy_2D_sharded_large_row or test_copy_uint16) and not test_copy_uint16_to_memory_config) or (test_copy and not test_copy_)'

# test_pad_subcoregrids: round 7 win — 57/58 pass with the IS_NOT_POW2 fix +
# InterleavedPow2AddrGenFast shim.
run_pytest "dm_test_pad_subcoregrids" "$DM_TEST_DIR/test_pad_subcoregrids.py" -k 'not test_pad_subcoregrids_rejects_sharded'

# indexed_fill_sharded is de-scoped: deterministically hangs on the companion build (cold JIT
# cache) in the emule parallel JIT compile — std::system forks clang from a ~130-thread process
# and the child wedges before exec (fork-in-multithreaded-process deadlock). Not a kernel/sharding
# bug: the exact kernel compiles standalone in ~8s. Tracking: tenstorrent/tt-emule#55.
# run_pytest "dm_test_indexed_fill_sharded"   "$DM_TEST_DIR/test_indexed_fill.py::test_indexed_fill_sharded" -k 'B8-b3-D64 or B6-b4-D128'

run_pytest "reduce_test_sum" "$REDUCE_TEST_DIR/test_sum.py" -k 'test_sum and not test_sum_global and not test_sum_4d and not test_sum_nd_shard and not test_sum_subcores'

run_pytest "dm_test_tilize_fp32_truncation" "$DM_TEST_DIR/test_tilize.py" -k 'test_tilize_fp32_truncation'

run_pytest "dm_test_tilizer" "$DM_TEST_DIR/test_tilizer.py"

run_pytest "dm_test_dropout" "$DM_TEST_DIR/test_dropout.py"

run_pytest "dm_test_reallocate" "$DM_TEST_DIR/test_reallocate.py" -k 'DRAM or sharded'

run_pytest "dm_test_creation_full_like_opt_rm" "$DM_TEST_DIR/test_creation.py::test_full_like_opt_tensor" -k 'ROW_MAJOR'

run_pytest "elt_test_activation" "$ELT_TEST_DIR/test_activation.py::test_hardtanh" "$ELT_TEST_DIR/test_activation.py::test_log_sigmoid" "$ELT_TEST_DIR/test_activation.py::test_threshold" "$ELT_TEST_DIR/test_activation.py::test_scalarB_hardshrink" "$ELT_TEST_DIR/test_activation.py::test_scalarB_softshrink" "$ELT_TEST_DIR/test_activation.py::test_xielu" "$ELT_TEST_DIR/test_activation.py::test_hardswish" "$ELT_TEST_DIR/test_activation.py::test_swish" "$ELT_TEST_DIR/test_activation.py::test_tanhshrink"
run_pytest "elt_test_math" "$ELT_TEST_DIR/test_math.py::test_cbrt" "$ELT_TEST_DIR/test_math.py::test_i0" "$ELT_TEST_DIR/test_math.py::test_erfinv" "$ELT_TEST_DIR/test_math.py::test_digamma" "$ELT_TEST_DIR/test_math.py::test_polygamma"
run_pytest "elt_test_elu" "$ELT_TEST_DIR/test_elu.py::test_elu_allclose" "$ELT_TEST_DIR/test_elu.py::test_elu_arange_masking"
run_pytest "elt_test_unary_i1" "$ELT_TEST_DIR/test_unary_i1.py::test_i1_zero" "$ELT_TEST_DIR/test_unary_i1.py::test_i1_clamp_boundary" "$ELT_TEST_DIR/test_unary_i1.py::test_i1_ood" "$ELT_TEST_DIR/test_unary_i1.py::test_i1_range"

# Eltwise comparison ops (#75): float (binary_comp_fp32), int/uint-output (binary_comp_init),
# bf16 + int32 relational, unary compare-to-scalar, typecast-output, and sharded col-bcast.
run_pytest "elt_test_binary_comp_init"  "$ELT_TEST_DIR/test_binary_comp_init.py"
run_pytest "elt_test_binary_comp_fp32"  "$ELT_TEST_DIR/test_binary_comp_fp32.py"
run_pytest "elt_test_relational"        "$ELT_TEST_DIR/test_relational.py" -k 'not isclose'
run_pytest "elt_test_unary_comp"        "$ELT_TEST_DIR/test_unary.py::test_unary_comp_ops"
run_pytest "elt_test_binary_ng_typecast_cmp" "$ELT_TEST_DIR/test_binary_ng_typecast.py" -k 'test_binary_ng_typecast_lt or (test_binary_w_typecast and (ge or gt or le or lt or eq or ne))'
run_pytest "elt_test_binary_sharded_col_major_cmp" "$ELT_TEST_DIR/test_binary_bcast.py::test_binary_sharded_col_major" -k 'eq or ne or gt or ge or lt or le'

run_pytest "fused_test_softmax" "$FUSED_TEST_DIR/test_softmax.py::test_large_fill_softmax" "$FUSED_TEST_DIR/test_softmax.py::test_softmax_accuracy" "$FUSED_TEST_DIR/test_softmax.py::test_softmax_stable_neg_values" "$FUSED_TEST_DIR/test_softmax.py::test_softmax_4096x4096_fp32" "$FUSED_TEST_DIR/test_softmax.py::test_softmax_large_kernel_block_size" "$FUSED_TEST_DIR/test_softmax.py::test_softmax_with_3D" "$FUSED_TEST_DIR/test_softmax.py::test_softmax_with_padded_tile_layout" "$FUSED_TEST_DIR/test_softmax.py::test_softmax_with_padded_tile_layout_large"
run_pytest "reduce_test_cumprod" "$REDUCE_TEST_DIR/test_cumprod.py::test_cumprod_backward" "$REDUCE_TEST_DIR/test_cumprod.py::test_cumprod_failing_cases"
run_pytest "reduce_test_cumsum_failing" "$REDUCE_TEST_DIR/test_cumsum.py::test_cumsum_failing_cases"

run_pytest "elt_test_celu_21f" "$ELT_TEST_DIR/test_celu_21f.py::test_celu_allclose" "$ELT_TEST_DIR/test_celu_21f.py::test_celu_arange"



run_pytest "dm_test_concat_size_switches" "$DM_TEST_DIR/test_concat.py::test_concat_size_switches"

run_pytest "dm_test_pad_not_sub_core" "$DM_TEST_DIR/test_pad.py::test_pad_tile" "$DM_TEST_DIR/test_pad.py::test_pad_rm" "$DM_TEST_DIR/test_pad.py::test_pad_with_program_cache" -k 'not sub_core'
run_pytest "dm_test_pad" "$DM_TEST_DIR/test_pad.py::test_pad_rm_small_to_large_width" "$DM_TEST_DIR/test_pad.py::test_pad_rm_small_to_large_width_with_program_cache" "$DM_TEST_DIR/test_pad.py::test_pad_program_cache_hit_updates_pad_value_buffer" "$DM_TEST_DIR/test_pad.py::test_pad_padding_validation_front_pad_not_supported" "$DM_TEST_DIR/test_pad.py::test_pad_padding_validation_length"

run_pytest "dm_test_permute_not_sharded" "$DM_TEST_DIR/test_permute.py::test_permute_4d_fixed_w" "$DM_TEST_DIR/test_permute.py::test_permute_4d_cn" "$DM_TEST_DIR/test_permute.py::test_permute_4d_cnwh" "$DM_TEST_DIR/test_permute.py::test_permute_4d_wh" "$DM_TEST_DIR/test_permute.py::test_permute_5d" "$DM_TEST_DIR/test_permute.py::test_permute_5d_wyh" "$DM_TEST_DIR/test_permute.py::test_permute_5d_xh_pad" "$DM_TEST_DIR/test_permute.py::test_permute_8d_swapped" "$DM_TEST_DIR/test_permute.py::test_permutations_5d_fixed_w" "$DM_TEST_DIR/test_permute.py::test_permute_identity" -k 'not sharded'
run_pytest "dm_test_permute" "$DM_TEST_DIR/test_permute.py::test_permute_5d_tiled_basic" "$DM_TEST_DIR/test_permute.py::test_permute_5d_tiled_swap" "$DM_TEST_DIR/test_permute.py::test_permute_squeeze" "$DM_TEST_DIR/test_permute.py::test_permute_for_specific_case" "$DM_TEST_DIR/test_permute.py::test_permute_on_4D_tensor_with_smaller_tuple_size" "$DM_TEST_DIR/test_permute.py::test_nil_volume_permute" "$DM_TEST_DIR/test_permute.py::test_transpose_wh_tiled_uint32"

run_pytest "dm_test_untilize_same_volume" "$DM_TEST_DIR/test_untilize.py::test_untilize_same_volume_different_shapes"
# Untilize sharded harvest from the NUM_L1_BANKS fix (+418 sharded variants).
# Excludes test_untilize_multi_core_{,nd_}sharded_to_interleaved — those have
# 8 residual ATOL≈3.2 failures on tensor_shape=[4,4,256,512]; the substring
# filter also excludes the matching _uneven_input_shard_spec variant.
# Adds ~418 of 476 passing sharded variants.
run_pytest "dm_test_untilize_sharded"       "$DM_TEST_DIR/test_untilize.py" -k 'sharded and not multi_core_sharded_to_interleaved and not multi_core_nd_sharded_to_interleaved'

run_pytest "reduce_test_reduction_mean" "$REDUCE_TEST_DIR/test_reduction_mean.py::test_mean" "$REDUCE_TEST_DIR/test_reduction_mean.py::test_mean_scaling" "$REDUCE_TEST_DIR/test_reduction_mean.py::test_mean_scaling_factor"
run_pytest "reduce_test_reduction_not_sharded" "$REDUCE_TEST_DIR/test_reduction.py::test_mean_2d_tensor_dims" "$REDUCE_TEST_DIR/test_reduction.py::test_mean_3d_tensor_dims" "$REDUCE_TEST_DIR/test_reduction.py::test_mean_4d_tensor_dims" "$REDUCE_TEST_DIR/test_reduction.py::test_sum_2d_tensor_dims" -k 'not sharded'
run_pytest "reduce_test_reduction_min_not_sharded" "$REDUCE_TEST_DIR/test_reduction_min.py::test_min" "$REDUCE_TEST_DIR/test_reduction_min.py::test_min_global" -k 'not sharded'
run_pytest "reduce_test_torch_compat" "$REDUCE_TEST_DIR/test_reduction.py::test_torch_compatibility"

run_pytest "dm_test_tosa_gather" "$DM_TEST_DIR/test_tosa_gather.py" \
    --deselect "tests/ttnn/unit_tests/operations/data_movement/test_tosa_gather.py::test_tosa_gather_general[N=128-K=64-C=128-W=32]" \
    --deselect "tests/ttnn/unit_tests/operations/data_movement/test_tosa_gather.py::test_tosa_gather_general[N=2-K=32-C=96-W=32]" \
    --deselect "tests/ttnn/unit_tests/operations/data_movement/test_tosa_gather.py::test_tosa_gather_general[N=64-K=128-C=256-W=128]" \
    --deselect "tests/ttnn/unit_tests/operations/data_movement/test_tosa_gather.py::test_tosa_gather_general[N=128-K=128-C=128-W=64]"

run_pytest "reduce_test_max" "$REDUCE_TEST_DIR/test_max.py"

run_pytest "matmul_test_linear" "$MATMUL_TEST_DIR/test_linear.py" -k 'test_linear_fp32_acc or test_vector_linear'

run_pytest "matmul_test_addmm" "$MATMUL_TEST_DIR/test_addmm.py" \
    -k 'test_alpha_zero_should_throw_error or test_input_tensor_with_invalid_shape or test_unsupported_dtype_should_throw_error'

echo ""
echo "========================================"
echo " Results: $PASS passed, $FAIL failed"
echo "========================================"

[ "$FAIL" -eq 0 ]
