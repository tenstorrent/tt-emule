#!/usr/bin/env bash
# Retry pass for the 8 files that died from SIGABRT (Fatal Python error)
# during the no-forked main sweep. Re-enable --forked here so an abort in
# one test counts as an error and the rest of the file's tests run.
#
# Wallclock: 7200s (120 min) per file — most of these are short, so the
# big budget mostly buys safety margin in case a slow test exists.
#
# OLD logs for these files (from the killed run) WILL be overwritten by
# this retry. The 8 XMLs will appear fresh.

set -uo pipefail

: "${CLUSTER_EXAMPLES:?CLUSTER_EXAMPLES must point at cluster_descriptor_examples}"
: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"

OUT="/localdev/arminale/tt-emule/snapshots/bh_sanity/bh_emule"
mkdir -p "$OUT"

run_one() {
    local rel="$1"
    local slug="$2"
    echo "::group::$rel"
    TT_METAL_EMULE_MODE=1 TT_METAL_SLOW_DISPATCH_MODE=1 MESH_DEVICE=P100 \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/blackhole_P100.yaml" \
        timeout 7200 pytest --forked --timeout 300 -v \
        -m "not disable_fast_runtime_mode" \
        "$rel" \
        --junitxml="$OUT/$slug.xml" \
        > "$OUT/$slug.log" 2>&1
    local rc=$?
    echo "rc=$rc $rel"
    echo "::endgroup::"
}

run_one tests/ttnn/unit_tests/operations/eltwise/test_unary_pow.py \
        tests-ttnn-unit-tests-operations-eltwise-test-unary-pow
run_one tests/ttnn/unit_tests/operations/eltwise/test_unary.py \
        tests-ttnn-unit-tests-operations-eltwise-test-unary
run_one tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py \
        tests-ttnn-unit-tests-operations-fused-test-distributed-layernorm-sharded
run_one tests/ttnn/unit_tests/operations/fused/test_group_norm_DRAM.py \
        tests-ttnn-unit-tests-operations-fused-test-group-norm-dram
run_one tests/ttnn/unit_tests/operations/fused/test_layer_norm_sharded.py \
        tests-ttnn-unit-tests-operations-fused-test-layer-norm-sharded
run_one tests/ttnn/unit_tests/operations/fused/test_rms_norm_sharded.py \
        tests-ttnn-unit-tests-operations-fused-test-rms-norm-sharded
run_one tests/ttnn/unit_tests/operations/reduce/test_argmax.py \
        tests-ttnn-unit-tests-operations-reduce-test-argmax
run_one tests/ttnn/unit_tests/operations/reduce/test_reduction_min.py \
        tests-ttnn-unit-tests-operations-reduce-test-reduction-min

echo "=== retry-forked pass complete ==="
