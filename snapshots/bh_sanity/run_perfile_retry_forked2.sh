#!/usr/bin/env bash
# Second --forked retry pass. Three files we discovered hang (not just slow)
# during the retry-wallclock pass:
#
#   test_unary_ops_ttnn.py — hung at test_unary_i0_ttnn after 160 tests
#   test_group_norm.py     — hung after 32 tests
#   test_sparse_matmul.py  — only 3/11 tests in prior 900s; suspect hang too
#
# All three need --forked so per-test timeout can interrupt C-level hangs.
# Wallclock 5400s (90 min) is plenty given the per-test timeout does the
# real work.

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
        timeout 5400 pytest --forked --timeout 300 -v \
        -m "not disable_fast_runtime_mode" \
        "$rel" \
        --junitxml="$OUT/$slug.xml" \
        > "$OUT/$slug.log" 2>&1
    local rc=$?
    echo "rc=$rc $rel"
    echo "::endgroup::"
}

run_one tests/ttnn/unit_tests/operations/eltwise/test_unary_ops_ttnn.py \
        tests-ttnn-unit-tests-operations-eltwise-test-unary-ops-ttnn
run_one tests/ttnn/unit_tests/operations/fused/test_group_norm.py \
        tests-ttnn-unit-tests-operations-fused-test-group-norm
run_one tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py \
        tests-ttnn-unit-tests-operations-matmul-test-sparse-matmul

echo "=== retry-forked2 complete ==="
