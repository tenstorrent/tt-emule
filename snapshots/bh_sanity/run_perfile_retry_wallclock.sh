#!/usr/bin/env bash
# Retry pass for the 5 files killed by wallclock (no segfault — just slow).
# No --forked needed (none of these crash); bumped wallclock to 7200s (120 min).
#
# Estimated runtimes per file (from killed-run pace):
#   test_binary_bcast.py       ~50 min  (3,848 tests)
#   test_unary_ops_ttnn.py     ~53 min  (377 tests, slow each)
#   test_batch_norm.py         ~29 min  (2,323 tests)
#   test_group_norm.py        ~145 min  (206 tests, very slow each) ← may still hit cap
#   test_sparse_matmul.py      ~83 min  (11 tests, ~7 min each)
#
# Sequential total: ~6 hours worst case. Most files finish well below cap.

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
        timeout 7200 pytest --timeout 300 -v \
        -m "not disable_fast_runtime_mode" \
        "$rel" \
        --junitxml="$OUT/$slug.xml" \
        > "$OUT/$slug.log" 2>&1
    local rc=$?
    echo "rc=$rc $rel"
    echo "::endgroup::"
}

run_one tests/ttnn/unit_tests/operations/eltwise/test_binary_bcast.py \
        tests-ttnn-unit-tests-operations-eltwise-test-binary-bcast
run_one tests/ttnn/unit_tests/operations/eltwise/test_unary_ops_ttnn.py \
        tests-ttnn-unit-tests-operations-eltwise-test-unary-ops-ttnn
run_one tests/ttnn/unit_tests/operations/fused/test_batch_norm.py \
        tests-ttnn-unit-tests-operations-fused-test-batch-norm
run_one tests/ttnn/unit_tests/operations/fused/test_group_norm.py \
        tests-ttnn-unit-tests-operations-fused-test-group-norm
run_one tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py \
        tests-ttnn-unit-tests-operations-matmul-test-sparse-matmul

echo "=== retry-wallclock pass complete ==="
