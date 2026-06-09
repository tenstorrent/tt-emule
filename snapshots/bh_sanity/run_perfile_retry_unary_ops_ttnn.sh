#!/usr/bin/env bash
# Final retry for test_unary_ops_ttnn.py — discovered to hang (not just be
# slow) during the retry-wallclock pass. Needs --forked so the per-test
# timeout can actually interrupt the hung test_unary_i0_ttnn variant.
#
# Wallclock: 5400s (90 min) — modest budget; per-test timeout 300s does the
# real work under --forked.

set -uo pipefail

: "${CLUSTER_EXAMPLES:?CLUSTER_EXAMPLES must point at cluster_descriptor_examples}"
: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"

OUT="/localdev/arminale/tt-emule/snapshots/bh_sanity/bh_emule"
mkdir -p "$OUT"

REL=tests/ttnn/unit_tests/operations/eltwise/test_unary_ops_ttnn.py
SLUG=tests-ttnn-unit-tests-operations-eltwise-test-unary-ops-ttnn

echo "::group::$REL"
TT_METAL_EMULE_MODE=1 TT_METAL_SLOW_DISPATCH_MODE=1 MESH_DEVICE=P100 \
    TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/blackhole_P100.yaml" \
    timeout 5400 pytest --forked --timeout 300 -v \
    -m "not disable_fast_runtime_mode" \
    "$REL" \
    --junitxml="$OUT/$SLUG.xml" \
    > "$OUT/$SLUG.log" 2>&1
echo "rc=$? $REL"
echo "::endgroup::"
