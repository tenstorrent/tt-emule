#!/usr/bin/env bash
# Dry-run target: `ttnn conv group` entry from the audited BH post-commit
# runner. Per docs/bh-sanity-coverage.md (Group C), all 7 conv files are
# NOT covered by emule today — this run is expected to surface a large
# F bucket and exercise the failure-classification path of the pipeline.
#
# Mirrors snapshots/bh_sanity/run_bh_sanity.sh line for `ttnn conv group`,
# with the two status-snapshot transformations applied (-x stripped,
# --forked injected).

set -uo pipefail

TT_METAL_DIR="${TT_METAL_DIR:-/localdev/arminale/tt-metal}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
PYTEST_BIN="${PYTEST_BIN:-/opt/ttmlir-toolchain/venv/bin/pytest}"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"
OUT_DIR="/localdev/arminale/tt-emule/snapshots/bh_sanity_dryrun"

# JIT cache stays warm from the previous test_pad run — don't clear it.
export PYTHONPATH="$TT_METAL_DIR/ttnn:$TT_METAL_DIR/tools:$BUILD_DIR/lib:$TT_METAL_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_HOME="$TT_METAL_DIR"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/blackhole_P100.yaml"
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export MESH_DEVICE=P100

XML_OUT="$OUT_DIR/test_conv.xml"
LOG_OUT="$OUT_DIR/test_conv.log"

cd "$TT_METAL_DIR"

{
    echo "=== conv dry-run harness ==="
    echo "date:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "env:"
    env | grep -E '^(TT_METAL_|MESH_DEVICE|PYTHONPATH|LD_LIBRARY_PATH)' | sort | sed 's/^/  /'
    echo
    echo "=== pytest output ==="
} > "$LOG_OUT"

t0=$(date +%s)
timeout 3600 "$PYTEST_BIN" --forked --timeout 300 -v \
    tests/ttnn/unit_tests/operations/conv \
    -m "not disable_fast_runtime_mode" \
    --junitxml="$XML_OUT" \
    >> "$LOG_OUT" 2>&1
rc=$?
t1=$(date +%s)
elapsed=$((t1 - t0))

{
    echo
    echo "=== summary ==="
    echo "rc:        $rc"
    echo "elapsed_s: $elapsed"
    echo "xml:       $XML_OUT"
} >> "$LOG_OUT"

echo "rc=$rc elapsed=${elapsed}s xml=$XML_OUT log=$LOG_OUT"
exit 0
