#!/usr/bin/env bash
# Supplementary dry-run target: test_tilize.py::test_tilize_test
# Documented as NOT covered by emule (docs/bh-sanity-coverage.md). Used to
# exercise the F bucket in the classifier since test_pad.py ran all-pass.

set -uo pipefail

TT_METAL_DIR="${TT_METAL_DIR:-/localdev/arminale/tt-metal}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
PYTEST_BIN="${PYTEST_BIN:-/opt/ttmlir-toolchain/venv/bin/pytest}"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"
OUT_DIR="/localdev/arminale/tt-emule/snapshots/bh_sanity_dryrun"

# DO NOT clear the JIT cache here — we want to reuse warmed state from the
# main harness run.
export PYTHONPATH="$TT_METAL_DIR/ttnn:$TT_METAL_DIR/tools:$BUILD_DIR/lib:$TT_METAL_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_HOME="$TT_METAL_DIR"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/blackhole_P100.yaml"
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export MESH_DEVICE=P100

TARGET="tests/ttnn/unit_tests/operations/data_movement/test_tilize.py::test_tilize_test"
XML_OUT="$OUT_DIR/test_tilize.xml"
LOG_OUT="$OUT_DIR/test_tilize.log"

cd "$TT_METAL_DIR"
t0=$(date +%s)
timeout 300 "$PYTEST_BIN" --timeout 60 -xv "$TARGET" --junitxml="$XML_OUT" \
    > "$LOG_OUT" 2>&1
rc=$?
t1=$(date +%s)
echo "rc=$rc elapsed=$((t1-t0))s xml=$XML_OUT"
exit 0
