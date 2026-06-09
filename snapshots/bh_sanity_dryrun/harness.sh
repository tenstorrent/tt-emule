#!/usr/bin/env bash
# DRY-RUN harness for /status-snapshot Phase 4.
#
# Mirrors the env-setup recipe from scripts/run_ttnn_pytests_blackhole.sh:81-88
# and runs ONE pytest invocation on tests/.../data_movement/test_pad.py
# (whole file) under the BH-emule variant. Captures JUnit XML + stdout/stderr.
#
# Inputs (env, override as needed):
#   TT_METAL_DIR   default /localdev/arminale/tt-metal
#   BUILD_DIR      default $TT_METAL_DIR/build_emule
#   PYTEST_BIN     default /opt/ttmlir-toolchain/venv/bin/pytest
#
# Outputs:
#   snapshots/bh_sanity_dryrun/test_pad.xml
#   snapshots/bh_sanity_dryrun/test_pad.log
#   snapshots/bh_sanity_dryrun/harness.invocation  (the exact pytest cmd, for repro)
#
# Once this proves out the env contract, the same pattern wraps run_bh_sanity.sh
# for the full sweep.

set -uo pipefail

TT_METAL_DIR="${TT_METAL_DIR:-/localdev/arminale/tt-metal}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
PYTEST_BIN="${PYTEST_BIN:-/opt/ttmlir-toolchain/venv/bin/pytest}"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"

OUT_DIR="/localdev/arminale/tt-emule/snapshots/bh_sanity_dryrun"
mkdir -p "$OUT_DIR"

# --- Pre-flight checks ------------------------------------------------------
for p in "$TT_METAL_DIR" "$BUILD_DIR" "$BUILD_DIR/lib" "$CLUSTER_EXAMPLES" "$PYTEST_BIN"; do
    if [ ! -e "$p" ]; then
        echo "FAIL preflight: missing $p" >&2
        exit 2
    fi
done

# --- Clear the JIT cache (per skill Phase 4 contract) -----------------------
rm -rf /tmp/tt_emule_jit_cache_$(id -u)* 2>/dev/null || true

# --- Export the env (matches run_ttnn_pytests_blackhole.sh:81-88) ----------
export PYTHONPATH="$TT_METAL_DIR/ttnn:$TT_METAL_DIR/tools:$BUILD_DIR/lib:$TT_METAL_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_HOME="$TT_METAL_DIR"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/blackhole_P100.yaml"
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export MESH_DEVICE=P100

# --- Mirrored invocation ----------------------------------------------------
# One line from the audited runner script, narrowed to test_pad.py whole-file.
# `timeout 600` matches the BH post-commit data-movement entry budget.
# `--timeout 60` matches the source cmd's per-test budget.
TEST_TARGET="tests/ttnn/unit_tests/operations/data_movement/test_pad.py"
XML_OUT="$OUT_DIR/test_pad.xml"
LOG_OUT="$OUT_DIR/test_pad.log"

INVOCATION="timeout 600 $PYTEST_BIN --timeout 60 -xv $TEST_TARGET --junitxml=$XML_OUT"
echo "$INVOCATION" > "$OUT_DIR/harness.invocation"

cd "$TT_METAL_DIR"

# Header in the log: env snapshot + the exact invocation, so the log is
# self-contained for repro.
{
    echo "=== DRY-RUN harness ==="
    echo "date:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host:         $(hostname)"
    echo "cwd:          $(pwd)"
    echo "TT_METAL_DIR: $TT_METAL_DIR"
    echo "BUILD_DIR:    $BUILD_DIR"
    echo "PYTEST_BIN:   $PYTEST_BIN"
    echo "env:"
    env | grep -E '^(TT_METAL_|MESH_DEVICE|PYTHONPATH|LD_LIBRARY_PATH)' | sort | sed 's/^/  /'
    echo
    echo "invocation:   $INVOCATION"
    echo "=== pytest output ==="
} > "$LOG_OUT"

t0=$(date +%s)
eval "$INVOCATION" >> "$LOG_OUT" 2>&1
rc=$?
t1=$(date +%s)
elapsed=$((t1 - t0))

{
    echo
    echo "=== harness summary ==="
    echo "rc:           $rc"
    echo "elapsed_s:    $elapsed"
    echo "xml:          $XML_OUT"
} >> "$LOG_OUT"

echo "rc=$rc elapsed=${elapsed}s xml=$XML_OUT log=$LOG_OUT"
exit 0  # never abort — even a failing pytest gives us a JUnit XML to parse
