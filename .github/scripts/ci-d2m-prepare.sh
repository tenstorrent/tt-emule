#!/usr/bin/env bash
# Generate ttrt-artifacts/system_desc.ttsys (required by tt-mlir's D2M
# conftest) by running `ttrt query --save-artifacts` under the same
# emulated env that the regression itself uses.
#
# Required env:
#   TT_EMULE_DIR (auto-derived), TT_MLIR_DIR, TT_METAL_DIR, BUILD_DIR

set -euo pipefail

: "${TT_MLIR_DIR:?}" ; : "${TT_METAL_DIR:?}" ; : "${BUILD_DIR:?}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_EMULE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CLUSTER_DESC="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml"
if [ ! -f "$CLUSTER_DESC" ]; then
    echo "ERROR: cluster descriptor missing at $CLUSTER_DESC" >&2
    exit 1
fi

# Snapshot our absolute BUILD_DIR (tt-metal build with tt-emule libs) before
# sourcing env/activate. env/activate does `$(pwd)/${BUILD_DIR:=build}/...`
# in PATH/PYTHONPATH — if BUILD_DIR is already set to an absolute path, the
# concatenation produces broken paths like /tt-mlir//abs/tt-metal/build/...
TT_EMULE_BUILD_DIR="$BUILD_DIR"
unset BUILD_DIR

cd "$TT_MLIR_DIR"

# env/activate references unbound vars; relax set -u.
set +u
# shellcheck disable=SC1091
source env/activate
set -u

# Restore our absolute build path under a distinct name and prepend the
# emule-built _ttnn.so location to PYTHONPATH so `import ttnn` resolves there.
export BUILD_DIR="$TT_EMULE_BUILD_DIR"
export PYTHONPATH="$TT_METAL_DIR/ttnn:$BUILD_DIR/lib:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
export TT_MLIR_HOME="$TT_MLIR_DIR"
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_DESC"
export TT_METAL_EMULATED_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1

# Some scripts overlay the built _ttnn.so into the source tree so that
# `import ttnn` resolves to the emule build. Mirror that.
if [ -f "$BUILD_DIR/lib/_ttnn.so" ] && [ -d "$TT_METAL_DIR/ttnn/ttnn" ]; then
    ln -sf "$BUILD_DIR/lib/_ttnn.so" "$TT_METAL_DIR/ttnn/ttnn/_ttnn.so"
fi

mkdir -p ttrt-artifacts

echo "== ttrt query =="
# `ttrt` is installed as a venv console_script during the build job, but venv
# state doesn't survive between jobs/containers. Invoke via `python -m ttrt`
# instead, which resolves from PYTHONPATH (tt-mlir/build/python_packages).
python -m ttrt query --save-artifacts --quiet || {
    echo "WARNING: ttrt query failed; tests may fail to load system_desc" >&2
}

echo ""
echo "== Generated artifacts =="
ls -la ttrt-artifacts/ || true
