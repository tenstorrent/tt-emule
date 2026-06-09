#!/usr/bin/env bash
# Sourceable env preamble for the BH post-commit full sweep.
#
# Mirrors scripts/run_ttnn_pytests_blackhole.sh:81-88 — sets the env
# vars the audited runner expects but does NOT itself set. Designed to
# be `source`d (not exec'd) from run_full_sweep.sh so the exports
# persist into the runner script.
#
# Inputs (env, with defaults):
#   TT_METAL_DIR   default /localdev/arminale/tt-metal
#   BUILD_DIR      default $TT_METAL_DIR/build_emule
#
# Side effects:
#   - exports the emule env + library/Python paths
#   - clears the JIT cache (start-of-sweep, per skill Phase 4 contract)
#   - cd's into $TT_METAL_DIR (pytest resolves paths from the metal root)

export TT_METAL_DIR="${TT_METAL_DIR:-/localdev/arminale/tt-metal}"
export BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
export CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"
# The audited runner invokes `pytest` bare (per the identical-invocation
# invariant against the source manifest); put the toolchain venv's pytest
# on PATH so that resolves to the right binary. The dryrun harness used
# $PYTEST_BIN explicitly; the audited runner can't, because the manifest
# doesn't.
export PYTEST_BIN="${PYTEST_BIN:-/opt/ttmlir-toolchain/venv/bin/pytest}"
export PATH="$(dirname "$PYTEST_BIN"):${PATH:-}"

# Pre-flight: every required path must exist.
for p in "$TT_METAL_DIR" "$BUILD_DIR" "$BUILD_DIR/lib" "$CLUSTER_EXAMPLES"; do
    if [ ! -e "$p" ]; then
        echo "env_setup FAIL: missing $p" >&2
        return 2 2>/dev/null || exit 2
    fi
done

# Clear the JIT cache — once per snapshot, before the first variant.
# RESUME=1 skips the clear so a mid-sweep restart keeps the warm cache.
if [ "${RESUME:-0}" != "1" ]; then
    rm -rf /tmp/tt_emule_jit_cache_$(id -u)* 2>/dev/null || true
fi

# Library + Python paths for tt-metal's ttnn imports.
export PYTHONPATH="$TT_METAL_DIR/ttnn:$TT_METAL_DIR/tools:$BUILD_DIR/lib:$TT_METAL_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_HOME="$TT_METAL_DIR"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"

# pytest must resolve test paths from the tt-metal root.
cd "$TT_METAL_DIR"

# Note: emule-mode envs (TT_METAL_EMULE_MODE, TT_METAL_SLOW_DISPATCH_MODE,
# MESH_DEVICE, TT_METAL_MOCK_CLUSTER_DESC_PATH) are NOT set here — the
# audited runner script (run_bh_sanity.sh) sets them inline on every
# pytest invocation as part of its identical-invocation layering.
