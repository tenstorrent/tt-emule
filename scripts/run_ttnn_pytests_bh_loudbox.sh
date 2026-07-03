#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# BH loudbox (8x P150, 2x4 mesh) CCL suite under emule. Reproduces the two ttnn-CCL legs that real CI runs
# on the BH-LoudBox sku in the SCHEDULED `blackhole-e2e-tests` workflow (every 8h) — both green on hardware:
#   * `bh-ttnn-ops-fast-unit`     -> pytest .../ccl/blackhole_CI/box/all_post_commit  (CI 116 passed / 0 failed)
#   * `bh-ccl-nightly-integration -> pytest .../ccl/blackhole_CI/box/nightly          (CI 237 passed / 0 failed)
# emule status (measured): apc all_gather failing_shapes (6) + all_to_all_dispatch (1) fail; nightly
# all_gather _broken (2) + all_to_all_dispatch (8) fail — tracked as emule #221 / #222. Everything else green.
# The mesh is selected from ttnn.get_num_devices()==8 (the bh_2d/bh_1d_mesh_device conftest fixtures) — no
# MESH_DEVICE token. Measured per-suite status + CI baseline: docs/fabric-ccl-emulation.md (Current state).
#
# Sub-device / trace nodes hard-TT_FATAL under slow dispatch (sub_device_manager_tracker.cpp:96-98). Trace
# PARAMETRIZATIONS (enable_trace/trace_mode=True) need fast dispatch; they are deselected under emule by the
# emule_ccl_trace_deselect pytest plugin (loaded via PYTEST_ADDOPTS below — #242; it replaces the emule-only
# box conftest that tt-metal a0f5d8e4 reverted). Trace-only FUNCTIONS and sub-device / perf files are skipped
# here (see docs/fabric-ccl-emulation.md).
#
# Usage: TT_METAL_DIR=<...> bash scripts/run_ttnn_pytests_bh_loudbox.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_DIR="${TT_METAL_DIR:?TT_METAL_DIR must be set}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
PYTEST_BIN="${PYTEST_BIN:-/opt/ttmlir-toolchain/venv/bin/pytest}"
GTEST_XML_DIR="${GTEST_XML_DIR:-}"
[ -n "$GTEST_XML_DIR" ] && mkdir -p "$GTEST_XML_DIR"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"
BOX_REL="tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box"
APC="$TT_METAL_DIR/$BOX_REL/all_post_commit"
NIGHTLY="$TT_METAL_DIR/$BOX_REL/nightly"

export TT_METAL_HOME="$TT_METAL_DIR" TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
# SCRIPT_DIR on PYTHONPATH so pytest can import the emule_ccl_trace_deselect plugin (#242).
export PYTHONPATH="$SCRIPT_DIR:$TT_METAL_DIR/ttnn:$TT_METAL_DIR/tools:$BUILD_DIR/lib:$TT_METAL_DIR"
export LD_LIBRARY_PATH="$BUILD_DIR/lib"
# Deselect trace parametrizations under emule (need fast dispatch); no-op off emule. See #242.
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+$PYTEST_ADDOPTS }-p emule_ccl_trace_deselect"
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/blackhole_8xP150.yaml"
export TT_METAL_EMULE_MODE=1 TT_METAL_SLOW_DISPATCH_MODE=1
export TT_EMULE_FIBER_WORKERS="${TT_EMULE_FIBER_WORKERS:-64}"
# Fabric dst-resolution (control-plane route table + multicast replay). Off → legacy single-neighbor
# (2-chip only). The 8-chip CCL nodes need it; it subsumes the 2-chip case (n300 green with it on).
export EMULE_FABRIC8="${EMULE_FABRIC8:-1}"

# Trace-only FUNCTIONS (not caught by the conftest's param-level hook), deselected per dir (rootdir-relative).
DESELECT_APC=(
  --deselect "$BOX_REL/all_post_commit/test_all_gather_apc.py::test_all_gather_subcore_grid"      # sub-device grid
  --deselect "$BOX_REL/all_post_commit/test_all_to_all_dispatch_bh.py::test_all_to_all_dispatch_trace"
)
DESELECT_NIGHTLY=(
  --deselect "$BOX_REL/nightly/test_all_to_all_dispatch_bh.py::test_all_to_all_dispatch_trace"
  --deselect "$BOX_REL/nightly/test_all_to_all_combine_bh.py::test_all_to_all_combine_trace"
)

# Out-of-scope files (see docs/fabric-ccl-emulation.md, Out of emule scope) — skipped with a logged reason so FAIL
# tracks only the fabric-targetable progression, not known blockers. Shared by both legs (base names match):
#   ⛔ sub-device under slow dispatch (TT_FATAL sub_device_manager_tracker.cpp:98) — out of emule scope
#   🔧 JIT shim gap 'is_ncrisc' in deepseek unified_kernels — separate /compute-llk-bringup-class gap
#   ⛔ perf / trace-only files (need fast dispatch) — nightly only
declare -A SKIP_FILES=(
  [test_all_broadcast_2d_fabric]="⛔ sub-device (slow dispatch)"
  [test_new_all_broadcast]="⛔ sub-device (slow dispatch)"
  [test_all_to_all]="⛔ sub-device (slow dispatch)"
  [test_all_reduce_2d_fabric]="⛔ sub-device (slow dispatch)"
  [test_all_reduce_2d_fabric_turning]="⛔ sub-device (slow dispatch)"
  [test_new_all_reduce]="⛔ sub-device (slow dispatch)"
  [test_deepseek_ccl_ops]="🔧 JIT shim gap: is_ncrisc"
  [test_ccl_perf]="⛔ perf / trace (fast dispatch)"
  [test_reduce_to_root_trace]="⛔ trace-only (fast dispatch)"
  # nightly reduce_scatter variants use a sub-device manager (TT_FATAL under slow dispatch) — CI-green on HW,
  # out of emule scope like the other sub-device CCLs.
  [test_minimal_reduce_scatter_async_bh]="⛔ sub-device (slow dispatch)"
  [test_reduce_scatter_batch_invariance_bh]="⛔ sub-device (slow dispatch)"
)

PASS=0; FAIL=0
run_pytest() {
  local name="$1"; shift
  echo "--- $name ---"
  local log="/tmp/loudbox_${name}.log"
  local junit_args=()
  [ -n "$GTEST_XML_DIR" ] && junit_args=(--junitxml="$GTEST_XML_DIR/${name}.xml")
  timeout 1800 "$PYTEST_BIN" -q "${junit_args[@]}" "$@" >"$log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "  PASS  $(grep -oE '[0-9]+ (passed|failed|xfailed|skipped)' "$log" | tr '\n' ' ')"
    PASS=$((PASS + 1))
  elif [ "$rc" -eq 5 ]; then
    # pytest exit 5 = no tests collected (all configs deselected — e.g. trace-only files under emule).
    echo "  SKIP  (no targetable configs — all deselected, e.g. trace-only)"
  else
    echo "  FAIL  $(grep -oE '[0-9]+ (passed|failed|error|xfailed)' "$log" | tr '\n' ' ') (log: $log)"
    FAIL=$((FAIL + 1))
  fi
}

# Run every test file in one box subdir, per-file (a failure isolates to one collective).
run_suite() {
  local tag="$1" dir="$2"; shift 2; local deselect=("$@")
  echo "============================================================"
  echo "# $tag  ($(basename "$dir"))"
  echo "============================================================"
  for f in "$dir"/test_*.py; do
    base=$(basename "$f" .py)
    if [ -n "${SKIP_FILES[$base]:-}" ]; then
      echo "--- ${tag}/${base} ---"; echo "  SKIP  ${SKIP_FILES[$base]}"
      continue
    fi
    run_pytest "${tag}_${base}" "$f" "${deselect[@]}"
  done
}

echo "############################################################"
echo "# BH loudbox CCL (descriptor: blackhole_8xP150.yaml, K=$TT_EMULE_FIBER_WORKERS)"
echo "############################################################"
rm -rf "/tmp/tt_emule_jit_cache_$(id -u)" 2>/dev/null

run_suite apc "$APC" "${DESELECT_APC[@]}"          # bh-ttnn-ops-fast-unit
run_suite nightly "$NIGHTLY" "${DESELECT_NIGHTLY[@]}"  # bh-ccl-nightly-integration

echo "------------------------------------------------------------"
echo "loudbox CCL: $PASS file(s) passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
