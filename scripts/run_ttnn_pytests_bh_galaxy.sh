#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# BH galaxy (32-chip UBB blackhole, 4x8 mesh) CCL suite under emule. Reproduces the real CI leg
#   bh-galaxy-ccl  (tests/pipeline_reorg/galaxy_e2e_tests.yaml) — sku bh_galaxy / topology-6u, daily cron:
#     pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/galaxy/galaxy_nightly
# The 32-chip mock descriptor (blackhole_galaxy.yaml) yields ttnn.get_num_devices()==32, which the
# root conftest's bh_2d_mesh_device fixture opens as MeshShape(4,8) — no MESH_DEVICE token, exactly as
# the 8-chip loudbox opens ==8 as (4,2). This is the first emule run whose aggregate worker L1
# (32 * ~280 MB ≈ 9 GB) exceeds the old 4 GB ceiling — it exercises the offset model above 4 GB.
#
# Modes:
#   (default) full galaxy_nightly leg, per-file.
#   BRING_UP=1 → only the simple-op ladder (simplest→hardest), for first 32-chip bring-up.
#
# Sub-device / trace nodes hard-TT_FATAL under slow dispatch; trace params are deselected by the
# emule_ccl_trace_deselect plugin (as in the loudbox runner). See docs/fabric-ccl-emulation.md.
#
# Usage: TT_METAL_DIR=<...> [BRING_UP=1] bash scripts/run_ttnn_pytests_bh_galaxy.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_DIR="${TT_METAL_DIR:?TT_METAL_DIR must be set}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
PYTEST_BIN="${PYTEST_BIN:-/opt/ttmlir-toolchain/venv/bin/pytest}"
GTEST_XML_DIR="${GTEST_XML_DIR:-}"
[ -n "$GTEST_XML_DIR" ] && mkdir -p "$GTEST_XML_DIR"
CLUSTER_EXAMPLES="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples"
BOX_REL="tests/ttnn/unit_tests/operations/ccl/blackhole_CI/galaxy"
GALAXY_NIGHTLY="$TT_METAL_DIR/$BOX_REL/galaxy_nightly"

export TT_METAL_HOME="$TT_METAL_DIR" TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
# SCRIPT_DIR on PYTHONPATH so pytest can import the emule_ccl_trace_deselect plugin (#242).
export PYTHONPATH="$SCRIPT_DIR:$TT_METAL_DIR/ttnn:$TT_METAL_DIR/tools:$BUILD_DIR/lib:$TT_METAL_DIR"
export LD_LIBRARY_PATH="$BUILD_DIR/lib"
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+$PYTEST_ADDOPTS }-p emule_ccl_trace_deselect"
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_EXAMPLES/blackhole_galaxy.yaml"
export TT_METAL_EMULE_MODE=1 TT_METAL_SLOW_DISPATCH_MODE=1
export TT_EMULE_FIBER_WORKERS="${TT_EMULE_FIBER_WORKERS:-24}"
# Multi-chip fabric dst-resolution (control-plane route table + multicast replay), built from the
# descriptor's ethernet_connections — subsumes the 8-chip case; the 4x8 galaxy is its first >8-chip run.
export EMULE_FABRIC8="${EMULE_FABRIC8:-1}"

# Out-of-scope files (same rationale as the loudbox runner): sub-device managers hard-TT_FATAL under
# slow dispatch. MoE all-to-all is emule-fragile (tracked emule #222) but stays in the full leg.
declare -A SKIP_FILES=(
  [test_new_all_broadcast]="⛔ sub-device (slow dispatch)"
)

PASS=0; FAIL=0
run_pytest() {
  local name="$1"; shift
  echo "--- $name ---"
  local log="/tmp/galaxy_${name}.log"
  local junit_args=()
  [ -n "$GTEST_XML_DIR" ] && junit_args=(--junitxml="$GTEST_XML_DIR/${name}.xml")
  timeout 2400 "$PYTEST_BIN" -q "${junit_args[@]}" "$@" >"$log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "  PASS  $(grep -oE '[0-9]+ (passed|failed|xfailed|skipped)' "$log" | tr '\n' ' ')"
    PASS=$((PASS + 1))
  elif [ "$rc" -eq 5 ]; then
    echo "  SKIP  (no targetable configs — all deselected, e.g. trace-only)"
  else
    echo "  FAIL  $(grep -oE '[0-9]+ (passed|failed|error|xfailed)' "$log" | tr '\n' ' ') (log: $log)"
    FAIL=$((FAIL + 1))
  fi
}

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
echo "# BH galaxy CCL (descriptor: blackhole_galaxy.yaml, 4x8, K=$TT_EMULE_FIBER_WORKERS)"
echo "############################################################"
rm -rf "/tmp/tt_emule_jit_cache_$(id -u)" 2>/dev/null

if [ -n "${BRING_UP:-}" ]; then
  # Simple-op bring-up ladder, simplest -> hardest (per-node so a failure isolates cleanly).
  echo "# BRING_UP: simple-op ladder"
  run_pytest bringup_ag_2D_line   "$GALAXY_NIGHTLY/test_all_gather_apc.py::test_all_gather_2D_line"
  run_pytest bringup_rs_row_2D    "$GALAXY_NIGHTLY/test_reduce_scatter_apc.py::test_rs_row_2D_nightly_linear"
  run_pytest bringup_ag_4D_line   "$GALAXY_NIGHTLY/test_all_gather_apc.py::test_all_gather_4D_line"
  run_pytest bringup_ag_ring      "$GALAXY_NIGHTLY/test_all_gather_apc.py::test_all_gather_everything_ring"
else
  run_suite galaxy_nightly "$GALAXY_NIGHTLY"  # bh-galaxy-ccl
fi

echo "------------------------------------------------------------"
echo "galaxy CCL: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
