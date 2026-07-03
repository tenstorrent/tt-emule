#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# No-comm (data-parallel) eltwise on a 1x2 mesh device under emule — verifies a dual-chip
# board (N300 = 2x Wormhole, P300 = 2x Blackhole) runs simple eltwise faithfully on BOTH
# chips with NO inter-chip communication (host scatter/gather only; no device-side CCL).
# Each chip emulates its own kernels; the slow-dispatch mesh CQ register/run split runs all
# chips' fibers concurrently (see docs/fiber-engine.md §9).
#
# Usage:
#   run_mesh_eltwise.sh <n300|p300> [--full]
#     default: curated functional subset (fast).
#     --full : also the stress-eltwise file (large shapes; slow on emule — set a high
#              TT_EMULE_FIBER_WORKERS and expect minutes).
# Env knobs: TT_EMULE_FIBER_WORKERS (worker count K; default 64 here for larger grids).
set -uo pipefail
BOARD="${1:?usage: $0 <n300|p300> [--full]}"
FULL=0; [ "${2:-}" = "--full" ] && FULL=1

# tt-metal root + build/pytest from env, matching the other scripts/ runners.
#   TT_METAL_DIR — path to tt-metal source tree (required)
#   BUILD_DIR    — default $TT_METAL_DIR/build_emule
#   PYTEST_BIN   — default /opt/ttmlir-toolchain/venv/bin/pytest
TT="${TT_METAL_DIR:?TT_METAL_DIR must be set}"
BUILD_DIR="${BUILD_DIR:-$TT/build_emule}"
CLUSTER="$TT/tt_metal/third_party/umd/tests/cluster_descriptor_examples"
case "$BOARD" in
  n300) DESC="$CLUSTER/wormhole_N300.yaml" ;;      # 2x Wormhole, board n300 -> 1x2 mesh
  p300) DESC="$CLUSTER/blackhole_P300_both_mmio.yaml" ;;  # 2x Blackhole, board p300 -> 1x2 mesh
  bh4)  DESC="$CLUSTER/blackhole_4xP150.yaml" ;;   # 4x Blackhole -> 2x2 mesh
  bh8)  DESC="$CLUSTER/blackhole_8xP150.yaml" ;;   # 8x Blackhole (loudbox) -> 2x4 mesh
  *) echo "unknown board '$BOARD' (use n300 | p300 | bh4 | bh8)"; exit 2 ;;
esac
[ -f "$DESC" ] || { echo "descriptor not found: $DESC"; exit 2; }

export TT_METAL_DIR=$TT TT_METAL_HOME=$TT TT_METAL_RUNTIME_ROOT=$TT
export PYTHONPATH=$TT/ttnn:$TT/tools:$BUILD_DIR/lib:$TT
export LD_LIBRARY_PATH=$BUILD_DIR/lib
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$DESC"
export TT_METAL_EMULE_MODE=1 TT_METAL_SLOW_DISPATCH_MODE=1
export TT_EMULE_FIBER_WORKERS="${TT_EMULE_FIBER_WORKERS:-64}"
P="${PYTEST_BIN:-/opt/ttmlir-toolchain/venv/bin/pytest}"
MD=$TT/tests/ttnn/unit_tests/base_functionality/test_multi_device.py
STRESS=$TT/tests/ttnn/stress_tests/test_eltwise.py

echo "############################################################"
echo "# mesh no-comm eltwise: board=$BOARD  K=$TT_EMULE_FIBER_WORKERS  full=$FULL"
echo "# descriptor: $DESC"
echo "############################################################"
# Arch switch / fresh kernels: wipe the JIT cache so WH<->BH don't cross-contaminate.
rm -rf "/tmp/tt_emule_jit_cache_$(id -u)" 2>/dev/null

# Curated functional subset (no device-side CCL): replicated + sharded eltwise, host-gathered.
SUITE=(
  "$MD::test_multi_device_single_op_binary"   # ShardTensorToMesh + add + ConcatMeshToTensor
  "$MD::test_multi_device_single_op_unary"    # ShardTensorToMesh + gelu + ConcatMeshToTensor
)
if [ "$FULL" = 1 ]; then
  SUITE+=(
    "$STRESS::test_stress_binary"   # replicated add (large shapes — slow on emule)
    "$STRESS::test_stress_unary"    # replicated silu
  )
fi

pass=0; fail=0
for sel in "${SUITE[@]}"; do
  timeout 1200 "$P" -q "$sel" >/tmp/mesh_${BOARD}_$(basename "$sel" | tr -c 'A-Za-z0-9' _).log 2>&1
  rc=$?
  res=$(grep -oE '[0-9]+ (passed|failed|error)' /tmp/mesh_${BOARD}_$(basename "$sel" | tr -c 'A-Za-z0-9' _).log | tr '\n' ' ')
  if [ "$rc" = 0 ]; then pass=$((pass+1)); else fail=$((fail+1)); fi
  printf '[rc=%s] %-44s %s\n' "$rc" "$(basename "$sel")" "$res"
done
echo "------------------------------------------------------------"
echo "mesh $BOARD: $pass test(s) passed, $fail failed"
[ "$fail" -eq 0 ]
