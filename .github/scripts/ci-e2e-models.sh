#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# CI wrapper for the end-to-end MODEL regression (simple_text_demo, etc.).
# Mirrors ci-ttnn-pytests.sh.
#
# Required env:
#   TT_METAL_DIR   path to tt-metal source tree (checked out at the pinned SHA)
#   BUILD_DIR      absolute path to the build tree (containing ttnn/_ttnn.so etc.)
#   TT_EMULE_ARCH  wormhole | blackhole — selects the per-arch runner
#
# Optional env:
#   GTEST_XML_DIR  default $RUNNER_TEMP/e2e-junit-xml (per-entry junit XML)
#   REGRESSION_LOG default $RUNNER_TEMP/e2e-models.log
#   HF_HOME        default $RUNNER_TEMP/hf-cache (HuggingFace weight cache)
#   SHARD_INDEX    1-based shard index (default 1; round-robin in runner)
#   SHARD_COUNT    total number of shards (default 1)

set -euo pipefail

: "${TT_METAL_DIR:?TT_METAL_DIR must be set}"
: "${BUILD_DIR:?BUILD_DIR must be set}"
: "${TT_EMULE_ARCH:?TT_EMULE_ARCH must be set (wormhole|blackhole)}"
case "$TT_EMULE_ARCH" in
    wormhole|blackhole) ;;
    *) echo "ERROR: TT_EMULE_ARCH must be wormhole|blackhole, got '$TT_EMULE_ARCH'" >&2; exit 1 ;;
esac
export SHARD_INDEX="${SHARD_INDEX:-1}"
export SHARD_COUNT="${SHARD_COUNT:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_EMULE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export GTEST_XML_DIR="${GTEST_XML_DIR:-${RUNNER_TEMP:-/tmp}/e2e-junit-xml}"
REGRESSION_LOG="${REGRESSION_LOG:-${RUNNER_TEMP:-/tmp}/e2e-models.log}"

# HuggingFace weight cache — the demos pull the ungated unsloth mirror (no token
# needed). Keep it under a stable path so a workflow-level actions/cache can
# persist it across runs.
export HF_HOME="${HF_HOME:-${RUNNER_TEMP:-/tmp}/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HUB_CACHE"

mkdir -p "$GTEST_XML_DIR"
rm -f "$GTEST_XML_DIR"/*.xml

# Ensure the post-build symlink that lets `import ttnn` resolve to the
# emule-built _ttnn.so (the test runner does a fresh tt-metal checkout without
# the BUILD_GUIDE "Post-build symlinks" step).
if [ ! -e "$TT_METAL_DIR/ttnn/ttnn/_ttnn.so" ]; then
    ln -sfn "$BUILD_DIR/ttnn/_ttnn.so" "$TT_METAL_DIR/ttnn/ttnn/_ttnn.so"
fi

echo "== ci-e2e-models.sh =="
echo "  TT_EMULE_DIR:   $TT_EMULE_DIR"
echo "  TT_METAL_DIR:   $TT_METAL_DIR"
echo "  BUILD_DIR:      $BUILD_DIR"
echo "  TT_EMULE_ARCH:  $TT_EMULE_ARCH"
echo "  GTEST_XML_DIR:  $GTEST_XML_DIR"
echo "  REGRESSION_LOG: $REGRESSION_LOG"
echo "  HF_HOME:        $HF_HOME"
echo "  SHARD:          $SHARD_INDEX of $SHARD_COUNT"
echo ""

set +e
bash "$TT_EMULE_DIR/scripts/run_e2e_models_${TT_EMULE_ARCH}.sh" 2>&1 | tee "$REGRESSION_LOG"
rc=${PIPESTATUS[0]}
set -e

xml_count=$(find "$GTEST_XML_DIR" -name '*.xml' 2>/dev/null | wc -l)
echo ""
echo "== E2E models done =="
echo "  exit code:    $rc"
echo "  XML files:    $xml_count in $GTEST_XML_DIR"
echo "  Log:          $REGRESSION_LOG"

# Propagate the runner's exit code: the curated entry list is all known-passing,
# so any failure is a real regression.
exit "$rc"
