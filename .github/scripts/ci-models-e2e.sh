#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# CI wrapper for the nightly model e2e suite (full model prefill+decode on emule).
# Mirrors ci-ttnn-pytests.sh but runs scripts/run_model_e2e.sh, which needs real
# model weights (HF_MODEL/HF_HOME staged by the workflow's actions/cache step).
#
# Required env:
#   TT_METAL_DIR   path to tt-metal source tree (checked out at the pinned SHA)
#   BUILD_DIR      absolute path to the build tree (containing ttnn/_ttnn.so etc.)
#   TT_EMULE_ARCH  wormhole | blackhole
#
# Optional env:
#   GTEST_XML_DIR  default $RUNNER_TEMP/models-e2e-junit-xml (per-entry junit XML)
#   REGRESSION_LOG default $RUNNER_TEMP/models-e2e.log
#   SHARD_INDEX    1-based shard index (default 1; round-robin in run_model_e2e.sh)
#   SHARD_COUNT    total number of shards (default 1)
#   HF_MODEL/HF_HOME/HF_HUB_OFFLINE   model weights (set by the workflow)

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

export GTEST_XML_DIR="${GTEST_XML_DIR:-${RUNNER_TEMP:-/tmp}/models-e2e-junit-xml}"
REGRESSION_LOG="${REGRESSION_LOG:-${RUNNER_TEMP:-/tmp}/models-e2e.log}"

mkdir -p "$GTEST_XML_DIR"
rm -f "$GTEST_XML_DIR"/*.xml

# Ensure the `import ttnn` -> emule _ttnn.so symlink (fresh tt-metal checkout
# lacks the post-build step; mirrors ci-ttnn-pytests.sh).
if [ ! -e "$TT_METAL_DIR/ttnn/ttnn/_ttnn.so" ]; then
    ln -sfn "$BUILD_DIR/lib/_ttnn.so" "$TT_METAL_DIR/ttnn/ttnn/_ttnn.so"
fi

echo "== ci-models-e2e.sh =="
echo "  TT_EMULE_DIR:   $TT_EMULE_DIR"
echo "  TT_METAL_DIR:   $TT_METAL_DIR"
echo "  BUILD_DIR:      $BUILD_DIR"
echo "  TT_EMULE_ARCH:  $TT_EMULE_ARCH"
echo "  HF_MODEL:       ${HF_MODEL:-<unset>}"
echo "  HF_HOME:        ${HF_HOME:-<unset>}"
echo "  GTEST_XML_DIR:  $GTEST_XML_DIR"
echo "  REGRESSION_LOG: $REGRESSION_LOG"
echo "  SHARD:          $SHARD_INDEX of $SHARD_COUNT"
echo ""

set +e
bash "$TT_EMULE_DIR/scripts/run_model_e2e.sh" 2>&1 | tee "$REGRESSION_LOG"
rc=${PIPESTATUS[0]}
set -e

xml_count=$(find "$GTEST_XML_DIR" -name '*.xml' 2>/dev/null | wc -l)
echo ""
echo "== model e2e done =="
echo "  exit code:    $rc"
echo "  XML files:    $xml_count in $GTEST_XML_DIR"
echo "  Log:          $REGRESSION_LOG"

exit "$rc"
