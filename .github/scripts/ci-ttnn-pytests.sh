#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# CI wrapper for the TTNN pytest regression.
#
# Required env:
#   TT_METAL_DIR   path to tt-metal source tree (checked out at the pinned SHA)
#   BUILD_DIR      absolute path to the build tree (containing ttnn/_ttnn.so etc.)
#
# Optional env:
#   GTEST_XML_DIR  default $RUNNER_TEMP/ttnn-junit-xml (per-entry junit XML)
#   REGRESSION_LOG default $RUNNER_TEMP/ttnn-pytests.log
#   SHARD_INDEX    1-based shard index (default 1; round-robin in run_ttnn_pytests.sh)
#   SHARD_COUNT    total number of shards (default 1)

set -euo pipefail

: "${TT_METAL_DIR:?TT_METAL_DIR must be set}"
: "${BUILD_DIR:?BUILD_DIR must be set}"
export SHARD_INDEX="${SHARD_INDEX:-1}"
export SHARD_COUNT="${SHARD_COUNT:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_EMULE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export GTEST_XML_DIR="${GTEST_XML_DIR:-${RUNNER_TEMP:-/tmp}/ttnn-junit-xml}"
REGRESSION_LOG="${REGRESSION_LOG:-${RUNNER_TEMP:-/tmp}/ttnn-pytests.log}"

mkdir -p "$GTEST_XML_DIR"
rm -f "$GTEST_XML_DIR"/*.xml

# Ensure the post-build symlink that lets `import ttnn` resolve to the
# emule-built _ttnn.so. tt-emule's BUILD_GUIDE.md "Post-build symlinks"
# subsection covers this — the test runner does a fresh tt-metal checkout
# without that step, so re-create it here.
if [ ! -e "$TT_METAL_DIR/ttnn/ttnn/_ttnn.so" ]; then
    ln -sfn "$BUILD_DIR/ttnn/_ttnn.so" "$TT_METAL_DIR/ttnn/ttnn/_ttnn.so"
fi

echo "== ci-ttnn-pytests.sh =="
echo "  TT_EMULE_DIR:   $TT_EMULE_DIR"
echo "  TT_METAL_DIR:   $TT_METAL_DIR"
echo "  BUILD_DIR:      $BUILD_DIR"
echo "  GTEST_XML_DIR:  $GTEST_XML_DIR"
echo "  REGRESSION_LOG: $REGRESSION_LOG"
echo "  SHARD:          $SHARD_INDEX of $SHARD_COUNT"
echo ""

set +e
bash "$TT_EMULE_DIR/scripts/run_ttnn_pytests.sh" 2>&1 | tee "$REGRESSION_LOG"
rc=${PIPESTATUS[0]}
set -e

xml_count=$(find "$GTEST_XML_DIR" -name '*.xml' 2>/dev/null | wc -l)
echo ""
echo "== TTNN pytest done =="
echo "  exit code:    $rc"
echo "  XML files:    $xml_count in $GTEST_XML_DIR"
echo "  Log:          $REGRESSION_LOG"

# Propagate the script's exit code: this CI job fails if any entry in the
# curated list fails. The script is self-classifying — only known-passing
# entries are listed, so any failure is a real regression.
exit "$rc"
