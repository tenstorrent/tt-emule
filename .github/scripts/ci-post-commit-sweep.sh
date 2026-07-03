#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# CI per-shard adapter for the nightly post-commit sweep: runs ONE shard under
# emule, writing per-entry XML+logs to $GTEST_XML_DIR for the aggregate job.
# Thin by design — the logic lives in scripts/post_commit_sweep/sweep.py so CI
# and local runs stay identical. See docs/post-commit-sweep.md.
#
# Required env: TT_METAL_DIR, BUILD_DIR, TT_EMULE_ARCH, SHARD_INDEX, SHARD_COUNT.
# Optional:     GTEST_XML_DIR, MANIFEST, PYTEST_BIN.

set -euo pipefail

: "${TT_METAL_DIR:?TT_METAL_DIR must be set}"
: "${BUILD_DIR:?BUILD_DIR must be set}"
: "${TT_EMULE_ARCH:?TT_EMULE_ARCH must be set (blackhole|wormhole)}"
: "${SHARD_INDEX:?SHARD_INDEX must be set}"
: "${SHARD_COUNT:?SHARD_COUNT must be set}"

MANIFEST="${MANIFEST:-$TT_METAL_DIR/tests/pipeline_reorg/ttnn_sanity_tests.yaml}"
PYTEST_BIN="${PYTEST_BIN:-/opt/ttmlir-toolchain/venv/bin/pytest}"
GTEST_XML_DIR="${GTEST_XML_DIR:-${RUNNER_TEMP:-/tmp}/sweep-xml}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_EMULE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

mkdir -p "$GTEST_XML_DIR"

echo "== ci-post-commit-sweep.sh =="
echo "  TT_EMULE_ARCH: $TT_EMULE_ARCH"
echo "  TT_METAL_DIR:  $TT_METAL_DIR"
echo "  BUILD_DIR:     $BUILD_DIR"
echo "  GTEST_XML_DIR: $GTEST_XML_DIR"
echo "  SHARD:         $SHARD_INDEX of $SHARD_COUNT"
echo ""

# pytest-timeout is required by the manifest cmds but absent from the toolchain
# venv. Install idempotently (pinned). pytest-forked / pytest-split are present.
# Use the venv's python (next to PYTEST_BIN) for pip, not the pytest binary.
VENV_PY="$(dirname "$PYTEST_BIN")/python"
if ! "$VENV_PY" -m pip show pytest-timeout >/dev/null 2>&1; then
    echo "Installing pytest-timeout into the toolchain venv…"
    "$VENV_PY" -m pip install --quiet pytest-timeout==2.4.0
fi

# Ensure the `import ttnn` -> emule _ttnn.so symlink (fresh tt-metal checkout
# lacks the post-build step; mirrors ci-ttnn-pytests.sh).
if [ ! -e "$TT_METAL_DIR/ttnn/ttnn/_ttnn.so" ]; then
    ln -sfn "$BUILD_DIR/lib/_ttnn.so" "$TT_METAL_DIR/ttnn/ttnn/_ttnn.so"
fi

# Use the venv python (has PyYAML; same interpreter as PYTEST_BIN) rather than
# the container's system python3, which may lack PyYAML.
"$VENV_PY" "$TT_EMULE_DIR/scripts/post_commit_sweep/sweep.py" run \
    --arch "$TT_EMULE_ARCH" --manifest "$MANIFEST" \
    --tt-metal-dir "$TT_METAL_DIR" --build-dir "$BUILD_DIR" \
    --out-dir "$GTEST_XML_DIR" \
    --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" \
    --pytest-bin "$PYTEST_BIN"

xml_count=$(find "$GTEST_XML_DIR" -name '*.xml' 2>/dev/null | wc -l)
echo ""
echo "== shard $SHARD_INDEX done — $xml_count XML files in $GTEST_XML_DIR =="
