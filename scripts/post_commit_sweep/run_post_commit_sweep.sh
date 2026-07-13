#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Local end-to-end driver: expand -> run -> parse, the same pipeline the nightly
# CI runs (CI shards it; this does all shards on one box). See
# docs/post-commit-sweep.md.
#
# Required env:
#   TT_METAL_DIR    path to the tt-metal source tree (built for emule)
#
# Optional env:
#   BUILD_DIR       default $TT_METAL_DIR/build_emule
#   TT_EMULE_ARCH   blackhole | wormhole          (default blackhole)
#   OUT_DIR         report + per-entry XML/log dir (default $TT_METAL_DIR/../sweep-out/<arch>)
#   MANIFEST        default $TT_METAL_DIR/tests/pipeline_reorg/ttnn_sanity_tests.yaml
#   PYTEST_BIN      default /opt/ttmlir-toolchain/venv/bin/pytest
#   SHARD_COUNT     split entries across N sequential local passes (default 1 = all at once)
#   SWEEP_ONLY      comma-separated entry slugs to run (debug / targeted reruns)
#   KEEP_XML        1 = don't wipe an existing XML dir first (resume) (default 0)
#
# Example (full local blackhole sweep — slow!):
#   TT_METAL_DIR=../tt-metal bash scripts/post_commit_sweep/run_post_commit_sweep.sh
#
# Example (fast debug loop — one small entry):
#   TT_METAL_DIR=../tt-metal SWEEP_ONLY=ttnn-data-movement-group \
#     bash scripts/post_commit_sweep/run_post_commit_sweep.sh

set -euo pipefail

: "${TT_METAL_DIR:?TT_METAL_DIR must be set}"
TT_METAL_DIR="$(cd "$TT_METAL_DIR" && pwd)"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
TT_EMULE_ARCH="${TT_EMULE_ARCH:-blackhole}"
MANIFEST="${MANIFEST:-$TT_METAL_DIR/tests/pipeline_reorg/ttnn_sanity_tests.yaml}"
PYTEST_BIN="${PYTEST_BIN:-/opt/ttmlir-toolchain/venv/bin/pytest}"
SHARD_COUNT="${SHARD_COUNT:-1}"
KEEP_XML="${KEEP_XML:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_EMULE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$TT_METAL_DIR/../sweep-out/$TT_EMULE_ARCH}"
XML_DIR="$OUT_DIR/xml"

mkdir -p "$XML_DIR"
# Also wipe the *.excluded.json sidecars: sweep.py reuses an existing one to
# skip re-collecting, so a stale sidecar from a prior run in the same OUT_DIR
# would cause incorrect ignores / missed import-error filtering.
[ "$KEEP_XML" = "1" ] || rm -f "$XML_DIR"/*.xml "$XML_DIR"/*.log "$XML_DIR"/*.excluded.json 2>/dev/null || true

echo "== post-commit sweep (local) =="
echo "  arch:        $TT_EMULE_ARCH"
echo "  TT_METAL_DIR:$TT_METAL_DIR"
echo "  BUILD_DIR:   $BUILD_DIR"
echo "  OUT_DIR:     $OUT_DIR"
echo "  SHARD_COUNT: $SHARD_COUNT"
echo "  SWEEP_ONLY:  ${SWEEP_ONLY:-<all>}"
echo ""

# pytest-timeout is required by the manifest cmds (`--timeout N`) but is not in
# the toolchain venv by default. Install idempotently (pinned, matches the
# original sweep). pytest-forked / pytest-split are already present. Use the
# venv's python (next to PYTEST_BIN) for pip, not the pytest binary.
VENV_PY="$(dirname "$PYTEST_BIN")/python"
if ! "$VENV_PY" -m pip show pytest-timeout >/dev/null 2>&1; then
    echo "Installing pytest-timeout into the toolchain venv…"
    "$VENV_PY" -m pip install --quiet pytest-timeout==2.4.0
fi

# Ensure the `import ttnn` -> emule _ttnn.so symlink exists (BUILD_GUIDE.md).
if [ ! -e "$TT_METAL_DIR/ttnn/ttnn/_ttnn.so" ]; then
    ln -sfn "$BUILD_DIR/lib/_ttnn.so" "$TT_METAL_DIR/ttnn/ttnn/_ttnn.so"
fi

# Pin SHA for the report header (best-effort).
PIN_SHA="$(grep -vE '^\s*(#|$)' "$TT_EMULE_DIR/tt-metal-pin.txt" 2>/dev/null | head -1 | awk '{print $1}')"
PIN_SHA="${PIN_SHA:-unknown}"

# 1) Expand + audit (also the expected-entry list for truncation detection).
"$VENV_PY" "$SCRIPT_DIR/sweep.py" expand \
    --arch "$TT_EMULE_ARCH" --manifest "$MANIFEST" \
    --out "$OUT_DIR/expected.json" >/dev/null

SKU="$("$VENV_PY" -c "import json;print(json.load(open('$OUT_DIR/expected.json'))['sku'])")"

# 2) Run each shard sequentially (shards share the JIT cache — never parallel).
for shard in $(seq 1 "$SHARD_COUNT"); do
    echo ">>> shard $shard / $SHARD_COUNT"
    "$VENV_PY" "$SCRIPT_DIR/sweep.py" run \
        --arch "$TT_EMULE_ARCH" --manifest "$MANIFEST" \
        --tt-metal-dir "$TT_METAL_DIR" --build-dir "$BUILD_DIR" \
        --out-dir "$XML_DIR" \
        --shard-index "$shard" --shard-count "$SHARD_COUNT" \
        --pytest-bin "$PYTEST_BIN" \
        ${SWEEP_ONLY:+--only "$SWEEP_ONLY"}
done

# 3) Parse + report.
"$VENV_PY" "$SCRIPT_DIR/parse_sweep_results.py" \
    --xml-dir "$XML_DIR" --log-dir "$XML_DIR" \
    --arch "$TT_EMULE_ARCH" --sku "$SKU" --pin-sha "$PIN_SHA" \
    --manifest "$MANIFEST" --tt-emule-dir "$TT_EMULE_DIR" \
    --expected-json "$OUT_DIR/expected.json" \
    --out-exec "$OUT_DIR/exec.md" \
    --out-dev "$OUT_DIR/dev.md" \
    --out-headline "$OUT_DIR/headline.json" \
    --out-summary "$OUT_DIR/summary.md"

echo ""
echo "== reports written to $OUT_DIR =="
echo "  exec.md / dev.md / headline.json / summary.md"
