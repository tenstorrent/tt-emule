#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# CI per-shard adapter for the nightly ASAN sweep: runs ONE shard of tt-metal's
# post-commit ttnn entry set with the emule sanitizers armed, to surface ASAN
# violations in tt-metal's kernels. Thin by design — the run logic is sweep.py's
# `--asan` mode so CI and local runs stay identical. See docs/asan-nightly-sweep.md.
#
# This job FAILS when its own entries hit any [ASAN ERROR], so a red shard in the
# run's job list points straight at the suites that violated. The verdict is
# rendered AT THE END: the sweep always runs every entry to completion first
# (a violation aborts only the offending test's forked child), and the findings
# grep + exit 1 below are the last thing this script does. An INVALID shard —
# one that ran but reached no emulator — also exits non-zero AND withholds its
# bucket-<id>.ok marker. The marker is written on every VALID run, findings or not:
# it tells the report job "this slice really executed", separately from the exit
# code, so the aggregate verdict can still distinguish a shard that found bugs
# from one that never ran its tests.
#
# Required env: TT_METAL_DIR, BUILD_DIR, TT_EMULE_ARCH.
# Bucket:       BUCKET_ID, SWEEP_ONLY (the bucket's manifest entries).
# Optional:     OUT_DIR, MANIFEST, PYTEST_BIN, ASAN_CHECKS, SHARD_INDEX/COUNT.

set -uo pipefail

: "${TT_METAL_DIR:?TT_METAL_DIR must be set}"
: "${BUILD_DIR:?BUILD_DIR must be set}"
: "${TT_EMULE_ARCH:?TT_EMULE_ARCH must be set (blackhole|wormhole)}"
# Bucket identity. SWEEP_ONLY carries the bucket's entries, so the shard index is
# a formality (1 of 1); BUCKET_ID names the completion marker the report checks.
SHARD_INDEX="${SHARD_INDEX:-1}"
SHARD_COUNT="${SHARD_COUNT:-1}"
BUCKET_ID="${BUCKET_ID:-shard-$SHARD_INDEX}"

MANIFEST="${MANIFEST:-$TT_METAL_DIR/tests/pipeline_reorg/ttnn_sanity_tests.yaml}"
PYTEST_BIN="${PYTEST_BIN:-/opt/ttmlir-toolchain/venv/bin/pytest}"
OUT_DIR="${OUT_DIR:-${RUNNER_TEMP:-/tmp}/asan-sweep-out}"
ASAN_CHECKS="${ASAN_CHECKS:-all}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_EMULE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

mkdir -p "$OUT_DIR"

# Per-shard JIT cache. ASAN is folded into the kernel cache key (it compiles with
# -g), so an ASAN lane never shares the plain sweep's cached .so files. Isolating
# per shard also keeps two shards on one runner from racing the same cache dir.
export TT_EMULE_JIT_CACHE_DIR="${TT_EMULE_JIT_CACHE_DIR:-${RUNNER_TEMP:-/tmp}/tt_emule_jit_asan_${SHARD_INDEX}}"

# Leave TT_METAL_EMULE_ASAN_ALLOW_CORE UNSET so __emule_asan_panic keeps its
# prctl(PR_SET_DUMPABLE, 0) suppression. Each abort would otherwise dump a
# multi-GB core, and a sweep aborts many times — that fills a runner's disk.
unset TT_METAL_EMULE_ASAN_ALLOW_CORE

# Per-check selection. Exported unconditionally (including "all") so the run does
# not inherit a narrowed list from the environment. An unrecognized name falls back
# to every check rather than to none — a typo must not silence the sweep and report
# clean — so a subset is echoed here to make what ran auditable from the log.
export TT_METAL_EMULE_ASAN_CHECKS="$ASAN_CHECKS"
if [ "$ASAN_CHECKS" != "all" ]; then
    echo "::notice::ASAN check subset active: '$ASAN_CHECKS' — checks outside this list will"\
         "not fire, so this run's '0 findings' only covers the listed checks."
fi
# Passed through from the workflow's skip-dirty-cb input (or the environment for a
# local run). Announced because it changes what a clean result means.
if [ -n "${TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB:-}" ] && [ "${TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB}" != "0" ]; then
    echo "::notice::Dirty CB check SKIPPED for this run — every other check is on. Findings"\
         "here are what Dirty CB would otherwise have masked; a clean result does not"\
         "mean the CB handshakes are balanced."
fi

echo "== ci-asan-sweep.sh =="
echo "  TT_EMULE_ARCH:  $TT_EMULE_ARCH"
echo "  TT_METAL_DIR:   $TT_METAL_DIR"
echo "  BUILD_DIR:      $BUILD_DIR"
echo "  OUT_DIR:        $OUT_DIR"
echo "  bucket:         $BUCKET_ID"
echo "  entries:        ${SWEEP_ONLY:-(all)}"
echo "  ASAN_CHECKS:    $ASAN_CHECKS"
echo "  skip Dirty CB:  ${TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB:-0}"
echo "  JIT cache:      $TT_EMULE_JIT_CACHE_DIR"
# Runner capacity, logged because it is the main lever on shard wall-clock: emule
# runs a fiber-worker pool per test, so cores translate fairly directly into
# throughput. Recorded here so a future decision to move this lane to a smaller,
# more plentiful runner class can be made from measurements rather than guesswork.
echo "  runner:         $(nproc) cores, $(free -g 2>/dev/null | awk '/^Mem:/{print $2"GB RAM"}') ${RUNNER_NAME:+($RUNNER_NAME)}"
echo ""

# pytest-timeout is required by the manifest cmds; pytest-forked by --asan's
# forced per-test forking. Install idempotently (pinned), mirroring
# ci-post-commit-sweep.sh. Use the venv's python, not the pytest binary.
VENV_PY="$(dirname "$PYTEST_BIN")/python"
for pkg in pytest-timeout==2.4.0 pytest-forked==1.6.0; do
    name="${pkg%%==*}"
    if ! "$VENV_PY" -m pip show "$name" >/dev/null 2>&1; then
        echo "Installing $pkg into the toolchain venv…"
        "$VENV_PY" -m pip install --quiet "$pkg"
    fi
done

# Ensure the `import ttnn` -> emule _ttnn.so symlink (a fresh tt-metal checkout
# lacks the post-build step; mirrors ci-post-commit-sweep.sh).
if [ ! -e "$TT_METAL_DIR/ttnn/ttnn/_ttnn.so" ]; then
    ln -sfn "$BUILD_DIR/lib/_ttnn.so" "$TT_METAL_DIR/ttnn/ttnn/_ttnn.so"
fi

"$VENV_PY" "$TT_EMULE_DIR/scripts/post_commit_sweep/sweep.py" run \
    --arch "$TT_EMULE_ARCH" --manifest "$MANIFEST" \
    --tt-metal-dir "$TT_METAL_DIR" --build-dir "$BUILD_DIR" \
    --out-dir "$OUT_DIR" \
    --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" \
    --pytest-bin "$PYTEST_BIN" \
    --asan
sweep_rc=$?

# ---------------------------------------------------------------------------
# Validity gate — the emule_postflight contract from
# tt-metal/tt_metal/impl/emulation/emule_setup.sh, adapted for --forked.
#
# emule_postflight looks for "execute_program_emulated", but that marker is
# emitted from INSIDE the test process, and pytest-forked runs each test in a
# forked child whose stdout/stderr it only replays when the child FAILS. On an
# all-passing entry every such line is discarded, so requiring it would fail a
# perfectly good run. Two markers that do survive are used instead:
#   * the emule-mode banner, logged by the PARENT at import (rtoptions.cpp);
#   * the executed-test count from the JUnit XML, which proves work happened
#     rather than just an import.
# An [ASAN ERROR] is unaffected by the child-output discard: the panic is
# [[noreturn]], so a violation always kills the child, which always makes
# pytest-forked replay its output.
# ---------------------------------------------------------------------------
log_count=$(find "$OUT_DIR" -name '*.log' 2>/dev/null | wc -l)
emule_mode=$(grep -rlE "simulator/emule target device" "$OUT_DIR" --include='*.log' 2>/dev/null | wc -l)
# Informational: present only for entries that had a failing/crashing child.
emulated=$(grep -rlF "execute_program_emulated" "$OUT_DIR" --include='*.log' 2>/dev/null | wc -l)
findings=$(grep -rhF "[ASAN ERROR]" "$OUT_DIR" --include='*.log' 2>/dev/null | wc -l)
# Real-hardware markers must never appear: emule is CPU-only, and their presence
# would mean the run bypassed the emulator (so every check was inert).
hw_markers=$(grep -rlE "Established firmware bundle version|Mapped hugepage|KMD version" \
    "$OUT_DIR" --include='*.log' 2>/dev/null | wc -l)
executed=$("$VENV_PY" - "$OUT_DIR" <<'PY'
import sys, glob, os, xml.etree.ElementTree as ET
total = 0
for p in glob.glob(os.path.join(sys.argv[1], "**", "*.xml"), recursive=True):
    try:
        r = ET.parse(p).getroot()
    except ET.ParseError:
        continue
    for s in ([r] if r.tag == "testsuite" else r.findall("testsuite")):
        if s.tag != "testsuite":
            continue
        t = int(s.get("tests") or 0)
        total += t - int(s.get("skipped") or 0)
print(max(total, 0))
PY
)

echo ""
echo "== bucket $BUCKET_ID done =="
echo "  sweep rc:               $sweep_rc"
echo "  entry logs:             $log_count"
echo "  logs in emule mode:     $emule_mode"
echo "  executed tests (XML):   $executed"
echo "  [ASAN ERROR] lines:     $findings"
echo "  logs w/ runner detail:  $emulated (only entries that had a failing child)"
echo "  real-HW marker logs:    $hw_markers"

if [ "$log_count" -eq 0 ]; then
    echo "::error::INVALID RUN — no entry logs produced; the shard ran nothing."
    exit 1
fi
if [ "$emule_mode" -eq 0 ]; then
    echo "::error::INVALID RUN — the emule-mode banner appears in no log, so tt-metal did not"\
         "select the emulator. Every sanitizer was inert; '0 findings' is meaningless."
    exit 1
fi
if [ "${executed:-0}" -eq 0 ]; then
    echo "::error::INVALID RUN — 0 tests executed (all skipped/deselected or collection failed);"\
         "'0 findings' says nothing about metal."
    exit 1
fi
if [ "$hw_markers" -ne 0 ]; then
    echo "::error::INVALID RUN — real-hardware markers present in $hw_markers log(s);"\
         "this did not run on the emulator."
    exit 1
fi

# Completion marker: written on every VALID run, findings or not. The report job
# fails if any shard's marker is missing — that is how a shard that died before
# this gate (an Actions hiccup in "Set up job") is still caught, since it leaves
# neither a marker nor results.
echo "bucket=$BUCKET_ID arch=$TT_EMULE_ARCH entries=$log_count findings=$findings" \
    > "$OUT_DIR/bucket-${BUCKET_ID}.ok"

if [ "$findings" -ne 0 ]; then
    echo "::error::$findings [ASAN ERROR] line(s) in this shard's entries — see the"\
         "per-entry annotations above and this shard's log artifact."
    exit 1
fi
echo "  VALID emule run — no ASAN findings in this shard."
exit 0
