#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Local CI driver. Runs emule regression categories (arch × suite) so a developer
# can cover, before merging, what the lightweight PR gate skips. Categories run
# as independent jobs from a queue, each with its own warm JIT cache.
#
# Usage:
#   TT_METAL_DIR=/path/to/tt-metal scripts/run_local_ci.sh [opts] [category...]
#
# Categories:  cpp:wormhole cpp:blackhole cpp:quasar ttnn:wormhole ttnn:blackhole
# Aliases:     cpp ttnn wormhole blackhole quasar all   (default: all)
#
# Options:
#   --tier deferred|full|pr   what to run within each category (default: deferred,
#                             i.e. the complement of each script's PR_TIER — the
#                             tests the PR gate skips). `full` runs everything.
#   -j N                      run up to N categories in parallel (default: 1)
#   --fail-fast               stop launching new categories after one fails
#   --clear-cache             wipe the selected categories' JIT caches first
#   -h | --help               this help
#
# Tier per category under `--tier deferred`: cpp:blackhole is skipped (it runs
# full on the PR gate, nothing deferred); ttnn:wormhole runs full (never on the
# gate); the rest run their deferred complement.
#
# Required env:  TT_METAL_DIR (tt-metal workspace built with build_emule/)
# Optional env:  BUILD_DIR    (default $TT_METAL_DIR/build_emule)
#                CACHE_ROOT   (default <repo>/.local_ci/cache; persistent, warm)
#
# Logs + a machine-readable summary.json land under <repo>/.local_ci/runs/<ts>/.

set -uo pipefail

usage() { sed -n '/^# Local CI driver\./,/^# Logs + a machine-readable/p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_EMULE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CI="$TT_EMULE_DIR/.github/scripts"

# --- args -------------------------------------------------------------------
TIER=deferred
JOBS=1
FAIL_FAST=0
CLEAR_CACHE=0
REQUESTED=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --tier)        [ "$#" -ge 2 ] || { echo "ERROR: --tier requires an argument" >&2; exit 2; }; TIER="$2"; shift 2 ;;
        -j)            [ "$#" -ge 2 ] || { echo "ERROR: -j requires an argument" >&2; exit 2; }; JOBS="$2"; shift 2 ;;
        --fail-fast)   FAIL_FAST=1; shift ;;
        --clear-cache) CLEAR_CACHE=1; shift ;;
        -h|--help)     usage 0 ;;
        -*)            echo "ERROR: unknown option '$1'" >&2; usage 2 ;;
        *)             REQUESTED+=("$1"); shift ;;
    esac
done
case "$TIER" in deferred|full|pr) ;; *) echo "ERROR: --tier must be deferred|full|pr" >&2; exit 2 ;; esac
[[ "$JOBS" =~ ^[0-9]+$ ]] && [ "$JOBS" -ge 1 ] || { echo "ERROR: -j must be a positive integer" >&2; exit 2; }

ALL_CATEGORIES=(cpp:wormhole cpp:blackhole cpp:quasar ttnn:wormhole ttnn:blackhole)

# Expand an alias (or pass through an explicit category), one per line.
expand() {
    case "$1" in
        all)       printf '%s\n' "${ALL_CATEGORIES[@]}" ;;
        cpp)       printf '%s\n' cpp:wormhole cpp:blackhole cpp:quasar ;;
        ttnn)      printf '%s\n' ttnn:wormhole ttnn:blackhole ;;
        wormhole)  printf '%s\n' cpp:wormhole ttnn:wormhole ;;
        blackhole) printf '%s\n' cpp:blackhole ttnn:blackhole ;;
        quasar)    printf '%s\n' cpp:quasar ;;
        cpp:wormhole|cpp:blackhole|cpp:quasar|ttnn:wormhole|ttnn:blackhole) printf '%s\n' "$1" ;;
        *)         echo "ERROR: unknown category '$1'" >&2; exit 2 ;;
    esac
}
[ "${#REQUESTED[@]}" -eq 0 ] && REQUESTED=(all)
CATEGORIES=()
for r in "${REQUESTED[@]}"; do
    # Command substitution (not <(...)) so expand's exit on a bad category aborts.
    out="$(expand "$r")" || exit 2
    while read -r c; do [ -n "$c" ] && CATEGORIES+=("$c"); done <<< "$out"
done
# de-dup, preserve order
declare -A seen; UNIQ=()
for c in "${CATEGORIES[@]}"; do [ -n "${seen[$c]:-}" ] || { UNIQ+=("$c"); seen[$c]=1; }; done
CATEGORIES=("${UNIQ[@]}")

: "${TT_METAL_DIR:?TT_METAL_DIR must be set (path to tt-metal with build_emule/)}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
CACHE_ROOT="${CACHE_ROOT:-$TT_EMULE_DIR/.local_ci/cache}"
export TT_METAL_DIR BUILD_DIR

# Resolve the CI_TIER a category runs under for the requested run tier, or "skip".
resolve_tier() {
    local cat="$1"
    case "$TIER" in
        full) echo full ;;
        deferred)
            case "$cat" in
                cpp:blackhole) echo skip ;;   # full on PR — nothing deferred
                ttnn:wormhole) echo full ;;   # never on the gate — full is its deferred set
                *)             echo deferred ;;
            esac ;;
        pr)
            case "$cat" in
                cpp:blackhole) echo full ;;   # PR full arch
                ttnn:wormhole) echo skip ;;   # not on the gate
                *)             echo pr ;;
            esac ;;
    esac
}

TS="$(date +%Y%m%d-%H%M%S)"
RUNDIR="$TT_EMULE_DIR/.local_ci/runs/$TS"
mkdir -p "$RUNDIR/xml"

echo "== run_local_ci.sh =="
echo "  tier:       $TIER"
echo "  jobs:       $JOBS"
echo "  fail-fast:  $FAIL_FAST"
echo "  categories: ${CATEGORIES[*]}"
echo "  cache root: $CACHE_ROOT"
echo "  run dir:    $RUNDIR"
echo ""

# Stream a leaf script's output: append every line to its log, and surface a
# live per-entry tick. Both leaf families mark entries with '--- <name> ---'
# then '  PASS' / '  FAIL', so this is category-agnostic. In-place (\r) for an
# interactive single-job run; plain tagged lines otherwise (parallel/redirected).
_progress_filter() {
    local tag="$1" inplace="$2" log="$3"
    local n=0 f=0 cur="" line mark
    while IFS= read -r line; do
        printf '%s\n' "$line" >> "$log"
        case "$line" in
            "--- "*" ---") cur="${line#--- }"; cur="${cur% ---}" ;;
            "  PASS"*|"  FAIL"*)
                n=$((n + 1)); case "$line" in "  FAIL"*) f=$((f + 1)); mark="✗" ;; *) mark="✓" ;; esac
                if [ "$inplace" = 1 ]; then
                    printf '\r\033[K  [%s] %d done, %d fail · last: %s %s' "$tag" "$n" "$f" "$mark" "$cur"
                else
                    printf '  [%s] %s %s  (%d done, %d fail)\n' "$tag" "$mark" "$cur" "$n" "$f"
                fi ;;
        esac
    done
    [ "$inplace" = 1 ] && [ "$n" -gt 0 ] && printf '\n'
    return 0
}

# Run one category: drive the matching CI wrapper with a per-category warm JIT
# cache, stream live per-entry progress, capture everything to a per-category
# log, record a status line.
run_category() {
    local cat="$1" tier="$2"
    local safe="${cat//:/-}" kind="${cat%%:*}" arch="${cat##*:}"
    local log="$RUNDIR/$safe.log" xml="$RUNDIR/xml/$safe" cache="$CACHE_ROOT/jit_$safe"
    mkdir -p "$xml" "$cache"; : > "$log"
    # In-place progress only for an interactive, single-job run (concurrent \r
    # from parallel workers would fight over the cursor).
    local inplace=0; { [ "$JOBS" -eq 1 ] && [ -t 1 ]; } && inplace=1
    local start rc; start="$(date +%s)"
    if [ "$kind" = cpp ]; then
        # ci-regression.sh always exits 0; classify-results.py is the pass/fail
        # authority (quasar uses its known-failures allowlist).
        CI_TIER="$tier" TT_EMULE_ARCH="$arch" GTEST_XML_DIR="$xml" \
            REGRESSION_LOG=/dev/null TT_EMULE_JIT_CACHE_DIR="$cache" \
            bash "$CI/ci-regression.sh" 2>&1 | _progress_filter "$cat" "$inplace" "$log"
        local allow=(); [ "$arch" = quasar ] && allow=(--allowlist "$TT_EMULE_DIR/.github/known-failures-quasar.txt")
        python3 "$CI/classify-results.py" --xml-dir "$xml" "${allow[@]}" --build-dir "$BUILD_DIR" >>"$log" 2>&1
        rc=$?
    else
        # ci-ttnn-pytests.sh is self-classifying (exits non-zero on any failure).
        CI_TIER="$tier" TT_EMULE_ARCH="$arch" SHARD_INDEX=1 SHARD_COUNT=1 \
            GTEST_XML_DIR="$xml" REGRESSION_LOG=/dev/null TT_EMULE_JIT_CACHE_DIR="$cache" \
            bash "$CI/ci-ttnn-pytests.sh" 2>&1 | _progress_filter "$cat" "$inplace" "$log"
        rc=${PIPESTATUS[0]}
    fi
    local dur=$(( $(date +%s) - start )) result
    [ "$rc" -eq 0 ] && result=PASS || result=FAIL
    printf '%s|%s|%s|%s\n' "$result" "$dur" "$tier" "$log" > "$RUNDIR/$safe.status"
    if [ "$result" = PASS ]; then
        printf '  [%s] PASS (%ss)\n' "$cat" "$dur"
    else
        printf '  [%s] FAIL (%ss) — see %s\n' "$cat" "$dur" "$log"
    fi
    # Result lives in the .status file; always exit 0 so the pool's `wait -n`
    # reflects "slot freed", not pass/fail.
    return 0
}

# --- build the work queue (skip categories that are no-ops under this tier) --
QUEUE=()
for cat in "${CATEGORIES[@]}"; do
    t="$(resolve_tier "$cat")"
    safe="${cat//:/-}"
    if [ "$t" = skip ]; then
        printf 'SKIP|0|%s|-\n' "$TIER" > "$RUNDIR/$safe.status"
        echo "  [$cat] SKIP (nothing to run under --tier $TIER)"
        continue
    fi
    [ "$CLEAR_CACHE" = 1 ] && rm -rf "$CACHE_ROOT/jit_$safe"
    QUEUE+=("$cat|$t")
done
echo ""

any_failed() { grep -lq '^FAIL' "$RUNDIR"/*.status 2>/dev/null; }

# --- job pool: up to JOBS categories at once; optional category-level fail-fast
running=0 aborted=0
for item in "${QUEUE[@]}"; do
    cat="${item%%|*}"; tier="${item##*|}"; safe="${cat//:/-}"
    if [ "$FAIL_FAST" = 1 ] && [ "$aborted" = 1 ]; then
        printf 'NOTRUN|0|%s|-\n' "$tier" > "$RUNDIR/$safe.status"
        echo "  [$cat] NOT RUN (fail-fast)"
        continue
    fi
    run_category "$cat" "$tier" &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then
        # `wait -n` (bash >=4.3) frees a single slot. Older bash has no `wait -n`,
        # so fall back to waiting for the whole batch and reset the counter to
        # match — batched rather than rolling, but never exceeds -j. (run_category
        # always exits 0, so a `then` here means a slot freed, not a passing test.)
        if wait -n 2>/dev/null; then running=$((running - 1)); else wait; running=0; fi
        [ "$FAIL_FAST" = 1 ] && any_failed && aborted=1
    fi
done
wait
[ "$FAIL_FAST" = 1 ] && any_failed && aborted=1

# --- summary ----------------------------------------------------------------
echo ""
echo "========================================================"
echo " Local CI summary  (tier=$TIER)"
echo "========================================================"
printf '  %-16s %-9s %-7s %s\n' CATEGORY TIER RESULT TIME
overall=0
json="$RUNDIR/summary.json"
printf '{\n  "timestamp": "%s",\n  "tier": "%s",\n  "categories": [\n' "$TS" "$TIER" > "$json"
first=1
for cat in "${CATEGORIES[@]}"; do
    safe="${cat//:/-}"; sf="$RUNDIR/$safe.status"
    [ -f "$sf" ] || continue
    IFS='|' read -r result dur ctier log < "$sf"
    printf '  %-16s %-9s %-7s %ss\n' "$cat" "$ctier" "$result" "$dur"
    [ "$result" = FAIL ] && overall=1
    [ "$first" = 1 ] || printf ',\n' >> "$json"; first=0
    printf '    {"category": "%s", "tier": "%s", "result": "%s", "duration_s": %s, "log": "%s"}' \
        "$cat" "$ctier" "$result" "$dur" "$log" >> "$json"
done
printf '\n  ],\n  "overall": "%s"\n}\n' "$([ "$overall" = 0 ] && echo PASS || echo FAIL)" >> "$json"
echo ""
echo "  logs + summary.json: $RUNDIR"
[ "$overall" = 0 ] && echo "  overall: PASS" || echo "  overall: FAIL"
exit "$overall"
