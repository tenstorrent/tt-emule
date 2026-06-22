#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Run every test that the on-PR gate no longer runs — the "deferred" complement
# of the PR gate — so a developer can cover locally, before merging, what the
# lightweight PR gate trims. (After merge, the on-push run executes the FULL
# suite and would catch these anyway; this script lets you find regressions
# pre-merge instead of reddening main.)
#
# What this runs (= on-push full suite MINUS the on-PR gate):
#   C++  : wormhole + quasar — every entry NOT in that script's PR_TIER smoke
#          set. Blackhole C++ is the PR gate's full arch, so it has nothing
#          deferred and is skipped here.
#   TTNN : blackhole — every entry NOT in its PR_TIER smoke set.
#          wormhole — the entire suite (no wormhole TTNN runs on the PR gate).
#
# This is the COMPLEMENT of the PR gate, not the whole suite. To run absolutely
# everything, invoke the per-arch scripts directly with CI_TIER unset (the
# default), e.g. `bash scripts/run_regression_wormhole.sh`.
#
# Reuses the CI wrappers (.github/scripts/) with CI_TIER=deferred so the set
# stays exactly the complement of the PR_TIER lists — one source of truth, no
# second list to keep in sync.
#
# Required env:
#   TT_METAL_DIR   path to tt-metal workspace (built with build_emule/ inside)
# Optional env:
#   BUILD_DIR      build tree; default $TT_METAL_DIR/build_emule
#   OUT_DIR        where logs + JUnit/gtest XML land; default ./_local_ci_out

set -uo pipefail

: "${TT_METAL_DIR:?TT_METAL_DIR must be set (path to tt-metal with build_emule/)}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_EMULE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CI="$TT_EMULE_DIR/.github/scripts"

OUT_DIR="${OUT_DIR:-$PWD/_local_ci_out}"
mkdir -p "$OUT_DIR"

# The CI wrappers default their log / XML paths under RUNNER_TEMP; point it at
# OUT_DIR so a local run drops everything in one place.
export RUNNER_TEMP="$OUT_DIR"
export TT_METAL_DIR BUILD_DIR

echo "== run_local_ci.sh (deferred complement of the PR gate) =="
echo "  TT_METAL_DIR: $TT_METAL_DIR"
echo "  BUILD_DIR:    $BUILD_DIR"
echo "  OUT_DIR:      $OUT_DIR"
echo ""

rc=0

# Switching architectures across runs requires a clean JIT cache: the cache
# (/tmp/tt_emule_jit_cache_$UID) is shared across archs and the TTNN scripts do
# NOT clear it (only the C++ run_regression_*.sh scripts do), so a stale kernel
# compiled for a different arch can be reused and hang/corrupt the next arch's
# run. CI never hits this (each arch's TTNN job is a separate runner with a
# fresh cache); locally we must clear it before each arch hand-off.
JIT_CACHE_DIR="/tmp/tt_emule_jit_cache_$(id -u)"
clear_jit_cache() { echo "Clearing JIT cache: $JIT_CACHE_DIR"; rm -rf "$JIT_CACHE_DIR"; }

echo "############################################################"
echo "# Deferred C++ regression (wormhole + quasar; blackhole skipped)"
echo "############################################################"
# CI_TIER=deferred: ci-regression-all.sh skips PR_FULL_ARCHES (blackhole) and
# runs wormhole + quasar with their CI_TIER=deferred complement. Quasar's
# known-failures allowlist still applies (the deferred set includes them).
# (run_regression_*.sh clears the JIT cache per-arch internally.)
CI_TIER=deferred bash "$CI/ci-regression-all.sh" || rc=1

echo ""
echo "############################################################"
echo "# Deferred TTNN — blackhole (complement of PR_TIER)"
echo "############################################################"
clear_jit_cache
CI_TIER=deferred TT_EMULE_ARCH=blackhole \
    GTEST_XML_DIR="$OUT_DIR/ttnn-xml-blackhole" \
    REGRESSION_LOG="$OUT_DIR/ttnn-blackhole.log" \
    bash "$CI/ci-ttnn-pytests.sh" || rc=1

echo ""
echo "############################################################"
echo "# Deferred TTNN — wormhole (entire suite; none runs on the PR gate)"
echo "############################################################"
clear_jit_cache
TT_EMULE_ARCH=wormhole \
    GTEST_XML_DIR="$OUT_DIR/ttnn-xml-wormhole" \
    REGRESSION_LOG="$OUT_DIR/ttnn-wormhole.log" \
    bash "$CI/ci-ttnn-pytests.sh" || rc=1

echo ""
echo "============================================================"
if [ "$rc" -eq 0 ]; then
    echo "Deferred suite: PASS"
else
    echo "Deferred suite: FAIL — see logs under $OUT_DIR"
fi
echo "============================================================"
exit "$rc"
