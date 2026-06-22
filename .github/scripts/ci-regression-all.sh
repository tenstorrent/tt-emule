#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Run the tt-emule C++ regression suite for ALL architectures sequentially in a
# single CI job, then classify each arch's results. Replaces the former 3-leg
# matrix (one xlarge runner per arch) with one runner that visits each arch in
# turn — trading wall-clock for fewer concurrent runners.
#
# Sequential (never parallel) is mandatory: the per-arch run_regression_*.sh
# scripts share the JIT cache (/tmp/tt_emule_jit_cache_$UID) and clear it at
# start, so running them concurrently corrupts the cache. Each arch writes its
# gtest XML to a per-arch subdir so results don't collide.
#
# Required env:
#   TT_METAL_DIR   path to tt-metal workspace (checked out at the right SHA)
#   BUILD_DIR      absolute path to the build tree (containing test/tt_metal/*)
#
# Optional env:
#   ARCHES         space-separated arch list; default "wormhole blackhole quasar"
#   GTEST_XML_ROOT root dir for per-arch XML; default $RUNNER_TEMP/gtest-xml
#   CI_TIER        full (default) | pr | deferred — selects which entries run:
#                    full      every entry, every arch (on-push / full local run)
#                    pr        PR gate: PR_FULL_ARCHES run full; others run only
#                              their script's PR_TIER (smoke)
#                    deferred  the complement of the PR gate: PR_FULL_ARCHES are
#                              skipped (they ran full on PR — nothing deferred);
#                              others run everything NOT in their PR_TIER
#   PR_FULL_ARCHES space-separated archs that run their FULL suite on the PR
#                  gate (and are skipped under `deferred`); default "blackhole"

set -uo pipefail

: "${TT_METAL_DIR:?TT_METAL_DIR must be set}"
: "${BUILD_DIR:?BUILD_DIR must be set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_EMULE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

ARCHES="${ARCHES:-wormhole blackhole quasar}"
GTEST_XML_ROOT="${GTEST_XML_ROOT:-${RUNNER_TEMP:-/tmp}/gtest-xml}"
CI_TIER="${CI_TIER:-full}"
PR_FULL_ARCHES="${PR_FULL_ARCHES:-blackhole}"

case "$CI_TIER" in
    full|pr|deferred) ;;
    *) echo "ERROR: CI_TIER must be full|pr|deferred, got '$CI_TIER'" >&2; exit 2 ;;
esac

# Is $1 present in the space-separated PR_FULL_ARCHES list?
_is_full_arch() {
    local a; for a in $PR_FULL_ARCHES; do [ "$a" = "$1" ] && return 0; done; return 1
}

declare -A results
overall_rc=0

for arch in $ARCHES; do
    # Resolve the per-arch tier from the run-wide CI_TIER. The "main" arch(s)
    # (PR_FULL_ARCHES) always run their full suite on the PR gate, so under the
    # complementary `deferred` run they have nothing to contribute and are
    # skipped entirely.
    arch_tier="$CI_TIER"
    if [ "$CI_TIER" = "pr" ] && _is_full_arch "$arch"; then
        arch_tier="full"
    elif [ "$CI_TIER" = "deferred" ] && _is_full_arch "$arch"; then
        results[$arch]="SKIPPED (full on PR — nothing deferred)"
        continue
    fi

    echo ""
    echo "########################################################"
    echo "# C++ regression: $arch (tier=$arch_tier)"
    echo "########################################################"

    # wormhole/blackhole: zero-tolerance (no allowlist). quasar: documented
    # known failures in known-failures-quasar.txt.
    allowlist=()
    if [ "$arch" = "quasar" ]; then
        allowlist=(--allowlist "$TT_EMULE_DIR/.github/known-failures-quasar.txt")
    fi

    # A smoke (pr-tier) run intentionally omits most allowlisted known-failure
    # tests, so their patterns match nothing — that is expected, not a stale
    # allowlist. Suppress the stale gate for smoke runs only.
    stale_arg=()
    if [ "$arch_tier" = "pr" ]; then
        stale_arg=(--ignore-stale)
    fi

    arch_xml_dir="$GTEST_XML_ROOT/$arch"

    # ci-regression.sh always exits 0 (classify-results.py is the pass/fail
    # authority); per-arch XML dir keeps results isolated.
    CI_TIER="$arch_tier" \
    TT_EMULE_ARCH="$arch" \
    GTEST_XML_DIR="$arch_xml_dir" \
    REGRESSION_LOG="${RUNNER_TEMP:-/tmp}/regression-${arch}.log" \
        bash "$SCRIPT_DIR/ci-regression.sh"

    echo ""
    echo "== Classifying $arch =="
    if python3 "$SCRIPT_DIR/classify-results.py" \
        --xml-dir "$arch_xml_dir" \
        "${allowlist[@]}" \
        "${stale_arg[@]}" \
        --build-dir "$BUILD_DIR"; then
        results[$arch]="PASS (tier=$arch_tier)"
    else
        results[$arch]="FAIL (tier=$arch_tier)"
        overall_rc=1
    fi
done

echo ""
echo "========================================================"
echo "# C++ regression summary"
echo "========================================================"
for arch in $ARCHES; do
    printf '  %-10s %s\n' "$arch" "${results[$arch]:-MISSING}"
done

exit "$overall_rc"
