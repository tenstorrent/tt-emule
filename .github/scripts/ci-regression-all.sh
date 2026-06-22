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

set -uo pipefail

: "${TT_METAL_DIR:?TT_METAL_DIR must be set}"
: "${BUILD_DIR:?BUILD_DIR must be set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_EMULE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

ARCHES="${ARCHES:-wormhole blackhole quasar}"
GTEST_XML_ROOT="${GTEST_XML_ROOT:-${RUNNER_TEMP:-/tmp}/gtest-xml}"

declare -A results
overall_rc=0

for arch in $ARCHES; do
    echo ""
    echo "########################################################"
    echo "# C++ regression: $arch"
    echo "########################################################"

    # wormhole/blackhole: zero-tolerance (no allowlist). quasar: documented
    # known failures in known-failures-quasar.txt.
    allowlist=()
    if [ "$arch" = "quasar" ]; then
        allowlist=(--allowlist "$TT_EMULE_DIR/.github/known-failures-quasar.txt")
    fi

    arch_xml_dir="$GTEST_XML_ROOT/$arch"

    # ci-regression.sh always exits 0 (classify-results.py is the pass/fail
    # authority); per-arch XML dir keeps results isolated.
    TT_EMULE_ARCH="$arch" \
    GTEST_XML_DIR="$arch_xml_dir" \
    REGRESSION_LOG="${RUNNER_TEMP:-/tmp}/regression-${arch}.log" \
        bash "$SCRIPT_DIR/ci-regression.sh"

    echo ""
    echo "== Classifying $arch =="
    if python3 "$SCRIPT_DIR/classify-results.py" \
        --xml-dir "$arch_xml_dir" \
        "${allowlist[@]}" \
        --build-dir "$BUILD_DIR"; then
        results[$arch]="PASS"
    else
        results[$arch]="FAIL"
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
