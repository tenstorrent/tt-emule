#!/usr/bin/env bash
# Run the full tt-emule regression suite with gtest XML output enabled.
#
# Required env:
#   TT_METAL_DIR    path to tt-metal workspace (checked out at the right SHA)
#   BUILD_DIR       absolute path to the build tree (containing test/tt_metal/*)
#
# Optional env:
#   GTEST_XML_DIR   where per-test XML lands; default: $RUNNER_TEMP/gtest-xml
#   REGRESSION_LOG  combined stdout/stderr log path; default: $RUNNER_TEMP/regression.log

set -euo pipefail

: "${TT_METAL_DIR:?TT_METAL_DIR must be set}"
: "${BUILD_DIR:?BUILD_DIR must be set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_EMULE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export GTEST_XML_DIR="${GTEST_XML_DIR:-${RUNNER_TEMP:-/tmp}/gtest-xml}"
export REGRESSION_LOG="${REGRESSION_LOG:-${RUNNER_TEMP:-/tmp}/regression.log}"

mkdir -p "$GTEST_XML_DIR"
rm -f "$GTEST_XML_DIR"/*.xml

echo "== ci-regression.sh =="
echo "  TT_EMULE_DIR:   $TT_EMULE_DIR"
echo "  TT_METAL_DIR:   $TT_METAL_DIR"
echo "  BUILD_DIR:      $BUILD_DIR"
echo "  GTEST_XML_DIR:  $GTEST_XML_DIR"
echo "  REGRESSION_LOG: $REGRESSION_LOG"
echo ""

cd "$TT_EMULE_DIR"

# Tee the full output to a log artifact while preserving the script's exit code.
# run_regression.sh always exits 0 (per-test PASS/FAIL is tracked in counters);
# real classification happens in classify-results.py against the XML output.
set +e
bash run_regression.sh 2>&1 | tee "$REGRESSION_LOG"
rc=${PIPESTATUS[0]}
set -e

xml_count=$(find "$GTEST_XML_DIR" -name '*.xml' 2>/dev/null | wc -l)
echo ""
echo "== Regression done =="
echo "  exit code:    $rc"
echo "  XML files:    $xml_count in $GTEST_XML_DIR"
echo "  Log:          $REGRESSION_LOG"

# We do NOT propagate run_regression.sh's exit code here. classify-results.py is
# the authority on pass/fail. ci-regression.sh always succeeds if it got this far.
exit 0
