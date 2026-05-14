#!/usr/bin/env bash
# Run the tt-mlir D2M regression suite via run_d2m_regression.sh, with pytest
# JUnit XML output enabled for downstream classification.
#
# Required env:
#   TT_MLIR_DIR    path to tt-mlir workspace (must contain build/, env/)
#   TT_METAL_DIR   path to tt-metal source tree (tt-mlir's third_party copy)
#
# Optional env:
#   BUILD_DIR       tt-metal-emule build dir (defaults to the one tt-mlir built)
#   D2M_XML_DIR     where pytest junit XML lands (default: $RUNNER_TEMP/d2m-xml)
#   D2M_LOG         combined log path     (default: $RUNNER_TEMP/d2m-regression.log)

set -euo pipefail

: "${TT_MLIR_DIR:?TT_MLIR_DIR must be set}"
: "${TT_METAL_DIR:?TT_METAL_DIR must be set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_EMULE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# run_d2m_regression.sh expects BUILD_DIR to be the tt-metal build that contains
# the tt-emule libs (_ttnn.so etc.). When tt-mlir builds tt-metal as a
# subproject, that build tree lives at:
#   $TT_MLIR_DIR/third_party/tt-metal/src/tt-metal/build (symlink to build_$BUILD_TYPE)
export BUILD_DIR="${BUILD_DIR:-$TT_MLIR_DIR/third_party/tt-metal/src/tt-metal/build}"

export D2M_XML_DIR="${D2M_XML_DIR:-${RUNNER_TEMP:-/tmp}/d2m-xml}"
export D2M_LOG="${D2M_LOG:-${RUNNER_TEMP:-/tmp}/d2m-regression.log}"

mkdir -p "$D2M_XML_DIR"
rm -f "$D2M_XML_DIR"/*.xml

echo "== ci-d2m-regression.sh =="
echo "  TT_EMULE_DIR:   $TT_EMULE_DIR"
echo "  TT_MLIR_DIR:    $TT_MLIR_DIR"
echo "  TT_METAL_DIR:   $TT_METAL_DIR"
echo "  BUILD_DIR:      $BUILD_DIR"
echo "  D2M_XML_DIR:    $D2M_XML_DIR"
echo "  D2M_LOG:        $D2M_LOG"
echo ""

# Activate tt-mlir env so pytest + python_packages are on PATH/PYTHONPATH.
cd "$TT_MLIR_DIR"
# shellcheck disable=SC1091
source env/activate
echo "  PYTHONPATH=$PYTHONPATH" | head -c 300; echo
echo ""

# run_d2m_regression.sh exits non-zero on FAIL/HUNG; we suppress that and let
# classify-d2m-results.py be the authority on pass/fail.
cd "$TT_EMULE_DIR"
set +e
bash run_d2m_regression.sh 2>&1 | tee "$D2M_LOG"
rc=${PIPESTATUS[0]}
set -e

xml_count=$(find "$D2M_XML_DIR" -name '*.xml' 2>/dev/null | wc -l)
echo ""
echo "== D2M regression done =="
echo "  exit code:   $rc"
echo "  XML files:   $xml_count in $D2M_XML_DIR"
echo "  Log:         $D2M_LOG"

# Authoritative pass/fail comes from classify-d2m-results.py later in the workflow.
exit 0
