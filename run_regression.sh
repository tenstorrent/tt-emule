#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Top-level regression dispatcher — delegates to per-architecture scripts in
# scripts/. Local-dev entry point; CI calls the per-arch scripts directly.
#
# TT_EMULE_ARCH:
#   wormhole  — scripts/run_regression_wormhole.sh
#   blackhole — scripts/run_regression_blackhole.sh
#   quasar    — scripts/run_regression_quasar.sh
#   all       — all three scripts, sequentially (default)

TT_EMULE_ARCH="${TT_EMULE_ARCH:-all}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "$TT_EMULE_ARCH" in
    wormhole|blackhole|quasar)
        exec bash "$SCRIPT_DIR/scripts/run_regression_${TT_EMULE_ARCH}.sh"
        ;;
    all)
        bash "$SCRIPT_DIR/scripts/run_regression_wormhole.sh"
        bash "$SCRIPT_DIR/scripts/run_regression_blackhole.sh"
        bash "$SCRIPT_DIR/scripts/run_regression_quasar.sh"
        ;;
    *)
        echo "ERROR: TT_EMULE_ARCH must be all|wormhole|blackhole|quasar, got '$TT_EMULE_ARCH'" >&2
        exit 1
        ;;
esac
