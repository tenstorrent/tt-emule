#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# tt-emule end-to-end MODEL regression — Wormhole (single-device N150).
# Thin wrapper: see scripts/run_e2e_models.sh for the runner + entry list.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_EMULE_ARCH=wormhole exec bash "$SCRIPT_DIR/run_e2e_models.sh"
