// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `noc/noc_parameters.h`. The upstream header is shipped per
// architecture under tt_metal/hw/inc/internal/tt-Nxx/<arch>/noc/. Some
// ttnn kernel-lib helpers (moreh_common.hpp, others) include it via the
// generic relative path. We re-export the wormhole copy by default since
// emule's primary target is N150.
//
// Defines NOC_X_SIZE / NOC_Y_SIZE / NOC_CMD_* / NOC_ADDR_* macros that
// kernels reference when building NOC addresses or issuing async ops.
// These are compile-time constants with no silicon dependency in emule.

#include "tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h"
