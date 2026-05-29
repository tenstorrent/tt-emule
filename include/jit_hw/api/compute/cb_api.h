// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Redirect shim for upstream `api/compute/cb_api.h`. The real upstream header
// at /localdev/arminale/tt-metal/tt_metal/hw/inc/api/compute/cb_api.h uses
// `ALWI` + `llk_wait_tiles` / `llk_pop_tiles` / `llk_wait_for_free_tiles` /
// `llk_push_tiles` directly — primitives that only exist on RISC-V/TRISC
// targets.
//
// emule's CB FIFO operations are already implemented in `api/cb_api.h` (via
// CBSyncState mutex + condition variables). Pull that in plus
// `common.h` for the `ALWI` macro so any code that includes this path resolves
// to emule's host-side CB ops.

#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/cb_api.h"
