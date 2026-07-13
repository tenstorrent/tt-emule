// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// DFB state for JIT-compiled kernels (parallel to emule_cb_state.h).
// Thread-local pointer to per-thread DFB interface array.

#include "tt_emule/dfb_sync_state.hpp"
#include "tt_emule/tile_counter.hpp"

using __emule_dfb_iface = tt_emule::EmuleDFBInterface;
