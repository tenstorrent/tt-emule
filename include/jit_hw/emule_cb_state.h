// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Circular buffer state for JIT-compiled kernels.
// __emule_cb_state is an alias for tt_emule::CBSyncState — the single source
// of truth for CB FIFO state used by JIT-compiled kernels.

#include "tt_emule/cb_sync_state.hpp"

using __emule_cb_state = tt_emule::CBSyncState;
