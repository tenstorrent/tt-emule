// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Redirect shim for upstream `internal/circular_buffer_interface.h`. The real
// header defines `LocalCBInterface` and a `get_local_cb_interface` whose return
// type clashes with emule's `CbInterface` (declared in `jit_hw/internal/
// llk_state.h`). Pulling in emule's version below short-circuits the include
// so the upstream header never reaches the parser.
#include "jit_hw/internal/llk_state.h"
