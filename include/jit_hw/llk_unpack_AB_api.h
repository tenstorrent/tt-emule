// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Emule shim for `llk_unpack_AB_api.h`. Upstream is a thin aggregator
// pulling in `llk_unpack_AB.h` and `llk_unpack_common_api.h`. Emule already
// shims those; surface them under this name as well for kernels that
// `#include "llk_unpack_AB_api.h"` directly.
//
// The canonical `llk_unpack_AB_init` / `llk_unpack_AB` (with the correct non-type
// `ckernel::BroadcastType` template param) live in `llk_unpack_a.h` (included below).
// Don't redeclare them here — a second overload set with a different template-param
// kind makes the call ambiguous once the kernel_lib instantiates it. `llk_unpack_AB`
// stashes its tile indices there (see `__emule_unpack_AB_state()`) for the binary
// math step (llk_math_eltwise_binary) to consume, since emule has no SRCA/SRCB banks.

#include <cstdint>
#include "llk_unpack_a.h"
#include "llk_unpack_common_api.h"
