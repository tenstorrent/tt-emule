// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Emule shim for `llk_unpack_AB_api.h`. Upstream is a thin aggregator
// pulling in `llk_unpack_AB.h` and `llk_unpack_common_api.h`. Emule already
// shims those; surface them under this name as well for kernels that
// `#include "llk_unpack_AB_api.h"` directly.
//
// The init/runtime entry points (`llk_unpack_AB_init`, `llk_unpack_AB`) are
// no-ops under emule: the unpacker state is implicit (kernels read CB
// data directly via the bridge), so init has no state to configure and
// the per-tile unpack is a no-op (the math/sfpu path handles its own
// CB reads).

#include <cstdint>
#include "llk_unpack_a.h"
#include "llk_unpack_common_api.h"

// Generic init/runtime templates. Drop the function bodies — emule's math
// path doesn't depend on unpacker state.
template <typename BroadcastType = void>
inline void llk_unpack_AB_init(
    const std::uint32_t /*operandA*/, const std::uint32_t /*operandB*/,
    const std::uint32_t /*transpose*/ = 0) {}

template <typename BroadcastType = void, bool acc_to_dest = false>
inline void llk_unpack_AB(
    const std::uint32_t /*operandA*/, const std::uint32_t /*operandB*/,
    const std::uint32_t /*tile_index_a*/, const std::uint32_t /*tile_index_b*/) {}
