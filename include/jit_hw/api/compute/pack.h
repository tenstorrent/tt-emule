// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Shadows upstream tt_metal/hw/inc/api/compute/pack.h to avoid a pack_tile
// redefinition; emule's pack_tile lives in jit_hw/api/compute/common.h.
//
// The shadow only works because the JIT command line lists `-I .../jit_hw`
// before any tt-metal include path. If that ordering ever changes, the
// upstream pack.h would resolve instead, ODR-conflicting with common.h's
// pack_tile.  Upstream does not expose a sentinel macro we can probe, so this
// invariant is documented rather than asserted.
//
// Also provides the packer reduce-mask surface the upstream ttnn kernel_lib
// (reduce_helpers_compute.inl) calls directly. emule's reduce_tile already
// produces correctly-masked reduce output in its DST model, so these are no-ops.

#include <cstdint>
#include "jit_hw/llk_types.h"  // ReduceDim, PackMode

namespace ckernel {

// PackMode is the single ckernel::PackMode defined in llk_types.h (included above);
// do not redefine it here — a second definition is an ODR redefinition for any TU
// that includes both (e.g. softmax / layernorm_sharded compute kernels).

template <ReduceDim reduce_dim, PackMode pack_mode = PackMode::Default>
inline void llk_pack_reduce_mask_config(uint32_t /*ocb*/) {}
inline void llk_pack_reduce_mask_clear() {}

}  // namespace ckernel
