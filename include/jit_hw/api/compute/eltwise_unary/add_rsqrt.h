// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"
// Emulator model for blaze's fused add+rsqrt SFPU tile op:
//   result = rsqrt(x + addend)
// Used by RMSNorm-family ops for `rsqrt(variance + epsilon)`. The `addend`
// argument is the bit-representation (as uint32) of the float to add before
// the reciprocal sqrt, matching the silicon LLK's scalar-immediate ABI.

#include <cmath>
#include <cstdint>
#include <cstring>

namespace ckernel {

// batched_qk_norm calls ckernel::sfpu::init_add_rsqrt<APPROX>() before the tile
// op; the LLK init only programs SFPU config regs, so it is a no-op here.
namespace sfpu {
template <bool APPROX = false>
inline void init_add_rsqrt() {}
}  // namespace sfpu

ALWI void add_rsqrt_tile_init() {}

// vec_mode is `auto` so the op accepts both the scoped `VectorMode::RC` enum
// (the RMSNorm-family call sites) and any plain-int vec-mode constant.
template <bool fast_and_approx = false, auto vec_mode = VectorMode::RC, int ITERATIONS = 8>
ALWI void add_rsqrt_tile(uint32_t idst, uint32_t addend) {
    __emule_dst_check(idst, "add_rsqrt_tile");
    float add_f;
    std::memcpy(&add_f, &addend, sizeof(add_f));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_dst[idst][i] = 1.0f / std::sqrt(__emule_dst[idst][i] + add_f);
    }
}

}  // namespace ckernel
