// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for SiLU (x * sigmoid(x)) tile operations.
//
// Provides ckernel::silu_tile_init / silu_tile + the lowered LLK
// silu/sigmoid init+apply templates used by the MatmulFusedAct.

#include <cmath>
#include <cstdint>

namespace ckernel {

template <bool approx = false>
ALWI void silu_tile_init() {}

template <int vector_mode = (int)VectorMode::RC, bool approx = false>
ALWI void silu_tile(uint32_t idst) {
    __emule_dst_check(idst, "silu_tile");
    // Real impl: x * sigmoid(x) per element in __emule_dst[idst].
    float* d = __emule_dst[idst];
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) {
        float x = d[i];
        d[i] = x / (1.0f + std::exp(-x));
    }
}

}  // namespace ckernel
