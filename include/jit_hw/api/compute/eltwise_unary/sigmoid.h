// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for sigmoid tile operations.
//
// TODO (real impl): emule sigmoid_tile should apply sigmoid(x) =
// 1 / (1 + exp(-x)) elementwise to __emule_dst[idst][0..1023], matching
// silicon's SFPU sigmoid output bit-for-bit (within bfloat16 precision).
// Current implementation is a no-op placeholder — kernels that depend on
// real sigmoid output will not produce correct results yet.

#include <cmath>
#include <cstdint>

namespace ckernel {

template <bool approx = false>
ALWI void sigmoid_tile_init() {}

template <int vector_mode = (int)VectorMode::RC, bool approx = false>
ALWI void sigmoid_tile(uint32_t idst) {
    __emule_dst_check(idst, "sigmoid_tile");
    // Real impl: 1 / (1 + exp(-x)) per element in __emule_dst[idst].
    float* d = __emule_dst[idst];
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) {
        d[i] = 1.0f / (1.0f + std::exp(-d[i]));
    }
}

}  // namespace ckernel
