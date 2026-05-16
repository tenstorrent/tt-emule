#pragma once
// Emulator stub for GELU SFPU tile op.
// Implements the exact GELU formula: 0.5 * x * (1 + erf(x / sqrt(2))).

#include <cmath>
#include <cstdint>

namespace ckernel {

template <bool fast_and_approx = true>
ALWI void gelu_tile_init() {}

template <bool fast_and_approx = true>
ALWI void gelu_tile(uint32_t idst) {
    __emule_dst_check(idst, "gelu_tile");
    static const float kInvSqrt2 = 1.0f / std::sqrt(2.0f);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = 0.5f * x * (1.0f + std::erf(x * kInvSqrt2));
    }
}

} // namespace ckernel
