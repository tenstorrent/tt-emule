#pragma once
// Emulator stub for reciprocal SFPU tile op.

#include <cstdint>

namespace ckernel {

#ifndef __EMULE_VECTOR_MODE_DEFINED
#define __EMULE_VECTOR_MODE_DEFINED
enum class VectorMode : uint8_t { RC = 0, R = 1, C = 2 };
#endif

template <bool legacy_compat = true>
ALWI void recip_tile_init() {}

template <bool legacy_compat = true>
ALWI void recip_tile(uint32_t idst, int /*vector_mode*/ = (int)VectorMode::RC) {
    __emule_dst_check(idst, "recip_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = 1.0f / __emule_dst[idst][i];
}

} // namespace ckernel
