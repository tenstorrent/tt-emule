#pragma once
#include "jit_hw/api/compute/common.h"
// Emulator stub for log1p SFPU tile op.

#include <cmath>
#include <cstdint>

namespace ckernel {

template <bool fast_and_approx = false>
ALWI void log1p_tile_init() {}

template <bool fast_and_approx = false>
ALWI void log1p_tile(uint32_t idst) {
    __emule_dst_check(idst, "log1p_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = std::log1p(__emule_compute_ctx().dst[idst][i]);
}

} // namespace ckernel
