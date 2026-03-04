// tt-emule stub: transpose_wh — 32×32 tile transpose in emulation
#pragma once

#include "api/compute/common.h"
#include <cstring>

namespace ckernel {

inline void transpose_wh_init(uint32_t icb, uint32_t ocb = 0) {}
inline void transpose_wh_init_short(uint32_t icb = 0) {}

// transpose_wh_tile: read tile from CB[icb][itile] into DST[dst_idx], then
// transpose the 32×32 tile in-place: swap [r][c] ↔ [c][r].
inline void transpose_wh_tile(uint32_t icb, uint32_t itile, uint32_t dst_idx) {
    // Copy tile into DST (handles bf16 → f32 conversion)
    ckernel::copy_tile(icb, itile, dst_idx);

    // Transpose 32×32 in-place
    float temp[1024];
    for (uint32_t r = 0; r < 32; r++)
        for (uint32_t c = 0; c < 32; c++)
            temp[c * 32 + r] = __emule_dst[dst_idx][r * 32 + c];
    std::memcpy(__emule_dst[dst_idx], temp, sizeof(temp));
}

}  // namespace ckernel

using namespace ckernel;
