// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tt-emule stub: transpose — 32×32 tile transpose in emulation.
// Shadows upstream tt_metal/hw/inc/api/compute/transpose.h (transpose_init /
// transpose_tile) to keep the LLK math/unpack path off the JIT compile.
#pragma once

#include "api/compute/common.h"
#include <cstring>

namespace ckernel {

inline void transpose_init(uint32_t icb) {}

// transpose_tile: read tile from CB[icb][itile] into DST[idst], then transpose
// the 32×32 tile in-place: swap [r][c] ↔ [c][r].
inline void transpose_tile(uint32_t icb, uint32_t itile, uint32_t idst) {
    // Copy tile into DST (handles bf16 → f32 conversion)
    ckernel::copy_tile(icb, itile, idst);

    // Transpose 32×32 in-place
    float temp[1024];
    for (uint32_t r = 0; r < 32; r++)
        for (uint32_t c = 0; c < 32; c++)
            temp[c * 32 + r] = __emule_compute_ctx().dst[idst][r * 32 + c];
    std::memcpy(__emule_compute_ctx().dst[idst], temp, sizeof(temp));
}

}  // namespace ckernel

using namespace ckernel;
