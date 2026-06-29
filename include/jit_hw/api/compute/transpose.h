// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tt-emule stub: transpose — 32×32 tile transpose in emulation.
#pragma once

#include "api/compute/common.h"
#include <cstring>

namespace ckernel {

inline void transpose_init(uint32_t icb, uint32_t call_line = __builtin_LINE()) {}

// B[w,h] = A[h,w]: copy CB tile into DST, then transpose the 32×32 in place.
inline void transpose_tile(uint32_t icb, uint32_t itile, uint32_t idst) {
    ckernel::copy_tile(icb, itile, idst);

    float temp[__EMULE_TILE_ELEMS];
    for (uint32_t r = 0; r < 32; r++)
        for (uint32_t c = 0; c < 32; c++)
            temp[c * 32 + r] = __emule_dst[idst][r * 32 + c];
    std::memcpy(__emule_dst[idst], temp, sizeof(temp));
}

}  // namespace ckernel

using namespace ckernel;
