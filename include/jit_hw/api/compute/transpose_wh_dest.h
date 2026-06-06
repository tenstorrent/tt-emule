// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tt-emule stub: transpose_wh_dest — in-place 32×32 transpose of a DST tile.
// Intercepts api/compute/transpose_wh_dest.h so the JIT compile doesn't pull
// the LLK-only llk_math_transpose_dest_api.h. Sibling of transpose_wh.h (which
// transposes CB→DST); this variant operates entirely in DST.
#pragma once

#include "api/compute/common.h"
#include <cstring>

namespace ckernel {

template <bool is_32bit = false>
ALWI void transpose_wh_dest_init_short() {}

// transpose_wh_dest: in-place 32×32 transpose B[w,h] = A[h,w] on DST[idst].
template <bool is_32bit = false>
ALWI void transpose_wh_dest(uint32_t idst) {
    __emule_dst_check(idst, "transpose_wh_dest");
    float temp[1024];
    for (uint32_t r = 0; r < 32; r++)
        for (uint32_t c = 0; c < 32; c++)
            temp[c * 32 + r] = __emule_dst[idst][r * 32 + c];
    std::memcpy(__emule_dst[idst], temp, sizeof(temp));
    __emule_dst_mark_dirty(idst);
}

}  // namespace ckernel

using namespace ckernel;
