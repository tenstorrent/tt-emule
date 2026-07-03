// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tt-emule stub: transpose_wh_dest operates on an already-acquired DST tile.
#pragma once

#include "api/compute/transpose.h"

namespace ckernel {

template <bool is_32bit = false, bool transpose_of_faces = true>
inline void transpose_wh_dest_init_short() {}

template <bool is_32bit = false, bool transpose_of_faces = true>
inline void transpose_wh_dest(uint32_t idst) {
    float temp[1024];
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t c = 0; c < 32; c++) {
            temp[c * 32 + r] = __emule_dst[idst][r * 32 + c];
        }
    }
    std::memcpy(__emule_dst[idst], temp, sizeof(temp));
}

}  // namespace ckernel

using namespace ckernel;
