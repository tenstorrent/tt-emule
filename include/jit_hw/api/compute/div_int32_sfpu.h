// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void div_int32_tile_init() {}

// rounding_mode=None: silicon computes a FLOAT quotient (float(a)*recip(float(b)))
// and the output CB is fp32 — store a float, not the integer division. b==0 → IEEE
// inf/nan (matches recip(0)). Real LLK: ckernel_sfpu_div_int32.h::calculate_div_int32.
template<DataFormat Fmt = DataFormat::Int32>
ALWI void div_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a = __emule_dst_load_i32(idst0, i);
        int32_t b = __emule_dst_load_i32(idst1, i);
        __emule_compute_ctx().dst[odst][i] = static_cast<float>(a) / static_cast<float>(b);
    }
}

} // namespace ckernel
