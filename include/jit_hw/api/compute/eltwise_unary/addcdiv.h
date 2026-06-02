// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/addcdiv.h`. Per element:
// odst = idst0 + v*(idst1/idst2); v is the packed-fp32 scalar `value`.

#include <cmath>
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void addcdiv_tile_init() {}

template <DataFormat data_format>
ALWI void addcdiv_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst, uint32_t value) {
    __emule_dst_check(idst0, "addcdiv_tile");
    __emule_dst_check(idst1, "addcdiv_tile");
    __emule_dst_check(idst2, "addcdiv_tile");
    __emule_dst_check(odst, "addcdiv_tile");
    float v;
    std::memcpy(&v, &value, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_dst[odst][i] = __emule_dst[idst0][i] + v * (__emule_dst[idst1][i] / __emule_dst[idst2][i]);
    }
}

}  // namespace ckernel
