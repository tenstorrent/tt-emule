// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Ternary SFPU shim: odst = dst0 + value*(dst1*dst2). `value` is the uint32
// bit-pattern of an fp32 scalar.
// Real LLK: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h

#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void addcmul_tile_init() {}

template <DataFormat data_format>
ALWI void addcmul_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst, uint32_t value) {
    __emule_dst_check(idst0, "addcmul_tile");
    __emule_dst_check(idst1, "addcmul_tile");
    __emule_dst_check(idst2, "addcmul_tile");
    __emule_dst_check(odst, "addcmul_tile");
    float v;
    std::memcpy(&v, &value, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_dst[odst][i] = __emule_dst[idst0][i] + v * (__emule_dst[idst1][i] * __emule_dst[idst2][i]);
    }
}

}  // namespace ckernel
