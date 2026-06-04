// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for elu SFPU tile op.
// Upstream: tt_metal/hw/inc/api/compute/eltwise_unary/elu.h
// LLK family: llk_math_eltwise_unary_sfpu_elu
//
// `param0` is the uint32 bit-pattern of an fp32 slope value. Per-element
// semantics: out = x > 0 ? x : slope * (exp(x) - 1).

#include <cmath>
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void elu_tile_init() {}

ALWI void elu_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "elu_tile");
    float slope;
    std::memcpy(&slope, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = x > 0.0f ? x : slope * (std::exp(x) - 1.0f);
    }
}

}  // namespace ckernel
