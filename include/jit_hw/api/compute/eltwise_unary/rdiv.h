// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"
// Emulator stub for reciprocal-divide SFPU tile op: result = value / x,
// then trunc/floor per the RoundingMode template arg.

#include <cmath>
#include <cstdint>
#include <cstring>

// VectorMode lives in ckernel::; see api/compute/vector_mode.h.
#include "jit_hw/api/compute/vector_mode.h"

namespace ckernel {

// RoundingMode mirrors the canonical silicon enum (ckernel_defs.h);
// rdiv_tile honors Trunc/Floor below.
#ifndef __EMULE_ROUNDING_MODE_DEFINED
#define __EMULE_ROUNDING_MODE_DEFINED
enum class RoundingMode : uint8_t {
    None  = 0,
    Trunc = 1,
    Floor = 2,
};
#endif

ALWI void rdiv_tile_init() {}

template <RoundingMode rounding_mode = RoundingMode::None>
ALWI void rdiv_tile(uint32_t dst_index, uint32_t value, VectorMode vector_mode = VectorMode::RC) {
    __emule_dst_check(dst_index, "rdiv_tile");
    float num;
    std::memcpy(&num, &value, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float r = num / __emule_compute_ctx().dst[dst_index][i];
        if (rounding_mode == RoundingMode::Trunc) {
            r = std::trunc(r);
        } else if (rounding_mode == RoundingMode::Floor) {
            r = std::floor(r);
        }
        __emule_compute_ctx().dst[dst_index][i] = r;
    }
}

} // namespace ckernel
