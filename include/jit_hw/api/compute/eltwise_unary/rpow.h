// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for rpow tile operations.

#include <cmath>
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/compute/vector_mode.h"

namespace ckernel {

ALWI void rpow_tile_init() {}

// rpow(exponent, base) = pow(base, exponent); base_val is an fp32 bit pattern.
ALWI void rpow_tile(uint32_t idst, uint32_t base_val, VectorMode vector_mode = VectorMode::RC) {
    __emule_dst_check(idst, "rpow_tile");
    float base;
    std::memcpy(&base, &base_val, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = std::pow(base, __emule_compute_ctx().dst[idst][i]);
}

} // namespace ckernel
