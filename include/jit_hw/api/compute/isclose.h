// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Real impl mirroring tt_metal/hw/inc/api/compute/isclose.h.
// Upstream reference: ckernel_sfpu_isclose.h.
//
//   result = |a - b| <= atol + rtol * |b|
//
// Special handling matches torch.isclose:
//   - matching-sign Inf:  result = 1
//   - mismatched Inf:     result = 0
//   - any NaN, EQUAL_NAN=false: result = 0
//   - both NaN, EQUAL_NAN=true:  result = 1
//   - one NaN, EQUAL_NAN=true:   result = 0
//
// rtol_bits / atol_bits are IEEE-754 bit patterns of fp32 tolerances passed
// as runtime args.

#include <cstdint>
#include <cmath>
#include <cstring>
#include "jit_hw/api/compute/common.h"

namespace ckernel {

template <bool EQUAL_NAN = false>
ALWI void isclose_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst,
                              uint32_t rtol_bits, uint32_t atol_bits) {
    __emule_dst_check(idst0, "isclose_binary_tile.a");
    __emule_dst_check(idst1, "isclose_binary_tile.b");
    __emule_dst_check(odst, "isclose_binary_tile.out");
    float atol, rtol;
    std::memcpy(&atol, &atol_bits, sizeof(float));
    std::memcpy(&rtol, &rtol_bits, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) {
        const float a = __emule_dst[idst0][i];
        const float b = __emule_dst[idst1][i];
        const bool a_nan = std::isnan(a);
        const bool b_nan = std::isnan(b);
        float r;
        if (a_nan || b_nan) {
            r = (EQUAL_NAN && a_nan && b_nan) ? 1.0f : 0.0f;
        } else if (std::isinf(a) || std::isinf(b)) {
            r = (a == b) ? 1.0f : 0.0f;  // matches sign and magnitude
        } else {
            r = (std::fabs(a - b) <= atol + rtol * std::fabs(b)) ? 1.0f : 0.0f;
        }
        __emule_dst[odst][i] = r;
    }
}
ALWI void isclose_binary_tile_init() {}

} // namespace ckernel
