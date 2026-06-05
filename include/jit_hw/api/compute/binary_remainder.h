// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Real impl mirroring tt_metal/hw/inc/api/compute/binary_remainder.h.
// Upstream reference: ckernel_sfpu_binary_remainder.h / ckernel_sfpu_remainder.h.
//
// Semantics match torch.remainder: result = a - b * floor(a / b).
// Sign of the result follows the divisor (Python-style modulo). std::fmod
// is C-style (sign follows dividend) so we compute floor-based explicitly.

#include <cstdint>
#include <cmath>
#include <cstring>
#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void remainder_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    __emule_dst_check(idst0, "remainder_int32_tile.a");
    __emule_dst_check(idst1, "remainder_int32_tile.b");
    __emule_dst_check(odst, "remainder_int32_tile.out");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) {
        int32_t a, b;
        std::memcpy(&a, &__emule_dst[idst0][i], sizeof(int32_t));
        std::memcpy(&b, &__emule_dst[idst1][i], sizeof(int32_t));
        int32_t r;
        if (b == 0) {
            r = 0;  // upstream LLK returns 0 for divide-by-zero
        } else {
            // Python/torch-style integer modulo: sign follows divisor.
            r = a - (a / b) * b;
            if ((r != 0) && ((r < 0) != (b < 0))) r += b;
        }
        std::memcpy(&__emule_dst[odst][i], &r, sizeof(int32_t));
    }
}
ALWI void remainder_int32_tile_init() {}

ALWI void remainder_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    __emule_dst_check(idst0, "remainder_binary_tile.a");
    __emule_dst_check(idst1, "remainder_binary_tile.b");
    __emule_dst_check(odst, "remainder_binary_tile.out");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) {
        const float a = __emule_dst[idst0][i];
        const float b = __emule_dst[idst1][i];
        // a - b * floor(a / b) — torch.remainder semantics (sign follows the
        // divisor); std::fmod would mismatch for a negative dividend.
        // By zero: silicon yields a NaN whose upper mantissa bits are zero, so it
        // packs to -inf in bf16 yet stays NaN in fp32. A plain std::nanf has the
        // mantissa MSB set and would pack to NaN, not inf.
        if (b == 0.0f) {
            const uint32_t nan_bits = 0xFF800001u;
            std::memcpy(&__emule_dst[odst][i], &nan_bits, sizeof(float));
        } else {
            __emule_dst[odst][i] = a - b * std::floor(a / b);
        }
    }
}
ALWI void remainder_binary_tile_init() {}

} // namespace ckernel
