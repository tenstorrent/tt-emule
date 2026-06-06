// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"
// Emulator stub for reciprocal SFPU tile op.

#include <cstdint>

// VectorMode lives in ckernel::; see api/compute/vector_mode.h.
#include "jit_hw/api/compute/vector_mode.h"

namespace ckernel {

template <bool legacy_compat = true>
ALWI void recip_tile_init() {}

template <bool legacy_compat = true>
ALWI void recip_tile(uint32_t idst, int /*vector_mode*/ = (int)VectorMode::RC) {
    __emule_dst_check(idst, "recip_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = 1.0f / __emule_dst[idst][i];
}

} // namespace ckernel
