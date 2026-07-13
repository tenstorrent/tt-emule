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

// Signature mirrors upstream api/compute/eltwise_unary/recip.h:
//   recip_tile<legacy_compat>(idst, VectorMode vector_mode = VectorMode::RC).
// vector_mode gates which faces are written (default RC = whole tile), honored via the
// shared face mask. Blackhole SDPA normalize calls recip_tile<false>(0, VectorMode::C)
// directly (the BH analogue of WH's recip_tile_first_column_wh_idst0_direct) to reciprocate
// the per-row softmax denominator, which lives in col 0; VectorMode::C → cols<16 of all 32
// rows (faces 0,2), so col 0 of every row is reciprocated. Unlike the WH SFPU-lane-walk
// path, emule's recip is a plain row-major loop, so there is no rows-16-31 coverage gap.
template <bool legacy_compat = true>
ALWI void recip_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC) {
    __emule_dst_check(idst, "recip_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        if (!__emule_vector_mode_active(i, vector_mode)) continue;
        __emule_compute_ctx().dst[idst][i] = 1.0f / __emule_compute_ctx().dst[idst][i];
    }
}

} // namespace ckernel
