// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Real impl mirroring tt_metal/hw/inc/api/compute/atan2.h.
// Upstream sfpu reference: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atan2.h
//   _sfpu_atan2_(y, x) - first arg is y, second is x, returns atan2(y, x).
// The kernel-side wrapper (compute/atan2.h) preserves this ordering when
// dispatching from `atan2_binary_tile(idst0, idst1, odst)`.

#include <cstdint>
#include <cmath>
#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void atan2_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    __emule_dst_check(idst0, "atan2_binary_tile.y");
    __emule_dst_check(idst1, "atan2_binary_tile.x");
    __emule_dst_check(odst, "atan2_binary_tile.out");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) {
        __emule_compute_ctx().dst[odst][i] = std::atan2(__emule_compute_ctx().dst[idst0][i], __emule_compute_ctx().dst[idst1][i]);
    }
}
ALWI void atan2_binary_tile_init() {}

} // namespace ckernel
