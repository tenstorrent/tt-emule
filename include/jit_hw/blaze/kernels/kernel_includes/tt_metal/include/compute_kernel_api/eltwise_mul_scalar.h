// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Emule shadow for blaze's `eltwise_mul_scalar` custom compute ops
// (deepseek_mul_tiles_bcast_scalar + deepseek_binary_dest_reuse_tiles, used by
// the RMSNorm scale-apply path). blaze's header routes them through
// `state_configure` + the raw `llk_math_eltwise_binary` / `_init_with_operands`
// LLK that emule can't consume; model the ops directly on DST/CBs instead.
// jit_hw is -I-first so this shadows blaze's header; blaze's tree stays pristine.

#include "jit_hw/api/compute/common.h"

#include <cstdint>

namespace ckernel {

ALWI void deepseek_mul_tiles_bcast_scalar_init_short(
    uint32_t /*icb0*/, uint32_t /*icb1*/, uint32_t /*call_line*/ = __builtin_LINE()) {}

// dst[idst] = cb0[itile0] * (scalar broadcast from cb1[itile1] element [0,0]).
template <bool fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void deepseek_mul_tiles_bcast_scalar(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_dst_check(idst, "deepseek_mul_tiles_bcast_scalar");
    float a[__EMULE_TILE_ELEMS];
    float b[__EMULE_TILE_ELEMS];
    __emule_unpack_cb_tile_to(icb0, itile0, a);
    __emule_unpack_cb_tile_to(icb1, itile1, b);
    const float scalar = b[0];  // face position [0,0]
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) __emule_dst[idst][i] = a[i] * scalar;
}

template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
ALWI void deepseek_binary_dest_reuse_tiles_init(uint32_t /*icb0*/, uint32_t /*call_line*/ = __builtin_LINE()) {}

// dst[idst] = dst[idst] * cb[in_tile_index]  (elementwise, dest-reuse).
template <bool fp32_dest_acc_en = DST_ACCUM_MODE,
          EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
ALWI void deepseek_binary_dest_reuse_tiles(uint32_t icb, uint32_t in_tile_index, uint32_t idst) {
    __emule_dst_check(idst, "deepseek_binary_dest_reuse_tiles");
    float b[__EMULE_TILE_ELEMS];
    __emule_unpack_cb_tile_to(icb, in_tile_index, b);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) __emule_dst[idst][i] *= b[i];
}

}  // namespace ckernel
