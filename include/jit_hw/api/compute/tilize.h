// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/tilize.h`. Owns the tilize init/uninit/block
// surface. Intercepts the include path so the JIT compile doesn't fall
// through to upstream's version (which references unstubbed LLK APIs).
//
// Real LLK reference:
//   tt_metal/hw/inc/api/compute/tilize.h
//   tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_tilize.h
//   tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack_fast_tilize.h
#include <cstdint>

#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/compute/compute_kernel_hw_startup.h"
#include "jit_hw/internal/llk_state.h"

inline void tilize_init(uint32_t, uint32_t, uint32_t) {
    __llk_unpack_is_tilize = true;
    __llk_pack_is_untilize = false;
}
inline void tilize_init_short(uint32_t, uint32_t) {
    __llk_unpack_is_tilize = true;
    __llk_pack_is_untilize = false;
}
inline void tilize_uninit(uint32_t = 0, uint32_t = 0) {
    __llk_unpack_is_tilize = false;
}

// tilize_block: read `ntiles` row-major tiles from `icb` and write them as
// nfaces tiles to `ocb`. emule's `pack_tile` already does the row-major→nfaces
// conversion when `__llk_pack_is_untilize=false` (which is what `tilize_init`
// sets), so the loop is just copy_tile + pack_tile.
inline void tilize_block(uint32_t icb, uint32_t ntiles, uint32_t ocb) {
    for (uint32_t t = 0; t < ntiles; ++t) {
        copy_tile(icb, t, 0);
        pack_tile(0, ocb);
    }
}

// fast_tilize_{init,block,uninit}: hardware-fast variants on the real device,
// equivalent to the regular tilize path in emule.
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
inline void fast_tilize_init(uint32_t icb, uint32_t /*block_dst_tiles*/, uint32_t ocb) {
    tilize_init(icb, full_ct_dim, ocb);
}
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
inline void fast_tilize_block(uint32_t icb, uint32_t ntiles, uint32_t ocb,
                              uint32_t /*block_c_index*/ = 0) {
    tilize_block(icb, ntiles, ocb);
}
inline void fast_tilize_uninit(uint32_t = 0, uint32_t = 0, uint32_t = 0) {
    tilize_uninit();
}
