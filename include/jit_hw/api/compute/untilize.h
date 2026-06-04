// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/untilize.h`. Owns `untilize_init`,
// `untilize_init_short`, and `untilize_block`. Intercepts the include path
// so the JIT compile doesn't fall through to upstream's version (which
// references unstubbed LLK APIs).
//
// `compute_kernel_hw_startup` is brought in via the same chain because
// upstream's `api/compute/untilize.h` pulls in common.h (and downstream
// kernel-lib headers like ttnn/cpp/ttnn/kernel_lib/untilize_helpers.{hpp,inl}
// rely on `compute_kernel_hw_startup` being declared by the time they parse).
//
// Real LLK reference:
//   tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack_untilize.h
#include <cstdint>

#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/compute/compute_kernel_hw_startup.h"
#include "jit_hw/internal/llk_state.h"

inline void untilize_init(uint32_t, uint32_t = 0) {
    __llk_unpack_is_tilize = false;
    __llk_pack_is_untilize = true;
}
inline void untilize_init_short(uint32_t) {
    __llk_unpack_is_tilize = false;
    __llk_pack_is_untilize = true;
}

// untilize_block: write `block_tile_count` tiles from `icb` to `ocb` in
// row-major layout. `copy_tile(icb, t, dst_slot)` loads nfaces→DST (DST
// is row-major float32 in emule); `__llk_pack_untilize` writes one DST
// slot to `ocb` at the position dictated by `__llk_pack_offset` /
// `__llk_pack_block_c`. cb_reserve_back resets `__llk_pack_offset` to 0
// for the output, so we just walk it forward.
//
// Real LLK reference:
//   tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack_untilize.h
template <uint32_t block_tile_count>
inline void untilize_block(uint32_t icb, uint32_t /*ntiles*/, uint32_t ocb) {
    __llk_pack_block_c = block_tile_count;
    __llk_pack_offset = 0;
    for (uint32_t t = 0; t < block_tile_count; ++t) {
        copy_tile(icb, t, 0);
        __llk_pack_untilize(0, ocb);
        __llk_pack_offset++;
    }
}
inline void untilize_block(uint32_t icb, uint32_t ntiles, uint32_t ocb) {
    __llk_pack_block_c = ntiles;
    __llk_pack_offset = 0;
    for (uint32_t t = 0; t < ntiles; ++t) {
        copy_tile(icb, t, 0);
        __llk_pack_untilize(0, ocb);
        __llk_pack_offset++;
    }
}
