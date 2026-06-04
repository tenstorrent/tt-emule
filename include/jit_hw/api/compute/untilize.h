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
// untilize_uninit reverts the packer to its default (tilize-pack) mode.
// In emule the only state to flip is `__llk_pack_is_untilize`.  Real
// callers: ssm_prefix_scan, rotary_embedding, and matmul ops that mix
// tilize/untilize within a single compute kernel.
inline void untilize_uninit(uint32_t = 0) {
    __llk_pack_is_untilize = false;
}

// untilize_block: write `ntiles` tiles from `icb` to `ocb` in row-major
// layout.  In silicon, the template arg `block_tile_count` is the DST
// sub-batch size (data-copy width); the runtime `ntiles` is the total
// tile count for this call.  emule has no DST sub-batching to model, so
// both overloads walk the same row-major scatter on the runtime arg.
//
// Real LLK reference:
//   tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack_untilize.h
inline void untilize_block(uint32_t icb, uint32_t ntiles, uint32_t ocb) {
    __llk_pack_block_c = ntiles;
    __llk_pack_offset = 0;
    for (uint32_t t = 0; t < ntiles; ++t) {
        copy_tile(icb, t, 0);
        __llk_pack_untilize(0, ocb);
        __llk_pack_offset++;
    }
}
template <uint32_t block_tile_count>
inline void untilize_block(uint32_t icb, uint32_t ntiles, uint32_t ocb) {
    untilize_block(icb, ntiles, ocb);
}
