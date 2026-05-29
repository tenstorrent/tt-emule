// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for pack_untilize. Provides emulator implementations of
// experimental::pack_untilize_block + a no-op surface for the upstream
// pack_untilize init/dst helpers. The macro define preempts verbatim
// injection of the experimental_pack_untilize_llks header chain.
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PACK_UNTILIZE_LLKS_H

#include "jit_hw/internal/llk_state.h"
#include "jit_hw/llk_pack.h"
#include "jit_hw/api/compute/common.h"

namespace ckernel {

inline void pack_untilize_init(uint32_t cb_out = 0) {}
inline void pack_untilize_init_short(uint32_t cb_out = 0) {}
inline void pack_untilize_uninit(uint32_t cb_out = 0) {}

// D2M emits pack_untilize_init<cols_per_dst_pass, total_col_tiles>(icb, ocb)
template <uint32_t cols_per_dst_pass, uint32_t total_col_tiles>
inline void pack_untilize_init(uint32_t /*icb*/, uint32_t /*ocb*/) {}
inline void pack_untilize_dst_init_short(uint32_t cb_out, uint32_t ct_dim = 0, uint32_t face_r_dim = 0) {}
inline void pack_untilize_dst(uint32_t cb_out, uint32_t out_subblock_h, uint32_t out_subblock_w,
                              uint32_t block_ct_dim = 0, uint32_t pack_dst_offset = 0) {}

// Template overloads used by bmm_large_block_zm_fused_bias_activation.cpp
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim,
          bool narrow_row = false, uint32_t row_num_datums = 32, bool dense = false>
inline void pack_untilize_dest_init(uint32_t ocb = 0, uint32_t face_r_dim = 16,
                                     uint32_t num_faces = 4, uint32_t call_line = 0) {}

template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim,
          bool diagonal = false, bool narrow_row = false,
          uint32_t row_num_datums = 32, uint32_t tile_dst_ct_offset = 0, bool dense = false>
inline void pack_untilize_dest(uint32_t ocb = 0, uint32_t block_rt_dim = 1,
                                uint32_t block_c_index = 0, uint32_t face_r_dim = 16,
                                uint32_t num_faces = 4, uint32_t tile_dst_rt_offset = 0) {}

}  // namespace ckernel

using namespace ckernel;

// ---- pack_untilize_block (global scope, templated) ----
// Matches tt_metal/hw/inc/api/compute/pack_untilize.h:150. Untilizes a
// block_rt_dim × block_ct_dim block of tiles from icb → ocb at offset
// block_c_index within the reserved CB region.
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim,
          bool diagonal = false, bool narrow_row = false,
          uint32_t row_num_datums = 32>
inline void pack_untilize_block(uint32_t icb, uint32_t block_rt_dim, uint32_t ocb,
                                uint32_t block_c_index = 0) {
    // Reuse the experimental variant's row-major scatter (DST → CB) — same
    // semantics: read block_rt_dim × block_ct_dim tiles from icb, untilize
    // into ocb at the column offset.
    __llk_pack_block_c = full_ct_dim;
    __llk_pack_offset = block_c_index * block_ct_dim;
    for (uint32_t r = 0; r < block_rt_dim; ++r) {
        for (uint32_t c = 0; c < block_ct_dim; ++c) {
            copy_tile(icb, r * block_ct_dim + c, c);
        }
        for (uint32_t c = 0; c < block_ct_dim; ++c) {
            __llk_pack_untilize(c, ocb);
            __llk_pack_offset++;
        }
    }
}

// ---- experimental::pack_untilize_block ----
// Implements the DST → row-major CB scatter that D2M-generated untilize
// kernels expect. Uses copy_tile (CB→DST) + __llk_pack_untilize (DST→CB);
// __llk_pack_block_c is the row stride in tiles, __llk_pack_offset is the
// linear tile position used to compute scatter coordinates.
namespace experimental {

template <uint32_t cols_per_dst_pass, uint32_t total_col_tiles>
inline void pack_untilize_block(uint32_t icb, uint32_t ocb,
                                uint32_t block_row_tiles,
                                uint32_t block_col_tiles) {
    __llk_pack_block_c = total_col_tiles;
    __llk_pack_offset = 0;

    const uint32_t num_col_blocks = block_col_tiles / cols_per_dst_pass;
    for (uint32_t r = 0; r < block_row_tiles; ++r) {
        for (uint32_t b = 0; b < num_col_blocks; ++b) {
            for (uint32_t c = 0; c < cols_per_dst_pass; ++c) {
                uint32_t src_tile = r * block_col_tiles + b * cols_per_dst_pass + c;
                copy_tile(icb, src_tile, c);
            }

            for (uint32_t c = 0; c < cols_per_dst_pass; ++c) {
                __llk_pack_untilize(c, ocb);
                __llk_pack_offset++;
            }
        }
    }
}

} // namespace experimental
