// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tt-emule stub: pack_untilize
#pragma once

// Preempt tt-mlir verbatim injection of experimental_pack_untilize_llks.
// We provide our own implementation using emulator primitives (copy_tile + __llk_pack_untilize).
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PACK_UNTILIZE_LLKS_H

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

// experimental::pack_untilize_block is implemented in llk_defs.h (which has full
// context for copy_tile, __llk_pack_untilize, __llk_pack_offset, etc.).
