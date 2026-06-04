// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for pack_untilize. Provides emulator implementations of
// `pack_untilize_block` (upstream signature) + `experimental::pack_untilize_block`
// + a no-op surface for the upstream pack_untilize init/dst helpers. The macro
// define preempts verbatim injection of the experimental_pack_untilize_llks
// header chain.
//
// Real LLK reference:
//   tt_metal/hw/inc/api/compute/pack_untilize.h (upstream `pack_untilize_block`)
//   tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack_untilize.h (PACK LLK)
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PACK_UNTILIZE_LLKS_H

#include "jit_hw/internal/llk_state.h"
#include "jit_hw/llk_pack.h"
#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/compute/compute_kernel_hw_startup.h"

namespace ckernel {

inline void pack_untilize_init(uint32_t cb_out = 0) {}
inline void pack_untilize_init_short(uint32_t cb_out = 0) {}
inline void pack_untilize_uninit(uint32_t cb_out = 0) {}

// D2M emits pack_untilize_init<cols_per_dst_pass, total_col_tiles>(icb, ocb)
template <uint32_t cols_per_dst_pass, uint32_t total_col_tiles>
inline void pack_untilize_init(uint32_t /*icb*/, uint32_t /*ocb*/) {}

// Skip-BH-DEST-remap variant — emule has no DEST remap state, so this is the
// same no-op as `pack_untilize_init` above. Required for the kernel-lib's
// `if constexpr (configure_remap)` else-branch to parse.
template <uint32_t cols_per_dst_pass, uint32_t total_col_tiles>
inline void pack_untilize_init_skip_remap(uint32_t /*icb*/, uint32_t /*ocb*/) {}
inline void pack_untilize_dst_init_short(uint32_t cb_out, uint32_t ct_dim = 0, uint32_t face_r_dim = 0) {}
inline void pack_untilize_dst(uint32_t cb_out, uint32_t out_subblock_h, uint32_t out_subblock_w,
                              uint32_t block_ct_dim = 0, uint32_t pack_dst_offset = 0) {}

// Upstream signature (tt_metal/hw/inc/api/compute/pack_untilize.h:150).
// `block_ct_dim` is the column tile-width of one PACK pass (also the DST
// slot count); `full_ct_dim` is the full row width in tiles. `block_rt_dim`
// is how many tile rows to process. `block_c_index` selects which
// `block_ct_dim`-wide column-block within the full row to write to.
//
// In emule: DST is row-major float32, so we copy_tile(icb, r*block_ct_dim+c, c)
// to load block_ct_dim tiles into DST slots, then call __llk_pack_untilize(c, ocb)
// per slot. `__llk_pack_block_c` is set to `full_ct_dim` so __llk_pack_untilize's
// row/col math (offset / block_c, offset % block_c) targets the correct tile
// position in the output CB. cb_reserve_back leaves the output CB un-touched,
// so we re-seed __llk_pack_offset per call.
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
inline void pack_untilize_block(uint32_t icb, uint32_t block_rt_dim, uint32_t ocb, uint32_t block_c_index = 0) {
    __llk_pack_block_c = full_ct_dim;
    for (uint32_t r = 0; r < block_rt_dim; ++r) {
        for (uint32_t c = 0; c < block_ct_dim; ++c) {
            copy_tile(icb, r * block_ct_dim + c, c);
        }
        __llk_pack_offset = r * full_ct_dim + block_c_index * block_ct_dim;
        for (uint32_t c = 0; c < block_ct_dim; ++c) {
            __llk_pack_untilize(c, ocb);
            __llk_pack_offset++;
        }
    }
}

// Template overloads used by bmm_large_block_zm_fused_bias_activation.cpp
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim,
          bool narrow_row = false, uint32_t row_num_datums = 32, bool dense = false>
inline void pack_untilize_dest_init(uint32_t ocb = 0, uint32_t face_r_dim = 16,
                                     uint32_t num_faces = 4, uint32_t call_line = 0) {}

// pack_untilize_dest: emule of silicon's pack_untilize_dest LLK.
//
// Silicon's hardware packer reads (block_rt_dim × block_ct_dim) DST tiles
// and writes them to cb_out untilized in ROW-MAJOR layout: the output
// region is (block_rt_dim * TILE_DIM) rows × (full_ct_dim * row_cols) cols
// laid out contiguously, where row_cols = TILE_DIM normally or
// row_num_datums when narrow_row is set.  This call writes the
// `block_ct_dim`-wide column-block at column-block index `block_c_index`
// of the full row.
//
// The cb's page_size is sync granularity only — it does NOT change the
// physical L1 layout.  Consumers (writers, downstream CBs) expect
// contiguous row-major sticks so that one logical output row sits at
// offset row * full_row_bytes regardless of page_size.
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim,
          bool diagonal = false, bool narrow_row = false,
          uint32_t row_num_datums = 32, uint32_t tile_dst_ct_offset = 0, bool dense = false>
inline void pack_untilize_dest(uint32_t ocb = 0, uint32_t block_rt_dim = 1,
                                uint32_t block_c_index = 0, uint32_t /*face_r_dim*/ = 16,
                                uint32_t /*num_faces*/ = 4, uint32_t /*tile_dst_rt_offset*/ = 0) {
    static_assert(!diagonal, "pack_untilize_dest: diagonal packer mode not modelled");
    static_assert(!dense,    "pack_untilize_dest: dense packer mode not modelled");

    constexpr uint32_t TILE_DIM = 32;
    constexpr uint32_t row_cols = narrow_row ? row_num_datums : TILE_DIM;

    const bool     is_32bit       = __emule_compute::cb_is_32bit_format(ocb);
    const uint32_t elem_size      = is_32bit ? 4 : 2;
    const uint32_t full_row_bytes = full_ct_dim * row_cols * elem_size;
    uint8_t* const base           = __emule_compute::cb_write_ptr_at(ocb, 0);

    auto target = [&](uint32_t rt, uint32_t ct, uint32_t r, uint32_t c) -> uint8_t* {
        const uint32_t out_row = rt * TILE_DIM + r;
        const uint32_t out_col = (block_c_index * block_ct_dim + ct) * row_cols + c;
        return base + out_row * full_row_bytes + out_col * elem_size;
    };

    for (uint32_t rt = 0; rt < block_rt_dim; ++rt) {
        for (uint32_t ct = 0; ct < block_ct_dim; ++ct) {
            const uint32_t dst_idx = tile_dst_ct_offset + rt * block_ct_dim + ct;
            for (uint32_t r = 0; r < TILE_DIM; ++r) {
                for (uint32_t c = 0; c < row_cols; ++c) {
                    const float src = __emule_dst[dst_idx][r * TILE_DIM + c];
                    uint8_t* dst_ptr = target(rt, ct, r, c);
                    if (is_32bit) {
                        std::memcpy(dst_ptr, &src, sizeof(uint32_t));
                    } else {
                        uint16_t bf = __emule_bf16::from_f32(src);
                        std::memcpy(dst_ptr, &bf, sizeof(uint16_t));
                    }
                }
            }
        }
    }
}

}  // namespace ckernel

using namespace ckernel;

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
