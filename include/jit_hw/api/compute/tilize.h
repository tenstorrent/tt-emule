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
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/compute/compute_kernel_hw_startup.h"
#include "jit_hw/internal/llk_state.h"
#include "jit_hw/api/compute/nfaces.h"
#include "jit_hw/api/bfp8.h"
#include "jit_hw/api/bfloat16.h"

inline void tilize_init(uint32_t, uint32_t, uint32_t) {
    __emule_compute_ctx().llk_unpack_is_tilize = true;
    __emule_compute_ctx().llk_pack_is_untilize = false;
}
inline void tilize_init_short(uint32_t, uint32_t) {
    __emule_compute_ctx().llk_unpack_is_tilize = true;
    __emule_compute_ctx().llk_pack_is_untilize = false;
}
inline void tilize_init_short_with_dt(uint32_t, uint32_t, uint32_t) {
    __emule_compute_ctx().llk_unpack_is_tilize = true;
    __emule_compute_ctx().llk_pack_is_untilize = false;
}
inline void tilize_uninit(uint32_t = 0, uint32_t = 0) {
    __emule_compute_ctx().llk_unpack_is_tilize = false;
    // Note: we do NOT clear __llk_pack_is_untilize here.  Silicon's
    // tilize_uninit reverts the unpacker config but does not touch the
    // packer state.  Clearing the pack flag here regresses kernels that
    // call tilize → tilize_uninit in the middle of a pack-untilize block.
    // Use the explicit `untilize_uninit` shim for the packer-side flag.
}
// _with_dt overloads mirror silicon's data-type-reconfig variants.  Emule
// reads CB format per-call, so the format change is implicit and the
// forwarders are no-op-equivalent.
inline void tilize_init_short_with_dt(uint32_t /*old_icb*/, uint32_t new_icb,
                                      uint32_t num_tiles, uint32_t /*ocb*/ = 0) {
    tilize_init_short(new_icb, num_tiles);
}
inline void tilize_uninit_with_dt(uint32_t /*old_icb*/ = 0, uint32_t /*new_icb*/ = 0,
                                  uint32_t /*ocb*/ = 0) {
    tilize_uninit();
}

// tilize_block: read `ntiles` tiles from `icb` (laid out as a single
// horizontal strip of TILE_HEIGHT rows × ntiles*TILE_WIDTH cols, stride
// ntiles*TILE_WIDTH*elem_size bytes per row) and write each tile to `ocb`
// in nfaces (face-row-major) format.  Matches silicon's llk_unpack_tilize:
// the input CB stores the block's H*W input as a horizontal strip, NOT as
// `ntiles` sequential 2048-byte tile blocks.  For ntiles=1 the strip is one
// tile wide, so the layout collapses to the same single-tile layout used
// elsewhere — backward compatible.
inline void tilize_block(uint32_t icb, uint32_t ntiles, uint32_t ocb) {
    constexpr uint32_t TILE_DIM = 32;
    const bool icb_is_32bit = __emule_compute::cb_is_32bit_format(icb);
    const bool ocb_is_32bit = __emule_compute::cb_is_32bit_format(ocb);
    const uint32_t in_elem_size  = icb_is_32bit ? 4 : 2;
    // Input row stride: silicon's tilize unpacker reads with a TILE_DIM (32) row
    // stride per tile regardless of output tile geometry. Callers pad input to
    // 32 rows even when the useful data is thinner.
    const uint32_t row_stride    = ntiles * TILE_DIM * in_elem_size;
    uint8_t* const in_base = __emule_compute::cb_read_ptr_at(icb, 0);

    // Output tile row count: thin tiles (rows ∈ {1,2,4,8,16}) are used by
    // retilize / create_q_heads / similar ops. Derive from the OCB page size.
    const uint32_t out_elem_size = ocb_is_32bit ? 4 : 2;
    const uint32_t out_rows =
        __emule_compute::cb_is_bfp8_b_format(ocb)
            ? TILE_DIM
            : __emule_nfaces::tile_rows_from_pagesize(
                  __emule_compute::cb_page_size(ocb), out_elem_size);

    // BFP8_b output: encode row-major bf16/fp32 input into 64 face-rows of
    // (shared exponent + 16 mantissas). Layout: 64 exp bytes, then 64*16
    // mantissa bytes (= 1088 bytes/tile). Mirrors the encoder used by
    // __llk_pack_tiled for the BFP8_b OCB path.
    if (__emule_compute::cb_is_bfp8_b_format(ocb)) {
        for (uint32_t t = 0; t < ntiles; ++t) {
            uint8_t* const out = __emule_compute::cb_write_ptr_at(ocb, __emule_compute_ctx().pack_offset[ocb]++);
            uint8_t* const exp_base  = out;
            uint8_t* const mant_base = out + 64;
            for (uint32_t fr = 0; fr < 64; ++fr) {
                float row16[16];
                for (uint32_t k = 0; k < 16; ++k) {
                    const uint32_t rm = __emule_nfaces::nfaces_to_rowmajor[fr * 16 + k];
                    const uint32_t rr = rm / TILE_DIM;
                    const uint32_t cc = rm % TILE_DIM;
                    const uint8_t* p = in_base + rr * row_stride + (t * TILE_DIM + cc) * in_elem_size;
                    if (icb_is_32bit) {
                        float v;
                        std::memcpy(&v, p, sizeof(float));
                        row16[k] = v;
                    } else {
                        uint16_t bf;
                        std::memcpy(&bf, p, sizeof(uint16_t));
                        row16[k] = __emule_bf16::to_f32(bf);
                    }
                }
                __emule_bfp8::encode_face_row(row16, exp_base[fr], &mant_base[fr * 16]);
            }
        }
        return;
    }

    for (uint32_t t = 0; t < ntiles; ++t) {
        uint8_t* const out = __emule_compute::cb_write_ptr_at(ocb, __emule_compute_ctx().pack_offset[ocb]++);
        for (uint32_t r = 0; r < out_rows; ++r) {
            const uint8_t* row_in =
                in_base + r * row_stride + t * TILE_DIM * in_elem_size;
            for (uint32_t c = 0; c < TILE_DIM; ++c) {
                const uint32_t out_pos = __emule_nfaces::tile_rm_to_nfaces(r * TILE_DIM + c, out_rows);
                if (icb_is_32bit && ocb_is_32bit) {
                    uint32_t v;
                    std::memcpy(&v, row_in + c * 4, 4);
                    std::memcpy(reinterpret_cast<uint32_t*>(out) + out_pos, &v, 4);
                } else if (!icb_is_32bit && !ocb_is_32bit) {
                    uint16_t v;
                    std::memcpy(&v, row_in + c * 2, 2);
                    reinterpret_cast<uint16_t*>(out)[out_pos] = v;
                } else if (!icb_is_32bit && ocb_is_32bit) {
                    uint16_t v;
                    std::memcpy(&v, row_in + c * 2, 2);
                    reinterpret_cast<float*>(out)[out_pos] = __emule_bf16::to_f32(v);
                } else {
                    float v;
                    std::memcpy(&v, row_in + c * 4, 4);
                    reinterpret_cast<uint16_t*>(out)[out_pos] = __emule_bf16::from_f32(v);
                }
            }
        }
    }
}

// fast_tilize_{init,block,uninit}: hardware-fast variants on the real device,
// equivalent to the regular tilize path in emule.
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
inline void fast_tilize_init(uint32_t icb, uint32_t /*block_dst_tiles*/, uint32_t ocb) {
    tilize_init(icb, full_ct_dim, ocb);
}
// fast_tilize_init_skip_remap is the BH-only variant that skips the LLK math
// tile-remap reconfig (used when the caller knows the operand is already in
// the right remap state).  In emule there's no remap state to skip — same
// path as fast_tilize_init.
inline void fast_tilize_init_skip_remap(uint32_t icb, uint32_t full_dim, uint32_t ocb,
                                        uint32_t /*call_line*/ = 0) {
    tilize_init(icb, full_dim, ocb);
}
// _with_dt forwarders mirror silicon's 3-arg reconfig API (tilize.h:330,334);
// emule reads CB format per-call so the format reconfig is implicit.
inline void fast_tilize_init_with_dt(uint32_t icb, uint32_t full_dim, uint32_t ocb) {
    tilize_init(icb, full_dim, ocb);
}
inline void fast_tilize_init_with_dt_skip_remap(uint32_t icb, uint32_t full_dim, uint32_t ocb) {
    tilize_init(icb, full_dim, ocb);
}
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
inline void fast_tilize_block(uint32_t icb, uint32_t ntiles, uint32_t ocb,
                              uint32_t /*block_c_index*/ = 0) {
    tilize_block(icb, ntiles, ocb);
}
inline void fast_tilize_uninit(uint32_t = 0, uint32_t = 0, uint32_t = 0) {
    tilize_uninit();
}

// Block variant with explicit tile-index args (some op code uses 5-arg form).
inline void tilize_block(uint32_t icb, uint32_t num_tiles, uint32_t ocb,
                         uint32_t /*itile_start*/, uint32_t /*otile_start*/) {
    tilize_block(icb, num_tiles, ocb);
}
