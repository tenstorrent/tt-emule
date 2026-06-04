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
    const uint32_t row_stride    = ntiles * TILE_DIM * in_elem_size;
    uint8_t* const in_base = __emule_compute::cb_read_ptr_at(icb, 0);

    for (uint32_t t = 0; t < ntiles; ++t) {
        uint8_t* const out = __emule_compute::cb_write_ptr_at(ocb, __emule_pack_offset[ocb]++);
        for (uint32_t r = 0; r < TILE_DIM; ++r) {
            const uint8_t* row_in =
                in_base + r * row_stride + t * TILE_DIM * in_elem_size;
            for (uint32_t c = 0; c < TILE_DIM; ++c) {
                const uint32_t out_pos = __emule_nfaces::rowmajor_to_nfaces[r * TILE_DIM + c];
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
// _with_dt fast-tilize forwarders mirror silicon's data-type-reconfig API
// (used by pool / conv2d-reuse / group_attn_matmul / upsample-bilinear).
// Emule reads CB format per-call so the format change is implicit.
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
inline void fast_tilize_init_with_dt(uint32_t /*old_icb*/, uint32_t new_icb,
                                     uint32_t /*block_dst_tiles*/, uint32_t ocb) {
    tilize_init(new_icb, full_ct_dim, ocb);
}
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
inline void fast_tilize_init_with_dt_skip_remap(uint32_t /*old_icb*/, uint32_t new_icb,
                                                uint32_t /*block_dst_tiles*/,
                                                uint32_t ocb,
                                                uint32_t /*call_line*/ = 0) {
    tilize_init(new_icb, full_ct_dim, ocb);
}
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
inline void fast_tilize_block(uint32_t icb, uint32_t ntiles, uint32_t ocb,
                              uint32_t /*block_c_index*/ = 0) {
    tilize_block(icb, ntiles, ocb);
}
inline void fast_tilize_uninit(uint32_t = 0, uint32_t = 0, uint32_t = 0) {
    tilize_uninit();
}
