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
// nfaces tiles to `ocb`.
//
// Direct rowmajor→nfaces conversion. Don't reuse `copy_tile + pack_tile`:
// copy_tile's unpack ALWAYS applies the nfaces→rowmajor permutation (assumes
// nfaces input); pack_tile then applies the inverse rowmajor→nfaces. With
// row-major source data, the unpack permutation produces a scrambled DST,
// so the round-trip is NOT identity.
inline void tilize_block(uint32_t icb, uint32_t ntiles, uint32_t ocb) {
    const bool icb_is_32bit = __emule_compute::cb_is_32bit_format(icb);
    const bool ocb_is_32bit = __emule_compute::cb_is_32bit_format(ocb);
    for (uint32_t t = 0; t < ntiles; ++t) {
        uint8_t* in = __emule_compute::cb_read_ptr_at(icb, t);
        uint8_t* out = __emule_compute::cb_write_ptr_at(ocb, __emule_pack_offset[ocb]++);
        if (icb_is_32bit && ocb_is_32bit) {
            const uint32_t* in_u = reinterpret_cast<const uint32_t*>(in);
            uint32_t* out_u = reinterpret_cast<uint32_t*>(out);
            for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
                std::memcpy(&out_u[__emule_nfaces::rowmajor_to_nfaces[i]], &in_u[i], sizeof(uint32_t));
            }
        } else if (!icb_is_32bit && !ocb_is_32bit) {
            const uint16_t* in_bf = reinterpret_cast<const uint16_t*>(in);
            uint16_t* out_bf = reinterpret_cast<uint16_t*>(out);
            for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
                out_bf[__emule_nfaces::rowmajor_to_nfaces[i]] = in_bf[i];
            }
        } else if (!icb_is_32bit && ocb_is_32bit) {
            // bf16 in → fp32 out
            const uint16_t* in_bf = reinterpret_cast<const uint16_t*>(in);
            float* out_f = reinterpret_cast<float*>(out);
            for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
                out_f[__emule_nfaces::rowmajor_to_nfaces[i]] = __emule_bf16::to_f32(in_bf[i]);
            }
        } else {
            // fp32 in → bf16 out
            const float* in_f = reinterpret_cast<const float*>(in);
            uint16_t* out_bf = reinterpret_cast<uint16_t*>(out);
            for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
                out_bf[__emule_nfaces::rowmajor_to_nfaces[i]] = __emule_bf16::from_f32(in_f[i]);
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
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
inline void fast_tilize_block(uint32_t icb, uint32_t ntiles, uint32_t ocb,
                              uint32_t /*block_c_index*/ = 0) {
    tilize_block(icb, ntiles, ocb);
}
inline void fast_tilize_uninit(uint32_t = 0, uint32_t = 0, uint32_t = 0) {
    tilize_uninit();
}
