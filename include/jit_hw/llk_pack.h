#pragma once
// Emule shim for upstream llk_lib/llk_pack.h — pack helpers and llk_pack
// templates.

#include "internal/llk_state.h"
#include "api/compute/common.h"

// ---- Pack helpers ----

// Tilize pack: PACK row-major DST → nfaces CB at pack_offset
inline void __llk_pack_tiled(uint32_t tile_idx, uint32_t ocb) {
    uint8_t* buf = __emule_compute::cb_write_ptr_at(ocb, __llk_pack_offset);
    if (__emule_compute::cb_is_bfp8_b_format(ocb)) {
        // Bfp8_b: 64 face-rows of 16 floats each, shared 8-bit exponent per
        // face-row. Walk in nfaces order so each row16 corresponds to one
        // exponent byte; DST is row-major fp32 so use the inverse LUT.
        uint8_t* exp_base  = buf;
        uint8_t* mant_base = buf + 64;
        for (uint32_t fr = 0; fr < 64; ++fr) {
            float row16[16];
            for (uint32_t k = 0; k < 16; ++k) {
                const uint32_t ni = fr * 16 + k;
                const uint32_t rm = __emule_nfaces::nfaces_to_rowmajor[ni];
                row16[k] = __emule_dst[tile_idx][rm];
            }
            uint8_t mant_row[16];
            __emule_bfp8::encode_face_row(row16, exp_base[fr], mant_row);
            std::memcpy(&mant_base[fr * 16], mant_row, 16);
        }
    } else if (__emule_compute::cb_is_32bit_format(ocb)) {
        uint32_t n = __emule_compute::cb_page_size(ocb) / sizeof(uint32_t);
        if (n > __EMULE_TILE_ELEMS) n = __EMULE_TILE_ELEMS;
        uint32_t* out = reinterpret_cast<uint32_t*>(buf);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            std::memcpy(&out[ni], &__emule_dst[tile_idx][i], sizeof(uint32_t));
        }
    } else {
        uint16_t* bf = reinterpret_cast<uint16_t*>(buf);
        uint32_t n = __emule_compute::cb_tile_elems(ocb);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            bf[ni] = __emule_bf16::from_f32(__emule_dst[tile_idx][i]);
        }
    }
}

// Untilize pack: scatter DST tile into row-major output CB
inline void __llk_pack_untilize(uint32_t tile_idx, uint32_t ocb) {
    uint32_t ps = __emule_compute::cb_page_size(ocb);
    uint32_t elem_size = ps / 1024;
    uint32_t block_c = __llk_pack_block_c;
    uint32_t tile_row = __llk_pack_offset / block_c;
    uint32_t tile_col = __llk_pack_offset % block_c;

    uint8_t* base = __emule_compute::cb_write_ptr_at(ocb, 0);
    uint32_t row_stride = block_c * 32 * elem_size;
    uint32_t tile_row_offset = tile_row * 32 * row_stride;
    uint32_t tile_col_offset = tile_col * 32 * elem_size;

    if (elem_size == 2) {  // bfloat16
        for (uint32_t r = 0; r < 32; r++) {
            uint16_t* dst = reinterpret_cast<uint16_t*>(
                base + tile_row_offset + r * row_stride + tile_col_offset);
            for (uint32_t c = 0; c < 32; c++)
                dst[c] = __emule_bf16::from_f32(__emule_dst[tile_idx][r * 32 + c]);
        }
    } else {  // float32 / int32: use memcpy to preserve exact bit patterns
        for (uint32_t r = 0; r < 32; r++) {
            uint32_t* dst = reinterpret_cast<uint32_t*>(
                base + tile_row_offset + r * row_stride + tile_col_offset);
            for (uint32_t c = 0; c < 32; c++)
                std::memcpy(&dst[c], &__emule_dst[tile_idx][r * 32 + c], sizeof(uint32_t));
        }
    }
}

// 2-param version (used by tilize)
template <bool Untilize, bool IsApprox>
inline void llk_pack(uint32_t tile_idx, uint32_t ocb) {
    if (__llk_pack_is_untilize) __llk_pack_untilize(tile_idx, ocb);
    else __llk_pack_tiled(tile_idx, ocb);
}

// 3-param version (used by untilize)
template <int AccumMode, bool Untilize, bool IsApprox>
inline void llk_pack(uint32_t tile_idx, uint32_t ocb) {
    if (__llk_pack_is_untilize) __llk_pack_untilize(tile_idx, ocb);
    else __llk_pack_tiled(tile_idx, ocb);
}

template <int AccumMode>
inline void llk_math_dest_section_done() {}

template <int AccumMode>
inline void llk_pack_dest_section_done() {
    __llk_pack_offset++;
}
