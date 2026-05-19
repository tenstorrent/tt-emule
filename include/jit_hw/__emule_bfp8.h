// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Block floating-point (Bfp8_b) tile encode/decode for tt-emule.
//
// Matches tt-metal's byte layout (see impl/data_format/bfloat8.cpp):
//   tile = 1088 bytes = 64 bytes exponent section + 1024 bytes mantissa section
//   exponent section: 4 faces × 16 shared exponents (one per 16-element row)
//   mantissa section: 4 faces × 16×16 sign-magnitude bytes (sign at bit 7,
//                     7-bit mantissa with implicit leading-1 in bits 6:0)
//
// Tile face layout (32x32 tile, 16x16 faces):
//   face 0: rows 0..15,  cols 0..15
//   face 1: rows 0..15,  cols 16..31
//   face 2: rows 16..31, cols 0..15
//   face 3: rows 16..31, cols 16..31
//
// Block = one row of one face = 16 elements sharing one exponent.

#include <cstdint>
#include <cstring>

namespace __emule_bfp8 {

constexpr uint32_t TILE_HW = 1024;          // 32x32
constexpr uint32_t TILE_W = 32;
constexpr uint32_t FACE_HW = 256;           // 16x16
constexpr uint32_t FACE_W = 16;
constexpr uint32_t NUM_FACES = 4;
constexpr uint32_t BLOCK_SIZE = 16;         // elements per shared exponent
constexpr uint32_t BLOCKS_PER_TILE = 64;    // 4 faces * 16 rows
constexpr uint32_t EXP_SECTION_BYTES = 64;  // 4 faces * 16 bytes
constexpr uint32_t MANT_SECTION_BYTES = 1024;
constexpr uint32_t TILE_BYTES = 1088;

// Encode a single bf16-style 8-bit exponent + 7-bit mantissa block from 16
// float values into one shared exponent byte and 16 sign-magnitude bytes.
inline void encode_block(const float* src, uint8_t* out_exp, uint8_t* out_mants) {
    uint8_t exps[BLOCK_SIZE];
    uint8_t mants_pre[BLOCK_SIZE];  // 7-bit mantissa with implicit leading-1, before shift
    uint8_t signs[BLOCK_SIZE];

    uint8_t shared_exp = 0;
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
        uint32_t bits;
        std::memcpy(&bits, &src[i], sizeof(uint32_t));
        uint32_t bf16 = (bits >> 16) & 0xFFFF;
        signs[i] = (bf16 >> 15) & 0x1;
        exps[i] = (bf16 >> 7) & 0xFF;
        // bf16 mantissa is bits 6:0; shift right by 1 then OR with 0x40
        // to set the implicit leading-1 in the 7-bit representation.
        mants_pre[i] = static_cast<uint8_t>(((bf16 & 0x7F) >> 1) | 0x40);
        if (exps[i] > shared_exp) shared_exp = exps[i];
    }
    *out_exp = shared_exp;

    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
        uint32_t delta = static_cast<uint32_t>(shared_exp - exps[i]);
        uint8_t mant = (delta >= 7) ? 0 : static_cast<uint8_t>(mants_pre[i] >> delta);
        out_mants[i] = static_cast<uint8_t>((signs[i] << 7) | (mant & 0x7F));
    }
}

// Decode 16 sign-magnitude bytes + 1 shared exponent → 16 floats.
// Matches tt-metal's unpack_bfp8_tiles_into_float_vec (is_exp_a=false / Bfp8_b):
//   the 7-bit mantissa is left-shifted until its high bit hits 0x40, the
//   shift count is added back to the exponent (re-bias to f32 form), and the
//   result is assembled as IEEE-754 float32 bits.
inline void decode_block(uint8_t shared_exp, const uint8_t* mants, float* dst) {
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
        uint8_t byte = mants[i];
        uint32_t sign = (byte >> 7) & 0x1;
        uint32_t mant = byte & 0x7F;

        if (mant == 0) {
            dst[i] = 0.0f;
            continue;
        }

        // Re-normalize: shift mant left until bit 6 (0x40) is set, counting shifts.
        uint32_t shift_cnt = 0;
        while (!(mant & 0x40) && shift_cnt < 7) {
            mant <<= 1;
            shift_cnt++;
        }
        // One extra shift to clear bit 6 (becomes the implicit leading 1 in f32).
        mant = (mant << 1) & 0x7F;

        // Re-bias exponent: bf16 exponent stored as-is for Bfp8_b. f32 has same
        // 8-bit bias of 127, so no rebias offset; subtract shift_cnt to undo
        // the normalisation. If shift_cnt > shared_exp → denormal → flush to zero.
        if (shift_cnt > shared_exp) {
            dst[i] = 0.0f;
            continue;
        }
        uint32_t f32_exp = static_cast<uint32_t>(shared_exp) - shift_cnt;
        uint32_t bits = (sign << 31) | (f32_exp << 23) | (mant << 16);
        std::memcpy(&dst[i], &bits, sizeof(float));
    }
}

// Encode a full tile: 1024-float row-major DST → 1088-byte tile buffer.
// Tile buffer layout: [64 bytes exp][1024 bytes mantissa], face-by-face.
inline void from_f32_tile(const float* dst_row_major, uint8_t* tile_buf) {
    uint8_t* exp_section = tile_buf;
    uint8_t* mant_section = tile_buf + EXP_SECTION_BYTES;

    for (uint32_t face = 0; face < NUM_FACES; face++) {
        uint32_t face_r = (face / 2) * FACE_W;
        uint32_t face_c = (face % 2) * FACE_W;
        for (uint32_t row = 0; row < FACE_W; row++) {
            float block[BLOCK_SIZE];
            for (uint32_t col = 0; col < BLOCK_SIZE; col++) {
                block[col] = dst_row_major[(face_r + row) * TILE_W + (face_c + col)];
            }
            uint32_t exp_idx = face * FACE_W + row;
            uint32_t mant_off = face * FACE_HW + row * BLOCK_SIZE;
            encode_block(block, &exp_section[exp_idx], &mant_section[mant_off]);
        }
    }
}

// Decode a full tile: 1088-byte tile buffer → 1024-float row-major DST.
inline void to_f32_tile(const uint8_t* tile_buf, float* dst_row_major) {
    const uint8_t* exp_section = tile_buf;
    const uint8_t* mant_section = tile_buf + EXP_SECTION_BYTES;

    for (uint32_t face = 0; face < NUM_FACES; face++) {
        uint32_t face_r = (face / 2) * FACE_W;
        uint32_t face_c = (face % 2) * FACE_W;
        for (uint32_t row = 0; row < FACE_W; row++) {
            uint32_t exp_idx = face * FACE_W + row;
            uint32_t mant_off = face * FACE_HW + row * BLOCK_SIZE;
            float block[BLOCK_SIZE];
            decode_block(exp_section[exp_idx], &mant_section[mant_off], block);
            for (uint32_t col = 0; col < BLOCK_SIZE; col++) {
                dst_row_major[(face_r + row) * TILE_W + (face_c + col)] = block[col];
            }
        }
    }
}

}  // namespace __emule_bfp8
