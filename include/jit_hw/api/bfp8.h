// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// BFP8_b codec for JIT-compiled kernels (emule software emulation).
//
// BFP8_b is a 32x32 tile compressed to 1088 bytes:
//   - 64 bytes of shared exponents (one byte per row of each 16x16 face,
//     4 faces × 16 rows = 64). Packed as 16 uint32 dwords (LE).
//   - 4 × 256 bytes of sign+mantissa data (one byte per element). Each face
//     stored row-major: 16 rows × 16 cols.
//
// Each element on host (FP32) is decomposed:
//   sign (1 bit) | mantissa (7 bits) using the row's shared exponent (FP32 bias).
//
// We implement the inverse `convert_u32_to_bfp<Bfp8_b,false>` and
// `convert_bfp_to_u32(Bfp8_b)` with is_exp_a=false.
//
// This is the kernel-side equivalent of the silicon UNPACK (BFP8 → BF16
// SrcA/B) and PACK (DST → BFP8) hardware engines.

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "jit_hw/api/bfloat16.h"

namespace __emule_bfp8 {

// Per-tile constants (32x32 tile, 16x16 faces, BFP8_b layout).
constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t TILE_COLS = 32;
constexpr uint32_t FACE_ROWS = 16;
constexpr uint32_t FACE_COLS = 16;
constexpr uint32_t NUM_FACES = 4;
constexpr uint32_t EXP_BYTES = 64;             // 4 faces × 16 rows
constexpr uint32_t FACE_DATA_BYTES = 256;      // 16 × 16
constexpr uint32_t TILE_BYTES = EXP_BYTES + NUM_FACES * FACE_DATA_BYTES;  // 1088

// Decode one BFP8 byte to a bf16 element value. Mirrors
// `convert_bfp_to_u32(Bfp8_b, data, exp, is_exp_a=false)`.
inline uint16_t bfp8_byte_to_bf16(uint8_t data, uint8_t shared_exp) {
    uint32_t sign = data >> 7;
    uint32_t man = data & 0x7f;
    uint32_t exp = shared_exp;
    if (man == 0) {
        // shared exp may be nonzero, but mantissa zero → return signed zero.
        exp = 0;
    } else {
        // Normalize: shift up until bit 6 is set.
        int shift_cnt = 0;
        while ((man & 0x40) == 0) {
            man <<= 1;
            ++shift_cnt;
        }
        // Drop the implicit leading-1 (shift one more, mask to 7 bits).
        man = (man << 1) & 0x7f;
        if (exp >= static_cast<uint32_t>(shift_cnt)) {
            exp -= shift_cnt;
        } else {
            // Denormal range — flush to zero.
            exp = 0;
            man = 0;
        }
    }
    // Build FP32, then take top 16 bits for bf16.
    uint32_t u32 = (sign << 31) | (exp << 23) | (man << 16);
    // BF16 is the upper 16 bits of FP32 (no rounding needed since the lower
    // 16 bits we've constructed are zero except for the partial mantissa).
    return static_cast<uint16_t>(u32 >> 16);
}

// Decode one BFP8 byte to FP32. Same as above but full FP32 output.
inline float bfp8_byte_to_f32(uint8_t data, uint8_t shared_exp) {
    uint16_t bf16 = bfp8_byte_to_bf16(data, shared_exp);
    return __emule_bf16::to_f32(bf16);
}

// Compute the shared exponent for a row of FP32 values (max FP32 exponent).
inline uint8_t row_max_exp_f32(const float* row, uint32_t n) {
    uint32_t max_exp = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t u;
        std::memcpy(&u, &row[i], sizeof(uint32_t));
        uint32_t exp = (u & 0x7f800000u) >> 23;
        if (exp > max_exp) max_exp = exp;
    }
    return static_cast<uint8_t>(max_exp);
}

// Encode one FP32 value to a BFP8 byte given the row's shared exponent.
// Mirrors `convert_u32_to_bfp<Bfp8_b, false>(u32, shared_exp, is_exp_a=false)`
// with round-to-nearest-even on the mantissa.
inline uint8_t f32_to_bfp8_byte(float val, uint32_t shared_exp) {
    uint32_t u32;
    std::memcpy(&u32, &val, sizeof(uint32_t));
    uint32_t mantissa = u32 & 0x007fffffu;
    uint32_t exp = (u32 & 0x7f800000u) >> 23;
    uint32_t sign = (u32 & 0x80000000u) >> 31;
    if (exp == 0) {
        // ±0 or denormal → return 0 (sign zero).
        return 0;
    }
    // Add implicit leading 1.
    mantissa |= (1u << 23);
    if (shared_exp > exp) {
        uint32_t diff = shared_exp - exp;
        while (diff > 31) {
            mantissa >>= 31;
            diff -= 31;
        }
        mantissa >>= diff;
    }
    // Round 24-bit mantissa down to 7 bits (shift 17). Mantissa now in bits [23:17].
    constexpr uint32_t MANTISSA_BFP_WIDTH = 7;
    constexpr uint32_t MANTISSA_BFP_SHIFT = 24 - MANTISSA_BFP_WIDTH;  // 17
    constexpr uint32_t MANTISSA_BFP_MAX_VAL = (1u << MANTISSA_BFP_WIDTH) - 1;  // 0x7f
    constexpr uint32_t ROUND_MASK = (1u << MANTISSA_BFP_SHIFT) - 1;
    constexpr uint32_t TIE_VALUE = 1u << (MANTISSA_BFP_SHIFT - 1);
    uint32_t round_value = mantissa & ROUND_MASK;
    mantissa >>= MANTISSA_BFP_SHIFT;
    uint32_t guard_bit = mantissa & 0x1u;
    if (round_value > TIE_VALUE || (round_value == TIE_VALUE && guard_bit == 1)) {
        mantissa += 1;
    }
    if (mantissa > MANTISSA_BFP_MAX_VAL) mantissa = MANTISSA_BFP_MAX_VAL;
    if (mantissa == 0) sign = 0;
    return static_cast<uint8_t>((sign << MANTISSA_BFP_WIDTH) | mantissa);
}

// Convert a row-major 32×32 BF16 tile to BFP8_b on-disk format.
// `rm` points to 1024 BF16 elements (row-major), `bfp8` points to 1088 bytes.
inline void rowmajor_bf16_tile_to_bfp8(const uint16_t* rm, uint8_t* bfp8) {
    // Stage 1: convert all 1024 BF16 to FP32 for max-exp + encode.
    float fp[1024];
    for (uint32_t i = 0; i < 1024; ++i) fp[i] = __emule_bf16::to_f32(rm[i]);

    // Exponents come first (64 bytes). For each face (tr,tc) and row i in
    // [0..15], compute max exp across 16 elements at row=tr*16+i, cols
    // tc*16..tc*16+15. Exponents are packed face-major (face0 16 exps,
    // face1 16 exps, ...).
    uint8_t exps[64];
    for (uint32_t tr = 0; tr < 2; ++tr) {
        for (uint32_t tc = 0; tc < 2; ++tc) {
            uint32_t face_idx = tr * 2 + tc;
            for (uint32_t i = 0; i < 16; ++i) {
                uint32_t row = tr * 16 + i;
                uint32_t col0 = tc * 16;
                exps[face_idx * 16 + i] =
                    row_max_exp_f32(&fp[row * 32 + col0], 16);
            }
        }
    }
    // Pack 64 exponents into 16 dwords (LE).
    uint32_t* exp_words = reinterpret_cast<uint32_t*>(bfp8);
    for (uint32_t w = 0; w < 16; ++w) {
        uint32_t v = 0;
        for (uint32_t b = 0; b < 4; ++b) v |= static_cast<uint32_t>(exps[w * 4 + b]) << (8 * b);
        exp_words[w] = v;
    }

    // Then 4 faces of 256 bytes mantissa data, row-major within each face.
    uint8_t* data = bfp8 + EXP_BYTES;
    for (uint32_t tr = 0; tr < 2; ++tr) {
        for (uint32_t tc = 0; tc < 2; ++tc) {
            uint32_t face_idx = tr * 2 + tc;
            uint8_t* face = data + face_idx * FACE_DATA_BYTES;
            for (uint32_t i = 0; i < 16; ++i) {
                uint8_t shared = exps[face_idx * 16 + i];
                uint32_t row = tr * 16 + i;
                uint32_t col0 = tc * 16;
                for (uint32_t j = 0; j < 16; ++j) {
                    face[i * 16 + j] = f32_to_bfp8_byte(fp[row * 32 + col0 + j], shared);
                }
            }
        }
    }
}

// Convert a BFP8_b on-disk 1088-byte tile to a row-major 1024-element BF16 tile.
inline void bfp8_tile_to_rowmajor_bf16(const uint8_t* bfp8, uint16_t* rm) {
    const uint32_t* exp_words = reinterpret_cast<const uint32_t*>(bfp8);
    uint8_t exps[64];
    for (uint32_t w = 0; w < 16; ++w) {
        for (uint32_t b = 0; b < 4; ++b) {
            exps[w * 4 + b] = static_cast<uint8_t>((exp_words[w] >> (8 * b)) & 0xff);
        }
    }
    const uint8_t* data = bfp8 + EXP_BYTES;
    for (uint32_t tr = 0; tr < 2; ++tr) {
        for (uint32_t tc = 0; tc < 2; ++tc) {
            uint32_t face_idx = tr * 2 + tc;
            const uint8_t* face = data + face_idx * FACE_DATA_BYTES;
            for (uint32_t i = 0; i < 16; ++i) {
                uint8_t shared = exps[face_idx * 16 + i];
                uint32_t row = tr * 16 + i;
                uint32_t col0 = tc * 16;
                for (uint32_t j = 0; j < 16; ++j) {
                    rm[row * 32 + col0 + j] = bfp8_byte_to_bf16(face[i * 16 + j], shared);
                }
            }
        }
    }
}

// Convenience: is this CB page size BFP8_b?
constexpr inline bool is_bfp8_page_size(uint32_t page_size) {
    return page_size == TILE_BYTES;
}

}  // namespace __emule_bfp8
