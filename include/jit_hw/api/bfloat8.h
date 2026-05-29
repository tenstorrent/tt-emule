// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Bfp8_b ↔ float32 conversion helpers for JIT-compiled kernels.
//
// Bfp8_b tile layout (32×32 = 4 faces of 16×16):
//   Bytes [0,   64): 64 exponent bytes — one per face-row (4 faces × 16 rows = 64).
//                    For each face-row of 16 elements, all 16 mantissas share this
//                    one 8-bit fp32-style exponent (biased by 127, no rebias since
//                    the Tenstorrent "_b" suffix means byte-wide exponent).
//   Bytes [64, 1088): 1024 mantissa bytes — one per element, in nfaces order:
//                     bit 7  = sign,
//                     bits 6..0 = top 7 bits of the fp32 mantissa.
// Total tile size: 1088 bytes.
//
// Decode (Bfp8_b → fp32):
//   sign   = (mantissa_byte >> 7) & 1
//   m_7bit = mantissa_byte & 0x7F
//   f32_bits = (sign << 31) | (exp_byte << 23) | (m_7bit << 16)
// Special: when exp_byte == 0 and m_7bit == 0, the value is 0.f.
//
// Encode (fp32 → Bfp8_b) is per-face-row: compute max exponent across the 16
// elements in the face-row, then quantize each element's mantissa relative to
// that shared exponent. See encode_face_row below.

#include <cstdint>
#include <cstring>
#include <algorithm>

namespace __emule_bfp8 {

constexpr uint32_t EXP_BLOCK_BYTES = 64;     // 64 face-rows, one exponent each.
constexpr uint32_t MANTISSA_BYTES   = 1024;  // 1024 elements, one mantissa each.

// Decode element `ni` from a Bfp8_b tile buffer.
//   tile_buf: pointer to the start of the tile in L1 (exponent block + mantissa block).
//   ni:       nfaces-order element index (0..1023).
inline float to_f32(const uint8_t* tile_buf, uint32_t ni) {
    const uint8_t exp_byte = tile_buf[ni / 16];        // shared exponent for this face-row
    const uint8_t mant_byte = tile_buf[EXP_BLOCK_BYTES + ni];
    const uint32_t sign = (mant_byte >> 7) & 1u;
    const uint32_t m_7bit = mant_byte & 0x7Fu;
    uint32_t bits = (sign << 31) | (static_cast<uint32_t>(exp_byte) << 23) | (m_7bit << 16);
    if (exp_byte == 0 && m_7bit == 0) bits = 0;  // canonical +0
    float v;
    std::memcpy(&v, &bits, sizeof(float));
    return v;
}

// Encode 16 fp32 values (one face-row) into one shared 8-bit exponent + 16
// mantissa bytes. Writes the exponent byte at exp_out and the 16 mantissa
// bytes at mant_out.
//
// Per-face-row quantization:
//   shared_exp = max over the 16 elements of their fp32 exponent field.
//   For each element: align its mantissa to shared_exp by right-shifting the
//   implicit-leading-1 normalized value by (shared_exp - element_exp). Keep
//   top 7 bits of the aligned 24-bit significand (1 implicit + 23 fractional).
inline void encode_face_row(const float* in16, uint8_t* exp_out, uint8_t* mant_out) {
    uint32_t shared_exp = 0;
    uint32_t bits[16];
    for (uint32_t i = 0; i < 16; ++i) {
        std::memcpy(&bits[i], &in16[i], sizeof(uint32_t));
        const uint32_t e = (bits[i] >> 23) & 0xFFu;
        if (e > shared_exp) shared_exp = e;
    }
    *exp_out = static_cast<uint8_t>(shared_exp);
    for (uint32_t i = 0; i < 16; ++i) {
        const uint32_t b = bits[i];
        const uint32_t sign = (b >> 31) & 1u;
        const uint32_t e = (b >> 23) & 0xFFu;
        const uint32_t mantissa23 = b & 0x7FFFFFu;
        // Significand with the implicit leading 1 (normalized) or 0 (subnormal).
        const uint32_t significand24 = (e == 0) ? mantissa23 : (mantissa23 | 0x800000u);
        uint32_t shift = (e == 0 || shared_exp == 0) ? 0u : (shared_exp - e);
        if (shift > 24) shift = 24;
        const uint32_t aligned = significand24 >> shift;
        // Take the top 7 bits below the implicit-1, with round-to-zero (truncate).
        // aligned has up to 24 significant bits; we want bits [22:16] of the
        // 23-bit fractional part, which is (aligned >> 16) & 0x7F when shared_exp
        // != 0. For shared_exp == 0 (all zeros face-row), result is zero already.
        const uint32_t m_7bit = (shared_exp == 0) ? 0u : ((aligned >> 16) & 0x7Fu);
        mant_out[i] = static_cast<uint8_t>((sign << 7) | m_7bit);
    }
}

}  // namespace __emule_bfp8
