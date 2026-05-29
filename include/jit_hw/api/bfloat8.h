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
//
// Bfp8_b stores mantissas as raw 7-bit unsigned values (NO implicit leading 1).
// Real fp32 has an implicit leading 1. To convert, find the position of the
// most-significant 1 in the 7-bit mantissa, left-shift it to that position
// (bit 6), strip the now-implicit leading 1, and decrement the exponent by
// the shift count. If the mantissa is 0 the value is 0. If the shift would
// drop the exponent below 0, the value underflows to 0 (matches the real
// `Flush denormals to zero` path in unpack_bfp8_tiles_into_float_vec).
inline float to_f32(const uint8_t* tile_buf, uint32_t ni) {
    const uint8_t exp_byte  = tile_buf[ni / 16];
    const uint8_t mant_byte = tile_buf[EXP_BLOCK_BYTES + ni];
    const uint32_t sign   = (mant_byte >> 7) & 1u;
    uint32_t m_7bit       = mant_byte & 0x7Fu;
    if (m_7bit == 0) {
        // Zero mantissa → exact zero regardless of exp_byte.
        uint32_t bits = sign << 31;
        float v; std::memcpy(&v, &bits, sizeof(float));
        return v;
    }
    // Normalize: shift left until bit 6 (0x40) is set, tracking shift count.
    uint32_t shift_cnt = 0;
    while ((m_7bit & 0x40u) == 0 && shift_cnt < 7) {
        m_7bit <<= 1;
        ++shift_cnt;
    }
    // Strip the leading 1 — one more left shift, then mask to 7 bits. The
    // result is the 6-bit fractional part placed at bits [6:1], i.e. the
    // top 6 bits of fp32's 23-bit mantissa once shifted by 16.
    const uint32_t mant_norm = static_cast<uint32_t>((m_7bit << 1) & 0x7Fu);
    // Subtract shift_cnt from the exponent. Underflow → flush to ±0.
    if (static_cast<uint32_t>(exp_byte) <= shift_cnt) {
        uint32_t bits = sign << 31;
        float v; std::memcpy(&v, &bits, sizeof(float));
        return v;
    }
    const uint32_t new_exp = static_cast<uint32_t>(exp_byte) - shift_cnt;
    uint32_t bits = (sign << 31) | (new_exp << 23) | (mant_norm << 16);
    float v; std::memcpy(&v, &bits, sizeof(float));
    return v;
}

// Encode 16 fp32 values (one face-row) into one shared 8-bit exponent + 16
// mantissa bytes (Bfp8_b layout).
//
// Per-face-row quantization:
//   shared_exp = max over the 16 elements of their fp32 exponent field.
//   For each element:
//     1. Build the 24-bit significand (implicit leading 1 OR mantissa[22:0]
//        for subnormals).
//     2. Align to shared_exp by right-shifting (shared_exp - element_exp)
//        positions. After this, the LEADING 1 of the aligned significand
//        sits at bit (23 - (shared_exp - element_exp)).
//     3. Extract bits [23:17] of the aligned value into the 7-bit mantissa.
//        This places the leading 1 at bit 6 (for the largest element) or
//        below (for smaller elements). Decoder normalizes by shifting until
//        bit 6 is set.
//   The decoder reverses this: locates the leading 1, strips it, scales
//   shared_exp down by the shift count.
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
        // 24-bit significand: prepend the implicit leading 1 for normals.
        const uint32_t significand24 = (e == 0) ? mantissa23 : (mantissa23 | 0x800000u);
        uint32_t shift = (e == 0 || shared_exp == 0) ? 0u : (shared_exp - e);
        if (shift > 24) shift = 24;
        const uint32_t aligned = significand24 >> shift;
        // Place the (possibly off-normalized) leading 1 of the aligned
        // significand at bit 6 of the 7-bit mantissa — that is, take
        // aligned[23:17]. For the element whose e == shared_exp, this gives
        // bit 6 set (canonical normalized form). For smaller elements the
        // leading 1 lands at a lower bit (which the decoder then re-shifts).
        const uint32_t m_7bit = (shared_exp == 0) ? 0u : ((aligned >> 17) & 0x7Fu);
        mant_out[i] = static_cast<uint8_t>((sign << 7) | m_7bit);
    }
}

}  // namespace __emule_bfp8
