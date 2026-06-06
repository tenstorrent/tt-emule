// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Bfp4_b codec for emule. Sibling of bfp8.h; mirrors tt-metal's
// /tt_metal/impl/data_format/blockfloat_common.cpp::convert_bfp_to_u32 /
// convert_u32_to_bfp<Bfp4_b> (and bfloat4.cpp). Bfp4_b differs from Bfp8_b only
// in mantissa width (3 vs 7 bits) and that two elements share one byte.
//
// Tile layout (32x32, num_faces=4, face=16x16):
//   bytes [0  .. 63]  — 64 face-row exponents (one per 16 elements, == Bfp8_b)
//   bytes [64 .. 575] — 512 mantissa bytes, two 4-bit elements per byte, in
//                       face-major (nfaces) order; element ni → byte 64+ni/2,
//                       low nibble if ni even, high nibble if ni odd.
//   total = 576 bytes (= tt-metalium tile_size(Bfp4_b) = 128*4 + 16*4).
//
// Each 4-bit element is [sign:1 | mantissa:3]; the shared exponent is the byte.
//
// Decode (per element): same shift-normalize as Bfp8_b but on a 3-bit mantissa
// (normalize until bit 2 set, strip leading 1, place at bits 22:20).
// Encode (per face-row of 16): shared exp = max fp32 exponent; per element
// align the 24-bit significand to it (truncating), then round to a 3-bit
// mantissa (nearest, ties to even) and saturate — matches silicon.

#include <cstdint>
#include <cstring>

namespace __emule_bfp4 {

// to_f32: decode one Bfp4_b element at nfaces position `ni`.
inline float to_f32(const uint8_t* tile_base, uint32_t ni) {
    const uint8_t  exp_byte  = tile_base[ni >> 4];               // [0..63], one per 16
    const uint8_t  data_byte = tile_base[64 + (ni >> 1)];        // two elements per byte
    const uint8_t  nib       = (ni & 1u) ? (data_byte >> 4) : (data_byte & 0x0Fu);
    const uint32_t sign      = (nib >> 3) & 0x1u;
    uint32_t       raw_man   = nib & 0x7u;

    if (raw_man == 0) {
        return 0.0f;
    }

    // Normalize: shift left until bit 2 is set, counting shifts.
    uint32_t shift_cnt = 0;
    while ((raw_man & 0x4u) == 0 && shift_cnt < 3) {
        raw_man = (raw_man << 1) & 0x7u;
        ++shift_cnt;
    }
    // Strip the leading 1 (bit 2) by one more left shift, mask to 3 bits.
    const uint32_t mant_norm = (raw_man << 1) & 0x7u;

    if (shift_cnt > exp_byte) {
        return 0.0f;
    }

    const uint32_t final_exp = static_cast<uint32_t>(exp_byte) - shift_cnt;
    const uint32_t bits = (sign << 31) | (final_exp << 23) | (mant_norm << 20);
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// encode_face_row: encode 16 row-major fp32 elements into one shared exponent
// byte + 8 packed mantissa bytes (two 4-bit elements per byte: element k →
// byte k/2, low nibble if k even, high nibble if k odd). Caller writes the 8
// bytes into the correct face-row slot of the tile.
inline void encode_face_row(const float in16[16], uint8_t& exp_out, uint8_t packed_out[8]) {
    uint32_t max_exp = 0;
    for (int k = 0; k < 16; ++k) {
        uint32_t bits;
        std::memcpy(&bits, &in16[k], sizeof(bits));
        const uint32_t e = (bits >> 23) & 0xFFu;
        if ((bits & 0x7FFFFFFFu) != 0 && e > max_exp) max_exp = e;
    }
    exp_out = static_cast<uint8_t>(max_exp);

    // Per element: align to the shared exponent (truncating), then round the
    // 3-bit mantissa to nearest, ties to even, and saturate — matches silicon
    // convert_u32_to_bfp (truncating the round drifts ~1 ULP, issue #51).
    constexpr uint32_t SHIFT = 21u;             // 24-bit significand -> 3-bit mantissa
    constexpr uint32_t TIE   = 1u << (SHIFT - 1);
    constexpr uint32_t MASK  = (1u << SHIFT) - 1u;
    constexpr uint32_t MAXV  = 0x7u;
    for (int j = 0; j < 8; ++j) packed_out[j] = 0;
    for (int k = 0; k < 16; ++k) {
        uint32_t bits;
        std::memcpy(&bits, &in16[k], sizeof(bits));
        uint32_t sign      = (bits >> 31) & 0x1u;
        const uint32_t exp = (bits >> 23) & 0xFFu;

        uint32_t raw_man = 0;
        if (exp != 0) {  // zero / denormal encode to 0 (silicon returns 0)
            uint32_t m = (1u << 23) | (bits & 0x7FFFFFu);
            uint32_t exp_diff = max_exp - exp;
            while (exp_diff > 31u) { m >>= 31; exp_diff -= 31u; }  // align (truncate)
            m >>= exp_diff;
            const uint32_t rv = m & MASK;                          // round-to-nearest-even
            m >>= SHIFT;
            if (rv > TIE || (rv == TIE && (m & 1u))) ++m;
            raw_man = (m > MAXV) ? MAXV : m;
        }
        if (raw_man == 0) sign = 0;
        const uint8_t nib = static_cast<uint8_t>((sign << 3) | raw_man);
        if (k & 1) {
            packed_out[k >> 1] |= static_cast<uint8_t>(nib << 4);
        } else {
            packed_out[k >> 1] |= nib;
        }
    }
}

} // namespace __emule_bfp4
