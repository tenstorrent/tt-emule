// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Bfp8_b codec for emule. Mirrors tt-metal's
// /tt_metal/impl/data_format/bfloat8.cpp::unpack_bfp8_tiles_into_float_vec
// and pack_as_bfp_tiles<Bfp8_b>. The reference uses simde/AVX2 intrinsics on
// the host; emule's JIT path is scalar — these helpers reproduce the per-lane
// algorithm without the SIMD packing.
//
// Tile layout (32x32 Wormhole-class default, num_faces=4, face=16x16):
//   bytes [0   .. 63]   — 64 face-row exponents (4 faces × 16 face rows)
//   bytes [64  .. 1087] — 1024 mantissa bytes, one per element, in face-major
//                         (nfaces) order: face0_row0_col0..15, face0_row1_..., ...
//   total = 1088 bytes (matches tt-metalium/constants.hpp::BFLOAT8_B_TILE_HW + 64).
//
// Each face-row of 16 elements shares one exponent byte.
//
// Decode (per element):
//   byte    = raw 8-bit value (sign:1 | mantissa:7)
//   exp     = shared face-row exponent byte
//   sign    = (byte >> 7) & 1
//   raw_man = byte & 0x7F     // NO implicit leading 1
//   if raw_man == 0: float = 0
//   else: shift raw_man left until bit 6 set (count = shift_cnt),
//         strip the leading 1 (one more left shift, mask 0x7F),
//         final_exp = exp - shift_cnt              (Bfp8_b rebias = 0)
//         denormal flush: if shift_cnt > exp, float = 0
//         else: float = sign | (final_exp << 23) | (mantissa_7bit << 16)
//
// Encode (per face-row of 16 floats):
//   exp_out = max of the elements' fp32 exponent bytes  (treat 0.0 as exp 0)
//   per element:
//     significand_24  = (1 << 23) | (bits & 0x7FFFFF)  // restore implicit 1
//     shift_amount    = exp_out - elem_exp + 17        // 17 lands bit-23 → bit-6
//     raw_man         = (significand_24 >> shift_amount) & 0x7F  (clamp shift ≥31 to 0)
//     byte            = (sign << 7) | raw_man
//
// Validated against tt-metal's pack_as_bfp8_tiles/unpack_bfp8_tiles_into_float_vec
// by tt-emule/tests/bfp8_roundtrip_test.cpp (bit-exact lane-by-lane).

#include <cstdint>
#include <cstring>

namespace __emule_bfp8 {

// Element index → (face_row [0..63], col_in_row [0..15]) in face-major (nfaces)
// order. Each face is 16x16, faces are stored sequentially in memory: face0
// occupies nfaces indices 0..255, face1 256..511, etc. Within a face, rows are
// sequential: face0_row0 = 0..15, face0_row1 = 16..31, ...
//
// Because face-rows are contiguous in the nfaces layout, this is just a
// 16-element division.
inline constexpr uint32_t face_row(uint32_t nfaces_idx) { return nfaces_idx >> 4; }   // / 16
inline constexpr uint32_t col_in_row(uint32_t nfaces_idx) { return nfaces_idx & 0xF; } // % 16

// to_f32: decode one Bfp8_b element at nfaces position `ni`.
inline float to_f32(const uint8_t* tile_base, uint32_t ni) {
    const uint8_t  exp_byte    = tile_base[face_row(ni)];                // [0..63]
    const uint8_t  mant_byte   = tile_base[64 + ni];                     // [64..1087]
    const uint32_t sign        = (mant_byte >> 7) & 0x1u;
    uint32_t       raw_man     = mant_byte & 0x7Fu;

    if (raw_man == 0) {
        // Zero mantissa → zero output (sign bit ignored per IEEE +0/-0 collapse).
        return 0.0f;
    }

    // Normalize: shift left until bit 6 is set, counting shifts.
    uint32_t shift_cnt = 0;
    while ((raw_man & 0x40u) == 0 && shift_cnt < 7) {
        raw_man = (raw_man << 1) & 0x7Fu;
        ++shift_cnt;
    }
    // Strip the leading 1 (bit 6) by one more left shift, then mask to 7 bits.
    const uint32_t mant_norm = (raw_man << 1) & 0x7Fu;

    // Denormal flush: tt-metal's reference zeros the output when shift_cnt > exp.
    if (shift_cnt > exp_byte) {
        return 0.0f;
    }

    const uint32_t final_exp = static_cast<uint32_t>(exp_byte) - shift_cnt;
    const uint32_t bits = (sign << 31) | (final_exp << 23) | (mant_norm << 16);
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// encode_face_row: encode 16 row-major fp32 elements into one shared exponent
// byte + 16 raw mantissa bytes. Caller is responsible for writing them into
// the correct nfaces slots in the tile.
inline void encode_face_row(const float in16[16], uint8_t& exp_out, uint8_t mant_out[16]) {
    // Pass 1: find max fp32 exponent across the 16 elements. Zeros are
    // treated as exp 0 — they encode to raw_man 0 regardless, so they don't
    // perturb the shared exponent if any non-zero element is present.
    uint32_t max_exp = 0;
    for (int k = 0; k < 16; ++k) {
        uint32_t bits;
        std::memcpy(&bits, &in16[k], sizeof(bits));
        const uint32_t e = (bits >> 23) & 0xFFu;
        // Skip zeros so they don't pin max_exp at 0 when other elements are
        // non-zero (already the case since 0 < any non-zero exp). Including
        // them is fine for correctness because exp=0 ⇒ value=0 ⇒ raw_man=0.
        if ((bits & 0x7FFFFFFFu) != 0 && e > max_exp) max_exp = e;
    }
    exp_out = static_cast<uint8_t>(max_exp);

    // Pass 2: encode each element relative to the shared exponent.
    for (int k = 0; k < 16; ++k) {
        uint32_t bits;
        std::memcpy(&bits, &in16[k], sizeof(bits));
        const uint32_t sign = (bits >> 31) & 0x1u;
        const uint32_t exp  = (bits >> 23) & 0xFFu;
        const uint32_t mant = bits & 0x7FFFFFu;

        uint32_t raw_man = 0;
        if ((bits & 0x7FFFFFFFu) != 0) {
            // Reconstruct 24-bit significand with implicit leading 1.
            const uint32_t signif24 = (1u << 23) | mant;
            // Shift right by 17 + (max_exp - exp) to land bit 23 of signif24
            // at bit 6 of the 7-bit raw mantissa. If the shift is ≥ 32 or
            // exp > max_exp (shouldn't happen since max_exp is the max), the
            // result rounds to zero.
            const uint32_t exp_delta = max_exp - exp;
            const uint32_t shift_amt = 17u + exp_delta;
            if (shift_amt < 32u) {
                raw_man = (signif24 >> shift_amt) & 0x7Fu;
            }
        }
        mant_out[k] = static_cast<uint8_t>((sign << 7) | raw_man);
    }
}

} // namespace __emule_bfp8
