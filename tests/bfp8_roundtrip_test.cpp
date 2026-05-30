// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Standalone validation for __emule_bfp8::{to_f32, encode_face_row}.
// Self-consistency test (encode → decode → compare). Ground truth for
// against-upstream bit-exact behavior is the pytest suite at
//   tests/ttnn/unit_tests/base_functionality/test_untilize_bfloat8_b.py
// which round-trips through tt-metal's pack_as_bfp8_tiles. This binary
// exists to catch algorithm-level bugs (normalization, denormal flush,
// sign handling, exponent sharing) quickly during dev iteration, without
// needing to spin up the full ttnn stack.

#include "jit_hw/api/compute/bfp8.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// One face-row buffer: 1 exponent byte + 16 mantissa bytes.
struct FaceRow {
    uint8_t exp;
    uint8_t mant[16];
};

// Pack a single face-row, then decode each lane and compare to the input.
// Returns max absolute relative error across the 16 lanes. Bfp8_b has ~7-bit
// mantissa precision per face-row, so relative error within ~2^-6 (≈0.016)
// is expected for non-pinned values.
double roundtrip_face_row(const float (&in16)[16]) {
    FaceRow fr{};
    __emule_bfp8::encode_face_row(in16, fr.exp, fr.mant);

    // Lay out a synthetic tile with one face-row's worth of data at fr=0.
    uint8_t tile[1088] = {};
    tile[0] = fr.exp;
    std::memcpy(&tile[64], fr.mant, 16);

    double worst = 0.0;
    for (int k = 0; k < 16; ++k) {
        float decoded = __emule_bfp8::to_f32(tile, /*nfaces_idx=*/k);
        // Relative error vs input — guard division by zero with abs threshold.
        const float ref = in16[k];
        const float err = std::fabs(decoded - ref);
        const float scale = std::fmax(std::fabs(ref), 1e-30f);
        const double rel = static_cast<double>(err) / static_cast<double>(scale);
        if (rel > worst) worst = rel;
    }
    return worst;
}

int run_case(const char* name, const float (&in16)[16], double tol) {
    const double err = roundtrip_face_row(in16);
    const bool pass = err <= tol;
    std::printf("[%s] max rel err = %.6g (tol %.6g)  %s\n",
                name, err, tol, pass ? "OK" : "FAIL");
    return pass ? 0 : 1;
}

int test_zero_mantissa_decodes_zero() {
    // All-zero raw mantissa, any exponent → 0.0 (per ref unpack: select_mask).
    uint8_t tile[1088] = {};
    tile[0] = 0x7F;  // arbitrary non-zero exponent
    for (int k = 0; k < 16; ++k) {
        if (__emule_bfp8::to_f32(tile, k) != 0.0f) {
            std::printf("[zero_mantissa] lane %d decoded non-zero — FAIL\n", k);
            return 1;
        }
    }
    std::printf("[zero_mantissa] all 16 lanes → 0.0  OK\n");
    return 0;
}

int test_sign_preservation() {
    // +1.0 and -1.0 alternated. Both have exp 127 (0x7F) and raw_man=0 (which
    // would decode to 0 — so use 1.5 to get a non-zero raw mantissa).
    float in16[16];
    for (int k = 0; k < 16; ++k) in16[k] = (k & 1) ? -1.5f : 1.5f;
    const double err = roundtrip_face_row(in16);
    // Verify sign of each decoded lane matches input.
    FaceRow fr{};
    __emule_bfp8::encode_face_row(in16, fr.exp, fr.mant);
    uint8_t tile[1088] = {};
    tile[0] = fr.exp;
    std::memcpy(&tile[64], fr.mant, 16);
    int fails = 0;
    for (int k = 0; k < 16; ++k) {
        float d = __emule_bfp8::to_f32(tile, k);
        if ((d < 0) != (in16[k] < 0)) {
            std::printf("[sign] lane %d input %.3f decoded %.3f — sign mismatch\n",
                        k, in16[k], d);
            ++fails;
        }
    }
    std::printf("[sign] max rel err = %.6g, sign-mismatch lanes = %d  %s\n",
                err, fails, fails == 0 ? "OK" : "FAIL");
    return fails == 0 ? 0 : 1;
}

} // namespace

int main() {
    int fails = 0;

    // Uniform values share an exponent perfectly; should round-trip exactly.
    {
        float in16[16] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                           1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        fails += run_case("uniform_1.0", in16, 1e-6);
    }

    // Mixed magnitudes within ~1 bit of each other; ~1/64 precision expected.
    {
        float in16[16];
        for (int k = 0; k < 16; ++k) in16[k] = 1.0f + (k * 0.05f);
        fails += run_case("near_1.0_spread", in16, 0.02);
    }

    // Modest dynamic range within Bfp8's representable window. The shared
    // exponent gives ~7 bits of precision relative to the max lane, so values
    // up to ~64× smaller than the max retain bf-style precision. Beyond that,
    // lanes flush to zero by design — exercised separately below.
    {
        float in16[16];
        for (int k = 0; k < 16; ++k) in16[k] = std::ldexp(1.0f, -(k % 6));
        fails += run_case("modest_dynamic_range", in16, 0.02);
    }

    // Tail flush: a single dominant lane forces all other lanes to flush.
    // Validates that the denormal-flush path doesn't NaN or wrap.
    {
        float in16[16] = {};
        in16[0] = 1.0e6f;
        for (int k = 1; k < 16; ++k) in16[k] = 1.0e-6f;  // ~10^12× smaller
        FaceRow fr{};
        __emule_bfp8::encode_face_row(in16, fr.exp, fr.mant);
        uint8_t tile[1088] = {};
        tile[0] = fr.exp;
        std::memcpy(&tile[64], fr.mant, 16);
        bool ok = true;
        // Lane 0 should decode close to 1e6; others should flush to 0.
        float d0 = __emule_bfp8::to_f32(tile, 0);
        if (std::fabs(d0 - 1.0e6f) / 1.0e6f > 0.02) {
            std::printf("[tail_flush] dominant lane decoded %.3e — FAIL\n", d0);
            ok = false;
        }
        for (int k = 1; k < 16; ++k) {
            float d = __emule_bfp8::to_f32(tile, k);
            if (d != 0.0f) {
                std::printf("[tail_flush] lane %d decoded %.3e, expected 0 — FAIL\n", k, d);
                ok = false;
            }
        }
        std::printf("[tail_flush] dominant %.3e + 15 flushed lanes  %s\n",
                    static_cast<double>(d0), ok ? "OK" : "FAIL");
        if (!ok) ++fails;
    }

    fails += test_zero_mantissa_decodes_zero();
    fails += test_sign_preservation();

    if (fails == 0) {
        std::printf("\nAll bfp8 roundtrip checks passed.\n");
        return 0;
    } else {
        std::printf("\n%d bfp8 roundtrip checks failed.\n", fails);
        return 1;
    }
}
