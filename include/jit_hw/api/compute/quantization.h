// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/quantization.h`. Intercepts the upstream
// include path which pulls in `llk_math_eltwise_binary_sfpu_quant.h` and the
// LLK SFPU implementation (replay-buffer + SFPLOAD/SFPSTORE intrinsics).
//
// Upstream semantics (see
//   tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_quant.h
// for the silicon implementation):
//
//   quant_tile(A_fp32, B_fp32, OUT_int32):
//       OUT = clamp_to_int8( round_even( A * B + zero_point_fp32 ) )
//   requant_tile(A_int32, B_fp32, OUT_int32):
//       OUT = clamp_to_int8( round_even( (float)A * B + zero_point_fp32 ) )
//   dequant_tile(A_int32, B_fp32, OUT_fp32):
//       OUT = ((float)A + zero_point_fp32) * B
//       (LLK convention: caller passes -zero_point through dequant_tile_init,
//        so the recorded body adds rather than subtracts.)
//
// The `zero_point` argument to each *_tile_init is the uint32 bit-pattern of
// a fp32 value (loaded into LREG2 via _sfpu_load_imm32_ on silicon). Each op
// keeps its own thread_local zero-point state so a single kernel can mix all
// three ops without their inits clobbering each other (mirrors the distinct
// replay-buffer slots on silicon).
//
// Per-op outputs that target int32 dst use __emule_dst_store_i32 (matching
// SFPSTORE with InstrModLoadStore::INT32_2S_COMP). The quant/requant clamp
// range is the int8 range because silicon uses SFPSTOCHRND_MOD1_FP32_TO_INT8.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

// Per-op zero-point state, populated by *_tile_init and consumed by *_tile.
// Stored as fp32 (decoded from the uint32 bit-pattern the caller supplies).

namespace {

inline float __emule_quant_decode_fp32(uint32_t bits) {
    float v;
    std::memcpy(&v, &bits, sizeof(float));
    return v;
}

// fp32 -> int8 (held in int32) with round-to-nearest-even and saturation.
// Matches SFPSTOCHRND_RND_EVEN + SFPSTOCHRND_MOD1_FP32_TO_INT8 on silicon.
inline int32_t __emule_quant_fp32_to_int8_sat(float x) {
    if (std::isnan(x)) {
        return 0;
    }
    float r = std::nearbyint(x);  // honours fenv rounding mode; default is RNE
    constexpr float kMin = -128.0f;
    constexpr float kMax = 127.0f;
    if (r < kMin) {
        return -128;
    }
    if (r > kMax) {
        return 127;
    }
    return static_cast<int32_t>(r);
}

}  // namespace

ALWI void quant_tile_init(uint32_t zero_point) {
    __emule_compute_ctx().quant_zero_point = __emule_quant_decode_fp32(zero_point);
}

ALWI void quant_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    __emule_dst_check(idst0, "quant_tile");
    __emule_dst_check(idst1, "quant_tile");
    __emule_dst_check(odst, "quant_tile");
    const float zp = __emule_compute_ctx().quant_zero_point;
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float a = __emule_compute_ctx().dst[idst0][i];
        float b = __emule_compute_ctx().dst[idst1][i];
        int32_t q = __emule_quant_fp32_to_int8_sat(a * b + zp);
        __emule_dst_store_i32(odst, i, q);
    }
}

ALWI void requant_tile_init(uint32_t zero_point) {
    __emule_compute_ctx().requant_zero_point = __emule_quant_decode_fp32(zero_point);
}

ALWI void requant_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    __emule_dst_check(idst0, "requant_tile");
    __emule_dst_check(idst1, "requant_tile");
    __emule_dst_check(odst, "requant_tile");
    const float zp = __emule_compute_ctx().requant_zero_point;
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a_int = __emule_dst_load_i32(idst0, i);
        float a = static_cast<float>(a_int);
        float b = __emule_compute_ctx().dst[idst1][i];
        int32_t q = __emule_quant_fp32_to_int8_sat(a * b + zp);
        __emule_dst_store_i32(odst, i, q);
    }
}

ALWI void dequant_tile_init(uint32_t zero_point) {
    // LLK convention: caller passes the negated zero point so the recorded
    // body can use SFPADD rather than SFPADD-with-negate. Mirror that here so
    // dequant_tile's add is correct without sign-flipping at the call site.
    __emule_compute_ctx().dequant_zero_point = __emule_quant_decode_fp32(zero_point);
}

ALWI void dequant_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    __emule_dst_check(idst0, "dequant_tile");
    __emule_dst_check(idst1, "dequant_tile");
    __emule_dst_check(odst, "dequant_tile");
    const float zp = __emule_compute_ctx().dequant_zero_point;
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t a_int = __emule_dst_load_i32(idst0, i);
        float a = static_cast<float>(a_int);
        float b = __emule_compute_ctx().dst[idst1][i];
        __emule_compute_ctx().dst[odst][i] = (a + zp) * b;
    }
}

}  // namespace ckernel
