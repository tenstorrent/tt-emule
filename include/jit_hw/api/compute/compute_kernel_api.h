// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Compute kernel API stub for emulated mode.
//
// tt-mlir's TTKernelIncludesMap.h maps a set of SFPU ops directly to
// `api/compute/compute_kernel_api.h` (rather than to a more specific
// per-op header). On real silicon those ops are pulled in through
// transitive includes; in tt-emule we declare them here directly.
#include "api/compute/common.h"
#include "api/compute/vector_mode.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "jit_hw/api/compute/experimental/fill_arange.h"
#include "api/compute/eltwise_unary/activations.h"  // abs_tile, abs_tile_int32
#include "api/compute/eltwise_unary/trigonometry.h"  // sole definer of tanh_tile (shared via #pragma once)

#include <algorithm>
// Minimal sfpi:: shim for SFPU kernels that use vFloat/vInt/v_if.
// TODO(#102): replace with real scalar semantics to produce correct outputs.
#include "jit_hw/sfpi.h"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <utility>

namespace ckernel {

// --- square (x*x) ---
ALWI void square_tile_init() {}
ALWI void square_tile(uint32_t idst) {
    __emule_dst_check(idst, "square_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = x * x;
    }
}

// --- exp2 (2^x) ---
ALWI void exp2_tile_init() {}
ALWI void exp2_tile(uint32_t idst) {
    __emule_dst_check(idst, "exp2_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::exp2(__emule_dst[idst][i]);
}

// --- expm1 (e^x - 1) ---
ALWI void expm1_tile_init() {}
ALWI void expm1_tile(uint32_t idst) {
    __emule_dst_check(idst, "expm1_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::expm1(__emule_dst[idst][i]);
}

// --- log ---
template <bool fast_and_approx = false>
ALWI void log_tile_init() {}
template <bool fast_and_approx = false>
ALWI void log_tile(uint32_t idst) {
    __emule_dst_check(idst, "log_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::log(__emule_dst[idst][i]);
}

// --- power (x^exponent_packed_float) ---
ALWI void power_tile_init() {}
ALWI void power_tile(uint32_t idst, uint32_t exponent_packed = 0) {
    __emule_dst_check(idst, "power_tile");
    if (exponent_packed == 0) {
        fprintf(stderr, "[EMULE] power_tile: exponent_packed=0; emule cannot "
                        "distinguish 'not provided' from packed 0.0f. Caller "
                        "must pass packed bits of the exponent.\n");
        std::abort();
    }
    float exponent;
    std::memcpy(&exponent, &exponent_packed, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::pow(__emule_dst[idst][i], exponent);
}

// --- sigmoid (1 / (1 + e^-x)) ---
template <uint32_t sigmoid_mode = 0>
ALWI void sigmoid_tile_init() {}
template <VectorMode vector_mode = VectorMode::RC, uint32_t sigmoid_mode = 0>
ALWI void sigmoid_tile(uint32_t idst) {
    __emule_dst_check(idst, "sigmoid_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = 1.0f / (1.0f + std::exp(-x));
    }
}

// --- sign (-1, 0, +1) ---
ALWI void sign_tile_init() {}
ALWI void sign_tile(uint32_t idst) {
    __emule_dst_check(idst, "sign_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = (x > 0.0f) - (x < 0.0f);
    }
}


// --- heaviside (step function; param0 is fp32 bit-pattern returned at x==0) ---
ALWI void heaviside_tile_init() {}
ALWI void heaviside_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "heaviside_tile");
    float value;
    std::memcpy(&value, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        if (x < 0.0f) { __emule_dst[idst][i] = 0.0f; }
        else if (x > 0.0f) { __emule_dst[idst][i] = 1.0f; }
        else { __emule_dst[idst][i] = value; }
    }
}

// --- signbit (1 if signed-bit set, else 0) ---
ALWI void signbit_tile_init() {}
ALWI void signbit_tile(uint32_t idst) {
    __emule_dst_check(idst, "signbit_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::signbit(__emule_dst[idst][i]) ? 1.0f : 0.0f;
}

// --- signbit int32 (1 if value negative, else 0) ---
ALWI void signbit_tile_int32_init() {}
ALWI void signbit_tile_int32(uint32_t idst) {
    __emule_dst_check(idst, "signbit_tile_int32");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, (v < 0) ? 1 : 0);
    }
}

// --- log_with_base (log(x) * scale; base_scale is bit-rep of 1/ln(base)) ---
template <bool fast_and_approx = false>
ALWI void log_with_base_tile_init() {}
template <bool fast_and_approx = false>
ALWI void log_with_base_tile(uint32_t idst, uint32_t base_scale) {
    __emule_dst_check(idst, "log_with_base_tile");
    float scale;
    std::memcpy(&scale, &base_scale, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::log(__emule_dst[idst][i]) * scale;
}

// --- power_iterative (x^n, n a non-negative integer via iterative multiply) ---
ALWI void power_iterative_tile_init() {}
ALWI void power_iterative_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "power_iterative_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        float result = 1.0f;
        for (uint32_t k = 0; k < param0; k++) result *= x;
        __emule_dst[idst][i] = result;
    }
}

// --- tiled_prod (cumulative product across the tile) ---
// emule uses row-major element order; SFPU lane order differs.
ALWI void tiled_prod_tile_init() {}
ALWI void tiled_prod_tile(uint32_t idst) {
    __emule_dst_check(idst, "tiled_prod_tile");
    float result = 1.0f;
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        result *= __emule_dst[idst][i];
        __emule_dst[idst][i] = result;
    }
}

// --- unary_max / unary_min (fp32, compared against fp32 bit-pattern param0) ---
ALWI void unary_max_tile_init() {}
ALWI void unary_max_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "unary_max_tile");
    float val;
    std::memcpy(&val, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::max(__emule_dst[idst][i], val);
}

ALWI void unary_min_tile_init() {}
ALWI void unary_min_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "unary_min_tile");
    float val;
    std::memcpy(&val, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::min(__emule_dst[idst][i], val);
}

// --- unary_max / unary_min (int32, param0 is the int32 value) ---
ALWI void unary_max_int32_tile_init() {}
ALWI void unary_max_int32_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "unary_max_int32_tile");
    int32_t val = (int32_t)param0;
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst_store_i32(idst, i, std::max(__emule_dst_load_i32(idst, i), val));
}

ALWI void unary_min_int32_tile_init() {}
ALWI void unary_min_int32_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "unary_min_int32_tile");
    int32_t val = (int32_t)param0;
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst_store_i32(idst, i, std::min(__emule_dst_load_i32(idst, i), val));
}

// --- unary_max / unary_min (uint32, DST slot reinterpreted as uint32) ---
ALWI void unary_max_uint32_tile_init() {}
ALWI void unary_max_uint32_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "unary_max_uint32_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        uint32_t x = (uint32_t)__emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, (int32_t)std::max(x, param0));
    }
}

ALWI void unary_min_uint32_tile_init() {}
ALWI void unary_min_uint32_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "unary_min_uint32_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        uint32_t x = (uint32_t)__emule_dst_load_i32(idst, i);
        __emule_dst_store_i32(idst, i, (int32_t)std::min(x, param0));
    }
}

// --- alt_complex_rotate90 (rotate adjacent 32-element SIMD chunk-pairs 90deg) ---
// Row-major chunk-pair interpretation: for chunk pair (a,b), new_a=-b, new_b=a.
ALWI void alt_complex_rotate90_tile_init() {}
ALWI void alt_complex_rotate90_tile(uint32_t idst) {
    __emule_dst_check(idst, "alt_complex_rotate90_tile");
    for (uint32_t base = 0; base < __EMULE_TILE_ELEMS; base += 64) {
        for (uint32_t j = 0; j < 32; j++) {
            float a = __emule_dst[idst][base + j];
            float b = __emule_dst[idst][base + 32 + j];
            __emule_dst[idst][base + j] = -b;
            __emule_dst[idst][base + 32 + j] = a;
        }
    }
}

// --- silu (x * sigmoid(x)) ---
ALWI void silu_tile_init() {}
ALWI void silu_tile(uint32_t idst) {
    __emule_dst_check(idst, "silu_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = x / (1.0f + std::exp(-x));
    }
}
// Upstream routes the same SFPU LLK on the PACK thread to overlap activation
// with the pack pipeline (matmul fused-activation, #79). emule has no
// MATH/PACK thread split — forward to the existing tile body.
ALWI void silu_tile_init_pack() { silu_tile_init(); }
ALWI void silu_tile_pack(uint32_t idst) { silu_tile(idst); }

// --- sfpu_reduce — SFPU-based reduction helper; emule reduces via other paths.
// No-op stubs let JIT compile. Upstream now calls these with non-type (enum)
// template args, e.g. sfpu_reduce<PoolType, reduce_format, ReduceDim>(idst, ct, rt)
// and sfpu_reduce_init<PoolType, reduce_format>(), so accept value template
// params plus any runtime args.
template <auto ...Vs, typename ...Args>
ALWI void sfpu_reduce(Args...) {}
template <auto ...Vs, typename ...Args>
ALWI void sfpu_reduce_init(Args...) {}

// --- topk (used by ttnn.topk via reduction/topk/device/kernels/compute/topk.cpp).
//
// DST layout per upstream contract:
//   DST[idst]     — data tile A (32 rows × 32 floats, row-major)
//   DST[idst+1]   — data tile B (used by merge/rebuild only)
//   DST[idst+2]   — index tile A (uint32_t reinterpreted as float in DST)
//   DST[idst+3]   — index tile B (merge/rebuild only)
//
// Sort axis is the COLUMN: the ttnn kernels run transpose_wh_tile on each tile
// first, moving the sort dim W onto the ROW index (DST[r*32+c] = input[H=c][W=r]),
// so topk along W is an independent sort over rows r at each fixed column c.
// SFPU does this with a bitonic network; emule stable_sorts (value,index) pairs.
// Strict comparator + stable_sort matches the SFPU no-swap-on-equal tie-break;
// ties pick a different-but-valid index (the test gathers indices for cosine and
// compares values, which are bf16-exact regardless of the tied index).
namespace __emule_topk {

// Sort the 64 datums of column `c` from tiles A (idst) ∪ B (idst+1) (+ index
// tiles idst+2/idst+3), top-32 → A, bottom-32 → B. `descending` = largest-first.
inline void merge_split_col(uint32_t idst, uint32_t c, bool descending) {
    uint32_t* idxA = reinterpret_cast<uint32_t*>(&__emule_dst[idst + 2][0]);
    uint32_t* idxB = reinterpret_cast<uint32_t*>(&__emule_dst[idst + 3][0]);
    std::pair<float, uint32_t> tmp[64];
    for (uint32_t r = 0; r < 32; ++r) {
        tmp[r]      = {__emule_dst[idst][r * 32 + c],     idxA[r * 32 + c]};
        tmp[32 + r] = {__emule_dst[idst + 1][r * 32 + c], idxB[r * 32 + c]};
    }
    std::stable_sort(tmp, tmp + 64, [descending](const auto& x, const auto& y) {
        return descending ? (x.first > y.first) : (x.first < y.first);
    });
    for (uint32_t r = 0; r < 32; ++r) {
        __emule_dst[idst][r * 32 + c]     = tmp[r].first;
        idxA[r * 32 + c]                  = tmp[r].second;
        __emule_dst[idst + 1][r * 32 + c] = tmp[32 + r].first;
        idxB[r * 32 + c]                  = tmp[32 + r].second;
    }
}

// Sort the 32 datums of a SINGLE value tile (idst) + index tile (idst+2) along
// column `c`. `descending` selects largest-first.
inline void sort_col(uint32_t idst, uint32_t c, bool descending) {
    uint32_t* idx = reinterpret_cast<uint32_t*>(&__emule_dst[idst + 2][0]);
    std::pair<float, uint32_t> tmp[32];
    for (uint32_t r = 0; r < 32; ++r) tmp[r] = {__emule_dst[idst][r * 32 + c], idx[r * 32 + c]};
    std::stable_sort(tmp, tmp + 32, [descending](const auto& x, const auto& y) {
        return descending ? (x.first > y.first) : (x.first < y.first);
    });
    for (uint32_t r = 0; r < 32; ++r) {
        __emule_dst[idst][r * 32 + c] = tmp[r].first;
        idx[r * 32 + c]               = tmp[r].second;
    }
}

} // namespace __emule_topk

ALWI void topk_tile_init() {}

// Local sort. i_end_phase >= 5 (single-core; multi-core K==64): A and B are one
// 64-element sequence per column → merge_split. i_end_phase < 5 (multi-core
// local stage): A and B are independent candidate sets → sort each in place
// (topk_merge/topk_rebuild combine them later). idir 0 = descending, 1 = ascending.
template <bool stable_sort = false>
ALWI void topk_local_sort(uint32_t idst, int idir, int i_end_phase,
                          int /*i_start_phase*/ = 0, int /*i_end_step*/ = 0,
                          int /*i_start_step*/ = 0) {
    __emule_dst_check(idst, "topk_local_sort.A");
    __emule_dst_check(idst + 1, "topk_local_sort.B");
    __emule_dst_check(idst + 2, "topk_local_sort.A_idx");
    __emule_dst_check(idst + 3, "topk_local_sort.B_idx");
    const bool desc = (idir == 0);
    if (i_end_phase >= 5) {
        for (uint32_t c = 0; c < 32; ++c) {
            __emule_topk::merge_split_col(idst, c, desc);
        }
    } else {
        for (uint32_t c = 0; c < 32; ++c) {
            __emule_topk::sort_col(idst, c, desc);
            __emule_topk::sort_col(idst + 1, c, desc);
        }
    }
}

// topk_merge (process_tiles) and topk_rebuild (process_tile_pair) both "move the
// larger 32 into the 0th dest, lower 32 into the 1st" (topk_common_funcs.hpp) —
// emulated as the order-independent full-sort merge_split, which converges to the
// global top-K over the kernel's reduction tree for all K without needing the
// SFPU's exact bitonic intermediate state. Template bool is `top_min` (driver
// passes false for largest, true for smallest).
template <bool top_min = false, bool stable_sort = false>
ALWI void topk_merge(uint32_t idst, int /*m_iter*/, int /*k*/) {
    __emule_dst_check(idst, "topk_merge.A");
    __emule_dst_check(idst + 1, "topk_merge.B");
    __emule_dst_check(idst + 2, "topk_merge.A_idx");
    __emule_dst_check(idst + 3, "topk_merge.B_idx");
    for (uint32_t c = 0; c < 32; ++c) {
        __emule_topk::merge_split_col(idst, c, /*descending=*/!top_min);
    }
}

// Rebuild: same merge-and-keep-larger-K as topk_merge, producing a fully sorted
// top-K run in A. `idir` (the driver's `ascending`): false → descending
// (largest), true → ascending (smallest). `skip_second`: when only the top-K
// run survives, B was not loaded — sort just the single tile A in place.
template <bool stable_sort = false>
ALWI void topk_rebuild(uint32_t idst, bool idir, int /*m_iter*/, int /*k*/,
                       int /*logk*/, int skip_second) {
    __emule_dst_check(idst, "topk_rebuild.A");
    __emule_dst_check(idst + 2, "topk_rebuild.A_idx");
    const bool desc = !idir;  // idir false → descending
    if (skip_second) {
        for (uint32_t c = 0; c < 32; ++c) {
            __emule_topk::sort_col(idst, c, desc);
        }
    } else {
        __emule_dst_check(idst + 1, "topk_rebuild.B");
        __emule_dst_check(idst + 3, "topk_rebuild.B_idx");
        for (uint32_t c = 0; c < 32; ++c) {
            __emule_topk::merge_split_col(idst, c, desc);
        }
    }
}

} // namespace ckernel
