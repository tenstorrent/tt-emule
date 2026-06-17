// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Reduce compute stub for emulated mode
// Implements reduce_tile with actual SUM/MAX math across ROW/COL/SCALAR dimensions.
//
// D2M-generated code calls:
//   reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL, false>(icb, icb_scaler, ocb);
//   reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL, false>(icb, icb_scaler, itile, itile_scaler, idst);
//   reduce_uninit<false>();
//
// The scaler CB contains a tile where every element = 1.0 (for SUM) or 1/N (for AVG).
// Results ACCUMULATE into DST (+=) to support multi-tile reductions.

#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/compute/nfaces.h"
#include "jit_hw/api/compute/compute_kernel_hw_startup.h"
#include "jit_hw/llk_defs.h"
#include <algorithm>
#include <limits>

// Default REDUCE_OP / REDUCE_DIM if not provided as JIT defines
#ifndef REDUCE_OP
#define REDUCE_OP PoolType::SUM
#endif
#ifndef REDUCE_DIM
#define REDUCE_DIM ReduceDim::REDUCE_COL
#endif

namespace ckernel {

// ---- Init / uninit (hardware pipeline config — no-ops in emulation) ----

template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false>
inline void reduce_init(uint32_t icb, uint32_t icb_scaler, uint32_t ocb, uint32_t = 0) {}

template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false>
inline void reduce_init_short(uint32_t icb, uint32_t icb_scaler = 0, uint32_t ocb = 0) {}

template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false>
inline void reduce_init_delta(uint32_t icb = 0, uint32_t icb_scaler = 0, uint32_t ocb = 0) {}

template <bool enforce_fp32_accumulation = false>
inline void reduce_uninit(uint32_t icb = 0) {}

inline void reduce_revert_delta(uint32_t ocb = 0) {}

// ---- reduce_tile: the core reduction operation ----
//
// Reads source tile from CB[icb][itile] and scaler tile from CB[icb_scaler][itile_scaler].
// Reduces along the specified dimension and ACCUMULATES into DST[idst].
//
// Output layout (matches real hardware):
//   REDUCE_ROW:    result[r][0] = reduce(src[r][0..31]) * scaler  (column vector, col 0 of each row)
//   REDUCE_COL:    result[0][c] = reduce(src[0..31][c]) * scaler  (row vector, row 0 of each col)
//   REDUCE_SCALAR: result[0][0] = reduce(all 1024 elements) * scaler  (single value)
//
// Non-result positions are zeroed on first write, but since DST is zeroed by
// tile_regs_acquire, only the result positions are written (accumulate).

template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false>
inline void reduce_tile(uint32_t icb, uint32_t icb_scaler,
                        uint32_t itile, uint32_t itile_scaler, uint32_t idst) {
    __emule_dst_check(idst, "reduce_tile");
    // UNPACK source tile via the central format-aware reader (fp32/bf16/int32/
    // uint16/Bfp8_b/Bfp4_b → row-major float).
    float src[1024];
    __emule_unpack_cb_tile_to(icb, itile, src);

    // UNPACK the scaler tile. Usually uniform (element 0 is the scale factor), but a
    // partial last tile (W % 32 != 0) carries a per-position MASK: scale value in the
    // valid reduce-axis positions, 0 in the padding. Silicon contracts the source
    // against this tile element-wise (GAPOOL/GMPOOL), so keep the whole tile rather
    // than just scaler[0] — that drops the padding garbage from the reduction.
    float scaler_tile[1024];
    __emule_unpack_cb_tile_to(icb_scaler, itile_scaler, scaler_tile);
    const float scaler = scaler_tile[0];

    // Real HW leaves DST undefined after tile_regs_acquire and the kernel is
    // responsible for staging multi-tile accumulation (e.g. via copy_tile from
    // a cb_acc CB).  emule zeroes DST in tile_regs_acquire (so other ops can
    // safely += into it), which breaks reduce_tile<MAX> when all input values
    // are negative — max(0, neg) clamps to 0.  Examples that hit this:
    // ttnn::min lowers to `min(x) = -max(-x)` via reduce_w_neg.cpp /
    // reduce_h_neg.cpp; the inner max sees only negative values.
    //
    // Fix: on the first reduce_tile call after tile_regs_acquire (slot is
    // "fresh"), overwrite DST at the result positions with the per-tile
    // result.  Subsequent calls see the slot dirty and max-accumulate against
    // the already-written value (or against the running accumulator the
    // kernel staged via copy_tile).
    const bool fresh = __emule_dst_take_fresh(idst);

    // Bound the reduction by the input CB tile's active region. On a full 32×32
    // tile th == tw == 32 (bit-identical to the historical hardcoded loops); on a
    // thin tile (e.g. Tile([1,16])) the inactive lanes carry uninitialized unpack
    // values that would otherwise pollute the reduction (e.g. an all-zero result
    // from a thin-tile reduction). The tile-local layout is still row-major over a
    // 32-stride, so the active region is rows [0,th) × cols [0,tw).
    const uint32_t th = get_tile_r_dim(icb);
    const uint32_t tw = get_tile_c_dim(icb);

    if constexpr (reduce_dim == ReduceDim::REDUCE_COL) {
        // Reduce over rows → row 0 of each column. scaler_tile[r] weights row r (see unpack note).
        for (uint32_t c = 0; c < tw; c++) {
            float result;
            if constexpr (reduce_type == PoolType::MAX) {
                float acc = -std::numeric_limits<float>::infinity();
                for (uint32_t r = 0; r < th; r++)
                    if (scaler_tile[r] != 0.0f) acc = std::max(acc, src[r * 32 + c]);
                result = acc * scaler;
            } else {  // SUM / AVG: fold the per-row scaler (scale + mask) into the sum
                float acc = 0.0f;
                for (uint32_t r = 0; r < th; r++)
                    acc += src[r * 32 + c] * scaler_tile[r];
                result = acc;
            }
            if constexpr (reduce_type == PoolType::MAX) {
                __emule_dst[idst][c] = fresh ? result : std::max(__emule_dst[idst][c], result);
            } else {
                __emule_dst[idst][c] += result;
            }
        }
        // Zero inactive output lanes on the fresh first write only, so downstream
        // broadcast / elementwise ops see deterministic 0 — but never clobber an
        // in-progress multi-tile accumulation.
        if (fresh)
            for (uint32_t c = tw; c < 32; c++) __emule_dst[idst][c] = 0.0f;
    } else if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
        // Reduce over columns → col 0 of each row. scaler_tile[c] weights column c (see unpack note).
        for (uint32_t r = 0; r < th; r++) {
            float result;
            if constexpr (reduce_type == PoolType::MAX) {
                float acc = -std::numeric_limits<float>::infinity();
                for (uint32_t c = 0; c < tw; c++)
                    if (scaler_tile[c] != 0.0f) acc = std::max(acc, src[r * 32 + c]);
                result = acc * scaler;
            } else {  // SUM / AVG: fold the per-column scaler (scale + mask) into the sum
                float acc = 0.0f;
                for (uint32_t c = 0; c < tw; c++)
                    acc += src[r * 32 + c] * scaler_tile[c];
                result = acc;
            }
            if constexpr (reduce_type == PoolType::MAX) {
                __emule_dst[idst][r * 32] = fresh ? result : std::max(__emule_dst[idst][r * 32], result);
            } else {
                __emule_dst[idst][r * 32] += result;
            }
        }
        // Zero inactive result rows (col 0) on the fresh first write only.
        if (fresh)
            for (uint32_t r = th; r < 32; r++) __emule_dst[idst][r * 32] = 0.0f;
    } else {
        // REDUCE_SCALAR: reduce the active region to a single value
        float acc;
        if constexpr (reduce_type == PoolType::MAX)
            acc = -std::numeric_limits<float>::infinity();
        else
            acc = 0.0f;

        for (uint32_t r = 0; r < th; r++)
            for (uint32_t c = 0; c < tw; c++) {
                float val = src[r * 32 + c];
                if constexpr (reduce_type == PoolType::MAX)
                    acc = std::max(acc, val);
                else
                    acc += val;
            }
        // Silicon's single-core HW reduce uses REDUCE_SCALAR, which applies the
        // scaler TWICE internally (once per dimension); the host pre-compensates
        // with sqrt(scaler) (reduce_op_single_core_hw_program_factory.cpp:48). So
        // square it here to match — applying it once over-counts by 1/sqrt(N).
        float result = acc * scaler * scaler;
        if constexpr (reduce_type == PoolType::MAX) {
            __emule_dst[idst][0] = fresh ? result : std::max(__emule_dst[idst][0], result);
        } else {
            __emule_dst[idst][0] += result;
        }
    }
}

// reduce_tile_math: intentional no-op — emulation performs math in reduce_tile directly.
template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false>
inline void reduce_tile_math(uint32_t idst, uint32_t num_faces = 4) {}

// ---- Non-template overloads for backward compatibility ----
inline void reduce_init(uint32_t icb = 0, uint32_t ocb = 0) {}
inline void reduce_init_short(uint32_t icb = 0, uint32_t ocb = 0) {}
inline void reduce_init_delta(uint32_t icb = 0, uint32_t ocb = 0) {}

} // namespace ckernel

using namespace ckernel;
