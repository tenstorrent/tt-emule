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
    // UNPACK source tile: nfaces→row-major + format conversion
    float src[1024];
    {
        uint8_t* buf = __emule_compute::cb_read_ptr_at(icb, itile);
        if (__emule_compute::cb_is_32bit_format(icb)) {
            const float* fbuf = reinterpret_cast<const float*>(buf);
            for (uint32_t i = 0; i < 1024; i++)
                src[i] = fbuf[__emule_nfaces::rowmajor_to_nfaces[i]];
        } else {
            uint16_t* bf = reinterpret_cast<uint16_t*>(buf);
            for (uint32_t i = 0; i < 1024; i++)
                src[i] = __emule_bf16::to_f32(bf[__emule_nfaces::rowmajor_to_nfaces[i]]);
        }
    }

    // UNPACK scaler tile: nfaces→row-major (scaler is uniform, but apply for correctness)
    float scaler = 1.0f;
    {
        uint8_t* sbuf = __emule_compute::cb_read_ptr_at(icb_scaler, itile_scaler);
        if (__emule_compute::cb_is_32bit_format(icb_scaler)) {
            const float* fsbuf = reinterpret_cast<const float*>(sbuf);
            scaler = fsbuf[__emule_nfaces::rowmajor_to_nfaces[0]];
        } else {
            uint16_t* sbf = reinterpret_cast<uint16_t*>(sbuf);
            scaler = __emule_bf16::to_f32(sbf[__emule_nfaces::rowmajor_to_nfaces[0]]);
        }
    }

    if constexpr (reduce_dim == ReduceDim::REDUCE_COL) {
        // Reduce columns: for each column c, sum/max across all rows → result in row 0
        for (uint32_t c = 0; c < 32; c++) {
            float acc;
            if constexpr (reduce_type == PoolType::MAX)
                acc = -std::numeric_limits<float>::infinity();
            else
                acc = 0.0f;

            for (uint32_t r = 0; r < 32; r++) {
                float val = src[r * 32 + c];
                if constexpr (reduce_type == PoolType::MAX)
                    acc = std::max(acc, val);
                else
                    acc += val;  // SUM and AVG both sum; scaler handles the 1/N
            }
            // Accumulate into DST (supports multi-tile reduction)
            float result = acc * scaler;
            if constexpr (reduce_type == PoolType::MAX)
                __emule_dst[idst][c] = std::max(__emule_dst[idst][c], result);
            else
                __emule_dst[idst][c] += result;
        }
    } else if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
        // Reduce rows: for each row r, sum/max across all cols → result in col 0
        for (uint32_t r = 0; r < 32; r++) {
            float acc;
            if constexpr (reduce_type == PoolType::MAX)
                acc = -std::numeric_limits<float>::infinity();
            else
                acc = 0.0f;

            for (uint32_t c = 0; c < 32; c++) {
                float val = src[r * 32 + c];
                if constexpr (reduce_type == PoolType::MAX)
                    acc = std::max(acc, val);
                else
                    acc += val;
            }
            // Result goes in column 0 of each row
            float result = acc * scaler;
            if constexpr (reduce_type == PoolType::MAX)
                __emule_dst[idst][r * 32] = std::max(__emule_dst[idst][r * 32], result);
            else
                __emule_dst[idst][r * 32] += result;
        }
    } else {
        // REDUCE_SCALAR: reduce all 1024 elements to a single value
        float acc;
        if constexpr (reduce_type == PoolType::MAX)
            acc = -std::numeric_limits<float>::infinity();
        else
            acc = 0.0f;

        for (uint32_t i = 0; i < 1024; i++) {
            if constexpr (reduce_type == PoolType::MAX)
                acc = std::max(acc, src[i]);
            else
                acc += src[i];
        }
        float result = acc * scaler;
        if constexpr (reduce_type == PoolType::MAX)
            __emule_dst[idst][0] = std::max(__emule_dst[idst][0], result);
        else
            __emule_dst[idst][0] += result;
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
