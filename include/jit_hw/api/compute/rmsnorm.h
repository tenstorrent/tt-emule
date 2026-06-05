// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Layer-1 emule shadow for the RMSNorm algorithm used by the upstream kernel.
//
// Silicon implements RMSNorm via a sequence of ~15 LLK calls (Layer-2,
// below the kernel-author tile-op API): mul_reduce_scalar_init / _uninit,
// add_rsqrt_tile, rmsnorm_mul_bcast_scalar_reuse_tiles, binary_dest_reuse_tiles,
// plus llk_unpack_AB, llk_math_eltwise_mul_reduce_scalar,
// llk_math_mul_reduce_column, llk_pack_reduce_mask_config, etc. The op.hpp
// body itself drops below Layer 1 — a pure Layer-1 lift would require
// shadowing the full LLK reduce-rsqrt chain.
//
// This partial lift packages the rmsnorm algorithm as a single Layer-1
// helper. The consumer op.hpp's `#ifdef __EMULE_JIT_MODE` block becomes
// one call to `rmsnorm_compute`. Algorithm: x / sqrt(mean(x²) + eps) * gamma,
// with optional padding (output may be wider than input).

#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/bfloat16.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace ckernel {

// Whole-op RMSNorm against direct CB pointers. Bypasses DST + LLK chain.
// `input_cb`: bf16 CB with `num_tiles` tiles of input.
// `gamma_cb` : bf16 CB with `num_tiles` tiles of per-element scale.
// `output_cb`: bf16 CB to receive `num_tiles` tiles of output (may be
//               wider than `input_cb` for padded variants).
// `epsilon_bits`: fp32 epsilon as raw bits (compile-time constant on caller).
//
// Caller is responsible for `cb_wait_front(input/gamma)` and any
// `cb_push_back / cb_pop_front` framing; this helper does only the
// arithmetic + the direct CB pointer writes.
// `valid_count_override`: if non-zero, compute mean over this many input
// elements instead of `total_elems_in`. Used by padded_rmsnorm where the
// last tile is zero-padded and contributes zeros to sum(x²) — those zeros
// shouldn't bias the divisor.
inline void rmsnorm_compute_impl(uint32_t input_cb, uint32_t gamma_cb,
                                 uint32_t output_cb, uint32_t num_tiles,
                                 uint32_t epsilon_bits,
                                 uint32_t valid_count_override = 0) {
    const uint32_t input_tile_bytes  = get_tile_size(input_cb);
    const uint32_t output_tile_bytes = get_tile_size(output_cb);
    const uint32_t gamma_tile_bytes  = get_tile_size(gamma_cb);
    const uint32_t total_elems_in  = (num_tiles * input_tile_bytes)  / 2;
    const uint32_t total_elems_g   = (num_tiles * gamma_tile_bytes)  / 2;
    const uint32_t total_elems_out = (num_tiles * output_tile_bytes) / 2;

    const uint16_t* in_ptr = reinterpret_cast<const uint16_t*>(
        static_cast<uintptr_t>(get_read_ptr(input_cb)));
    const uint16_t* g_ptr = reinterpret_cast<const uint16_t*>(
        static_cast<uintptr_t>(get_read_ptr(gamma_cb)));
    uint16_t* out_ptr = reinterpret_cast<uint16_t*>(
        static_cast<uintptr_t>(get_write_ptr(output_cb)));

    float eps;
    {
        uint32_t bits = epsilon_bits;
        std::memcpy(&eps, &bits, sizeof(float));
    }

    // sum(x²) in f64 for numerical headroom; libm sqrt for rsqrt; bf16
    // round-trip on store. Output beyond `total_elems_in` is zero (handles
    // padded variants).
    const uint32_t valid_count =
        (valid_count_override > 0) ? valid_count_override : total_elems_in;
    double sum_sq = 0.0;
    for (uint32_t i = 0; i < valid_count; i++) {
        float v = __emule_bf16::to_f32(in_ptr[i]);
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
    }
    const double mean_sq = sum_sq / static_cast<double>(valid_count);
    const float inv_rms = static_cast<float>(
        1.0 / std::sqrt(mean_sq + static_cast<double>(eps)));

    for (uint32_t j = 0; j < total_elems_out; j++) {
        float x = (j < total_elems_in) ? __emule_bf16::to_f32(in_ptr[j]) : 0.0f;
        float g = (j < total_elems_g)  ? __emule_bf16::to_f32(g_ptr[j])  : 0.0f;
        out_ptr[j] = __emule_bf16::from_f32(x * inv_rms * g);
    }
}

} // namespace ckernel
