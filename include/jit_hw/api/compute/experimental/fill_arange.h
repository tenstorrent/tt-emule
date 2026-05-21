// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulation stub for experimental::fill_arange_tile<DataFormat>(cb_id).
//
// Writes values [0, 1, 2, ..., 1023] into page 0 of the given CB in nfaces
// layout, matching what pack_dst_to_buf would produce for a DST tile that
// contains float(i) at row-major index i.  A subsequent copy_tile(cb_id, 0,
// dst_slot) therefore unpacks to DST[i] = float(i) for i in [0, 1023].
//
// This implements the "index tile" produced by the D2M arange decomposition
// (DecomposeArange.cpp → FillArangeTileOp → ExperimentalFillArangeTileOp).
// The calling sequence in a generated compute kernel is:
//
//   experimental::fill_arange_tile<DataFormat::Float16_b>(scratch_cb);
//   copy_tile(scratch_cb, 0, dst);          // DST[i] = float(i)
//   binop_with_scalar_tile_init();
//   add_unary_tile(dst, tile_offset_bits);  // DST[i] += tile_offset
//   binop_with_scalar_tile_init();
//   mul_unary_tile(dst, step_bits);         // DST[i] *= step
//   binop_with_scalar_tile_init();
//   add_unary_tile(dst, start_bits);        // DST[i] += start
//   pack_tile<true>(dst, out_cb, out_slot);

#include "jit_hw/api/compute/common.h"

namespace experimental {

template <DataFormat DATA_FORMAT = DataFormat::Float16_b>
ALWI void fill_arange_tile(uint32_t cb_id) {
    uint8_t* buf = __emule_compute::cb_write_ptr_at(cb_id, 0);

    if constexpr (DATA_FORMAT == DataFormat::Float32) {
        float* out = reinterpret_cast<float*>(buf);
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            out[ni] = static_cast<float>(i);
        }
    } else if constexpr (DATA_FORMAT == DataFormat::Int32 ||
                         DATA_FORMAT == DataFormat::UInt32) {
        uint32_t* out = reinterpret_cast<uint32_t*>(buf);
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            out[ni] = i;
        }
    } else {
        // Float16_b (bfloat16) and other 16-bit formats.
        uint16_t* out = reinterpret_cast<uint16_t*>(buf);
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            out[ni] = __emule_bf16::from_f32(static_cast<float>(i));
        }
    }
}

} // namespace experimental
