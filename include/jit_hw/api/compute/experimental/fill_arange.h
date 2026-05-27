// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator override for the experimental padding / arange LLK snippet that
// TTKernelToCpp embeds verbatim into generated D2M compute kernels.
//
// The real LLK (include/ttmlir/Target/TTKernel/LLKs/experimental_padding_llks.h)
// guards all function bodies with `#ifdef TRISC_UNPACK`, which is never
// defined in the emulator (all TRISC threads run in one host thread).
// Defining the LLK's own header guard here causes the verbatim snippet to be
// silently skipped — so only these software implementations are used.

// Block the hardware LLK verbatim snippet so our definitions win.
#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PADDING_LLKS_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_PADDING_LLKS_H
#endif

#include "jit_hw/api/compute/common.h"

namespace experimental {

// fill_arange_tile<DataFormat>(cb_id)
//
// Writes values [0, 1, ..., 1023] into page 0 of the given CB in nfaces
// layout, so that a subsequent copy_tile(cb_id, 0, dst) unpacks to
// DST[i] = float(i) in row-major order.
//
// D2M arange flow:
//   fill_arange_tile(scratch_cb)            -- write index tile [0..1023]
//   copy_tile(scratch_cb, 0, dst)           -- unpack to DST
//   add_unary_tile(dst, tile_offset_bits)   -- DST[i] += tile_offset
//   mul_unary_tile(dst, step_bits)          -- DST[i] *= step
//   add_unary_tile(dst, start_bits)         -- DST[i] += start
//   pack_tile(dst, out_cb)                  -- write result
template <DataFormat DATA_FORMAT = DataFormat::Float16_b>
ALWI void fill_arange_tile(uint32_t cb_id) {
    uint8_t* buf = __emule_compute::cb_read_ptr_at(cb_id, 0);
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
        // Float16_b and other 16-bit formats.
        uint16_t* out = reinterpret_cast<uint16_t*>(buf);
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            out[ni] = __emule_bf16::from_f32(static_cast<float>(i));
        }
    }
}

// write_row_mask_tile<DataFormat>(validRows, cb_id)
//
// Writes a mask tile to page 0 of the given CB:
//   row < validRows  →  1.0 (or 1 for Int32)
//   row >= validRows →  0.0 (or 0 for Int32)
// The tile is written in nfaces layout.
template <DataFormat DATA_FORMAT = DataFormat::Float32>
ALWI void write_row_mask_tile(uint32_t validRows, uint32_t cb_id) {
    uint8_t* buf = __emule_compute::cb_read_ptr_at(cb_id, 0);
    if constexpr (DATA_FORMAT == DataFormat::Float32) {
        float* out = reinterpret_cast<float*>(buf);
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            out[ni] = (i / 32 < validRows) ? 1.0f : 0.0f;
        }
    } else if constexpr (DATA_FORMAT == DataFormat::Int32) {
        uint32_t* out = reinterpret_cast<uint32_t*>(buf);
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            out[ni] = (i / 32 < validRows) ? 1u : 0u;
        }
    } else {
        // Float16_b
        uint16_t* out = reinterpret_cast<uint16_t*>(buf);
        constexpr uint16_t one_bf16 = 0x3F80u; // 1.0 in bfloat16
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            out[ni] = (i / 32 < validRows) ? one_bf16 : uint16_t{0};
        }
    }
}

// write_col_mask_tile<DataFormat>(validCols, cb_id)
//
// Writes a mask tile to page 0 of the given CB:
//   col < validCols  →  1.0 (or 1 for Int32)
//   col >= validCols →  0.0 (or 0 for Int32)
// The tile is written in nfaces layout.
template <DataFormat DATA_FORMAT = DataFormat::Float32>
ALWI void write_col_mask_tile(uint32_t validCols, uint32_t cb_id) {
    uint8_t* buf = __emule_compute::cb_read_ptr_at(cb_id, 0);
    if constexpr (DATA_FORMAT == DataFormat::Float32) {
        float* out = reinterpret_cast<float*>(buf);
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            out[ni] = (i % 32 < validCols) ? 1.0f : 0.0f;
        }
    } else if constexpr (DATA_FORMAT == DataFormat::Int32) {
        uint32_t* out = reinterpret_cast<uint32_t*>(buf);
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            out[ni] = (i % 32 < validCols) ? 1u : 0u;
        }
    } else {
        // Float16_b
        uint16_t* out = reinterpret_cast<uint16_t*>(buf);
        constexpr uint16_t one_bf16 = 0x3F80u;
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            out[ni] = (i % 32 < validCols) ? one_bf16 : uint16_t{0};
        }
    }
}

} // namespace experimental
