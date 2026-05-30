// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tt-emule stub: bcast (no-op in emulation)
#pragma once

#include "api/compute/common.h"

namespace ckernel {

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void init_bcast(uint32_t icb0 = 0, uint32_t icb1 = 1) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void add_bcast_rows_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void add_bcast_cols_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void add_bcast_scalar_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void any_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void add_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void sub_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void mul_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {}

// `unary_bcast<BroadcastType>` — broadcasts a single tile across DST positions.
// Added by tt-mlir PR #7926 era D2M codegen for scalar/row/col broadcasts.
// On emule we don't track per-thread DST layout; treat as copy_tile.
template <BroadcastType BCAST_T>
inline void unary_bcast_init(uint32_t /*icb0*/, uint32_t /*icb1*/) {}

template <BroadcastType BCAST_T>
inline void unary_bcast(uint32_t icb, uint32_t in_tile_index, uint32_t idst) {
    copy_tile(icb, in_tile_index, idst);
}

// Single-template-arg overloads matching upstream bcast.h:408-412 — the
// `template <BroadcastType tBcastDim>` form, op_type implicit:
// upstream's signature is `mul_tiles_bcast<COL>(...)` which forwards to
// `any_tiles_bcast<EltwiseBinaryType::ELWMUL, COL>(...)`. softmax.cpp:303
// calls this form.
template <BroadcastType BCAST_T>
ALWI void add_tiles_bcast(uint32_t /*icb0*/, uint32_t /*icb1*/,
                          uint32_t /*itile0*/, uint32_t /*itile1*/,
                          uint32_t /*idst*/, uint32_t /*bcast_row_idx*/ = 0) {}
template <BroadcastType BCAST_T>
ALWI void sub_tiles_bcast(uint32_t /*icb0*/, uint32_t /*icb1*/,
                          uint32_t /*itile0*/, uint32_t /*itile1*/,
                          uint32_t /*idst*/, uint32_t /*bcast_row_idx*/ = 0) {}
template <BroadcastType BCAST_T>
ALWI void mul_tiles_bcast(uint32_t /*icb0*/, uint32_t /*icb1*/,
                          uint32_t /*itile0*/, uint32_t /*itile1*/,
                          uint32_t /*idst*/, uint32_t /*bcast_row_idx*/ = 0) {}

// ---- Non-templated row/col/scalar variants (upstream signatures) ----
//
// Upstream tt_metal/hw/inc/api/compute/bcast.h provides specific entry points
// like `add_tiles_bcast_rows(icb0, icb1, itile0, itile1, idst)` and
// `add_bcast_rows_init_short(icb0, icb1, call_line)` that kernels call
// WITHOUT a template arg (bmm_large_block_zm_fused_bias_activation.cpp lines
// 455, 474). The generic templated forms above can't deduce template args
// from those call sites — clang reports "couldn't infer template argument
// 'op_type'". Add the specific non-templated variants matching upstream
// signatures; bodies are no-ops because host doesn't model bcast.

ALWI void add_bcast_rows_init_short(uint32_t /*icb0*/, uint32_t /*icb1*/,
                                    uint32_t /*call_line*/ = 0) {}
ALWI void sub_bcast_rows_init_short(uint32_t /*icb0*/, uint32_t /*icb1*/,
                                    uint32_t /*call_line*/ = 0) {}
ALWI void mul_bcast_rows_init_short(uint32_t /*icb0*/, uint32_t /*icb1*/,
                                    uint32_t /*call_line*/ = 0) {}
ALWI void add_bcast_cols_init_short(uint32_t /*icb0*/, uint32_t /*icb1*/,
                                    uint32_t /*call_line*/ = 0) {}
ALWI void sub_bcast_cols_init_short(uint32_t /*icb0*/, uint32_t /*icb1*/,
                                    uint32_t /*call_line*/ = 0) {}
ALWI void mul_bcast_cols_init_short(uint32_t /*icb0*/, uint32_t /*icb1*/,
                                    uint32_t /*call_line*/ = 0) {}
ALWI void add_bcast_scalar_init_short(uint32_t /*icb0*/, uint32_t /*icb1*/,
                                      uint32_t /*call_line*/ = 0) {}
ALWI void mul_tiles_bcast_scalar_init_short(uint32_t /*icb0*/, uint32_t /*icb1*/,
                                            uint32_t /*call_line*/ = 0) {}
ALWI void sub_tiles_bcast_scalar_init_short(uint32_t /*icb0*/, uint32_t /*icb1*/,
                                            uint32_t /*call_line*/ = 0) {}

ALWI void add_tiles_bcast_rows(uint32_t /*icb0*/, uint32_t /*icb1*/,
                               uint32_t /*itile0*/, uint32_t /*itile1*/,
                               uint32_t /*idst*/,
                               uint32_t /*bcast_row_idx*/ = 0) {}
ALWI void sub_tiles_bcast_rows(uint32_t /*icb0*/, uint32_t /*icb1*/,
                               uint32_t /*itile0*/, uint32_t /*itile1*/,
                               uint32_t /*idst*/,
                               uint32_t /*bcast_row_idx*/ = 0) {}
ALWI void mul_tiles_bcast_rows(uint32_t /*icb0*/, uint32_t /*icb1*/,
                               uint32_t /*itile0*/, uint32_t /*itile1*/,
                               uint32_t /*idst*/,
                               uint32_t /*bcast_row_idx*/ = 0) {}
ALWI void add_tiles_bcast_cols(uint32_t /*icb0*/, uint32_t /*icb1*/,
                               uint32_t /*itile0*/, uint32_t /*itile1*/,
                               uint32_t /*idst*/) {}
ALWI void sub_tiles_bcast_cols(uint32_t /*icb0*/, uint32_t /*icb1*/,
                               uint32_t /*itile0*/, uint32_t /*itile1*/,
                               uint32_t /*idst*/) {}
ALWI void mul_tiles_bcast_cols(uint32_t /*icb0*/, uint32_t /*icb1*/,
                               uint32_t /*itile0*/, uint32_t /*itile1*/,
                               uint32_t /*idst*/) {}
ALWI void add_tiles_bcast_scalar(uint32_t /*icb0*/, uint32_t /*icb1*/,
                                 uint32_t /*itile0*/, uint32_t /*itile1*/,
                                 uint32_t /*idst*/) {}
ALWI void sub_tiles_bcast_scalar(uint32_t /*icb0*/, uint32_t /*icb1*/,
                                 uint32_t /*itile0*/, uint32_t /*itile1*/,
                                 uint32_t /*idst*/) {}
ALWI void mul_tiles_bcast_scalar(uint32_t /*icb0*/, uint32_t /*icb1*/,
                                 uint32_t /*itile0*/, uint32_t /*itile1*/,
                                 uint32_t /*idst*/) {}

}  // namespace ckernel
