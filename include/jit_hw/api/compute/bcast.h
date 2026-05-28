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

// ===========================================================================
// Non-templated tile-level broadcast wrappers
// ===========================================================================
//
// Real tt-metal exposes these in `tt_metal/hw/inc/api/compute/bcast.h`
// (e.g. mul_tiles_bcast_cols at line 201, add_tiles_bcast_rows at line 228).
// They wrap UNPACK+MATH llk_* calls that read CB[icb1] with a broadcast
// pattern: for `_cols` variants, the value at column 0 of cb_b's tile is
// replicated across all columns; for `_rows` variants, the value at the
// given `bcast_row_idx` row of cb_b is replicated across all rows. The
// kernels in `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/*.cpp`
// invoke these (non-templated) names directly; emule's existing
// `mul_tiles_bcast<op, bcast>` templated form doesn't match.
//
// Host implementation mirrors the row-major→nfaces walk used by emule's
// add_tiles / mul_tiles in common.h:232-301.

namespace __emule_bcast {

// Broadcast index pattern: given the row-major index `i` in the OUTPUT tile
// (0..1023, with r = i>>5, c = i & 31), what row-major index in cb_b should
// we read?
//   bcast_cols: r*32 + 0   (broadcast col 0 across all cols of the row)
//   bcast_rows: bcast_row_idx*32 + c   (broadcast a specific row across all rows)
constexpr uint32_t bcast_cols_src_idx(uint32_t i) {
    return i & ~31u;  // zero out the column bits
}
constexpr uint32_t bcast_rows_src_idx(uint32_t i, uint32_t bcast_row_idx) {
    return (bcast_row_idx << 5) | (i & 31u);
}

// Common driver: for each i in [0, n), dst[i] = op(a[ni(i)], b[ni(bcast(i))])
// where ni() is the row-major→nfaces remap.
template <typename Op, typename BcastFn>
inline void apply_bcast(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1,
    uint32_t idst, BcastFn bcast_fn) {
    if (__emule_compute::cb_is_32bit_format(icb0)) {
        const float* buf0 = reinterpret_cast<const float*>(__emule_compute::cb_read_ptr_at(icb0, itile0));
        const float* buf1 = reinterpret_cast<const float*>(__emule_compute::cb_read_ptr_at(icb1, itile1));
        uint32_t n = __emule_compute::cb_page_size(icb0) / sizeof(float);
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t ni_a = __emule_nfaces::rowmajor_to_nfaces[i];
            uint32_t ni_b = __emule_nfaces::rowmajor_to_nfaces[bcast_fn(i)];
            __emule_dst[idst][i] = Op{}(buf0[ni_a], buf1[ni_b]);
        }
    } else {
        uint16_t* buf0 = reinterpret_cast<uint16_t*>(__emule_compute::cb_read_ptr_at(icb0, itile0));
        uint16_t* buf1 = reinterpret_cast<uint16_t*>(__emule_compute::cb_read_ptr_at(icb1, itile1));
        uint32_t n = __emule_compute::cb_tile_elems(icb0);
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t ni_a = __emule_nfaces::rowmajor_to_nfaces[i];
            uint32_t ni_b = __emule_nfaces::rowmajor_to_nfaces[bcast_fn(i)];
            __emule_dst[idst][i] = Op{}(__emule_bf16::to_f32(buf0[ni_a]),
                                        __emule_bf16::to_f32(buf1[ni_b]));
        }
    }
}

struct __op_add { float operator()(float a, float b) const { return a + b; } };
struct __op_sub { float operator()(float a, float b) const { return a - b; } };
struct __op_mul { float operator()(float a, float b) const { return a * b; } };

}  // namespace __emule_bcast

// ---- _init_short stubs (non-templated; emule's compute path is format-agnostic) ----
ALWI void mul_bcast_cols_init_short(uint32_t /*icb0*/, uint32_t /*icb1*/) {}
ALWI void mul_bcast_rows_init_short(uint32_t /*icb0*/, uint32_t /*icb1*/) {}
ALWI void add_bcast_rows_init_short(uint32_t /*icb0*/, uint32_t /*icb1*/) {}
ALWI void sub_bcast_cols_init_short(uint32_t /*icb0*/, uint32_t /*icb1*/) {}

// ---- Tile-level bcast math (real) ----
ALWI void mul_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_dst_check(idst, "mul_tiles_bcast_cols");
    __emule_dst_mark_dirty(idst);
    __emule_bcast::apply_bcast<__emule_bcast::__op_mul>(
        icb0, icb1, itile0, itile1, idst, __emule_bcast::bcast_cols_src_idx);
}

ALWI void mul_tiles_bcast_rows(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst,
    uint32_t bcast_row_idx = 0) {
    __emule_dst_check(idst, "mul_tiles_bcast_rows");
    __emule_dst_mark_dirty(idst);
    __emule_bcast::apply_bcast<__emule_bcast::__op_mul>(
        icb0, icb1, itile0, itile1, idst,
        [bcast_row_idx](uint32_t i) { return __emule_bcast::bcast_rows_src_idx(i, bcast_row_idx); });
}

ALWI void add_tiles_bcast_rows(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst,
    uint32_t bcast_row_idx = 0) {
    __emule_dst_check(idst, "add_tiles_bcast_rows");
    __emule_dst_mark_dirty(idst);
    __emule_bcast::apply_bcast<__emule_bcast::__op_add>(
        icb0, icb1, itile0, itile1, idst,
        [bcast_row_idx](uint32_t i) { return __emule_bcast::bcast_rows_src_idx(i, bcast_row_idx); });
}

ALWI void sub_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_dst_check(idst, "sub_tiles_bcast_cols");
    __emule_dst_mark_dirty(idst);
    __emule_bcast::apply_bcast<__emule_bcast::__op_sub>(
        icb0, icb1, itile0, itile1, idst, __emule_bcast::bcast_cols_src_idx);
}

}  // namespace ckernel
