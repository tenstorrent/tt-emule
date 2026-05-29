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
// Read element ni from CB `cb` at tile `itile`, converted to float, regardless
// of the CB's stored format. Mirrors emule's other compute primitives — each
// CB's format is independent and may differ between icb0 and icb1 (e.g. an
// fp32 ex2pe tile read alongside a bf16 input tile in the layernorm
// normalize step).
inline float __emule_read_cb_elem(uint32_t cb_id, uint8_t* page_base, uint32_t elem_idx) {
    if (__emule_compute::cb_is_32bit_format(cb_id)) {
        return reinterpret_cast<const float*>(page_base)[elem_idx];
    }
    uint16_t bf = reinterpret_cast<const uint16_t*>(page_base)[elem_idx];
    return __emule_bf16::to_f32(bf);
}

template <typename Op, typename BcastFn>
inline void apply_bcast(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1,
    uint32_t idst, BcastFn bcast_fn) {
    uint8_t* base0 = __emule_compute::cb_read_ptr_at(icb0, itile0);
    uint8_t* base1 = __emule_compute::cb_read_ptr_at(icb1, itile1);
    // DST is a 1024-element tile; iterate the full row-major span. Each CB
    // is read in its own format (bf16 or fp32), promoted to float for the
    // op, and written into DST (which is always float).
    for (uint32_t i = 0; i < 1024; ++i) {
        uint32_t ni_a = __emule_nfaces::rowmajor_to_nfaces[i];
        uint32_t ni_b = __emule_nfaces::rowmajor_to_nfaces[bcast_fn(i)];
        float a = __emule_read_cb_elem(icb0, base0, ni_a);
        float b = __emule_read_cb_elem(icb1, base1, ni_b);
        __emule_dst[idst][i] = Op{}(a, b);
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

// ---- Single-template-param shorthands (mirror real tt-metal's `template <BroadcastType>` API) ----
//
// Real tt_metal/hw/inc/api/compute/bcast.h exposes `add_tiles_bcast<BroadcastType>`,
// `mul_tiles_bcast<BroadcastType>`, `sub_tiles_bcast<BroadcastType>` as
// shorthand for the two-param `any_tiles_bcast<ELW{ADD,MUL,SUB}, ...>` form.
// Kernels like minimal_matmul/compute.cpp call them by the one-param name
// (e.g. `add_tiles_bcast<BroadcastType::ROW>(...)` for fused bias). emule's
// existing two-param `add_tiles_bcast<EltwiseBinaryType, BroadcastType>`
// no-op stub above doesn't match this signature, hence the dispatch here.
// We route the active paths through the per-CB-format apply_bcast driver,
// which is correct for the mixed bf16/fp32/Bfp8 CBs Llama exercises.
template <BroadcastType tBcastDim>
ALWI void add_tiles_bcast(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx = 0) {
    if constexpr (tBcastDim == BroadcastType::ROW) {
        add_tiles_bcast_rows(icb0, icb1, itile0, itile1, idst, bcast_row_idx);
    } else if constexpr (tBcastDim == BroadcastType::COL) {
        __emule_dst_check(idst, "add_tiles_bcast<COL>");
        __emule_dst_mark_dirty(idst);
        __emule_bcast::apply_bcast<__emule_bcast::__op_add>(
            icb0, icb1, itile0, itile1, idst, __emule_bcast::bcast_cols_src_idx);
    } else {
        // SCALAR: bcast a single scalar from cb_b across the entire tile.
        __emule_dst_check(idst, "add_tiles_bcast<SCALAR>");
        __emule_dst_mark_dirty(idst);
        __emule_bcast::apply_bcast<__emule_bcast::__op_add>(
            icb0, icb1, itile0, itile1, idst, [](uint32_t /*i*/) { return 0u; });
    }
}

template <BroadcastType tBcastDim>
ALWI void mul_tiles_bcast(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx = 0) {
    if constexpr (tBcastDim == BroadcastType::ROW) {
        mul_tiles_bcast_rows(icb0, icb1, itile0, itile1, idst, bcast_row_idx);
    } else if constexpr (tBcastDim == BroadcastType::COL) {
        mul_tiles_bcast_cols(icb0, icb1, itile0, itile1, idst);
    } else {
        __emule_dst_check(idst, "mul_tiles_bcast<SCALAR>");
        __emule_dst_mark_dirty(idst);
        __emule_bcast::apply_bcast<__emule_bcast::__op_mul>(
            icb0, icb1, itile0, itile1, idst, [](uint32_t /*i*/) { return 0u; });
    }
}

template <BroadcastType tBcastDim>
ALWI void sub_tiles_bcast(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx = 0) {
    if constexpr (tBcastDim == BroadcastType::COL) {
        sub_tiles_bcast_cols(icb0, icb1, itile0, itile1, idst);
    } else if constexpr (tBcastDim == BroadcastType::ROW) {
        __emule_dst_check(idst, "sub_tiles_bcast<ROW>");
        __emule_dst_mark_dirty(idst);
        __emule_bcast::apply_bcast<__emule_bcast::__op_sub>(
            icb0, icb1, itile0, itile1, idst,
            [bcast_row_idx](uint32_t i) { return __emule_bcast::bcast_rows_src_idx(i, bcast_row_idx); });
    } else {
        __emule_dst_check(idst, "sub_tiles_bcast<SCALAR>");
        __emule_dst_mark_dirty(idst);
        __emule_bcast::apply_bcast<__emule_bcast::__op_sub>(
            icb0, icb1, itile0, itile1, idst, [](uint32_t /*i*/) { return 0u; });
    }
}

}  // namespace ckernel
