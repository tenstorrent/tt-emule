// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// bcast tile ops. Real implementations for row/col/scalar broadcast — the
// second input tile is reshaped to one row / one column / one scalar and
// broadcast across the first input tile, then the elementwise op is applied.
// Init helpers stay no-op (HW packer/unpacker state emule doesn't model).

#include "api/compute/common.h"
#include <cstdint>

namespace ckernel {

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void init_bcast(uint32_t icb0 = 0, uint32_t icb1 = 1) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void add_bcast_rows_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void add_bcast_cols_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void add_bcast_scalar_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1) {}

// Pre-existing 2-template-arg no-op generic forms used by D2M-generated code.
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

// ---- Non-templated row/col/scalar variants (real implementations) ----
//
// Row-major layout: element (r, c) of a 32x32 tile is at index r*32 + c
// in the unpacked float[1024]. `__emule_unpack_cb_tile_to` handles format
// dispatch (bf16 / fp32 / Bfp8_b) so the bcast bodies stay format-agnostic.

namespace __emule_bcast {
enum class Dim { Rows, Cols, Scalar };

// Map a row-major output element (r, c) to the row-major index of the
// broadcast-tile element it pulls from.
//   Rows:   B selects row `bcast_row_idx`, broadcast across all 32 rows
//   Cols:   B selects col 0,               broadcast across all 32 cols
//   Scalar: B uses element [0][0],         broadcast everywhere
inline uint32_t src_idx(Dim d, uint32_t r, uint32_t c, uint32_t bcast_row_idx) {
    switch (d) {
        case Dim::Rows:   return bcast_row_idx * 32 + c;
        case Dim::Cols:   return r * 32;
        case Dim::Scalar: return 0;
    }
    return 0;
}

template <Dim D, typename Op>
inline void apply(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1,
                  uint32_t idst, uint32_t bcast_row_idx, Op op) {
    __emule_dst_check(idst, "bcast");
    __emule_dst_mark_dirty(idst);
    float tile_a[__EMULE_TILE_ELEMS];
    float tile_b[__EMULE_TILE_ELEMS];
    __emule_unpack_cb_tile_to(icb0, itile0, tile_a);
    __emule_unpack_cb_tile_to(icb1, itile1, tile_b);
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t out_i = r * 32 + c;
            __emule_dst[idst][out_i] = op(tile_a[out_i], tile_b[src_idx(D, r, c, bcast_row_idx)]);
        }
    }
}
} // namespace __emule_bcast

ALWI void add_tiles_bcast_rows(uint32_t icb0, uint32_t icb1,
                               uint32_t itile0, uint32_t itile1,
                               uint32_t idst, uint32_t bcast_row_idx = 0) {
    __emule_bcast::apply<__emule_bcast::Dim::Rows>(
        icb0, icb1, itile0, itile1, idst, bcast_row_idx,
        [](float a, float b) { return a + b; });
}
ALWI void sub_tiles_bcast_rows(uint32_t icb0, uint32_t icb1,
                               uint32_t itile0, uint32_t itile1,
                               uint32_t idst, uint32_t bcast_row_idx = 0) {
    __emule_bcast::apply<__emule_bcast::Dim::Rows>(
        icb0, icb1, itile0, itile1, idst, bcast_row_idx,
        [](float a, float b) { return a - b; });
}
ALWI void mul_tiles_bcast_rows(uint32_t icb0, uint32_t icb1,
                               uint32_t itile0, uint32_t itile1,
                               uint32_t idst, uint32_t bcast_row_idx = 0) {
    __emule_bcast::apply<__emule_bcast::Dim::Rows>(
        icb0, icb1, itile0, itile1, idst, bcast_row_idx,
        [](float a, float b) { return a * b; });
}

ALWI void add_tiles_bcast_cols(uint32_t icb0, uint32_t icb1,
                               uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_bcast::apply<__emule_bcast::Dim::Cols>(
        icb0, icb1, itile0, itile1, idst, 0,
        [](float a, float b) { return a + b; });
}
ALWI void sub_tiles_bcast_cols(uint32_t icb0, uint32_t icb1,
                               uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_bcast::apply<__emule_bcast::Dim::Cols>(
        icb0, icb1, itile0, itile1, idst, 0,
        [](float a, float b) { return a - b; });
}
ALWI void mul_tiles_bcast_cols(uint32_t icb0, uint32_t icb1,
                               uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_bcast::apply<__emule_bcast::Dim::Cols>(
        icb0, icb1, itile0, itile1, idst, 0,
        [](float a, float b) { return a * b; });
}

ALWI void add_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1,
                                 uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_bcast::apply<__emule_bcast::Dim::Scalar>(
        icb0, icb1, itile0, itile1, idst, 0,
        [](float a, float b) { return a + b; });
}
ALWI void sub_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1,
                                 uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_bcast::apply<__emule_bcast::Dim::Scalar>(
        icb0, icb1, itile0, itile1, idst, 0,
        [](float a, float b) { return a - b; });
}
ALWI void mul_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1,
                                 uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_bcast::apply<__emule_bcast::Dim::Scalar>(
        icb0, icb1, itile0, itile1, idst, 0,
        [](float a, float b) { return a * b; });
}

// Single-template-arg `<BroadcastType>` form — dispatches to the row/col/scalar
// variant above based on BCAST_T. Upstream signature: bcast.h:408-412.
template <BroadcastType BCAST_T>
ALWI void add_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0,
                          uint32_t itile1, uint32_t idst,
                          uint32_t bcast_row_idx = 0) {
    if constexpr (BCAST_T == BroadcastType::ROW) {
        add_tiles_bcast_rows(icb0, icb1, itile0, itile1, idst, bcast_row_idx);
    } else if constexpr (BCAST_T == BroadcastType::COL) {
        add_tiles_bcast_cols(icb0, icb1, itile0, itile1, idst);
    } else if constexpr (BCAST_T == BroadcastType::SCALAR) {
        add_tiles_bcast_scalar(icb0, icb1, itile0, itile1, idst);
    } else {
        add_tiles(icb0, icb1, itile0, itile1, idst);  // NONE — element-wise
    }
}
template <BroadcastType BCAST_T>
ALWI void sub_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0,
                          uint32_t itile1, uint32_t idst,
                          uint32_t bcast_row_idx = 0) {
    if constexpr (BCAST_T == BroadcastType::ROW) {
        sub_tiles_bcast_rows(icb0, icb1, itile0, itile1, idst, bcast_row_idx);
    } else if constexpr (BCAST_T == BroadcastType::COL) {
        sub_tiles_bcast_cols(icb0, icb1, itile0, itile1, idst);
    } else if constexpr (BCAST_T == BroadcastType::SCALAR) {
        sub_tiles_bcast_scalar(icb0, icb1, itile0, itile1, idst);
    } else {
        sub_tiles(icb0, icb1, itile0, itile1, idst);
    }
}
template <BroadcastType BCAST_T>
ALWI void mul_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0,
                          uint32_t itile1, uint32_t idst,
                          uint32_t bcast_row_idx = 0) {
    if constexpr (BCAST_T == BroadcastType::ROW) {
        mul_tiles_bcast_rows(icb0, icb1, itile0, itile1, idst, bcast_row_idx);
    } else if constexpr (BCAST_T == BroadcastType::COL) {
        mul_tiles_bcast_cols(icb0, icb1, itile0, itile1, idst);
    } else if constexpr (BCAST_T == BroadcastType::SCALAR) {
        mul_tiles_bcast_scalar(icb0, icb1, itile0, itile1, idst);
    } else {
        mul_tiles(icb0, icb1, itile0, itile1, idst);
    }
}

// Init helpers — HW packer/unpacker state config that emule doesn't model.
// Match upstream signatures (call_line arg ignored; default 0 emule-side,
// __builtin_LINE() upstream — semantically equivalent for a no-op).
ALWI void add_bcast_rows_init_short(uint32_t, uint32_t, uint32_t = 0) {}
ALWI void sub_bcast_rows_init_short(uint32_t, uint32_t, uint32_t = 0) {}
ALWI void mul_bcast_rows_init_short(uint32_t, uint32_t, uint32_t = 0) {}
ALWI void add_bcast_cols_init_short(uint32_t, uint32_t, uint32_t = 0) {}
ALWI void sub_bcast_cols_init_short(uint32_t, uint32_t, uint32_t = 0) {}
ALWI void mul_bcast_cols_init_short(uint32_t, uint32_t, uint32_t = 0) {}
ALWI void add_bcast_scalar_init_short(uint32_t, uint32_t, uint32_t = 0) {}
ALWI void mul_tiles_bcast_scalar_init_short(uint32_t, uint32_t, uint32_t = 0) {}
ALWI void sub_tiles_bcast_scalar_init_short(uint32_t, uint32_t, uint32_t = 0) {}

}  // namespace ckernel
