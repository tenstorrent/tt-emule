// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// bcast tile ops. Real implementations for row/col/scalar broadcast — the
// second input tile is reshaped to one row / one column / one scalar and
// broadcast across the first input tile, then the elementwise op is applied.
// Init helpers stay no-op (HW packer/unpacker state emule doesn't model).
//
// Row-major layout: element (r, c) of a 32x32 tile is at index r*32 + c
// in the unpacked float[1024]. __emule_unpack_cb_tile_to handles format
// dispatch (bf16 / fp32 / Bfp8_b) so the bcast bodies stay format-agnostic.

#include "api/compute/common.h"
#include <cstdint>

namespace ckernel {

namespace __emule_bcast {
enum class Dim { Rows, Cols, Scalar };

// Map a row-major output element (r, c) to the row-major index of the
// broadcast-tile element it pulls from.
inline uint32_t src_idx(Dim d, uint32_t r, uint32_t c, uint32_t bcast_row_idx) {
    switch (d) {
        case Dim::Rows:   return bcast_row_idx * 32 + c;
        case Dim::Cols:   return r * 32;
        case Dim::Scalar: return 0;
    }
    return 0;
}

template <Dim D, bool acc_to_dest, typename Op>
inline void apply(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1,
                  uint32_t idst, uint32_t bcast_row_idx, Op op) {
    __emule_dst_check(idst, "bcast");
    __emule_dst_mark_dirty(idst);
    // Block-float (Bfp8_b/Bfp4_b) inputs carry a shared exponent per face-row
    // and cannot be read element-by-element: they MUST go through the central
    // format-aware reader, which decodes the shared exponent. (They are always
    // full 32-row tiles — faces are 16×16 — so the thin-tile path never applies
    // to them; note rows_a/rows_b below are mis-derived for block-float since
    // the page size doesn't map to a 2-byte element stride.) Route either-side
    // block-float through __emule_unpack_cb_tile_to, matching main's behavior.
    if (__emule_compute::cb_is_bfp8_b_format(icb0) ||
        __emule_compute::cb_is_bfp4_b_format(icb0) ||
        __emule_compute::cb_is_bfp8_b_format(icb1) ||
        __emule_compute::cb_is_bfp4_b_format(icb1)) {
        float tile_a[__EMULE_TILE_ELEMS];
        float tile_b[__EMULE_TILE_ELEMS];
        __emule_unpack_cb_tile_to(icb0, itile0, tile_a);
        __emule_unpack_cb_tile_to(icb1, itile1, tile_b);
        for (uint32_t r = 0; r < 32; ++r) {
            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t out_i = r * 32 + c;
                float v = op(tile_a[out_i], tile_b[src_idx(D, r, c, bcast_row_idx)]);
                if constexpr (acc_to_dest) {
                    __emule_dst[idst][out_i] += v;
                } else {
                    __emule_dst[idst][out_i] = v;
                }
            }
        }
        return;
    }
    // Tile-shape-aware: thin tiles (rows < 32) use a 2-column-face layout
    // rather than the 4-face 32×32 layout that the rowmajor_to_nfaces LUT
    // assumes. Required for Tile([1,32]) / Tile([2,32]) inputs used by rope,
    // broadcast_rmsnorm, etc. We can't reuse __emule_unpack_cb_tile_to here
    // because it always applies the 32×32 LUT — for thin tiles that would
    // produce a scrambled row-major buffer. Inline the load with the
    // shape-aware tile_rm_to_nfaces helper instead.
    const bool in0_is_32b = __emule_compute::cb_is_32bit_format(icb0);
    const bool in1_is_32b = __emule_compute::cb_is_32bit_format(icb1);
    const uint8_t* a_ptr = __emule_compute::cb_read_ptr_at(icb0, itile0);
    const uint8_t* b_ptr = __emule_compute::cb_read_ptr_at(icb1, itile1);
    const uint32_t elem_a = in0_is_32b ? 4u : 2u;
    const uint32_t elem_b = in1_is_32b ? 4u : 2u;
    const uint32_t rows_a = __emule_nfaces::tile_rows_from_pagesize(
        __emule_compute::cb_page_size(icb0), elem_a);
    const uint32_t rows_b = __emule_nfaces::tile_rows_from_pagesize(
        __emule_compute::cb_page_size(icb1), elem_b);
    auto load_a = [&](uint32_t r, uint32_t c) -> float {
        uint32_t ni = __emule_nfaces::tile_rm_to_nfaces(r * 32 + c, rows_a);
        return in0_is_32b
            ? reinterpret_cast<const float*>(a_ptr)[ni]
            : __emule_bf16::to_f32(reinterpret_cast<const uint16_t*>(a_ptr)[ni]);
    };
    auto load_b = [&](uint32_t r, uint32_t c) -> float {
        uint32_t ni = __emule_nfaces::tile_rm_to_nfaces(r * 32 + c, rows_b);
        return in1_is_32b
            ? reinterpret_cast<const float*>(b_ptr)[ni]
            : __emule_bf16::to_f32(reinterpret_cast<const uint16_t*>(b_ptr)[ni]);
    };
    for (uint32_t r = 0; r < rows_a; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t out_i = r * 32 + c;
            // src_idx maps the output (r,c) to a logical (r',c') in the bcast
            // tile B: ROW→(bcast_row_idx, c), COL→(r, 0), SCALAR→(0, 0),
            // NONE→(r, c). Decompose the returned linear index to call load_b.
            const uint32_t b_i = src_idx(D, r, c, bcast_row_idx);
            const uint32_t b_r = b_i / 32;
            const uint32_t b_c = b_i % 32;
            float v = op(load_a(r, c), load_b(b_r, b_c));
            if constexpr (acc_to_dest) {
                __emule_dst[idst][out_i] += v;
            } else {
                __emule_dst[idst][out_i] = v;
            }
        }
    }
}
} // namespace __emule_bcast

// ---- Init helpers (templated 2-arg form) — HW config, no-op on emule. ----
template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void init_bcast(uint32_t icb0 = 0, uint32_t icb1 = 1) {}
template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void add_bcast_rows_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1) {}
template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void add_bcast_cols_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1) {}
template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void add_bcast_scalar_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1) {}

// ---- 2-template-arg generic forms used by D2M-generated code. ----
// Dispatch on both template args through __emule_bcast::apply. ROW broadcast
// in this signature has no bcast_row_idx (the single-template-arg form does);
// defaults to row 0.
// `acc_to_dest` (silicon's `llk_math_eltwise_binary<...>` arg): when true the
// op accumulates into DST instead of overwriting. Needed by deepseek MoE gate.
template <EltwiseBinaryType op_type, BroadcastType bcast_type, bool acc_to_dest = false>
inline void any_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    auto op = [](float a, float b) {
        if constexpr (op_type == EltwiseBinaryType::ELWADD) return a + b;
        else if constexpr (op_type == EltwiseBinaryType::ELWSUB) return a - b;
        else /* ELWMUL */ return a * b;
    };
    if constexpr (bcast_type == BroadcastType::ROW) {
        __emule_bcast::apply<__emule_bcast::Dim::Rows, acc_to_dest>(icb0, icb1, itile0, itile1, idst, 0, op);
    } else if constexpr (bcast_type == BroadcastType::COL) {
        __emule_bcast::apply<__emule_bcast::Dim::Cols, acc_to_dest>(icb0, icb1, itile0, itile1, idst, 0, op);
    } else if constexpr (bcast_type == BroadcastType::SCALAR) {
        __emule_bcast::apply<__emule_bcast::Dim::Scalar, acc_to_dest>(icb0, icb1, itile0, itile1, idst, 0, op);
    } else {  // NONE — elementwise (acc_to_dest path not supported here; falls back to overwrite)
        if constexpr (op_type == EltwiseBinaryType::ELWADD)      add_tiles(icb0, icb1, itile0, itile1, idst);
        else if constexpr (op_type == EltwiseBinaryType::ELWSUB) sub_tiles(icb0, icb1, itile0, itile1, idst);
        else /* ELWMUL */                                        mul_tiles(icb0, icb1, itile0, itile1, idst);
    }
}
// Named forwarders. The function name and op_type template arg are redundant
// by upstream convention (kernels always pass matching pair); behave like
// any_tiles_bcast<op_type, bcast_type> regardless of the chosen name.
template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void add_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    any_tiles_bcast<op_type, bcast_type>(icb0, icb1, itile0, itile1, idst);
}
template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void sub_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    any_tiles_bcast<op_type, bcast_type>(icb0, icb1, itile0, itile1, idst);
}
template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void mul_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    any_tiles_bcast<op_type, bcast_type>(icb0, icb1, itile0, itile1, idst);
}

// `unary_bcast<BroadcastType>` — loads one tile and broadcasts it across DST.
// SCALAR replicates element (0,0); ROW replicates row 0 down all rows; COL
// replicates column 0 across all columns. Used by the binary_ng *_scalar/row/
// col_bcast compute kernels when the host picks the LLK-broadcast path
// (BCAST_LLK=1, so the reader does NOT pre-fill the tile). Broadcast on the raw
// DST bits so it is format-agnostic (fp32 and int32 share the path). Layout:
// element (r,c) at r*32 + c (see __emule_bcast above).
template <BroadcastType BCAST_T>
inline void unary_bcast_init(uint32_t /*icb0*/, uint32_t /*icb1*/) {}

template <BroadcastType BCAST_T>
inline void unary_bcast(uint32_t icb, uint32_t in_tile_index, uint32_t idst) {
    copy_tile(icb, in_tile_index, idst);  // load tile into DST honoring format
    if constexpr (BCAST_T == BroadcastType::SCALAR) {
        const int32_t v = __emule_dst_load_i32(idst, 0);
        for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) __emule_dst_store_i32(idst, i, v);
    } else if constexpr (BCAST_T == BroadcastType::ROW) {
        for (uint32_t r = 1; r < 32; ++r)
            for (uint32_t c = 0; c < 32; ++c)
                __emule_dst_store_i32(idst, r * 32 + c, __emule_dst_load_i32(idst, c));
    } else if constexpr (BCAST_T == BroadcastType::COL) {
        for (uint32_t r = 0; r < 32; ++r)
            for (uint32_t c = 1; c < 32; ++c)
                __emule_dst_store_i32(idst, r * 32 + c, __emule_dst_load_i32(idst, r * 32));
    }
    // NONE: plain copy already done.
}

// ---- Non-templated row/col/scalar variants ----

ALWI void add_tiles_bcast_rows(uint32_t icb0, uint32_t icb1,
                               uint32_t itile0, uint32_t itile1,
                               uint32_t idst, uint32_t bcast_row_idx = 0) {
    __emule_bcast::apply<__emule_bcast::Dim::Rows, false>(
        icb0, icb1, itile0, itile1, idst, bcast_row_idx,
        [](float a, float b) { return a + b; });
}
ALWI void sub_tiles_bcast_rows(uint32_t icb0, uint32_t icb1,
                               uint32_t itile0, uint32_t itile1,
                               uint32_t idst, uint32_t bcast_row_idx = 0) {
    __emule_bcast::apply<__emule_bcast::Dim::Rows, false>(
        icb0, icb1, itile0, itile1, idst, bcast_row_idx,
        [](float a, float b) { return a - b; });
}
ALWI void mul_tiles_bcast_rows(uint32_t icb0, uint32_t icb1,
                               uint32_t itile0, uint32_t itile1,
                               uint32_t idst, uint32_t bcast_row_idx = 0) {
    __emule_bcast::apply<__emule_bcast::Dim::Rows, false>(
        icb0, icb1, itile0, itile1, idst, bcast_row_idx,
        [](float a, float b) { return a * b; });
}

ALWI void add_tiles_bcast_cols(uint32_t icb0, uint32_t icb1,
                               uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_bcast::apply<__emule_bcast::Dim::Cols, false>(
        icb0, icb1, itile0, itile1, idst, 0,
        [](float a, float b) { return a + b; });
}
ALWI void sub_tiles_bcast_cols(uint32_t icb0, uint32_t icb1,
                               uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_bcast::apply<__emule_bcast::Dim::Cols, false>(
        icb0, icb1, itile0, itile1, idst, 0,
        [](float a, float b) { return a - b; });
}
ALWI void mul_tiles_bcast_cols(uint32_t icb0, uint32_t icb1,
                               uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_bcast::apply<__emule_bcast::Dim::Cols, false>(
        icb0, icb1, itile0, itile1, idst, 0,
        [](float a, float b) { return a * b; });
}

ALWI void add_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1,
                                 uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_bcast::apply<__emule_bcast::Dim::Scalar, false>(
        icb0, icb1, itile0, itile1, idst, 0,
        [](float a, float b) { return a + b; });
}
ALWI void sub_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1,
                                 uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_bcast::apply<__emule_bcast::Dim::Scalar, false>(
        icb0, icb1, itile0, itile1, idst, 0,
        [](float a, float b) { return a - b; });
}
ALWI void mul_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1,
                                 uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_bcast::apply<__emule_bcast::Dim::Scalar, false>(
        icb0, icb1, itile0, itile1, idst, 0,
        [](float a, float b) { return a * b; });
}

// Single-template-arg `<BroadcastType>` form — upstream signature bcast.h:408-412.
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

// ---- Init helpers (non-templated) — HW config, no-op on emule. ----
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
