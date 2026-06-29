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

template <Dim D, bool acc_to_dest, typename Op>
inline void apply(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1,
                  uint32_t idst, uint32_t bcast_row_idx, Op op) {
    __emule_dst_check(idst, "bcast");
    __emule_dst_mark_dirty(idst);
    // Output / operand-0 shape (tiny-tile aware); DST stays the 32-strided grid
    // (out_i = r*32+c). The broadcast operand maps (r,c) → (b_r,b_c) per dim:
    // ROW→(bcast_row_idx, c), COL→(r, 0), SCALAR→(0, 0).
    const uint32_t th0 = get_tile_r_dim(icb0);
    const uint32_t tw0 = get_tile_c_dim(icb0);
    const uint32_t th1 = get_tile_r_dim(icb1);
    const uint32_t tw1 = get_tile_c_dim(icb1);
    auto bsrc = [&](uint32_t r, uint32_t c, uint32_t& b_r, uint32_t& b_c) {
        if constexpr (D == Dim::Rows)      { b_r = bcast_row_idx; b_c = c; }
        else if constexpr (D == Dim::Cols) { b_r = r;             b_c = 0; }
        else                               { b_r = 0;             b_c = 0; }  // Scalar
    };
    // Block-float (Bfp8_b/Bfp4_b) inputs carry a shared exponent per face-row and
    // can't be read element-by-element — route through the central format-aware
    // reader, which decodes the shared exponent into a row-major float[1024].
    // (__emule_unpack_cb_tile_to decodes tiny Bfp8_b/Bfp4_b as well.) NOTE:
    // silicon's llk_unpack_AB does NOT support ROW/COL broadcast on a narrow tile
    // (tile_w=16, num_faces_c < num_faces_r) — it asserts (llk_unpack_AB.h:78,99).
    // emule computes it regardless; a faithful ASSERT is tracked in #178.
    if (__emule_compute::cb_is_bfp8_b_format(icb0) ||
        __emule_compute::cb_is_bfp4_b_format(icb0) ||
        __emule_compute::cb_is_bfp8_b_format(icb1) ||
        __emule_compute::cb_is_bfp4_b_format(icb1)) {
        float tile_a[__EMULE_TILE_ELEMS];
        float tile_b[__EMULE_TILE_ELEMS];
        __emule_unpack_cb_tile_to(icb0, itile0, tile_a);
        __emule_unpack_cb_tile_to(icb1, itile1, tile_b);
        for (uint32_t r = 0; r < th0; ++r)
            for (uint32_t c = 0; c < tw0; ++c) {
                uint32_t b_r, b_c; bsrc(r, c, b_r, b_c);
                float v = op(tile_a[r * 32u + c], tile_b[b_r * 32u + b_c]);
                if constexpr (acc_to_dest) __emule_dst[idst][r * 32u + c] += v;
                else                       __emule_dst[idst][r * 32u + c] = v;
            }
        return;
    }
    // Non-block-float: read each operand directly via the shape-aware nfaces map
    // (narrow tiles use a single column-face; partial-height tiles use shorter faces).
    const bool in0_is_32b = __emule_compute::cb_is_32bit_format(icb0);
    const bool in1_is_32b = __emule_compute::cb_is_32bit_format(icb1);
    const uint8_t* a_ptr = __emule_compute::cb_read_ptr_at(icb0, itile0);
    const uint8_t* b_ptr = __emule_compute::cb_read_ptr_at(icb1, itile1);
    auto load_a = [&](uint32_t r, uint32_t c) -> float {
        uint32_t ni = __emule_nfaces::tile_rc_to_nfaces(r, c, th0, tw0);
        return in0_is_32b ? reinterpret_cast<const float*>(a_ptr)[ni]
                          : __emule_bf16::to_f32(reinterpret_cast<const uint16_t*>(a_ptr)[ni]);
    };
    auto load_b = [&](uint32_t r, uint32_t c) -> float {
        uint32_t ni = __emule_nfaces::tile_rc_to_nfaces(r, c, th1, tw1);
        return in1_is_32b ? reinterpret_cast<const float*>(b_ptr)[ni]
                          : __emule_bf16::to_f32(reinterpret_cast<const uint16_t*>(b_ptr)[ni]);
    };
    for (uint32_t r = 0; r < th0; ++r) {
        for (uint32_t c = 0; c < tw0; ++c) {
            uint32_t b_r, b_c; bsrc(r, c, b_r, b_c);
            float v = op(load_a(r, c), load_b(b_r, b_c));
            if constexpr (acc_to_dest) {
                __emule_dst[idst][r * 32u + c] += v;
            } else {
                __emule_dst[idst][r * 32u + c] = v;
            }
        }
    }
}
} // namespace __emule_bcast

// ---- Init helpers (templated 2-arg form) — HW config, no-op on emule. ----
// init_bcast's 3rd silicon arg (ocb) is pack-config only; ignored here.
// Defaulted so existing 2-arg callers still resolve.
template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void init_bcast(uint32_t icb0 = 0, uint32_t icb1 = 1, uint32_t ocb = 0) {}
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
