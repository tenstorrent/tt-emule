#pragma once
// Emule-specific LLK state symbols:
//   - thread_local pack/unpack trackers used by tilize/untilize stubs
//   - CbInterface stub + get_local_cb_interface
//   - operand/output query helpers
//   - re-exports the per-CB format arrays via jit_hw/api/cb_api.h (see below)
// Must not include common.h (would cycle through llk_defs.h); cb_api.h is safe — it does
// not include this header.
#include <cstdint>

// Operand CB interface stub
struct CbInterface {
    uint32_t fifo_page_size = 0;
    uint32_t fifo_rd_ptr = 0;
    uint32_t fifo_wr_ptr = 0;
    uint32_t fifo_size = 0;
    uint32_t fifo_limit = 0;
    uint32_t fifo_num_pages = 0;
};

// Global operand interface stubs.
// `__emule_cbs[cb_id]` provides real CB state when available (declared in
// emule_cb_state.h, populated per-core by the program runner). When the
// compute kernel uses `fifo_rd_ptr << 4` as a raw L1 address (e.g. the
// `experimental::write_row_mask_tile` helper that fires under TRISC_UNPACK),
// we need to return a real pointer rather than a dummy zero. Returning by
// reference into thread_local storage keeps callers that take a reference
// happy.
//
// Guarded: dataflow_api.h provides a CB-backed version with a different
// struct type (`CBInterface`, capital B); if both headers are included the
// dataflow version takes precedence for non-tilize call sites.
#ifndef __EMULE_GET_LOCAL_CB_INTERFACE_DEFINED
#define __EMULE_GET_LOCAL_CB_INTERFACE_DEFINED
#include "jit_hw/emule_cb_state.h"
inline CbInterface& get_local_cb_interface(uint32_t cb_id) {
    thread_local CbInterface ci{};
    if (__emule_cbs != nullptr) {
        auto& cb = __emule_cbs[cb_id];
        uint32_t r = cb.page_mask ? (cb.read_idx & cb.page_mask)
                                  : (cb.num_pages ? (cb.read_idx % cb.num_pages) : 0);
        uint32_t w = cb.page_mask ? (cb.write_idx & cb.page_mask)
                                  : (cb.num_pages ? (cb.write_idx % cb.num_pages) : 0);
        // Encode addresses as 16-byte indices so callers' `<< 4` reconstitutes the byte addr.
        uintptr_t rp = cb.base ? reinterpret_cast<uintptr_t>(cb.base + r * cb.page_size) >> 4 : 0;
        uintptr_t wp = cb.base ? reinterpret_cast<uintptr_t>(cb.base + w * cb.page_size) >> 4 : 0;
        ci.fifo_page_size = cb.page_size;
        ci.fifo_rd_ptr    = static_cast<uint32_t>(rp);
        ci.fifo_wr_ptr    = static_cast<uint32_t>(wp);
        ci.fifo_num_pages = cb.num_pages;
    }
    return ci;
}
#endif

inline uint32_t get_operand_id(uint32_t operand) { return operand; }
inline uint32_t get_output_id(uint32_t output) { return output; }
inline uint32_t get_operand_face_r_dim(uint32_t) { return 16; }
inline uint32_t get_operand_num_faces(uint32_t) { return 4; }
inline bool get_operand_narrow_tile(uint32_t) { return false; }
inline bool get_output_partial_face(uint32_t) { return false; }
inline bool get_output_narrow_tile(uint32_t) { return false; }
inline uint32_t get_output_face_r_dim(uint32_t) { return 16; }
inline uint32_t get_output_num_faces(uint32_t) { return 4; }

// All four per-CB format arrays (unpack_src_format / pack_dst_format / unpack_dst_format /
// pack_src_format) live in jit_hw/api/cb_api.h — single source of truth, populated from the
// EMULE_CB_DATA_FORMATS JIT define (mirroring the device's chlkc_descriptors.h). Pulled in
// here so kernel-lib helpers like `constexpr auto format = unpack_src_format[icb]`
// (tilize_helpers.inl is_fp32_input_format) resolve them as constant expressions. cb_api.h
// does not include this header, so there is no include cycle.
#include "jit_hw/api/cb_api.h"

// ---- Tilize/Untilize state ----
// Unpack state: tracks source CB and tile position for datacopy calls
static thread_local uint32_t __llk_unpack_src_cb = 0;
static thread_local uint32_t __llk_unpack_start_tile_idx = 0;
static thread_local uint32_t __llk_unpack_block_c = 0;
static thread_local uint32_t __llk_unpack_current_tile = 0;
static thread_local bool __llk_unpack_is_tilize = false;

// Pack state: tracks output position and layout mode
static thread_local uint32_t __llk_pack_offset = 0;
static thread_local bool __llk_pack_is_untilize = false;
static thread_local uint32_t __llk_pack_block_c = 0;

// Matmul state: `transpose=1` passed to mm_init / mm_block_init means the
// unpacker delivers the IN1 (SrcB) tile transposed before the math op
// (see llk_unpack_AB_matmul.h:176 — `THCON_SEC0_REG2_Haloize_mode_RMW`).
// Equivalent: compute A * B^T. emule has no SrcB modelling — `matmul_tiles`
// reads this flag and transposes its decoded IN1 tile before the FMA loop.
static thread_local bool __llk_matmul_transpose = false;
