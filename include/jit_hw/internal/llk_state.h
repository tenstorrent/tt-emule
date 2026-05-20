#pragma once
// Emule-specific LLK state symbols:
//   - thread_local pack/unpack trackers used by tilize/untilize stubs
//   - inline format arrays (unpack_src_format / unpack_dst_format)
//   - CbInterface stub + get_local_cb_interface
//   - operand/output query helpers
// Kept free of common.h to avoid a cycle through llk_defs.h.
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

// Global operand interface stubs
// Guarded: dataflow_api.h provides a CB-backed version; if both headers are
// included, the dataflow version (which reads real CB state) should win.
#ifndef __EMULE_GET_LOCAL_CB_INTERFACE_DEFINED
#define __EMULE_GET_LOCAL_CB_INTERFACE_DEFINED
inline CbInterface& get_local_cb_interface(uint32_t) {
    static CbInterface dummy;
    return dummy;
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

// Format arrays (stub)
inline uint32_t unpack_src_format[32] = {};
inline uint32_t unpack_dst_format[32] = {};

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
