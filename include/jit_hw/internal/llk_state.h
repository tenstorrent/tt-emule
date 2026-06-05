#pragma once
// Emule-specific LLK state symbols:
//   - thread_local pack/unpack trackers used by tilize/untilize stubs
//   - CbInterface stub + get_local_cb_interface
//   - operand/output query helpers
//   - re-exports the per-CB format arrays via jit_hw/api/cb_api.h (see below)
// Must not include common.h (would cycle through llk_defs.h); cb_api.h is safe — it does
// not include this header.
#include <cstdint>

#include "internal/cb_interface.h"

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
