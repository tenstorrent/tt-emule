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

// Per-operand unpack destination data format. Silicon (tt_metal/hw/ckernels/
// <arch>/metal/llk_io/llk_operands.h) returns `unpack_dst_format[operand_id]`.
// The kernel_lib chain (ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl) calls this
// to decide `enable_unpack_to_dest` (true only for Float32/UInt32/Int32), which
// selects the A2D vs B2D datacopy path. unpack_dst_format[] is the same per-CB
// format table (populated from the EMULE_CB_DATA_FORMATS JIT define) the other
// emule format helpers read, so this is faithful to the silicon decision.
inline uint32_t get_operand_dst_format(uint32_t operand_id) { return unpack_dst_format[operand_id]; }

// ---- Tilize/Untilize/Matmul trackers ----
// The pack/unpack tilize trackers and the matmul-transpose flag are per-compute-
// thread state, held in ComputeThreadCtx (emule_thread_ctx.h) and reached via
// __emule_compute_ctx().llk_*.
