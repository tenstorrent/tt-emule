#pragma once
// Emule shim for upstream `llk_lib/llk_unpack_A.h` and friends — variadic
// tilize stub, unpack tilize/untilize state-setters, and `llk_unpack_A`
// template.

#include "jit_hw/llk_types.h"          // type-surface include kept explicit for self-containment
#include "internal/llk_state.h"
#include "api/compute/common.h"        // ckernel::BroadcastType, ckernel::EltwiseBinaryReuseDestType

// ---- Hardware-level stubs (not called in emulation) ----
template <typename... Args>
inline void _llk_unpack_tilize_(Args&&...) {}

// ---- Unpack tilize/untilize: set source CB and tile position ----

inline void llk_unpack_tilize(uint32_t, uint32_t, uint32_t, uint32_t) {}

inline void llk_unpack_tilize_block(uint32_t icb, uint32_t block_c, uint32_t start_tile_idx) {
    __llk_unpack_src_cb = icb;
    __llk_unpack_start_tile_idx = start_tile_idx;
    __llk_unpack_block_c = block_c;
    __llk_unpack_current_tile = 0;
}

inline void llk_unpack_untilize(uint32_t icb, uint32_t block_c, uint32_t start_tile_idx) {
    __llk_unpack_src_cb = icb;
    __llk_unpack_start_tile_idx = start_tile_idx;
    __llk_unpack_block_c = block_c;
    __llk_unpack_current_tile = 0;
    __llk_pack_block_c = block_c;
}

// ---- llk_unpack_A: general-purpose tile unpack (no-op in emulation) ----
// In emulation, actual data movement happens in copy_tile / datacopy helpers.
// This stub exists so that code paths referencing llk_unpack_A compile.
template <ckernel::BroadcastType BType = ckernel::BroadcastType::NONE,
          bool acc_to_dest = false,
          ckernel::EltwiseBinaryReuseDestType binary_reuse_dest = ckernel::EltwiseBinaryReuseDestType::NONE,
          bool unpack_to_dest = false>
inline void llk_unpack_A(uint32_t /*operand*/, uint32_t /*tile_index*/) {}

// LLK unpack init — silicon configures unpack source format / SrcA/SrcB
// layout here; no-op stub in emule.
template <ckernel::BroadcastType BType = ckernel::BroadcastType::NONE,
          bool acc_to_dest = false,
          ckernel::EltwiseBinaryReuseDestType binary_reuse_dest = ckernel::EltwiseBinaryReuseDestType::NONE>
inline void llk_unpack_A_init(
    uint32_t /*transpose_of_faces*/ = 0,
    uint32_t /*within_face_16x16_transpose*/ = 0,
    uint32_t /*operand*/ = 0) {}

template <ckernel::BroadcastType BType = ckernel::BroadcastType::NONE>
inline void llk_unpack_AB_init(
    uint32_t /*icb0*/ = 0,
    uint32_t /*icb1*/ = 0,
    ckernel::Transpose /*transpose*/ = ckernel::Transpose::NoneT) {}

template <ckernel::BroadcastType BType = ckernel::BroadcastType::NONE>
inline void llk_unpack_AB(uint32_t /*icb0*/, uint32_t /*icb1*/,
                          uint32_t /*tile_idx0*/, uint32_t /*tile_idx1*/) {}

inline void llk_unpack_tilize_init(
    uint32_t /*operand*/ = 0,
    uint32_t /*ct_dim*/ = 0,
    uint32_t /*face_r_dim*/ = 16,
    uint32_t /*narrow_tile*/ = 0) {}

// Dual-operand tilize init (pool/upsample tilizeA_B_reduce path). HW unpacker
// config — no-op in emule (the tilize+reduce math runs in unpack_tilizeA_B_block).
template <bool neginf_srcA = true, std::uint32_t reload_srcB = true,
          bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B_init(
    uint32_t /*operandA*/, uint32_t /*operandB*/, uint32_t /*ct_dim*/,
    uint32_t /*num_faces*/ = 4, uint32_t /*unpA_face_r_dim*/ = 16,
    uint32_t /*unpB_face_r_dim*/ = 16) {}

// llk_math_wait_for_dest_available lives in llk_sync_stubs.h.
