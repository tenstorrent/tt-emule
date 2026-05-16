#pragma once
// emule shim for upstream `llk_lib/llk_unpack_A.h` and friends — variadic
// tilize stub, unpack tilize/untilize state-setters, and `llk_unpack_A`
// template. Split out of `llk_defs.h`.

#include "jit_hw/llk_types.h"          // (defensive — types this header uses live in common.h, but keeping the type-surface include explicit for self-containment)
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
