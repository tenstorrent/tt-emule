#pragma once
// Emule shim for upstream `llk_lib/llk_math_eltwise_unary_datacopy.h`.
// Provides the tilize/untilize datacopy helpers + 3-/4-param
// `llk_math_eltwise_unary_datacopy` templates.
#include "jit_hw/llk_types.h"          // ckernel::DataCopyType
#include "internal/llk_state.h"
#include "api/compute/common.h"        // ckernel::BroadcastType + __emule_dst / __emule_compute / __emule_bf16 / __emule_nfaces

using ckernel::BroadcastType;

// ---- Datacopy helpers ----

// Tilize mode: extract 32x32 tile from row-major input CB into DST
inline void __llk_tilize_datacopy(uint32_t dst_idx) {
    uint32_t cb = __llk_unpack_src_cb;
    uint32_t ps = __emule_compute::cb_page_size(cb);
    uint32_t elem_size = ps / 1024;  // 2 for bf16, 4 for f32
    uint32_t tile_col = __llk_unpack_current_tile;
    uint32_t block_c = __llk_unpack_block_c;

    uint8_t* base = __emule_compute::cb_read_ptr_at(cb, 0);
    uint32_t tile_row_offset = __llk_unpack_start_tile_idx * ps;
    uint32_t tile_col_offset = tile_col * 32 * elem_size;
    uint32_t row_stride = block_c * 32 * elem_size;

    if (elem_size == 2) {  // bfloat16
        for (uint32_t r = 0; r < 32; r++) {
            uint16_t* src = reinterpret_cast<uint16_t*>(
                base + tile_row_offset + r * row_stride + tile_col_offset);
            for (uint32_t c = 0; c < 32; c++)
                __emule_dst[dst_idx][r * 32 + c] = __emule_bf16::to_f32(src[c]);
        }
    } else {  // float32 / int32: use memcpy to preserve exact bit patterns
        for (uint32_t r = 0; r < 32; r++) {
            uint32_t* src = reinterpret_cast<uint32_t*>(
                base + tile_row_offset + r * row_stride + tile_col_offset);
            for (uint32_t c = 0; c < 32; c++)
                std::memcpy(&__emule_dst[dst_idx][r * 32 + c], &src[c], sizeof(uint32_t));
        }
    }
    __llk_unpack_current_tile++;
}

// Untilize mode: copy individual tile page from input CB into DST
inline void __llk_untilize_datacopy(uint32_t dst_idx) {
    uint32_t tile_idx = __llk_unpack_start_tile_idx + __llk_unpack_current_tile;
    copy_tile(__llk_unpack_src_cb, tile_idx, dst_idx);
    __llk_unpack_current_tile++;
}

// 4-param version (used by tilize). CopyType is `enum class DataCopyType`.
template <ckernel::DataCopyType CopyType, int AccumMode, BroadcastType Bcast, bool UnpackToDest>
inline void llk_math_eltwise_unary_datacopy(uint32_t dst_idx) {
    if (__llk_unpack_is_tilize) __llk_tilize_datacopy(dst_idx);
    else __llk_untilize_datacopy(dst_idx);
}

// 3-param version (used by untilize)
template <ckernel::DataCopyType CopyType, int AccumMode, BroadcastType Bcast>
inline void llk_math_eltwise_unary_datacopy(uint32_t dst_idx) {
    if (__llk_unpack_is_tilize) __llk_tilize_datacopy(dst_idx);
    else __llk_untilize_datacopy(dst_idx);
}
