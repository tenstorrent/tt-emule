// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Emule shim for tt-metal's `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`.
//
// Real header lives at
//   /localdev/arminale/tt-metal/ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp
// (+ tilize_helpers.inl). The real implementation wraps Tensix LLKs
// (`tilize_init`, `tilize_block`, `tilize_uninit`, `fast_tilize_*`) which
// require unpack/pack engines and SrcA/DST register orchestration that emule
// does not model. Rather than shim each LLK separately, intercept the
// kernel-lib header — its API is small (one templated `tilize<>` plus the
// enum classes in `tilize_config`) and pure data-rearrangement on the CB
// byte pool, so the host equivalent is straightforward.
//
// Out of scope here:
//   - `fast_tilize_*` variants (Blackhole-only). The real header dispatches
//     between fast and standard via `if constexpr (use_fast)`. We force the
//     standard path by making `can_use_fast_tilize()` return false. The fast
//     LLK names never need to exist.
//   - Reconfiguration of register data formats (NoReconfigure is the only
//     mode the embedding kernel uses). The other modes are no-ops since
//     emule's `reconfig_data_format_*` are already no-ops in common.h.
//   - Block-float input formats (Bfp8/4/2) — `tilize_helpers.inl` asserts
//     these out upstream; mirror with no special handling.
//
// Behavioral model (matches real WH-b0 tile layout):
//   - Input CB stores 32 contiguous rows × (block_width_tiles*32*elem_size)
//     bytes per block, row-major, with each "page" being one bf16 tile of
//     2048 bytes (or 4096 for fp32). Reader kernels (e.g.
//     embeddings_tilize.cpp) write 32 noc_async_reads of the row stride
//     into the CB before cb_push_back.
//   - Output CB stores `block_width_tiles` tiles per block, each tile is
//     32×32 = 1024 elements split into 4 face-major 16×16 faces in
//     {(top-left), (top-right), (bottom-left), (bottom-right)} order.

#include <cstdint>
#include <cstring>
#include <optional>

#include "jit_hw/api/cb_api.h"
#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/compute/cb_api.h"
#include "jit_hw/internal/llk_state.h"

namespace compute_kernel_lib {

// Config enums — same names + members as the real
//   ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp `tilize_config` namespace.
namespace tilize_config {
enum class ReconfigureRegisterDatatypeMode : uint8_t {
    NoReconfigure,
    UnpackReconfigure,
    PackReconfigure,
    UnpackAndPackReconfigure,
};
enum class InitUninitMode : uint8_t {
    InitAndUninit,
    InitOnly,
    UninitOnly,
    NoInit,
};
enum class WaitMode : uint8_t {
    WaitBlock,
    WaitUpfront,
    NoWait,
};
enum class Fp32Mode : uint8_t {
    Lossless,
    Fast,
};
}  // namespace tilize_config

// is_fp32_input_format<cb_id>() — used by the embedding kernel to pick
// Fp32Mode at compile time. The JIT define EMULE_TILE_SIZES populates
// `unpack_tile_size[]` (in api/cb_api.h); page > 2048 bytes ⇒ fp32.
//
// Note: the real header uses `unpack_src_format[cb_id] == DataFormat::Float32`
// which is the same signal via the format array. emule doesn't materialize a
// full DataFormat array at JIT time, so probe the byte size instead.
template <uint32_t cb_id>
constexpr bool is_fp32_input_format() {
    return unpack_tile_size[cb_id] > 2048;
}

template <uint32_t cb_id>
constexpr bool is_fp32_output_format() {
    return unpack_tile_size[cb_id] > 2048;
}

// =============================================================================
// Host-side row-major → tile-layout rearrangement
// =============================================================================
//
// One block of `block_width_tiles` tiles. Input CB has 32 rows × (block * 32)
// elements row-major (tile_height=32 hardcoded by WH-b0). Output CB receives
// `block_width_tiles` tiles in face-major layout.
//
// Mirrors the real
//   tt_metal/hw/inc/api/compute/tilize.h::tilize_block
// at the bytes-on-the-wire level — the real LLK rearranges via SrcA register
// (UNPACK + PACK paths); we do it directly in memory.
inline void tilize_block_host(uint32_t icb, uint32_t block_width_tiles, uint32_t ocb) {
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t FACE_HEIGHT = 16;
    constexpr uint32_t FACE_WIDTH = 16;
    constexpr uint32_t NUM_FACES = 4;

    const uint32_t in_page_bytes = __emule_compute::cb_page_size(icb);
    const uint32_t out_page_bytes = __emule_compute::cb_page_size(ocb);
    // bf16 (2) or fp32 (4) — match by tile page size vs 1024 elements.
    const uint32_t elem_size = (in_page_bytes >= 4096) ? 4u : 2u;

    // Input row stride: full row spans `block_width_tiles` × 32 elements.
    const uint32_t row_stride_bytes = block_width_tiles * TILE_WIDTH * elem_size;
    const uint32_t face_row_bytes = FACE_WIDTH * elem_size;  // 32 (bf16) or 64 (fp32)
    const uint32_t face_bytes = FACE_HEIGHT * face_row_bytes;  // 512 (bf16) or 1024 (fp32)

    uint8_t* src_base = __emule_compute::cb_read_ptr_at(icb, /*tile_offset=*/0);
    uint8_t* dst_base = __emule_compute::cb_write_ptr_at(ocb, /*tile_offset=*/0);

    for (uint32_t t = 0; t < block_width_tiles; ++t) {
        uint8_t* dst_tile = dst_base + t * out_page_bytes;
        const uint32_t src_col_off = t * TILE_WIDTH * elem_size;  // byte offset to start of tile's column block

        for (uint32_t face = 0; face < NUM_FACES; ++face) {
            const uint32_t face_r = face / 2;       // 0 (top) or 1 (bottom)
            const uint32_t face_c = face % 2;       // 0 (left) or 1 (right)
            uint8_t* dst_face = dst_tile + face * face_bytes;

            for (uint32_t fr = 0; fr < FACE_HEIGHT; ++fr) {
                const uint32_t src_row = face_r * FACE_HEIGHT + fr;
                const uint8_t* src_face_row =
                    src_base + src_row * row_stride_bytes + src_col_off + face_c * face_row_bytes;
                std::memcpy(dst_face + fr * face_row_bytes, src_face_row, face_row_bytes);
            }
        }
    }
}

// =============================================================================
// compute_kernel_lib::tilize<>(num_blocks, total_input_pages = nullopt)
// =============================================================================
//
// Matches the signature of the real upstream template at
//   tt_metal/ttnn/cpp/ttnn/kernel_lib/tilize_helpers.{hpp,inl}
// but bypasses the real LLK calls (`tilize_init`, `tilize_block`,
// `tilize_uninit`, `fast_*` variants) by performing the rearrangement
// directly on emule's CB byte pool.
//
// Template params kept for ABI compatibility with caller code; only
// `block_width_tiles`, `input_cb`, `output_cb`, and `wait_mode` change
// runtime behavior. `init_uninit_mode`, `reconfig_mode`, and `fp32_mode`
// would matter on Tensix but are no-ops on host.
template <
    uint32_t block_width_tiles,
    uint32_t input_cb,
    uint32_t output_cb,
    tilize_config::InitUninitMode init_uninit_mode = tilize_config::InitUninitMode::InitAndUninit,
    tilize_config::WaitMode wait_mode = tilize_config::WaitMode::WaitBlock,
    tilize_config::ReconfigureRegisterDatatypeMode reconfig_mode =
        tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
    tilize_config::Fp32Mode fp32_mode = tilize_config::Fp32Mode::Fast>
inline void tilize(uint32_t num_blocks, std::optional<uint32_t> total_input_pages = std::nullopt) {
    static_assert(block_width_tiles > 0, "block_width_tiles must be > 0");
    static_assert(input_cb != output_cb, "tilize cannot be in-place");
    static_assert(input_cb < 32 && output_cb < 32, "cb id must be < 32");

    const bool asymmetric_cb_pages = total_input_pages.has_value();

    // Upfront wait (when requested)
    if constexpr (wait_mode == tilize_config::WaitMode::WaitUpfront) {
        const uint32_t total_wait =
            asymmetric_cb_pages ? *total_input_pages : (block_width_tiles * num_blocks);
        cb_wait_front(input_cb, total_wait);
    }

    uint32_t pages_left = asymmetric_cb_pages ? *total_input_pages : 0u;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        uint32_t input_pages = block_width_tiles;
        if (asymmetric_cb_pages) {
            input_pages = (pages_left < 32) ? pages_left : 32;
        }

        if constexpr (wait_mode == tilize_config::WaitMode::WaitBlock) {
            cb_wait_front(input_cb, input_pages);
        }

        cb_reserve_back(output_cb, block_width_tiles);

        tilize_block_host(input_cb, block_width_tiles, output_cb);

        cb_push_back(output_cb, block_width_tiles);
        cb_pop_front(input_cb, input_pages);

        if (asymmetric_cb_pages) {
            pages_left -= input_pages;
        }
    }
}

}  // namespace compute_kernel_lib
