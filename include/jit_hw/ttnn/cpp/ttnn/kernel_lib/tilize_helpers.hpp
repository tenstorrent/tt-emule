// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shadow for ttnn `kernel_lib/tilize_helpers.hpp`.
//
// Provides a minimal compute_kernel_lib::tilize<>() that copies pages
// from input CB to output CB. Real silicon does fp32→bf16 conversion
// and tile-format layout. In emule, page data is already in tile layout
// (host-side tilization happened in from_torch), so a memcpy-equivalent
// is correct for bfloat16 inputs.

#include <optional>
#include <cstdint>
#include <cstring>

#include "api/compute/tilize.h"
#include "api/compute/cb_api.h"

namespace compute_kernel_lib {

namespace tilize_config {

constexpr uint32_t INVALID_CB = 32;  // NUM_CIRCULAR_BUFFERS

enum class ReconfigureRegisterDatatypeMode : uint8_t {
    NoReconfigure,
    UnpackOnlyReconfigure,
    UnpackAndPackReconfigure,
};
enum class InitUninitMode : uint8_t {
    InitAndUninit,
    InitOnly,
    UninitOnly,
    Neither,
};
enum class WaitMode : uint8_t {
    WaitBlock,
    NoWait,
};
enum class Fp32Mode : uint8_t {
    Fast,
    Lossless,
};

}  // namespace tilize_config

template <uint32_t input_cb>
ALWI constexpr bool is_fp32_input_format() {
    return false;  // emule: assume bfloat16 inputs unless explicit reconfigure
}

template <
    uint32_t block_width_tiles,
    uint32_t input_cb,
    uint32_t output_cb,
    tilize_config::InitUninitMode init_uninit_mode = tilize_config::InitUninitMode::InitAndUninit,
    tilize_config::WaitMode wait_mode = tilize_config::WaitMode::WaitBlock,
    tilize_config::ReconfigureRegisterDatatypeMode reconfig_mode =
        tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
    tilize_config::Fp32Mode fp32_mode = tilize_config::Fp32Mode::Fast>
ALWI void tilize(uint32_t num_blocks, std::optional<uint32_t> total_input_pages = std::nullopt) {
    (void)total_input_pages;
    // For each block: wait for block_width_tiles input pages, reserve
    // block_width_tiles output pages, copy tile-for-tile, then push/pop.
    for (uint32_t b = 0; b < num_blocks; ++b) {
        if constexpr (wait_mode == tilize_config::WaitMode::WaitBlock) {
            cb_wait_front(input_cb, block_width_tiles);
        } else {
            // NoWait: caller already waited
        }
        cb_reserve_back(output_cb, block_width_tiles);

        uint32_t in_rd = get_read_ptr(input_cb);
        uint32_t out_wr = get_write_ptr(output_cb);
        // Emule encodes fifo_page_size in bytes directly (not 16-byte words).
        uint32_t page_size_in = get_local_cb_interface(input_cb).fifo_page_size;
        uint32_t page_size_out = get_local_cb_interface(output_cb).fifo_page_size;
        uint32_t copy_bytes = (page_size_in < page_size_out) ? page_size_in : page_size_out;
        for (uint32_t t = 0; t < block_width_tiles; ++t) {
            std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(out_wr) + t * page_size_out),
                        reinterpret_cast<const void*>(static_cast<uintptr_t>(in_rd) + t * page_size_in),
                        copy_bytes);
        }

        cb_push_back(output_cb, block_width_tiles);
        cb_pop_front(input_cb, block_width_tiles);
    }
}

}  // namespace compute_kernel_lib
