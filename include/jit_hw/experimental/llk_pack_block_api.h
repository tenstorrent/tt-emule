// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Emule shim for `experimental/llk_pack_block_api.h`. Silicon-specific
// block-contiguous pack helpers used by `api/compute/experimental/
// pack_block.h` (Blackhole-only `pack_block_contiguous_init` and
// `pack_block_contiguous`). Under emule the pack engine is a memcpy
// loop, so block-contiguous packing is the same as a regular pack —
// these init helpers can be no-ops.
//
// `pack_block_contiguous` itself is rarely invoked in the
// matmul_fused_act / matmul flow today (it is on the Blackhole-only
// tiny-tile path); emule currently provides a no-op stub so the symbol
// resolves if a kernel includes/references it.

#include <cstdint>

template <bool zero_output = false>
inline void llk_pack_block_contiguous_mop_config(const std::uint32_t /*output*/) {
    // No-op under emule: there is no MOP/addr-mod hardware to configure.
}

template <bool is_fp32_dest_acc_en>
inline void llk_pack_block_contiguous(std::uint32_t /*tile_index*/, std::uint32_t /*output*/,
                                      std::uint32_t /*num_tiles*/) {
    // No-op under emule. The block-contiguous pack path is only taken
    // by Blackhole tiny-tile kernels that haven't been brought up in
    // emule yet; when a future emule allowlist op uses this, replace
    // this body with a `pack_tile`-based memcpy loop over `num_tiles`.
}
