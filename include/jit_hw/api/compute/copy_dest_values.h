// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Copy dest values stub for emulated mode
// Delegates to common.h where copy_tile/copy_tile_to_dst_init_short are implemented.
// This header exists because D2M-generated untilize kernels include it directly.
#include "api/compute/common.h"

// copy_block: batch version — copy ntiles starting at start_tile into DST[start_dst..]
inline void copy_block(uint32_t icb, uint32_t start_tile, uint32_t ntiles, uint32_t start_dst) {
    for (uint32_t i = 0; i < ntiles; i++)
        ckernel::copy_tile(icb, start_tile + i, start_dst + i);
}
