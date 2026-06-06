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

namespace ckernel {

// copy_dest_values — DST-to-DST tile copy. Mapped here by tt-mlir's
// TTKernelIncludesMap.h. On real silicon this routes through SFPU; on
// emule, DST is a flat float array so it's a per-element copy.
ALWI void copy_dest_values_init() {}

ALWI void copy_dest_values(uint32_t dst_src, uint32_t dst_dst) {
    __emule_dst_check(dst_src, "copy_dest_values src");
    __emule_dst_check(dst_dst, "copy_dest_values dst");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[dst_dst][i] = __emule_dst[dst_src][i];
}

// Templated DataFormat overload (MPWI copies UInt16/UInt32 index tiles). The
// copy is bit-preserving in the flat float DST, so integer bit-patterns survive
// regardless of DATA_FORMAT — same body as above.
template <DataFormat DATA_FORMAT>
ALWI void copy_dest_values(uint32_t dst_src, uint32_t dst_dst) {
    __emule_dst_check(dst_src, "copy_dest_values src");
    __emule_dst_check(dst_dst, "copy_dest_values dst");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[dst_dst][i] = __emule_dst[dst_src][i];
}

} // namespace ckernel
