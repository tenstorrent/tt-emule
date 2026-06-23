// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Emule shadow for api/compute/experimental/pack_block.h (Blackhole
// block-contiguous pack of tiny tiles). The real header routes to the
// llk_pack_block_contiguous{,_mop_config} LLKs, which are only declared under
// TRISC_PACK — but emule's PACK(x)=x references them in every TRISC variant, so
// the real header fails to compile (undeclared identifier). The net effect is a
// straight DST→CB block pack, already provided by emule's pack_tile_block.
#pragma once

#include "api/compute/common.h"

namespace ckernel {

ALWI void pack_block_contiguous_init(uint32_t /*ocb*/) {}

// Pack num_tiles tiles from DEST[ifrom_dst ..] to consecutive CB[ocb] slots.
ALWI void pack_block_contiguous(uint32_t ifrom_dst, uint32_t ocb, uint32_t num_tiles) {
    pack_tile_block(ifrom_dst, ocb, num_tiles);
}

}  // namespace ckernel
