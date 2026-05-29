// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for clamp SFPU tile op.
// Math impl deferred; no-op stub compiles and lets D2M-emitted kernels link.

#include <cstdint>

namespace ckernel {

ALWI void clamp_tile_init() {}

ALWI void clamp_tile(uint32_t idst, uint32_t /*param0*/, uint32_t /*param1*/) {
    __emule_dst_check(idst, "clamp_tile");
    // TODO: implement actual clamp math against param0/param1 (float-packed).
}

ALWI void clamp_tile_int32(uint32_t idst, uint32_t /*param0*/, uint32_t /*param1*/) {
    __emule_dst_check(idst, "clamp_tile_int32");
    // TODO: implement actual clamp math against param0/param1 (int32-packed).
}

} // namespace ckernel
