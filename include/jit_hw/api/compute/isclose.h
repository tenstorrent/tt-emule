// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Emule shim for `api/compute/isclose.h`. Same rationale as atan2.h: the
// binary_ng eltwise_binary_sfpu_scalar.cpp kernel transitively #includes this
// header, but only one binary SFPU op is selected per JIT instantiation.
// Provide no-op definitions so unselected ops parse. Real header at
//   /localdev/arminale/tt-metal/tt_metal/hw/inc/api/compute/isclose.h

#include "jit_hw/api/compute/common.h"

namespace ckernel {

template <bool EQUAL_NAN = false>
ALWI void isclose_binary_tile(uint32_t /*idst0*/, uint32_t /*idst1*/, uint32_t /*odst*/,
                              uint32_t /*rtol_bits*/, uint32_t /*atol_bits*/) {
    // No-op stub.
}

ALWI void isclose_binary_tile_init() {
    // No-op stub.
}

}  // namespace ckernel
