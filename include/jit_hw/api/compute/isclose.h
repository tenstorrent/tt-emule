// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// No-op shim mirroring tt_metal/hw/inc/api/compute/isclose.h.
//
// Same fat-dispatcher dead-code pattern as atan2.h — pulled in transitively
// by eltwise_binary_sfpu kernels but only executed when the kernel's
// BINARY_SFPU_OP selects isclose. If a future kernel selects this, swap the
// empty bodies for real semantics.

#include <cstdint>
#include "jit_hw/api/compute/common.h"

namespace ckernel {

template <bool EQUAL_NAN = false>
ALWI void isclose_binary_tile(uint32_t /*idst0*/, uint32_t /*idst1*/,
                              uint32_t /*odst*/, uint32_t /*rtol_bits*/,
                              uint32_t /*atol_bits*/) {}
ALWI void isclose_binary_tile_init() {}

} // namespace ckernel
