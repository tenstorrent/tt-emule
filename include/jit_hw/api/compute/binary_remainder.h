// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// No-op shim mirroring tt_metal/hw/inc/api/compute/binary_remainder.h.
//
// Fat-dispatcher dead-code: pulled in by eltwise_binary_sfpu kernels;
// executed only when BINARY_SFPU_OP selects (remainder_int32, remainder).
// Upstream's body references undefined `APPROX` + unstubbed
// llk_math_eltwise_binary_sfpu_remainder_*. Empty bodies parse and never
// execute. Swap for real semantics if a future test selects this op.

#include <cstdint>
#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void remainder_int32_tile(uint32_t /*idst0*/, uint32_t /*idst1*/, uint32_t /*odst*/) {}
ALWI void remainder_int32_tile_init() {}

ALWI void remainder_binary_tile(uint32_t /*idst0*/, uint32_t /*idst1*/, uint32_t /*odst*/) {}
ALWI void remainder_binary_tile_init() {}

} // namespace ckernel
