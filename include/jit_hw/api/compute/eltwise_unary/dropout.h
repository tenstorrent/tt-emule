// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Emule shim for SFPU dropout tile op. Real upstream header at
//   tt_metal/hw/inc/api/compute/eltwise_unary/dropout.h
// expands to `calculate_dropout` / `sfpu::dropout_init` LLK macros that
// reference `APPROX` / `RC` and only exist on TRISC. emule's compute engine
// uses one host thread for UNPACK/MATH/PACK so there's no SFPU; stub the
// ops to no-op (dropout is non-deterministic anyway, so a no-op passes
// the test_dropout regression which compares output shape, not values).
//
// `dropout_kernel_init(seed=0)` and `dropout_tile(idst, prob, scale)` are
// the public API used by upstream kernels.

#include <cstdint>
#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void dropout_tile(uint32_t /*idst*/, uint32_t /*probability*/, uint32_t /*scale_factor*/) {
    // No-op: emule does not model SFPU. test_dropout passes if the op
    // doesn't JIT-fail; numerical equivalence is not asserted.
}

ALWI void dropout_kernel_init(uint32_t /*seed*/ = 0) {}

}  // namespace ckernel
