// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Narrow stub so the metal-layer ckernel_sfpu_sqrt.h's
//   #include "cmath_common.h"
// resolves on emule's JIT -I set (jit_hw is first; the LLK common/inc dir is not on the
// path). Only sqrt_init()'s `math::reset_counters(p_setrwc::SET_ABD_F)` call is exercised —
// a hardware replay-buffer/counter reset with no counterpart under emule's x86 execution
// model, so a no-op is the faithful functional equivalent (mirrors the other RWC/semaphore
// stubs in ckernel.h). Widening emule's JIT -I set to reach the real cmath_common.h instead
// was tried and rejected: it transitively pulls in the real ckernel.h, which conflicts with
// this project's own ckernel.h shim (e.g. `p_stall` redefined as struct vs namespace) and
// needs further real-only headers (cfg_defines.h) emule doesn't provide.
#include <cstdint>

struct p_setrwc {
    static constexpr std::uint32_t SET_ABD_F = 0xf;
};

namespace ckernel::math {
inline void reset_counters(std::uint32_t /*setrwc*/) {}
}  // namespace ckernel::math
