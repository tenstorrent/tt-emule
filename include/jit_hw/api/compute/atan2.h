// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Emule shim for `api/compute/atan2.h`. The real header lives at
//   /localdev/arminale/tt-metal/tt_metal/hw/inc/api/compute/atan2.h
// and references undefined APPROX + the unstubbed llk_math_eltwise_binary_sfpu_atan2.
//
// atan2 is one of ~15 binary SFPU ops that
//   ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp
// transitively #includes — but only ONE op is selected per JIT instantiation
// via the BINARY_SFPU_OP macro. For ops that aren't selected (e.g. atan2 when
// the kernel is JIT'd for multiply) the function bodies are dead code, yet
// C++ still parses them.
//
// Provide a no-op body so the parse succeeds. If atan2 ever becomes the
// selected op, the no-op will silently produce zero output and a downstream
// PCC check would catch it; at that point implement the real SFPU semantics.

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void atan2_binary_tile(uint32_t /*idst0*/, uint32_t /*idst1*/, uint32_t /*odst*/) {
    // No-op stub. See header comment.
}

ALWI void atan2_binary_tile_init() {
    // No-op stub. See header comment.
}

}  // namespace ckernel
