// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// No-op shim mirroring tt_metal/hw/inc/api/compute/atan2.h.
//
// atan2 is one of ~15 binary SFPU op headers transitively #included by
// eltwise_sfpu fat-dispatcher kernels (eltwise_binary_sfpu_no_bcast.cpp,
// eltwise_binary_sfpu_scalar.cpp). Each JIT instantiation selects exactly one
// op via the BINARY_SFPU_OP define; the unselected op headers are dead code
// but still parsed by C++. Upstream's atan2.h references undefined `APPROX`
// and unstubbed `llk_math_eltwise_binary_sfpu_atan2*`, so the real header
// can't compile on host — emule provides empty bodies that parse and never
// execute.
//
// If a future kernel ever selects atan2 (BINARY_SFPU_OP=atan2 or similar),
// PCC will fail at that test — implement the real semantics here then.

#include <cstdint>
#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void atan2_binary_tile(uint32_t /*idst0*/, uint32_t /*idst1*/, uint32_t /*odst*/) {}
ALWI void atan2_binary_tile_init() {}

} // namespace ckernel
