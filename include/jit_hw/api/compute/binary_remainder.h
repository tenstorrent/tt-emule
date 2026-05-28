// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Emule shim for `api/compute/binary_remainder.h`. The real header at
//   tt_metal/hw/inc/api/compute/binary_remainder.h
// references undefined APPROX (a build-time switch on real Tensix) and the
// unstubbed llk_math_eltwise_binary_sfpu_(binary_)remainder.
//
// Same situation as atan2.h: the binary_ng SFPU dispatcher kernel
//   ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp
// transitively #includes all ~15 binary SFPU ops but only one is selected per
// JIT instantiation via the BINARY_SFPU_OP macro. Provide no-op bodies so the
// unselected paths parse; if remainder is ever the selected op, the no-op
// will silently produce zero and a downstream PCC check would catch it.

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void remainder_int32_tile(uint32_t /*idst0*/, uint32_t /*idst1*/, uint32_t /*odst*/) {}
ALWI void remainder_int32_tile_init() {}

ALWI void remainder_binary_tile(uint32_t /*idst0*/, uint32_t /*idst1*/, uint32_t /*odst*/) {}
ALWI void remainder_binary_tile_init() {}

}  // namespace ckernel
