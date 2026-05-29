// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for llk_math_eltwise_unary_sfpu_mask.h. Provides no-op LLK
// symbols so mask.h parses and SDPA compute kernels JIT-compile. Real
// masking logic is unstubbed — kernels needing functional masks must
// route through an emule-aware path (currently none).

#include <cstdint>
#include "jit_hw/api/compute/common_globals.h"

namespace ckernel {

inline void llk_math_eltwise_unary_sfpu_mask_init() {}

template <bool APPROX = true>
inline void llk_math_eltwise_unary_sfpu_mask(uint32_t /*idst*/,
                                              DataFormat /*data_format*/ = DataFormat::Float16_b) {}

template <bool APPROX = true>
inline void llk_math_eltwise_unary_sfpu_mask_posinf(uint32_t /*idst*/) {}

}  // namespace ckernel
