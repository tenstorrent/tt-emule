// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Resolver so metal-layer SFPU headers' (e.g. ckernel_sfpu_piecewise_rational.h,
// ckernel_sfpu_threshold.h)
//   #include "sfpu/ckernel_sfpu_converter.h"
// resolves on emule's JIT -I set (jit_hw is first; the LLK common/inc dir is not
// on the path), same gap that ckernel_sfpu_rsqrt_compat.h fills.
//
// Intentionally EMPTY: emule's sfpi backend already defines
// ckernel::sfpu::Converter (see sfpi.h) with the same as_float(uint32_t)->float
// contract as the real tt-llk header. Forwarding to the real header would
// redefine that class. The include only needs to exist here; the symbol is
// supplied by sfpi.h, which the metal-layer SFPU headers already pull in.
