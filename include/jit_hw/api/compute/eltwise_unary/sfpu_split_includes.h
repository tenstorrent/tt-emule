// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Conditional SFPU includes for emulation (mirrors real sfpu_split_includes.h).
// Only includes headers for operations that have emule stub implementations.

#if SFPU_OP_BINOP_WITH_SCALAR_INCLUDE
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

#if SFPU_OP_FILL_INCLUDE
#include "api/compute/eltwise_unary/fill.h"
#endif

#if SFPU_OP_UNARY_COMP_INCLUDE
#include "api/compute/eltwise_unary/comp.h"
#endif

// SILU, SIGMOID, TANH, and many other "common" SFPU ops are routed through
// the default macro `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` by
// ttnn/.../unary/common/unary_op_utils.cpp:get_macro_definition (default
// arm). Map that to emule's compute_kernel_api.h, which exposes the
// canonical `*_tile` / `*_tile_init` symbols (silu_tile, gelu_tile, etc.).
#if SFPU_OP_COMPUTE_KERNEL_API_INCLUDE
#include "api/compute/compute_kernel_api.h"
#endif
