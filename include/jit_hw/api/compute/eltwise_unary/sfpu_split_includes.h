// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Conditional SFPU includes for emulation. Mirrors upstream
//   tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
// — same SFPU_OP_*_INCLUDE guards, same header paths. emule only conditionally
// includes the subset of headers it has shims for. Other guards' headers are
// genuine LLK gaps tracked per-op; the guard simply not firing here means a
// kernel that triggers it will surface the gap at JIT-compile time.
//
// The guards are JIT-time #defines emitted by upstream's
//   unary_op_utils::get_macro_definition()
// when the host lowers a specific SFPU op into the eltwise_sfpu / eltwise_sfpu
// fat-dispatcher kernels (compute/eltwise_sfpu.cpp + binary_sfpu_*.cpp). The
// kernel calls into the op's `_tile` / `_tile_init` symbols which only exist
// when this header pulls the right op header in.
//
// Adding a missing guard here unblocks every kernel that ever JIT-instantiates
// that op family without needing to touch the kernels themselves.

#if SFPU_OP_EXP_INCLUDE
#include "api/compute/eltwise_unary/exp.h"
#endif

#if SFPU_OP_ERF_ERFC_INCLUDE
#include "api/compute/eltwise_unary/erf_erfc.h"
#endif

#if SFPU_OP_LOGICAL_NOT_INCLUDE
#include "api/compute/eltwise_unary/logical_not.h"
#endif

#if SFPU_OP_GELU_INCLUDE
#include "api/compute/eltwise_unary/gelu.h"
#endif

#if SFPU_OP_SQRT_INCLUDE
#include "api/compute/eltwise_unary/sqrt.h"
#endif

#if SFPU_OP_RSQRT_INCLUDE
#include "api/compute/eltwise_unary/rsqrt.h"
#endif

#if SFPU_OP_RECIP_INCLUDE
#include "api/compute/eltwise_unary/recip.h"
#endif

#if SFPU_OP_RELU_FAMILY_INCLUDE
#include "api/compute/eltwise_unary/relu.h"
#endif

#if SFPU_OP_NEG_INCLUDE
#include "api/compute/eltwise_unary/negative.h"
#endif

#if SFPU_OP_TRIG_FAMILY_INCLUDE
#include "api/compute/eltwise_unary/trigonometry.h"
#endif

#if SFPU_OP_TYPECAST_INCLUDE
#include "api/compute/eltwise_unary/typecast.h"
#endif

#if SFPU_OP_BITWISE_NOT_INCLUDE
#include "api/compute/eltwise_unary/bitwise_not.h"
#endif

#if SFPU_OP_ROUND_FAMILY_INCLUDE
#include "api/compute/eltwise_unary/rounding.h"
#endif

#if SFPU_OP_BINOP_WITH_SCALAR_INCLUDE
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

#if SFPU_OP_SELU_INCLUDE
#include "api/compute/eltwise_unary/selu.h"
#endif

#if SFPU_OP_FILL_INCLUDE
#include "api/compute/eltwise_unary/fill.h"
#endif

#if SFPU_OP_LOG1P_INCLUDE
#include "api/compute/eltwise_unary/log1p.h"
#endif

#if SFPU_OP_UNARY_COMP_INCLUDE
#include "api/compute/eltwise_unary/comp.h"
#endif

#if SFPU_OP_ACTIVATIONS_INCLUDE
#include "api/compute/eltwise_unary/activations.h"
#endif

#if SFPU_OP_WHERE_INCLUDE
#include "api/compute/eltwise_unary/where.h"
#endif

#if SFPU_OP_CLAMP_INCLUDE
#include "api/compute/eltwise_unary/clamp.h"
#endif

#if SFPU_OP_RPOW_INCLUDE
#include "api/compute/eltwise_unary/rpow.h"
#endif

#if SFPU_OP_COMPUTE_KERNEL_API_INCLUDE
// Catch-all for "common" SFPU activations (silu, tanh, gelu, etc.) that
// upstream routes through the default arm of unary_op_utils::get_macro_definition.
#include "api/compute/compute_kernel_api.h"
#endif
