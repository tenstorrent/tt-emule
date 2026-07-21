// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Emule shim for `llk_math_binary_api.h`. Upstream is a thin aggregator that
// pulls in `llk_math_common_api.h` and `llk_math_eltwise_binary.h` plus a
// handful of `llk_math_eltwise_binary_init<>` wrappers.  Emule already has
// `llk_math_eltwise_binary.h`; surface it under this name as well so kernel
// includes that say `#include "llk_math_binary_api.h"` resolve.

#include <cstdint>
#include "llk_math_eltwise_binary.h"

// `llk_math_eltwise_binary_init<>` (the operand/acc form) lives in
// `llk_math_eltwise_binary.h` (included above) with the correct non-type enum
// template params. Don't redeclare it here — the template params are
// `ckernel::EltwiseBinaryType` / `BroadcastType` / `MathFidelity` VALUES at the
// call site, not `typename`s.

// `llk_math_eltwise_binary_init_with_operands<op, bcast, fidelity>(icb_a, icb_b)`
// — silicon configures the unpacker/math pipeline for the binary op; a no-op under
// emule (the math is done inline by llk_math_eltwise_binary<>). The kernel_lib
// (binary_op_helpers.inl) instantiates this with enum VALUES, so the template
// parameters are non-type, mirroring llk_math_eltwise_binary_init.
template <
    ckernel::EltwiseBinaryType /*eltwise_binary_type*/,
    ckernel::BroadcastType /*src_b_bcast_type*/,
    ckernel::MathFidelity /*math_fidelity*/>
inline void llk_math_eltwise_binary_init_with_operands(
    std::uint32_t /*src_a_operand*/, std::uint32_t /*src_b_operand*/,
    const std::uint32_t /*acc_to_dest*/ = 0) {}
