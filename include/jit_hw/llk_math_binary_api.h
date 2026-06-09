// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Emule shim for `llk_math_binary_api.h`. Upstream is a thin aggregator that
// pulls in `llk_math_common_api.h` and `llk_math_eltwise_binary.h` plus a
// handful of `llk_math_eltwise_binary_init<>` wrappers.  Emule already has
// `llk_math_eltwise_binary.h`; surface it under this name as well so kernel
// includes that say `#include "llk_math_binary_api.h"` resolve.

#include "llk_math_eltwise_binary.h"

// `llk_math_eltwise_binary_init<>` — silicon configures the unpacker/math
// state for the binary op.  Under emule the math is performed inline by the
// `llk_math_eltwise_binary<>` body (in `llk_math_eltwise_binary.h`); this
// init can be a no-op.
template <
    typename EltwiseBinaryType,
    typename BroadcastType,
    int math_fidelity,
    typename EltwiseBinaryReuseDestType>
inline void llk_math_eltwise_binary_init(const std::uint32_t /*acc_to_dest*/ = 0) {}

// Variants taking concrete cb_ids and full shape, used by some kernel
// templates.  All no-ops under emule.
template <typename EltwiseBinaryType, typename BroadcastType, int math_fidelity>
inline void llk_math_eltwise_binary_init_with_operands(
    std::uint32_t /*src_a_operand*/, std::uint32_t /*src_b_operand*/,
    const std::uint32_t /*acc_to_dest*/ = 0) {}
