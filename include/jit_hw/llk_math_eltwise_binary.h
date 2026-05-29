// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Empty shim. The reduce_w_neg / reduce_h_neg compute kernels include this
// header but never reference any symbol from it (likely a refactor leftover
// in upstream). Providing a no-op resolution lets the JIT compile succeed
// without pulling in the real ~700-line LLK header.
//
// TODO: file an upstream cleanup PR to drop the include from
// ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_*_neg.cpp.

// moreh_common.hpp calls `llk_math_eltwise_binary_init<EltwiseBinaryType,
// BroadcastType, MathFidelity>()`. Accept enums via `auto` template params
// (C++17) so we don't need to include the enum definitions here.
#include <cstdint>
namespace ckernel {
template <auto EltwiseBinaryType_v = 0, auto BroadcastType_v = 0, auto MathFidelity_v = 0>
inline void llk_math_eltwise_binary_init(uint32_t /*transpose*/ = 0,
                                          uint32_t /*acc_to_dest*/ = 0) {}
}
