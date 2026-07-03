// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// LLK math binary shim. Originally an empty no-op for reduce_w_neg /
// reduce_h_neg (which include this header but never reference any symbol).
// Now also provides llk_math_eltwise_binary_init — the init function
// tt-metal's compute kernels (e.g. ttnn/.../moreh_common.hpp) call to set up
// binary-op state. Silicon configures the LLK hardware; emule has no LLK
// vector hardware, so init is a no-op — the actual binary math happens in
// the api/compute/eltwise_*/*.h tile-ops.

#include <cstdint>

#include "jit_hw/api/compute/common.h"   // EltwiseBinaryType, BroadcastType, EltwiseBinaryReuseDestType
#include "jit_hw/api/compute/bcast.h"    // any_tiles_bcast
#include "jit_hw/llk_types.h"             // MathFidelity

// Mirrors tt-metal's tt_metal/hw/ckernels/blackhole/metal/llk_api/
// llk_math_binary_api.h:31 signature. All template params are forwarded
// from the call site verbatim; runtime args are accepted-and-ignored.
template <
    ckernel::EltwiseBinaryType /*eltwise_binary_type*/,
    ckernel::BroadcastType /*src_b_bcast_type*/,
    ckernel::MathFidelity /*math_fidelity*/,
    ckernel::EltwiseBinaryReuseDestType /*binary_reuse_dest*/ = ckernel::EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary_init(
    const std::uint32_t /*operand_A*/,
    const std::uint32_t /*operand_B*/,
    const std::uint32_t /*acc_to_dest*/ = 0) {}

template <
    ckernel::EltwiseBinaryType eltwise_binary_type,
    ckernel::BroadcastType src_b_bcast_type,
    bool /*dst_sync*/,
    ckernel::MathFidelity /*math_fidelity*/,
    ckernel::EltwiseBinaryReuseDestType /*binary_reuse_dest*/ = ckernel::EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary(
    const std::uint32_t operand_A,
    const std::uint32_t operand_B,
    const std::uint32_t dst_index,
    const bool acc_to_dest = false) {
    auto& ctx = __emule_compute_ctx();
    const std::uint32_t icb0 = ctx.llk_binary_icb0;
    const std::uint32_t icb1 = ctx.llk_binary_icb1;
    const std::uint32_t itile0 = ctx.llk_binary_itile0;
    const std::uint32_t itile1 = ctx.llk_binary_itile1;
    (void)operand_A;
    (void)operand_B;
    if (acc_to_dest) {
        ckernel::any_tiles_bcast<eltwise_binary_type, src_b_bcast_type, true>(
            icb0, icb1, itile0, itile1, dst_index);
    } else {
        ckernel::any_tiles_bcast<eltwise_binary_type, src_b_bcast_type, false>(
            icb0, icb1, itile0, itile1, dst_index);
    }
}
