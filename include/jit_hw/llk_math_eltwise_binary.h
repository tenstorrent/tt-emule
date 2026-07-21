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
#include "jit_hw/llk_types.h"             // MathFidelity
#include "jit_hw/api/compute/bcast.h"    // ckernel::any_tiles_bcast (broadcast-aware binary math)
#include "jit_hw/llk_unpack_AB_api.h"    // __emule_unpack_AB_state() — tile indices stashed by llk_unpack_AB

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

// The actual binary math. Silicon computes SRCA op SRCB → DST[dst_index]; the two
// operands were loaded into the SRC banks by the preceding llk_unpack_AB. Emule has
// no SRC banks — read the two CB tiles directly using the operands + tile indices
// stashed by llk_unpack_AB (see llk_unpack_AB_api.h) and route through the
// broadcast-aware primitive ckernel::any_tiles_bcast. `clear_fp32_dst_acc` true =
// overwrite DST (fresh op, e.g. binary_op_helpers.inl); false = accumulate onto DST
// (the reduce-scaler mask path in reduce_helpers_compute.inl sums products into DST).
template <
    ckernel::EltwiseBinaryType eltwise_binary_type,
    ckernel::BroadcastType src_b_bcast_type,
    ckernel::EltwiseBinaryReuseDestType binary_reuse_dest>
inline void __emule_llk_binary_math(
    std::uint32_t operand_A, std::uint32_t operand_B,
    std::uint32_t tile_a, std::uint32_t tile_b, std::uint32_t dst_index, bool clear_fp32_dst_acc) {
    // BroadcastType::NONE routes through add/sub/mul_tiles, which honour dest_accum_en
    // from the ctx (not a template flag), so set it to make accumulate/overwrite
    // correct regardless of the broadcast dim.
    __emule_compute_ctx().dest_accum_en = !clear_fp32_dst_acc;
    if constexpr (binary_reuse_dest == ckernel::EltwiseBinaryReuseDestType::NONE) {
        if (clear_fp32_dst_acc) {
            ckernel::any_tiles_bcast<eltwise_binary_type, src_b_bcast_type, false>(
                operand_A, operand_B, tile_a, tile_b, dst_index);
        } else {
            ckernel::any_tiles_bcast<eltwise_binary_type, src_b_bcast_type, true>(
                operand_A, operand_B, tile_a, tile_b, dst_index);
        }
    } else {
        // DEST_TO_SRC{A,B}: combine the CB-side operand-B tile with DST[dst_index] in
        // place (not exercised by the current kernel_lib, which always uses NONE).
        ckernel::binary_dest_reuse_tiles<eltwise_binary_type, binary_reuse_dest>(operand_B, tile_b, dst_index);
    }
}

// Overload 1 (dst_index, clear): both operands come from the unpack stash — used by
// reduce_helpers_compute.inl's masked-reduce scaler path.
template <
    ckernel::EltwiseBinaryType eltwise_binary_type,
    ckernel::BroadcastType src_b_bcast_type,
    bool /*is_fp32_dest_acc_en*/,
    ckernel::MathFidelity /*math_fidelity*/,
    ckernel::EltwiseBinaryReuseDestType binary_reuse_dest = ckernel::EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary(std::uint32_t dst_index, const bool clear_fp32_dst_acc = true) {
    const auto& s = __emule_unpack_AB_state();
    __emule_llk_binary_math<eltwise_binary_type, src_b_bcast_type, binary_reuse_dest>(
        s.operand_a, s.operand_b, s.tile_a, s.tile_b, dst_index, clear_fp32_dst_acc);
}

// Overload 2 (operand_A, operand_B, dst_index, clear): operands explicit, tile
// indices from the stash — used by binary_op_helpers.inl's binary_exec.
template <
    ckernel::EltwiseBinaryType eltwise_binary_type,
    ckernel::BroadcastType src_b_bcast_type,
    bool /*is_fp32_dest_acc_en*/,
    ckernel::MathFidelity /*math_fidelity*/,
    ckernel::EltwiseBinaryReuseDestType binary_reuse_dest = ckernel::EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary(
    std::uint32_t operand_A, std::uint32_t operand_B, std::uint32_t dst_index, const bool clear_fp32_dst_acc) {
    const auto& s = __emule_unpack_AB_state();
    __emule_llk_binary_math<eltwise_binary_type, src_b_bcast_type, binary_reuse_dest>(
        operand_A, operand_B, s.tile_a, s.tile_b, dst_index, clear_fp32_dst_acc);
}
