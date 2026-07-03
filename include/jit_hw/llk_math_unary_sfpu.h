// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// LLK unary SFPU stubs. Silicon's init functions configure SFPU op tables
// in firmware; emule has no SFPU vector hardware, so all inits are no-ops.
// The actual SFPU math lives in the api/compute/eltwise_unary/*.h tile-ops
// (exp_tile, sigmoid_tile, etc.) which operate directly on __emule_compute_ctx().dst.

#include <cstdint>

#include "jit_hw/ckernel.h"     // ckernel::SfpuType
#include "jit_hw/llk_types.h"   // ckernel::DataCopyType

// Generic LLK unary SFPU init — parameterized by the SfpuType being set up.
template <ckernel::SfpuType /*Op*/ = ckernel::SfpuType::unused, int /*AccumMode*/ = 0>
inline void llk_math_eltwise_unary_sfpu_init(uint32_t /*operand*/ = 0) {}

// SFPU "param init" form — extra runtime arg (e.g. scaling factor).
template <ckernel::SfpuType /*Op*/, int /*AccumMode*/ = 0>
inline void llk_math_eltwise_unary_sfpu_init_param(uint32_t /*param*/) {}

// kernel-specific moe_gate topk init. The real silicon impl configures SFPU
// state for the deepseek MoE gate top-k op. Emule no-op; the actual math
// must run in the consumer SFPU tile-op (TODO: identify and implement that).
template <bool /*APPROX*/ = false, int /*DST_ACCUM_MODE*/ = 0>
inline void llk_math_sfpu_deepseek_moe_gate_topk_init() {}
