// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/ema.h`. Exponential moving average down tile rows:
//   out = beta*x + alpha*prev   (first sample of a batch: out = beta*x).
// `prev` is a per-column carry persisted across tiles within a batch and reset
// by ema_clear_previous_output. alpha/beta are fp32 bit-patterns from ema_init.
// Real LLK: tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_ema.h
// Unblocks ema_compute.cpp compilation; ttnn.ema PCC is deferred (the failure is
// in emule's surrounding transpose/pack path, not this recurrence) — see #53.
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

static thread_local float __emule_ema_alpha = 0.0f;
static thread_local float __emule_ema_beta = 0.0f;
static thread_local float __emule_ema_prev[32] = {};
static thread_local bool __emule_ema_has_prev = false;

ALWI void ema_init(uint32_t alpha, uint32_t beta) {
    std::memcpy(&__emule_ema_alpha, &alpha, sizeof(float));
    std::memcpy(&__emule_ema_beta, &beta, sizeof(float));
}

ALWI void ema_clear_previous_output() { __emule_ema_has_prev = false; }

ALWI void ema_tile(uint32_t input_dst_index) {
    __emule_dst_check(input_dst_index, "ema_tile");
    for (uint32_t row = 0; row < 32; ++row) {
        for (uint32_t col = 0; col < 32; ++col) {
            float x = __emule_dst[input_dst_index][row * 32 + col];
            float out = __emule_ema_has_prev ? __emule_ema_beta * x + __emule_ema_alpha * __emule_ema_prev[col]
                                             : __emule_ema_beta * x;
            __emule_dst[input_dst_index][row * 32 + col] = out;
            __emule_ema_prev[col] = out;
        }
        __emule_ema_has_prev = true;
    }
}

}  // namespace ckernel
