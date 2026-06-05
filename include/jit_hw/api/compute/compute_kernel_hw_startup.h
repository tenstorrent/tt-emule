#pragma once
// Emule entry point for D2M-generated compute kernels. Owns
// `compute_kernel_hw_startup` and pulls in the per-domain LLK function
// headers a compute kernel might transitively need. Intercepts the include
// path so the JIT compile's `-I` search doesn't fall through to the
// upstream tt-metal header (which references unstubbed LLK setup APIs).
#include "jit_hw/llk_types.h"
#include "jit_hw/internal/llk_state.h"
#include "jit_hw/api/compute/common.h"
#include "jit_hw/llk_math_eltwise_unary_datacopy.h"
#include "jit_hw/llk_pack.h"
#include "jit_hw/llk_unpack_a.h"
#include "jit_hw/llk_sync_stubs.h"
#include <cstring>

// ---- Compute startup and init ----
// Guard against double-definition: jit_kernel_stubs.hpp also has stub
// overloads of the same signatures (for kernels that don't include this
// header directly). If both files are included, first-wins via this macro.
#ifndef __EMULE_COMPUTE_KERNEL_HW_STARTUP_DEFINED
#define __EMULE_COMPUTE_KERNEL_HW_STARTUP_DEFINED
inline void compute_kernel_hw_startup(uint32_t, uint32_t) {
    __llk_pack_offset = 0;
    __llk_pack_is_untilize = false;
    __llk_unpack_is_tilize = false;
    // Reset PACK engine auto-advance offsets and L1 acc flag to prevent
    // stale state from prior kernel invocations in the same thread.
    std::memset(__emule_pack_offset, 0, sizeof(__emule_pack_offset));
    __emule_l1_acc_enabled = false;
}
inline void compute_kernel_hw_startup(uint32_t a, uint32_t b, uint32_t) {
    compute_kernel_hw_startup(a, b);
}
#endif
