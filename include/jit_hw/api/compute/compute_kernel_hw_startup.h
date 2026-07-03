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
// #46346: SrcOrder selects how icb0/icb1 map onto SrcA/SrcB (matmul uses Reverse).
// emule's startup is a state reset, so the mapping is accepted-and-ignored.
enum class SrcOrder : uint8_t { Regular = 0, Reverse = 1 };
inline void compute_kernel_hw_startup(uint32_t, uint32_t) {
    __emule_compute_ctx().llk_pack_offset = 0;
    __emule_compute_ctx().llk_pack_is_untilize = false;
    __emule_compute_ctx().llk_unpack_is_tilize = false;
    __emule_compute_ctx().llk_matmul_transpose = false;
    // Reset PACK engine auto-advance offsets and L1 acc flag to prevent
    // stale state from prior kernel invocations in the same thread.
    std::memset(__emule_compute_ctx().pack_offset, 0, sizeof(__emule_compute_ctx().pack_offset));
    __emule_compute_ctx().l1_acc_enabled = false;
    // Same rationale for the pack-subrect (TTI_SETADCXX / _llk_pack_mop_config_)
    // capture — restore the full-tile defaults so a prior kernel's sub-rectangle
    // config doesn't leak into this one.
    __emule_reset_pack_subrect();
}
template <SrcOrder src_order = SrcOrder::Regular>
inline void compute_kernel_hw_startup(uint32_t a, uint32_t b, uint32_t) {
    (void)src_order;  // mapping is ignored — emule resets the same state regardless
    compute_kernel_hw_startup(a, b);
}
#endif
