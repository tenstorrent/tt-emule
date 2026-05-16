#pragma once
// tt-emule shim for tt-mlir-emitted `#include "api/compute/compute_kernel_hw_startup.h"`.
//
// This shim used to be an empty include-only intercept of the upstream
// `api/compute/compute_kernel_hw_startup.h`, delegating the body to
// `jit_hw/llk_defs.h`. It is now the OWNER of `compute_kernel_hw_startup`
// and the canonical entry point for D2M-generated compute kernels into
// tt-emule's per-domain LLK surface (split out of the original monolithic
// `llk_defs.h`).
//
// D2M compute kernels `#include` this shim per tt-mlir PR #7926 (Apr 15
// 2026, per-API include emission), so all the LLK function headers a
// compute kernel might transitively need are pulled in below.
#include "jit_hw/llk_types.h"
#include "jit_hw/internal/llk_state.h"
#include "jit_hw/api/compute/common.h"
#include "jit_hw/llk_math_eltwise_unary_datacopy.h"
#include "jit_hw/llk_pack.h"
#include "jit_hw/llk_unpack_a.h"
#include "jit_hw/llk_sync_stubs.h"
#include <cstring>

// ---- Compute startup and init ----
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
