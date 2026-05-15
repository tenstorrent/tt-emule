#pragma once
// tt-emule shim for tt-mlir-emitted `#include "api/compute/compute_kernel_hw_startup.h"`.
//
// tt-mlir PR #7926 (Apr 15 2026) added per-API include emission, so D2M-generated
// matmul/eltwise kernels now ship this `#include`. The upstream tt-metal version
// pulls in the entire LLK setup surface (`llk_*_hw_configure`, `llk_pack_init`,
// `llk_*_set_fp32_dest_acc`, …) which doesn't apply to the emulator.
//
// This shim is intentionally empty: `compute_kernel_hw_startup` is defined in
// `llk_defs.h`, which is already pulled in by `jit_kernel_stubs.hpp` at the top
// of every JIT wrapper. The only purpose of this file is to intercept the
// `#include` so the JIT compile's `-I` search doesn't fall through to the
// upstream tt-metal header.
