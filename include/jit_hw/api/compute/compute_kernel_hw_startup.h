#pragma once
// tt-emule shim for tt-mlir-emitted `#include "api/compute/compute_kernel_hw_startup.h"`.
//
// tt-mlir PR #7926 (Apr 15 2026) added per-API include emission, so D2M-generated
// matmul/eltwise kernels now ship this `#include`. The upstream tt-metal version
// pulls in the entire LLK setup surface (`llk_*_hw_configure`, `llk_pack_init`,
// `llk_*_set_fp32_dest_acc`, …) which doesn't apply to the emulator.
//
// We pull in `llk_defs.h` here (rather than from jit_kernel_stubs.hpp) so the
// LLK stub surface (DataCopyType, UnpackToDestEn, llk_math_eltwise_unary_datacopy,
// __llk_pack_*/__llk_unpack_* state, experimental::pack_untilize_block, …) is
// only visible to kernels that explicitly `#include "compute_kernel_hw_startup.h"`
// — i.e. D2M-generated kernels. Including llk_defs.h in every TU regressed
// the SFPU INT32 unary path (AddUnary/SubUnary), so we scope it here instead.
#include "jit_hw/llk_defs.h"
