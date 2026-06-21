// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Emule shadow for blaze's custom `topk_xl` compute op (bitonic-sort TopK over
// 512/1024/2048 elements, value|index fused as bf16|u16 in fp32).
//
// Post-#1383 ("Get rid of all relative includes"), blaze kernels include this
// header by its repo-root-rooted path
//   blaze/kernels/kernel_includes/tt_metal/include/compute_kernel_api/topk_xl.h
// resolved via the tt-blaze root that the KernelDescriptor passes in
// compiler_include_paths. The emule JIT puts jit_hw FIRST on the -I list, so
// this file shadows blaze's silicon header (which pulls in the per-TRISC
// llk_*_topk_xl_* / sfpu topk_mop_config LLK sort network emule can't consume)
// and instead routes to the unconditional high-level emule model. blaze's tree
// stays pristine.
#include "api/compute/topk_xl.h"
