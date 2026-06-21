// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Emule shadow for blaze's custom `kimi_moe_gate` compute op.
//
// Post-#1383 ("Get rid of all relative includes"), blaze kernels include this
// header by its repo-root-rooted path
//   blaze/kernels/kernel_includes/tt_metal/include/compute_kernel_api/kimi_moe_gate.h
// resolved via the tt-blaze root that the KernelDescriptor passes in
// compiler_include_paths. The emule JIT puts jit_hw FIRST on the -I list, so
// this file shadows blaze's silicon header (which decomposes into a per-TRISC
// LLK / SFPU sort network emule can't consume) and instead routes to the
// unconditional high-level emule model. blaze's tree stays pristine.
#include "api/compute/kimi_moe_gate.h"
