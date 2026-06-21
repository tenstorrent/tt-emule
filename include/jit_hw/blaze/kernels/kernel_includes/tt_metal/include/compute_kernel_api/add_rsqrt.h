// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Emule shadow for blaze's custom `add_rsqrt` compute op (rsqrt(x + addend),
// used by the RMSNorm family for rsqrt(variance + epsilon)).
//
// blaze's silicon header includes the per-TRISC SFPU LLK
// (llk_math_eltwise_unary_sfpu_add_rsqrt) under TRISC_MATH, which emule can't
// consume. The emule JIT puts jit_hw FIRST on the -I list, so this file shadows
// blaze's header (resolved by its repo-root-rooted include path) and routes to
// the unconditional high-level emule model. blaze's tree stays pristine.
#include "api/compute/eltwise_unary/add_rsqrt.h"
