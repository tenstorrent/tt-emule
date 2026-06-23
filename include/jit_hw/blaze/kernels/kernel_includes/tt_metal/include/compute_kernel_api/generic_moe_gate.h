// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Emule shadow for blaze's custom `generic_moe_gate` compute op. Sibling of the
// deepseek/glm/kimi moe_gate shadows. The emule JIT puts jit_hw FIRST on the -I
// list, so this shadows blaze's silicon header (which pulls the
// llk_math_deepseek_moe_gate_eltwise_binary + generic-moe-gate topk SFPU LLKs
// emule can't consume) and routes to the high-level emule model. blaze stays
// pristine.
#include "api/compute/generic_moe_gate.h"
