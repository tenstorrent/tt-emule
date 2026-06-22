// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Emule shadow for blaze/ops/local_top_k/kernels/run_top32_llk.hpp.
// cross_core_topk_merge and cross_device_top32_merge include the header by this
// full blaze/ops path (local_top_k itself uses the bare name — see
// jit_hw/run_top32_llk.hpp). The jit_hw -I root resolves ahead of the blaze
// tree, so this shadows the real header (which pulls the row-major top-32 SFPU
// LLKs emule can't consume) with the high-level emule model. See
// api/compute/experimental/top32_rm.h.
#pragma once

#include "api/compute/experimental/top32_rm.h"
