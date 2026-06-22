// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Emule shadow for blaze's blaze/ops/local_top_k/kernels/run_top32_llk.hpp.
// local_top_k's kernel includes it by the bare name "run_top32_llk.hpp", which
// resolves here via the jit_hw -I root (ahead of the runner-relocated kernel
// dir). Routes the kernel-author API to the high-level emule model, bypassing
// the row-major top-32 SFPU LLK headers emule can't consume. See
// api/compute/experimental/top32_rm.h.
#pragma once

#include "api/compute/experimental/top32_rm.h"
