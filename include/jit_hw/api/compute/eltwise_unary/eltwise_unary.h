// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// SFPU unary init — no-op in emulation.
#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void unary_op_init_common(uint32_t /*cb_in*/, uint32_t /*cb_out*/) {}
ALWI void init_sfpu(uint32_t /*cb_in*/, uint32_t /*cb_out*/) {}

} // namespace ckernel
