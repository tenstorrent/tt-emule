// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"

namespace ckernel {
ALWI void quant_tile_init(uint32_t) {}
ALWI void quant_tile(uint32_t, uint32_t, uint32_t) {}
ALWI void requant_tile_init(uint32_t) {}
ALWI void requant_tile(uint32_t, uint32_t, uint32_t) {}
ALWI void dequant_tile_init(uint32_t) {}
ALWI void dequant_tile(uint32_t, uint32_t, uint32_t) {}
} // namespace ckernel
