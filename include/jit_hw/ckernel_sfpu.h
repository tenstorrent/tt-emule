// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Umbrella SFPU header shim. The real `ckernel_sfpu.h` pulls in every SFPU
// `_calculate_*` template and emits raw Tensix instructions that don't exist on
// x86. Kernels that include it otherwise get their SFPU symbols from the
// dedicated api/compute shims; the only symbol consumed from here is
// `sfpu::_init_sfpu_config_reg()`.

#include "jit_hw/sfpi.h"

namespace ckernel {
namespace sfpu {

// SFPU config-register reset. emule applies SFPU constants directly as each
// scalar shim runs, so there is no config register to reset — no-op.
inline void _init_sfpu_config_reg() {}

}  // namespace sfpu
}  // namespace ckernel
