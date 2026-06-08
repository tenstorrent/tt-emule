// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Stub of the silicon SFPU exp ckernel. Real silicon defines
// SFPU-side exp calculate templates that operate on sfpi vectors. emule's
// scalar exp lives in api/compute/eltwise_unary/exp.h; this stub exists so
// upstream headers that #include this transitively parse cleanly. The
// functional emule exp is via exp_tile_init/exp_tile.

#include "jit_hw/sfpi.h"
