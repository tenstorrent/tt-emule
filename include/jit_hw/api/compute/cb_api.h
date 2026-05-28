// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Path shim: tt-metal kernels include "api/compute/cb_api.h" (e.g.
// ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp). emule's CB shim lives at
// "api/cb_api.h" (one level up). Redirect so the include resolves to emule
// rather than falling through to the real tt-metal header at
// tt_metal/hw/inc/api/compute/cb_api.h (which references undefined ALWI,
// UNPACK/PACK/MATH wrappers around llk_* symbols that don't exist on host).
//
// Real LLK source: /localdev/arminale/tt-metal/tt_metal/hw/inc/api/compute/cb_api.h
#include "jit_hw/api/cb_api.h"
