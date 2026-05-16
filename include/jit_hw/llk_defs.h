// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// LLK definitions aggregator for emulated mode. Per-domain headers below
// own the actual LLK surface; this aggregator is kept for back-compat with
// any consumer that `#include`s `jit_hw/llk_defs.h` directly.
#include "llk_types.h"
#include "api/compute/common.h"
#include "jit_hw/api/compute/nfaces.h"
#include "internal/firmware_common.h"
#include "internal/llk_state.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_pack.h"
#include "llk_sync_stubs.h"
#include "llk_unpack_a.h"
