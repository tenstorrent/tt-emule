// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// LLK definitions for emulated mode — the enums/types silicon's
// tt_llk_<arch>/common/inc/llk_defs.h provides (ReduceDim, PoolType, …).
// Deliberately does NOT pull the compute API (common.h, pack/unpack/datacopy):
// dataflow kernels include this (via reduce_helpers_dataflow.hpp) only for the
// enums, and those compute headers' `using namespace ckernel` would otherwise
// leak into a dataflow kernel and collide with its own bare constants (e.g. the
// sampling writer's FACE_WIDTH). Compute consumers include the compute headers
// directly.
#include "llk_types.h"
#include "jit_hw/api/compute/nfaces.h"
#include "internal/firmware_common.h"
#include "internal/llk_state.h"
#include "llk_sync_stubs.h"
