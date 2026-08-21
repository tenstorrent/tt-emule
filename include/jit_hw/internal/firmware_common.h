// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Firmware common stub for emulated mode
// Most macros (PACK/MATH/UNPACK/ALWI) are already defined in api/compute/common.h
// This header provides additional firmware-level stubs.
#include <cstdint>
#include "jit_hw/internal/emule_fiber_bridge.h"  // __emule_fiber_yield

// Cache ops. emule has no hardware cache; invalidate_l1_cache() is the per-iteration
// re-read barrier a kernel busy-poll spins on. The cooperative fiber engine has no
// preemption, so a spin that never calls a fiber primitive would hog its worker and
// starve the peer fiber updating the polled value. Yielding here cedes the worker.
inline void invalidate_l1_cache() { __emule_fiber_yield(); }
inline void flush_l1_cache() {}

// Debug waypoints - no-op in emulation
#ifndef WAYPOINT
#define WAYPOINT(...)
#endif

// UnpackToDestEn lives in llk_types.h as `inline constexpr bool` (not a
// macro). D2M-emitted kernels use it as a non-type template argument; the
// constexpr form preserves type information at the template-arg site.
