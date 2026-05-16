// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Firmware common stub for emulated mode
// Most macros (PACK/MATH/UNPACK/ALWI) are already defined in api/compute/common.h
// This header provides additional firmware-level stubs.
#include <cstdint>

// Cache ops — no-op in emulation (no hardware cache).
inline void invalidate_l1_cache() {}
inline void flush_l1_cache() {}

// Debug waypoints - no-op in emulation
#ifndef WAYPOINT
#define WAYPOINT(...)
#endif

// UnpackToDestEn lives in llk_types.h as `inline constexpr bool` (not a
// macro). D2M-emitted kernels use it as a non-type template argument; the
// constexpr form preserves type information at the template-arg site.
