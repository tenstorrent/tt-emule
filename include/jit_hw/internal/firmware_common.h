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

// UnpackToDestEn moved from `#define UnpackToDestEn false` to the
// `inline constexpr bool UnpackToDestEn = false;` declared in llk_defs.h.
// D2M-emitted kernels use it as a template argument; constexpr form gives
// proper type information while the macro form (any name expansion at template
// arg position) prevented the enum-class `DataCopyType` declaration from
// parsing.
