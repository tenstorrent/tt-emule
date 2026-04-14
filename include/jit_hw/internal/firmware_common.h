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

#ifndef UnpackToDestEn
#define UnpackToDestEn false
#endif
