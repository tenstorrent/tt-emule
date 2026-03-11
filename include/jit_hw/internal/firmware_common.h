#pragma once
// Firmware common stub for emulated mode
// Most macros (PACK/MATH/UNPACK/ALWI) are already defined in api/compute/common.h
// This header provides additional firmware-level stubs.
#include <cstdint>

// Debug waypoints - no-op in emulation
#ifndef WAYPOINT
#define WAYPOINT(...)
#endif

#ifndef UnpackToDestEn
#define UnpackToDestEn false
#endif
