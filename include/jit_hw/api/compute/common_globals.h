#pragma once
// Minimal DataFormat enum for JIT-compiled SFPU kernels.
// Matches the real tt-metal enum values we need.
#include <cstdint>

enum class DataFormat : uint8_t {
    Float32   = 0,
    Float16   = 1,
    Bfp8      = 2,
    Bfp4      = 3,
    Float16_b = 5,
    Bfp8_b    = 6,
    Bfp4_b    = 7,
    Int32     = 8,
    UInt16    = 9,
    Lf8       = 10,
    Int8      = 14,
    UInt32    = 24,
    UInt8     = 30,
    Invalid   = 0xff
};
