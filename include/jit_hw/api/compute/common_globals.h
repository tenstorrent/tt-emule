#pragma once
// Minimal DataFormat enum for JIT-compiled SFPU kernels.
// Matches the real tt-metal enum values we need.
#include <cstdint>

enum class DataFormat : uint8_t {
    Float32   = 0,
    Float16   = 1,
    Bfp8      = 2,
    Bfp4      = 3,
    Tf32      = 4,
    Float16_b = 5,
    Bfp8_b    = 6,
    Bfp4_b    = 7,
    Int32     = 8,
    UInt16    = 9,
    Lf8       = 10,
    Bfp2      = 11,
    Int8      = 14,
    Bfp2_b    = 15,
    UInt32    = 24,
    UInt8     = 30,
    Fp8_e4m3  = 0x1A,
    RawUInt8  = 0xF0,
    RawUInt16 = 0xF1,
    RawUInt32 = 0xF2,
    Invalid   = 0xff
};

// PACK engine auto-advance: tracks the write offset within a reserve_back batch.
// On real hardware the PACK engine auto-advances its L1 write pointer after each
// pack_tile.  In emulation, cb_write_ptr is write_idx-based and only advances on
// push_back.  This counter emulates the hardware auto-advance: reset to 0 on
// reserve_back, incremented by pack_tile.
static thread_local uint32_t __emule_pack_offset[32] = {};
