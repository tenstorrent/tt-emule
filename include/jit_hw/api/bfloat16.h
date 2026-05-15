// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// bfloat16 ↔ float32 conversion helpers for JIT-compiled kernels.
// Used by compute stubs and available for future SFPU/tilize ops.

#include <cstdint>
#include <cstring>

namespace __emule_bf16 {

inline float to_f32(uint16_t bf16) {
    uint32_t f32 = static_cast<uint32_t>(bf16) << 16;
    float val;
    std::memcpy(&val, &f32, sizeof(float));
    return val;
}

inline uint16_t from_f32(float val) {
    uint32_t f32;
    std::memcpy(&f32, &val, sizeof(float));
    // Round to nearest even (add 0x7FFF + bit 16 for tie-breaking)
    f32 += 0x7FFF + ((f32 >> 16) & 1);
    return static_cast<uint16_t>(f32 >> 16);
}

} // namespace __emule_bf16
