#pragma once
// Emulator stub for SELU SFPU tile op.
// Upstream takes packed alpha/scale as uint32 params; for now we use
// the standard fixed constants (alpha=1.6733, scale=1.0507).

#include <cmath>
#include <cstdint>
#include <cstring>

namespace ckernel {

ALWI void selu_tile_init() {}

ALWI void selu_tile(uint32_t idst, uint32_t param_scale = 0, uint32_t param_alpha = 0) {
    __emule_dst_check(idst, "selu_tile");
    // Decode packed float params if provided, else use canonical SELU constants.
    float scale = 1.0507009873554805f;
    float alpha = 1.6732632423543772f;
    if (param_scale != 0) std::memcpy(&scale, &param_scale, sizeof(float));
    if (param_alpha != 0) std::memcpy(&alpha, &param_alpha, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = scale * (x > 0.0f ? x : alpha * (std::exp(x) - 1.0f));
    }
}

} // namespace ckernel
