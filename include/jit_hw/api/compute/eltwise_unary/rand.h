#pragma once
// Emulator stub for SFPU rand tile op.
// Each element of DST is overwritten with a pseudorandom float in [from, from+scale].

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ckernel {

ALWI void rand_tile_init(uint32_t seed = 0) {
    if (seed != 0) std::srand(seed);
}

ALWI void rand_tile(uint32_t idst, uint32_t from = 0, uint32_t scale = 0) {
    __emule_dst_check(idst, "rand_tile");
    float from_f = 0.0f, scale_f = 1.0f;
    if (from != 0) std::memcpy(&from_f, &from, sizeof(float));
    if (scale != 0) std::memcpy(&scale_f, &scale, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        __emule_dst[idst][i] = from_f + r * scale_f;
    }
}

} // namespace ckernel
