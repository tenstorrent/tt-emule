#pragma once
// Emulator stub for SFPU rand tile op.
// Each element of DST is overwritten with a pseudorandom float in [from, from+scale].

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ckernel {

ALWI void rand_tile_init(uint32_t seed = 0) {
    if (seed != 0) std::srand(seed);
}

ALWI void rand_tile(uint32_t idst, uint32_t from = 0, uint32_t scale = 0) {
    __emule_dst_check(idst, "rand_tile");
    if (from == 0 || scale == 0) {
        fprintf(stderr, "[EMULE] rand_tile: from=%u scale=%u; emule cannot "
                        "distinguish 'not provided' from packed 0.0f. Caller "
                        "must pass packed bits of both arguments.\n", from, scale);
        std::abort();
    }
    float from_f, scale_f;
    std::memcpy(&from_f, &from, sizeof(float));
    std::memcpy(&scale_f, &scale, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        __emule_dst[idst][i] = from_f + r * scale_f;
    }
}

} // namespace ckernel
