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
    // Abort only when *both* args are the default zero — that is the
    // unambiguous "caller relied on defaults and forgot to pass packed bits"
    // case. A zero `from` alone is a legitimate value (e.g. uniform [0, 1]),
    // and so is a zero `scale` alone (a degenerate but well-defined point
    // distribution). Distinguishable bit patterns mean the caller was
    // explicit; trust them.
    if (from == 0 && scale == 0) {
        fprintf(stderr, "[EMULE] rand_tile: both from and scale are 0; emule "
                        "cannot tell whether the caller forgot to pass packed "
                        "bits or actually meant a zero-width range starting "
                        "at zero. Pass both arguments explicitly.\n");
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
