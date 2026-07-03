#pragma once
#include "jit_hw/api/compute/common.h"
// Emulator stub for the SFPU rand tile op.
// Each element of DST is overwritten with a pseudorandom float in [from, from+scale].

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

namespace ckernel {

// Per-core RNG: every emule core is a host thread, so a thread_local engine is
// one independent stream per core. emule matches the test's contract (seeded →
// reproducible, seed 0 → random), not silicon's exact RNG bit sequence.
static thread_local std::mt19937 __emule_rand_engine;
// Persistent mode set by rand_tile_init: deterministic (nonzero seed) keeps
// drawing from the seeded stream so a multi-tile sequence stays reproducible;
// random (seed 0) reseeds from entropy per draw.
static thread_local bool __emule_rand_deterministic = false;
static thread_local uint64_t __emule_rand_nonce = 0;

// Seed from (op seed, core coords): same seed reproduces byte-identical output
// (re-seeding resets the stream), distinct cores diverge. seed 0 selects the
// random mode (each draw reseeds from entropy).
ALWI void rand_tile_init(uint32_t seed = 0) {
    if (seed != 0) {
        const uint32_t cx = get_absolute_logical_x();
        const uint32_t cy = get_absolute_logical_y();
        std::seed_seq seq{seed, cx, cy, 0x9E3779B9u, 0x85EBCA6Bu};
        __emule_rand_engine.seed(seq);
        __emule_rand_deterministic = true;
    } else {
        __emule_rand_deterministic = false;
    }
}

ALWI void rand_tile(uint32_t idst, uint32_t from = 0, uint32_t scale = 0) {
    __emule_dst_check(idst, "rand_tile");
    // Both args zero is the unambiguous "caller forgot to pass packed bits"
    // case (a zero from or zero scale alone is a legitimate range); fail loud.
    if (from == 0 && scale == 0) {
        fprintf(stderr, "[EMULE] rand_tile: both from and scale are 0; pass both "
                        "arguments explicitly (packed float bits).\n");
        std::abort();
    }
    float from_f, scale_f;
    std::memcpy(&from_f, &from, sizeof(float));
    std::memcpy(&scale_f, &scale, sizeof(float));

    // Deterministic mode draws from the continuous seeded stream (multi-tile
    // sequences stay reproducible). Random mode reseeds from entropy each draw
    // so two runs differ (silicon's seed 0 pulls from a free-running counter).
    if (!__emule_rand_deterministic) {
        std::random_device rd;
        std::seed_seq seq{rd(), rd(),
                          static_cast<uint32_t>(__emule_rand_nonce & 0xFFFFFFFFu),
                          static_cast<uint32_t>(__emule_rand_nonce >> 32),
                          get_absolute_logical_x(), get_absolute_logical_y()};
        __emule_rand_engine.seed(seq);
        ++__emule_rand_nonce;
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_compute_ctx().dst[idst][i] = from_f + dist(__emule_rand_engine) * scale_f;
    }
}

} // namespace ckernel
