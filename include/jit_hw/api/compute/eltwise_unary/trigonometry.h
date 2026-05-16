#pragma once
// Emulator stubs for trigonometric SFPU tile ops.
// All ops operate on DST (float32 storage) element-wise via std:: math.

#include <cmath>
#include <cstdint>

namespace ckernel {

ALWI void sin_tile_init() {}
ALWI void sin_tile(uint32_t idst) {
    __emule_dst_check(idst, "sin_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::sin(__emule_dst[idst][i]);
}

ALWI void cos_tile_init() {}
ALWI void cos_tile(uint32_t idst) {
    __emule_dst_check(idst, "cos_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::cos(__emule_dst[idst][i]);
}

ALWI void tan_tile_init() {}
ALWI void tan_tile(uint32_t idst) {
    __emule_dst_check(idst, "tan_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::tan(__emule_dst[idst][i]);
}

ALWI void tanh_tile_init() {}
ALWI void tanh_tile(uint32_t idst) {
    __emule_dst_check(idst, "tanh_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::tanh(__emule_dst[idst][i]);
}

ALWI void asin_tile_init() {}
ALWI void asin_tile(uint32_t idst) {
    __emule_dst_check(idst, "asin_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::asin(__emule_dst[idst][i]);
}

ALWI void acos_tile_init() {}
ALWI void acos_tile(uint32_t idst) {
    __emule_dst_check(idst, "acos_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::acos(__emule_dst[idst][i]);
}

ALWI void atan_tile_init() {}
ALWI void atan_tile(uint32_t idst) {
    __emule_dst_check(idst, "atan_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::atan(__emule_dst[idst][i]);
}

} // namespace ckernel
