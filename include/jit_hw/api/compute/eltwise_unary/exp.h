#pragma once
// Emulator stub for exponential tile operations.

#include <cmath>
#include <cstdint>

enum class InputClamping : uint8_t {
    ClampToNegative = 1,
    None = 0,
};

// VectorMode — mirrors tt-metal's SFPU vector mode enum.
#ifndef __EMULE_VECTOR_MODE_DEFINED
#define __EMULE_VECTOR_MODE_DEFINED
enum class VectorMode : uint8_t { RC = 0, R = 1, C = 2 };
#endif

// p_sfpu constants used by real exp_tile callers
namespace p_sfpu {
constexpr uint16_t kCONST_1_FP16B = 0x3F80;
} // namespace p_sfpu

namespace ckernel {

template <bool approx = false, bool fast_and_approx = true,
          bool scale_en = false, bool skip_positive_check = false,
          InputClamping input_clamping = InputClamping::ClampToNegative,
          int iterations = 8>
ALWI void exp_tile_init(uint32_t = 0, uint32_t = 0) {}

template <bool approx = false, bool fast_and_approx = true,
          bool scale_en = false, bool skip_positive_check = false,
          InputClamping input_clamping = InputClamping::ClampToNegative,
          int iterations = 8>
ALWI void exp_tile(uint32_t idst, int vector_mode = (int)VectorMode::RC,
                   uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    __emule_dst_check(idst, "exp_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::exp(__emule_dst[idst][i]);
}

} // namespace ckernel
