#pragma once
// Mirrors tt_metal/api/tt-metalium/circular_buffer_constants.h. Local copy so
// kernels can reach it via the jit_hw -I path without exposing tt_metal/api/.

#include <cstdint>

#if defined(ARCH_WORMHOLE)
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
#else
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 64;
#endif
constexpr static std::uint32_t UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG  = 4;
constexpr static std::uint32_t UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG = 2;
constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_WORD_SIZE = 16;
constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT = 4;
