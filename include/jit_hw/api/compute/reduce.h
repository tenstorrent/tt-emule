#pragma once
// Reduce compute stub for emulated mode
#include "api/compute/common.h"

inline void reduce_init(uint32_t icb = 0, uint32_t ocb = 0) {}
inline void reduce_init_short(uint32_t icb = 0, uint32_t ocb = 0) {}
inline void reduce_init_delta(uint32_t icb = 0, uint32_t ocb = 0) {}
inline void reduce_tile(uint32_t icb, uint32_t itile, uint32_t dst_idx) {}
inline void reduce_revert_delta(uint32_t ocb = 0) {}
