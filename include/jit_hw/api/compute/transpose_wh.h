// tt-emule stub: transpose_wh (no-op in emulation)
#pragma once

namespace ckernel {

inline void transpose_wh_init(uint32_t icb, uint32_t ocb = 0) {}
inline void transpose_wh_init_short(uint32_t icb = 0) {}
inline void transpose_wh_tile(uint32_t icb, uint32_t itile, uint32_t dst_idx) {}

}  // namespace ckernel

using namespace ckernel;
