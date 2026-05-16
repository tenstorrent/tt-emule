#pragma once
// Emule shim for upstream LLK CB / sync surface — llk_wait_tiles /
// llk_pop_tiles / llk_push_tiles / llk_wait_for_free_tiles delegate to
// cb_* from cb_api.h; math/packer sync stubs are no-ops.

#include "api/cb_api.h"
#include <cstdint>

// ---- LLK CB stubs (delegate to cb_api.h functions) ----
// The real tt-metal cb_api.h inlines these as calls to llk_wait_tiles etc.
// In emulation, the CB functions are the real implementation; llk_* wrappers delegate.
inline void llk_wait_tiles(int operand, std::int32_t num_tiles) {
    cb_wait_front(static_cast<uint32_t>(operand), static_cast<uint32_t>(num_tiles));
}
inline void llk_pop_tiles(std::int32_t operand, std::int32_t num_tiles, std::int32_t = 0) {
    cb_pop_front(static_cast<uint32_t>(operand), static_cast<uint32_t>(num_tiles));
}
template <bool = false, bool = false, bool = false>
inline void llk_wait_for_free_tiles(std::int32_t operand, std::int32_t num_tiles) {
    cb_reserve_back(static_cast<uint32_t>(operand), static_cast<uint32_t>(num_tiles));
}
template <bool = false, bool = false>
inline void llk_push_tiles(std::int32_t operand, std::int32_t num_tiles) {
    cb_push_back(static_cast<uint32_t>(operand), static_cast<uint32_t>(num_tiles));
}

// ---- Sync stubs (no-op in single-threaded compute) ----
inline void llk_math_wait_for_dest_available() {}
inline void llk_packer_wait_for_math_done() {}

// ---- TRISC sync primitives ----
// Referenced by the verbatim-inlined body of `experimental::unpack_stall_on_pack`
// (D2M emits the contents of `experimental_reg_api.h` into the kernel). On real
// silicon these gate PACK/UNPACK threads on TRISC semaphores; on emule
// (single-threaded compute) they are no-ops.
namespace semaphore {
    constexpr int PACK_DONE = 0;
    constexpr int UNPACK_SYNC = 1;
    constexpr int FPU_SFPU = 2;
    constexpr int MATH_DONE = 3;
}
namespace p_stall {
    constexpr int STALL_SYNC = 0;
    constexpr int WAIT_SFPU = 1;
    constexpr int MATH = 2;
    constexpr int PACK = 3;
    constexpr int UNPACK = 4;
}
template <int Stall = 0>
inline void t6_semaphore_post(int /*sem*/ = 0) {}
template <int Stall = 0>
inline void t6_semaphore_wait_on_zero(int /*sem*/ = 0) {}
template <int Stall = 0>
inline void t6_semaphore_get(int /*sem*/ = 0) {}
