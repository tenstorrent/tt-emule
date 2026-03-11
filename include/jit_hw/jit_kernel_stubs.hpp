#pragma once
// Stubs included at the TOP of every JIT wrapper before the kernel source.
// Provides everything a kernel needs that isn't in its own includes.

// Mark that we're in JIT compilation mode.  Headers like dataflow_api.hpp
// use this to call extern "C" bridge functions instead of inlining Device
// methods (which would have wrong ABI when TT_EMULE_USE_XY_PAIR is active
// in the host process but not in the JIT .so).
#define __EMULE_JIT_MODE 1

#include "internal/risc_attribs.h"
#include "api/compile_time_args.h"
#include "emule_cb_state.h"

#include <vector>
#include <cstdint>
#include <cstring>

// Forward declarations matching kernel_runner.cpp definitions.
// The main executable exports these with -rdynamic; the JIT .so resolves them
// at dlopen() time.
namespace tt_emule { class Core; class Device; }
extern thread_local std::vector<uint32_t> __rt_args;
extern thread_local std::vector<uint32_t> __common_rt_args;
extern thread_local tt_emule::Core*       __core;
extern thread_local tt_emule::Device*     __device;

// C-linkage bridge functions — resolve at dlopen time to the host process's
// implementations, avoiding ABI mismatch with Device layout.
extern "C" uint8_t* __emule_dram_ptr(uint64_t offset);
extern "C" uint8_t* __emule_noc_resolve(uint32_t x, uint32_t y, uint64_t addr);

// L1 bridge pointer — host pointer to L1 memory start.
// Used to convert between uint32_t L1 "addresses" (really: truncated pointers
// from mmap'd-below-4GB L1) and real host pointers.
extern thread_local uint8_t* __emule_bridge_l1;

// get_arg_addr(idx) — mirrors tt-metal's rta_l1_base-based implementation.
// Returns a pointer to the idx-th runtime arg (held in __rt_args).
inline void* get_arg_addr(uint32_t idx) {
    return static_cast<void*>(&__rt_args[idx]);
}

// get_common_arg_addr(idx) — pointer to common runtime arg.
inline void* get_common_arg_addr(uint32_t idx) {
    return static_cast<void*>(&__common_rt_args[idx]);
}

// Per-core and common runtime argument value access.
template<typename T = uint32_t>
inline T get_arg_val(uint32_t idx) {
    static_assert(sizeof(T) <= sizeof(uint32_t));
    T val;
    std::memcpy(&val, &__rt_args[idx], sizeof(T));
    return val;
}

template<typename T = uint32_t>
inline T get_common_arg_val(uint32_t idx) {
    static_assert(sizeof(T) <= sizeof(uint32_t));
    T val;
    std::memcpy(&val, &__common_rt_args[idx], sizeof(T));
    return val;
}

// CBIndex — matches tt::CBIndex from tt-metal.
namespace tt {
struct CBIndex {
    static constexpr uint32_t c_0  = 0;
    static constexpr uint32_t c_1  = 1;
    static constexpr uint32_t c_2  = 2;
    static constexpr uint32_t c_3  = 3;
    static constexpr uint32_t c_4  = 4;
    static constexpr uint32_t c_5  = 5;
    static constexpr uint32_t c_6  = 6;
    static constexpr uint32_t c_7  = 7;
    static constexpr uint32_t c_8  = 8;
    static constexpr uint32_t c_9  = 9;
    static constexpr uint32_t c_10 = 10;
    static constexpr uint32_t c_11 = 11;
    static constexpr uint32_t c_12 = 12;
    static constexpr uint32_t c_13 = 13;
    static constexpr uint32_t c_14 = 14;
    static constexpr uint32_t c_15 = 15;
    static constexpr uint32_t c_16 = 16;
    static constexpr uint32_t c_17 = 17;
    static constexpr uint32_t c_18 = 18;
    static constexpr uint32_t c_19 = 19;
    static constexpr uint32_t c_20 = 20;
    static constexpr uint32_t c_21 = 21;
    static constexpr uint32_t c_22 = 22;
    static constexpr uint32_t c_23 = 23;
    static constexpr uint32_t c_24 = 24;
    static constexpr uint32_t c_25 = 25;
    static constexpr uint32_t c_26 = 26;
    static constexpr uint32_t c_27 = 27;
    static constexpr uint32_t c_28 = 28;
    static constexpr uint32_t c_29 = 29;
    static constexpr uint32_t c_30 = 30;
    static constexpr uint32_t c_31 = 31;
};
} // namespace tt

// No-op assertion macro used in some kernel headers.
#define ASSERT(...) ((void)0)

// ---- ThreadId enum + mailbox stubs (used by both compute and dataflow kernels) ----
namespace ckernel {
enum ThreadId {
    BriscThreadId  = 0,
    UnpackThreadId = 1,
    MathThreadId   = 2,
    PackThreadId   = 3
};
inline uint32_t mailbox_read(uint8_t /*thread*/) { return 1; }
inline void mailbox_write(uint8_t /*thread*/, uint32_t /*data*/) {}
} // namespace ckernel
