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
#include "dev_mem_map.h"
#include "emule_cb_state.h"
#include "emule_dfb_state.h"
#include "tools/profiler/kernel_profiler.hpp"

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

// Translate a raw L1 firmware offset (or already-absolute host pointer) to a
// host uint8_t*.  Available to ALL JIT kernels so the l1_arg_ptr regex patch in
// emulated_program_runner can inject calls without requiring dataflow_api.h.
#ifndef __EMULE_LOCAL_L1_TO_PTR_DEFINED
#define __EMULE_LOCAL_L1_TO_PTR_DEFINED
inline uint8_t* __emule_local_l1_to_ptr(uint32_t l1_addr) {
    uint32_t l1_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__emule_bridge_l1));
    if (l1_addr >= l1_base) {
        return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(l1_addr));
    }
    return __emule_bridge_l1 + l1_addr;
}
#endif

// Bank mapping arrays — populated by emulated_program_runner, resolved at dlopen.
// Match firmware declarations from dataflow_api_common.h / firmware_common.h.
extern uint16_t dram_bank_to_noc_xy[2][32];
extern int32_t bank_to_dram_offset[32];
extern uint16_t l1_bank_to_noc_xy[2][32];
extern int32_t bank_to_l1_offset[32];

// Per-core NOC coordinates (set per kernel thread by program runner).
extern thread_local uint8_t my_x[2];
extern thread_local uint8_t my_y[2];

// Per-core logical coordinates (for D2M get_absolute_logical_x/y).
extern thread_local uint32_t __emule_logical_x;
extern thread_local uint32_t __emule_logical_y;

// Processor ID — substitutes RISC-V mhartid CSR in emulation.
extern thread_local uint8_t __processor_id;

// CSR emulation: NEO engine ID and TRISC core ID within engine.
extern thread_local uint8_t __emule_neo_id;
extern thread_local uint8_t __emule_trisc_id;

// Thread ID: number of compute/DM engines and this engine's index.
extern thread_local uint32_t __emule_num_threads;
extern thread_local uint32_t __emule_my_thread_id;

// tt_l1_ptr: type qualifier for L1 pointers. On real HW, this adds volatile.
// In emulation, L1 is normal host memory — the qualifier is empty.
#ifndef tt_l1_ptr
#define tt_l1_ptr
#endif

// NOC index — always 0 for emulation (real firmware sets this per core).
constexpr uint8_t noc_index = 0;

// get_arg_addr(idx) — mirrors tt-metal's rta_l1_base-based implementation.
// Returns a pointer to the idx-th runtime arg (held in __rt_args).
// Signature matches real firmware: int param, uintptr_t return.
static inline uintptr_t get_arg_addr(int arg_idx) {
    return reinterpret_cast<uintptr_t>(&__rt_args[arg_idx]);
}

// get_common_arg_addr(idx) — pointer to common runtime arg.
static inline uintptr_t get_common_arg_addr(int arg_idx) {
    return reinterpret_cast<uintptr_t>(&__common_rt_args[arg_idx]);
}

// Per-core and common runtime argument value access.
// Signatures match real firmware: int param, template return.
template<typename T = uint32_t>
inline T get_arg_val(int arg_idx) {
    static_assert(sizeof(T) <= sizeof(uint32_t));
    if (arg_idx < 0 || static_cast<size_t>(arg_idx) >= __rt_args.size()) {
        fprintf(stderr, "EMULE BUG: get_arg_val(%d) out of bounds (size=%zu)\n",
                arg_idx, __rt_args.size());
        std::abort();
    }
    T val;
    std::memcpy(&val, &__rt_args[arg_idx], sizeof(T));
    return val;
}

template<typename T = uint32_t>
inline T get_common_arg_val(int arg_idx) {
    static_assert(sizeof(T) <= sizeof(uint32_t));
    if (arg_idx < 0 || static_cast<size_t>(arg_idx) >= __common_rt_args.size()) {
        fprintf(stderr, "EMULE BUG: get_common_arg_val(%d) out of bounds (size=%zu)\n",
                arg_idx, __common_rt_args.size());
        std::abort();
    }
    T val;
    std::memcpy(&val, &__common_rt_args[arg_idx], sizeof(T));
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

// No-op assertion macro — kernel asserts reference hardware state not available in emulation.
#define ASSERT(...) ((void)0)

// Semaphore address helper — returns a uint32_t L1 address for the given
// semaphore ID.  Defined here so both compute and dataflow kernels can use it.
// EMULE_SEM_BASE should be passed as a JIT compiler define (e.g. -DEMULE_SEM_BASE=0xFFE00).
// If not defined here, dataflow_api.h provides the default (0xFFE00).
#ifndef EMULE_SEM_ALIGN
#define EMULE_SEM_ALIGN 16
#endif
#if defined(EMULE_SEM_BASE)
#ifndef __EMULE_GET_SEMAPHORE_DEFINED
#define __EMULE_GET_SEMAPHORE_DEFINED
inline uint32_t get_semaphore(uint32_t semaphore_id) {
    uint32_t l1_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__emule_bridge_l1));
    return l1_base + EMULE_SEM_BASE + semaphore_id * EMULE_SEM_ALIGN;
}
#endif
#endif

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
