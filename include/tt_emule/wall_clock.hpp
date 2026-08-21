// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Host-backed stand-in for the RISCV_DEBUG_REG_WALL_CLOCK_{L,H} hardware registers (declared in
// jit_hw/internal/risc_attribs.h, used by common.hpp::spin() / device_delay_spin.cpp). Kernels
// busy-wait by re-reading these as a live 64-bit counter until it advances past a target, so the
// backing storage must genuinely tick — and must outlive the JIT-compiled kernel .so that reads
// it, which gets dlclose()'d once its Program is torn down. Hosted here (compiled into the
// long-lived runtime via emulated_program_runner.cpp's existing device.hpp include chain), not in
// jit_hw (recompiled fresh into every ephemeral kernel .so, where a thread ticking it would
// crash the moment its owning .so is unloaded out from under it).
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

namespace tt_emule {

struct WallClock {
    alignas(8) std::atomic<uint32_t> lo{0};
    std::atomic<uint32_t> hi{0};
    WallClock() {
        std::thread([this] {
            using namespace std::chrono;
            for (;;) {
                // Coarser unit than nanoseconds so the 32-bit lo half wraps far less often
                // (~71 min at microsecond resolution vs ~4.3s at nanosecond resolution) —
                // narrows (doesn't eliminate) the tearing window between the two separate,
                // unsynchronized 32-bit stores/loads, matching a real split hi/lo register pair.
                auto us = static_cast<uint64_t>(duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());
                lo.store(static_cast<uint32_t>(us), std::memory_order_relaxed);
                hi.store(static_cast<uint32_t>(us >> 32), std::memory_order_relaxed);
                std::this_thread::sleep_for(microseconds(1));
            }
        }).detach();
    }
};

// Intentionally leaked: the ticker thread must outlive this object, but a function-local static
// would run WallClock's (trivial but non-zero-cost-to-reason-about) destructor at static-teardown
// time while the detached thread may still be writing to it. Heap-allocating and never freeing
// means the memory stays valid until the process's address space itself goes away.
inline WallClock& wall_clock() {
    static WallClock* instance = new WallClock();
    return *instance;
}

}  // namespace tt_emule

// Kernel .so's resolve these via dlopen, so they must stay exported: `visibility("default")`
// counters the project-wide -fvisibility-inlines-hidden, and `used` stops the compiler from
// eliding the definition entirely since nothing in this TU calls them itself (only external
// kernel .so's do, dynamically).
extern "C" __attribute__((visibility("default"), used)) inline uintptr_t __emule_wall_clock_lo_addr() {
    return reinterpret_cast<uintptr_t>(&tt_emule::wall_clock().lo);
}
extern "C" __attribute__((visibility("default"), used)) inline uintptr_t __emule_wall_clock_hi_addr() {
    return reinterpret_cast<uintptr_t>(&tt_emule::wall_clock().hi);
}
