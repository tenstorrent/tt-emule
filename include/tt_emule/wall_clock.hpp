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
                auto ns = static_cast<uint64_t>(steady_clock::now().time_since_epoch().count());
                lo.store(static_cast<uint32_t>(ns), std::memory_order_relaxed);
                hi.store(static_cast<uint32_t>(ns >> 32), std::memory_order_relaxed);
                std::this_thread::sleep_for(microseconds(1));
            }
        }).detach();
    }
};

inline WallClock& wall_clock() {
    static WallClock instance;
    return instance;
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
