#include "tt_emule/host_api.hpp"
#include "tt_emule/program.hpp"
#include "tt_emule/device.hpp"
#include "tt_emule/circular_buffer.hpp"
#include <thread>
#include <vector>
#include <stdexcept>

// Thread-local context used by kernel_api headers
thread_local std::vector<uint32_t> __rt_args;
thread_local tt_emule::Core*       __core   = nullptr;
thread_local tt_emule::Device*     __device = nullptr;

// C-linkage helper exported for JIT kernels.  JIT kernels are compiled without
// TT_EMULE_USE_XY_PAIR, so they cannot inline Device::dram_ptr() correctly
// (Device gains a vtable pointer when TT_EMULE_USE_XY_PAIR is active, shifting
// member offsets).  By calling through this C-linkage function the JIT kernel
// never touches Device's layout — the call resolves at dlopen time via -rdynamic.
extern "C" uint8_t* __emule_dram_ptr(uint64_t offset) {
    return __device->dram_ptr(offset);
}

namespace tt_emule {

void EnqueueProgram(Device& device, Program& program, bool /*blocking*/) {
    // 1. Instantiate all CBs on the core
    Core& core = device.core();
    for (auto& cfg : program.cb_configs()) {
        core.cb(cfg.cb_index) =
            std::make_shared<CircularBuffer>(cfg.num_pages, cfg.page_size);
    }

    // 2. Launch one thread per kernel descriptor
    std::vector<std::thread> threads;
    threads.reserve(program.kernels().size());

    for (KernelDescriptor& kd : program.kernels()) {
        threads.emplace_back([&kd, &core, &device]() {
            // Inject thread-local context
            __rt_args = kd.rt_args;
            __core    = &core;
            __device  = &device;
            // Run kernel body
            kd.fn();
        });
    }

    // 3. Join all threads
    for (auto& t : threads) {
        t.join();
    }

    // 4. Tear down CBs so next EnqueueProgram starts fresh
    for (auto& cfg : program.cb_configs()) {
        core.cb(cfg.cb_index).reset();
    }
}

} // namespace tt_emule
