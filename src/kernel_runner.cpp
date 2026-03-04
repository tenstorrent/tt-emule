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

namespace tt_emule {

void EnqueueProgram(Device& device, Program& program, bool /*blocking*/) {
    // 1. Instantiate all CBs on the core
    Core& core = device.core();
    for (auto& cfg : program.cb_configs()) {
        core.cb(cfg.cb_index) =
            std::make_shared<CircularBuffer>(cfg.num_pages);
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
