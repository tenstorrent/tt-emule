#include "tt_emule/host_api.hpp"
#include "tt_emule/program.hpp"
#include "tt_emule/device.hpp"
#include "tt_emule/circular_buffer.hpp"
#include "tt_emule/dfb_sync_state.hpp"
#include "tt_emule/tile_counter.hpp"
#include <thread>
#include <vector>
#include <barrier>
#include <stdexcept>

// Thread-local context used by kernel_api headers
thread_local std::vector<uint32_t> __rt_args;
thread_local tt_emule::Core*       __core   = nullptr;
thread_local tt_emule::Device*     __device = nullptr;
thread_local uint8_t               __processor_id = 0;

// DFB thread-locals (standalone path uses __dfb_ifaces; JIT path uses __emule_dfbs)
thread_local tt_emule::EmuleDFBInterface* __dfb_ifaces = nullptr;
thread_local tt_emule::TileCounterArray*  __emule_tc_array = nullptr;

// JIT path also uses __emule_dfbs (same as __dfb_ifaces, aliased in emule_dfb_state.h)
thread_local tt_emule::EmuleDFBInterface* __emule_dfbs = nullptr;

extern "C" uint8_t* __emule_dram_ptr(uint64_t offset) {
    return __device->dram_ptr(offset);
}

namespace tt_emule {

namespace {

// Build per-thread DFB interface arrays for all kernel threads.
// Returns [num_kernels][MAX_DFBS] array of EmuleDFBInterface.
std::vector<std::vector<EmuleDFBInterface>> build_dfb_interfaces(
    Core& core, Program& program)
{
    auto& dfb_cfgs = program.dfb_configs();
    auto& kernels = program.kernels();
    size_t nk = kernels.size();

    std::vector<std::vector<EmuleDFBInterface>> result(
        nk, std::vector<EmuleDFBInterface>(MAX_DFBS));

    if (dfb_cfgs.empty()) return result;

    for (auto& cfg : dfb_cfgs) {
        auto& dsync = core.dfb_sync_array()[cfg.dfb_index];

        for (size_t ki = 0; ki < nk; ++ki) {
            auto& kd = kernels[ki];
            auto& iface = result[ki][cfg.dfb_index];
            iface.entry_size  = cfg.entry_size;
            iface.num_entries = cfg.num_entries;

            bool is_producer = false;
            bool is_consumer = false;

            uint16_t proc_bit = static_cast<uint16_t>(1u << kd.processor_id);
            if (cfg.producer_risc_mask & proc_bit) is_producer = true;
            if (cfg.consumer_risc_mask & proc_bit) is_consumer = true;

            if (!is_producer && !is_consumer) continue;
            iface.active = true;

            uint8_t neo_id = 0;
            uint8_t counter_base = cfg.dfb_index;

            if (is_producer) {
                iface.num_tcs_to_rr = cfg.num_consumers;
                iface.stride_size = cfg.entry_size * cfg.num_producers;
                for (uint8_t c = 0; c < cfg.num_consumers && c < MAX_TC_SLOTS_PER_DFB; ++c) {
                    auto& slot = iface.tc_slots[c];
                    slot.neo_id = neo_id;
                    slot.counter_id = counter_base + c;
                    slot.base_addr = static_cast<uint32_t>(
                        reinterpret_cast<uintptr_t>(dsync.base));
                    slot.limit = slot.base_addr + cfg.num_entries * cfg.entry_size;
                    slot.wr_ptr = slot.base_addr;
                    slot.rd_ptr = slot.base_addr;
                }
                if (cfg.num_consumers > 1)
                    iface.broadcast_tc = true;
            } else {
                iface.num_tcs_to_rr = 1;
                iface.stride_size = cfg.entry_size * cfg.num_consumers;
                auto& slot = iface.tc_slots[0];
                slot.neo_id = neo_id;
                // Each consumer gets its own TC slot index
                uint8_t consumer_ordinal = 0;
                uint16_t mask = cfg.consumer_risc_mask;
                for (uint8_t b = 0; b < 16 && mask; ++b) {
                    if (mask & (1u << b)) {
                        if (b == kd.processor_id) break;
                        ++consumer_ordinal;
                    }
                    mask &= ~(1u << b);
                }
                slot.counter_id = counter_base + consumer_ordinal;
                slot.base_addr = static_cast<uint32_t>(
                    reinterpret_cast<uintptr_t>(dsync.base));
                slot.limit = slot.base_addr + cfg.num_entries * cfg.entry_size;
                slot.wr_ptr = slot.base_addr;
                slot.rd_ptr = slot.base_addr;
            }
        }
    }
    return result;
}

} // anonymous namespace

void EnqueueProgram(Device& device, Program& program, bool /*blocking*/) {
    Core& core = device.core();

    // 1. Instantiate all CBs on the core
    for (auto& cfg : program.cb_configs()) {
        core.cb(cfg.cb_index) =
            std::make_shared<CircularBuffer>(cfg.num_pages, cfg.page_size);
    }

    // 2. Set up DFBs if present
    bool has_dfbs = program.has_dfbs();
    std::vector<std::vector<EmuleDFBInterface>> dfb_iface_per_thread;

    if (has_dfbs) {
        if (!core.tile_counters())
            core.init_tile_counters(4);  // 4 neos default

        // Initialize DFB sync state + tile counters on core
        for (auto& cfg : program.dfb_configs()) {
            uint32_t total_bytes = cfg.entry_size * cfg.num_entries;
            uint32_t base_addr = core.l1_alloc(total_bytes);
            uint8_t* base = reinterpret_cast<uint8_t*>(
                static_cast<uintptr_t>(base_addr));
            uint32_t capacity = cfg.num_entries;
            core.init_dfb_sync(cfg.dfb_index, base, cfg.entry_size,
                               cfg.num_entries, capacity);

            // Set up tile counter capacities
            uint8_t counter_base = static_cast<uint8_t>(cfg.dfb_index);
            uint8_t num_tcs = std::max(cfg.num_producers, cfg.num_consumers);
            for (uint8_t c = 0; c < num_tcs; ++c) {
                auto& tc = core.tile_counters()->get(0, counter_base + c);
                tc.capacity = capacity;
                tc.posted.store(0, std::memory_order_relaxed);
                tc.acked.store(0, std::memory_order_relaxed);
            }
        }

        dfb_iface_per_thread = build_dfb_interfaces(core, program);
    }

    // 3. Launch one thread per kernel descriptor
    size_t num_threads = program.kernels().size();
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // Init barrier: all threads wait until context is set up before running kernels
    std::barrier init_barrier(static_cast<ptrdiff_t>(num_threads));

    for (size_t i = 0; i < num_threads; ++i) {
        KernelDescriptor& kd = program.kernels()[i];
        threads.emplace_back([&kd, &core, &device, &init_barrier,
                              has_dfbs, i, &dfb_iface_per_thread]() {
            __rt_args      = kd.rt_args;
            __core         = &core;
            __device       = &device;
            __processor_id = kd.processor_id;

            if (has_dfbs) {
                __dfb_ifaces    = dfb_iface_per_thread[i].data();
                __emule_dfbs    = dfb_iface_per_thread[i].data();
                __emule_tc_array = core.tile_counters();
            }

            init_barrier.arrive_and_wait();
            kd.fn();
        });
    }

    // 4. Join all threads
    for (auto& t : threads) {
        t.join();
    }

    // 5. Tear down CBs so next EnqueueProgram starts fresh
    for (auto& cfg : program.cb_configs()) {
        core.cb(cfg.cb_index).reset();
    }

    // 6. Tear down DFBs
    if (has_dfbs) {
        core.reset_dfb_sync();
    }
}

} // namespace tt_emule
