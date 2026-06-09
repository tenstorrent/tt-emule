// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

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

// Thread-local context consumed by JIT kernel stubs in include/jit_hw/.
thread_local std::vector<uint32_t> __rt_args;
thread_local tt_emule::Core*       __core   = nullptr;
thread_local tt_emule::Device*     __device = nullptr;
thread_local uint8_t               __processor_id = 0;

// DFB thread-locals consumed by jit_hw/cb_api.h and jit_hw/dfb_api.h.
thread_local tt_emule::TileCounterArray*  __emule_tc_array = nullptr;
thread_local tt_emule::EmuleDFBInterface* __emule_dfbs     = nullptr;

// Sanitizer thread-locals consumed by jit_hw headers. Storage must live in
// this library so JIT-compiled kernel .so files can resolve the symbols at
// dlopen time, including from test binaries that don't link tt-metal's
// emulated_program_runner.cpp. The host writes these via extern declarations
// before invoking each kernel; when the host leaves them at their zero/null
// default, the inline sanitizer checks in jit_kernel_stubs.hpp / cb_api.h /
// dataflow_api.h short-circuit as no-ops.
thread_local uint32_t __emule_sem_l1_range_start = 0;
thread_local uint32_t __emule_sem_l1_range_end = 0;
thread_local const char* __emule_kernel_name = nullptr;
thread_local uint32_t __emule_pending_noc_reads = 0;
thread_local uint32_t __emule_l1_unreserved_base = 0;
thread_local const uint64_t* __emule_l1_tensor_ranges = nullptr;
thread_local uint32_t __emule_l1_tensor_ranges_count = 0;
thread_local const uint64_t* __emule_l1_padding_ranges = nullptr;
thread_local uint32_t __emule_l1_padding_ranges_count = 0;
thread_local uint64_t* __emule_l1_resolved_ranges = nullptr;
thread_local uint32_t* __emule_l1_resolved_ranges_count = nullptr;
thread_local uint32_t __emule_l1_resolved_ranges_capacity = 0;
thread_local uint32_t __emule_cb_reserved_pages[32] = {};
thread_local uint32_t __emule_cb_waited_pages[32] = {};
thread_local bool __emule_cb_boundary_strict = false;

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
    auto& kernels  = program.kernels();
    size_t nk      = kernels.size();

    std::vector<std::vector<EmuleDFBInterface>> result(
        nk, std::vector<EmuleDFBInterface>(MAX_DFBS));

    if (dfb_cfgs.empty()) return result;

    for (auto& cfg : dfb_cfgs) {
        auto& dsync = core.dfb_sync_array()[cfg.dfb_index];
        bool is_blocked = (cfg.consumer_access_pattern == AccessPattern::BLOCKED);
        // STRIDED: M = max(P, C); BLOCKED: M = P (only producers interleave)
        uint32_t M = is_blocked ? cfg.num_producers
                                : std::max(cfg.num_producers, cfg.num_consumers);
        uint32_t stride_size = M * cfg.entry_size;
        // Space counter IDs by MAX_TC_SLOTS_PER_DFB per DFB to prevent
        // cross-DFB TC index collision in multi-DFB programs.
        // With neo_id=0, effective limit is TILE_COUNTERS_PER_NEO / MAX_TC_SLOTS_PER_DFB = 8 DFBs.
        if (cfg.dfb_index >= (TILE_COUNTERS_PER_NEO / MAX_TC_SLOTS_PER_DFB)) {
            throw std::out_of_range(
                "dfb_index exceeds safe TC range (max 8 DFBs per NEO with neo_id=0)");
        }
        uint8_t counter_base = static_cast<uint8_t>(
            cfg.dfb_index * MAX_TC_SLOTS_PER_DFB);

        for (size_t ki = 0; ki < nk; ++ki) {
            auto& kd    = kernels[ki];
            auto& iface = result[ki][cfg.dfb_index];
            iface.entry_size  = cfg.entry_size;
            iface.num_entries = cfg.num_entries;
            iface.stride_size = stride_size;

            // WH/BH ComputeKernel sets bit 2 in risc_mask; Quasar uses bits 8+.
            bool is_tensix = (kd.type == KernelType::Compute || kd.type == KernelType::QuasarCompute);
            uint8_t my_bit;  // bit position in the risc_mask for this kernel
            if (is_tensix) {
                bool quasar_masks = ((cfg.producer_risc_mask | cfg.consumer_risc_mask) & 0xFF00u) != 0;
                my_bit = quasar_masks
                    ? static_cast<uint8_t>(kd.processor_id + 8)
                    : 2;  // WH/BH convention
            } else {
                my_bit = kd.processor_id;
            }
            uint16_t proc_bit = static_cast<uint16_t>(1u << my_bit);
            bool is_producer  = (cfg.producer_risc_mask & proc_bit) != 0;
            bool is_consumer  = (cfg.consumer_risc_mask & proc_bit) != 0;

            if (!is_producer && !is_consumer) continue;
            iface.active       = true;
            iface.broadcast_tc = false;  // STRIDED: round-robin, not broadcast

            uint32_t base_addr = static_cast<uint32_t>(
                reinterpret_cast<uintptr_t>(dsync.base));

            if (is_producer) {
                // Determine this producer's ordinal among all producers
                uint8_t p = 0;
                for (uint8_t b = 0; b < 16; ++b) {
                    if (!(cfg.producer_risc_mask & (1u << b))) continue;
                    if (b == my_bit) break;
                    ++p;
                }
                if (is_blocked) {
                    // BLOCKED DM-DM: producer broadcasts to all consumer TCs.
                    // TC for (producer p, consumer c) has counter_id = p*C + c.
                    // Each producer's TC slots cover its own contiguous block.
                    iface.broadcast_tc = true;
                    iface.num_tcs_to_rr = static_cast<uint8_t>(cfg.num_consumers);
                    iface.stride_size = cfg.entry_size;  // BLOCKED: stride_in_entries = 1
                    uint32_t capacity_per_p = cfg.num_entries / cfg.num_producers;
                    uint32_t block_off = p * capacity_per_p * cfg.entry_size;
                    for (uint32_t c = 0; c < cfg.num_consumers && c < MAX_TC_SLOTS_PER_DFB; ++c) {
                        auto& slot       = iface.tc_slots[c];
                        slot.neo_id      = 0;
                        slot.counter_id  = counter_base + static_cast<uint8_t>(p * cfg.num_consumers + c);
                        slot.base_addr   = base_addr + block_off;
                        slot.limit       = slot.base_addr + capacity_per_p * cfg.entry_size;
                        slot.wr_ptr      = slot.base_addr;
                        slot.rd_ptr      = slot.base_addr;
                    }
                } else {
                    // STRIDED: existing code
                    uint32_t num_tcs = M / cfg.num_producers;
                    iface.num_tcs_to_rr = static_cast<uint8_t>(num_tcs);
                    for (uint32_t k = 0; k < num_tcs && k < MAX_TC_SLOTS_PER_DFB; ++k) {
                        uint8_t tc_idx   = static_cast<uint8_t>(p + k * cfg.num_producers);
                        auto& slot       = iface.tc_slots[k];
                        slot.neo_id      = 0;
                        slot.counter_id  = counter_base + tc_idx;
                        slot.base_addr   = base_addr;
                        slot.limit       = base_addr + cfg.num_entries * cfg.entry_size;
                        uint32_t offset  = tc_idx * cfg.entry_size;
                        slot.wr_ptr      = base_addr + offset;
                        slot.rd_ptr      = base_addr + offset;
                    }
                }
            } else {
                // Determine this consumer's ordinal among all consumers
                uint8_t c = 0;
                for (uint8_t b = 0; b < 16; ++b) {
                    if (!(cfg.consumer_risc_mask & (1u << b))) continue;
                    if (b == my_bit) break;
                    ++c;
                }
                if (is_blocked) {
                    // BLOCKED consumer: drain each TC fully before advancing.
                    // Each TC slot covers one producer's contiguous block.
                    iface.num_tcs_to_rr = static_cast<uint8_t>(cfg.num_producers);
                    iface.stride_size = cfg.entry_size;  // BLOCKED: stride_in_entries = 1
                    iface.drain_per_tc = true;
                    uint32_t capacity_per_p = cfg.num_entries / cfg.num_producers;
                    for (uint32_t pi = 0; pi < cfg.num_producers && pi < MAX_TC_SLOTS_PER_DFB; ++pi) {
                        auto& slot       = iface.tc_slots[pi];
                        slot.neo_id      = 0;
                        slot.counter_id  = counter_base + static_cast<uint8_t>(pi * cfg.num_consumers + c);
                        uint32_t block_off = pi * capacity_per_p * cfg.entry_size;
                        slot.base_addr   = base_addr + block_off;
                        slot.limit       = slot.base_addr + capacity_per_p * cfg.entry_size;
                        slot.rd_ptr      = slot.base_addr;
                        slot.wr_ptr      = slot.base_addr;
                    }
                } else {
                    // STRIDED: existing code
                    uint32_t num_tcs = M / cfg.num_consumers;
                    iface.num_tcs_to_rr = static_cast<uint8_t>(num_tcs);
                    for (uint32_t k = 0; k < num_tcs && k < MAX_TC_SLOTS_PER_DFB; ++k) {
                        uint8_t tc_idx   = static_cast<uint8_t>(c + k * cfg.num_consumers);
                        auto& slot       = iface.tc_slots[k];
                        slot.neo_id      = 0;
                        slot.counter_id  = counter_base + tc_idx;
                        slot.base_addr   = base_addr;
                        slot.limit       = base_addr + cfg.num_entries * cfg.entry_size;
                        uint32_t offset  = tc_idx * cfg.entry_size;
                        slot.rd_ptr      = base_addr + offset;
                        slot.wr_ptr      = base_addr + offset;
                    }
                }
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
            // Per-TC capacity = total entries divided by interleaving factor M
            bool is_blocked = (cfg.consumer_access_pattern == AccessPattern::BLOCKED);
            uint32_t M = is_blocked ? cfg.num_producers
                                    : std::max(cfg.num_producers, cfg.num_consumers);
            uint32_t capacity = cfg.num_entries / M;
            core.init_dfb_sync(cfg.dfb_index, base, cfg.entry_size,
                               cfg.num_entries, capacity);

            // Set up tile counter capacities; space counter IDs by
            // MAX_TC_SLOTS_PER_DFB per DFB (same formula as build_dfb_interfaces).
            // With neo_id=0, effective limit is TILE_COUNTERS_PER_NEO / MAX_TC_SLOTS_PER_DFB = 8 DFBs.
            if (cfg.dfb_index >= (TILE_COUNTERS_PER_NEO / MAX_TC_SLOTS_PER_DFB)) {
                throw std::out_of_range(
                    "dfb_index exceeds safe TC range (max 8 DFBs per NEO with neo_id=0)");
            }
            uint8_t counter_base = static_cast<uint8_t>(
                cfg.dfb_index * MAX_TC_SLOTS_PER_DFB);
            // BLOCKED DM-DM uses P*C TCs (one per producer-consumer pair)
            uint8_t num_tcs = is_blocked
                ? static_cast<uint8_t>(cfg.num_producers * cfg.num_consumers)
                : static_cast<uint8_t>(M);
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
    std::vector<std::exception_ptr> thread_exceptions(num_threads);

    // Init barrier: all threads wait until context is set up before running kernels
    std::barrier init_barrier(static_cast<ptrdiff_t>(num_threads));

    for (size_t i = 0; i < num_threads; ++i) {
        KernelDescriptor& kd = program.kernels()[i];
        threads.emplace_back([&kd, &core, &device, &init_barrier,
                              has_dfbs, i, &dfb_iface_per_thread,
                              &thread_exceptions]() {
            __rt_args      = kd.rt_args;
            __core         = &core;
            __device       = &device;
            __processor_id = kd.processor_id;

            if (has_dfbs) {
                __emule_dfbs     = dfb_iface_per_thread[i].data();
                __emule_tc_array = core.tile_counters();
            }

            init_barrier.arrive_and_wait();
            try {
                kd.fn();
            } catch (...) {
                thread_exceptions[i] = std::current_exception();
            }
        });
    }

    // 4. Join all threads
    for (auto& t : threads) {
        t.join();
    }

    // 4b. Rethrow first kernel exception
    for (size_t i = 0; i < num_threads; ++i) {
        if (thread_exceptions[i]) {
            std::rethrow_exception(thread_exceptions[i]);
        }
    }

    // 5. Tear down CBs so next EnqueueProgram starts fresh
    for (auto& cfg : program.cb_configs()) {
        core.cb(cfg.cb_index).reset();
    }

    // 6. Tear down DFBs
    if (has_dfbs) {
        core.reset_dfb_sync();
    }

    // 7. Reset L1 bump allocator so next EnqueueProgram starts fresh
    core.reset_l1_bump();
}

} // namespace tt_emule
