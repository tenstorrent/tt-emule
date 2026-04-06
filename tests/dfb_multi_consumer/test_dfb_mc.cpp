#include "tt_emule/host_api.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

extern void dfb_mc_producer_kernel_main();
extern void dfb_mc_consumer_kernel_main();

using namespace tt_emule;

static constexpr uint32_t NUM_CONSUMERS  = 4;
static constexpr uint32_t NUM_ENTRIES    = 8;   // total; 2 per consumer
static constexpr uint32_t ENTRY_WORDS    = 256; // 1 KB entries
static constexpr uint32_t ENTRY_BYTES    = ENTRY_WORDS * sizeof(uint32_t);
static constexpr uint32_t SEED           = 0xAB00;

// Per-consumer output region (NUM_ENTRIES/NUM_CONSUMERS entries each)
static constexpr uint32_t ENTRIES_PER_CONSUMER = NUM_ENTRIES / NUM_CONSUMERS; // 2

int main() {
    Device* device = CreateDevice(0);

    // Allocate per-consumer output regions in L1
    uint32_t output_addrs[NUM_CONSUMERS];
    for (uint32_t c = 0; c < NUM_CONSUMERS; ++c) {
        output_addrs[c] = device->l1_alloc(
            ENTRIES_PER_CONSUMER * ENTRY_BYTES);
    }

    // Build program
    Program prog = CreateProgram();

    // DFB 0: 1P-4C STRIDED configuration
    // producer: processor 0 (bit 0 = 0x01)
    // consumers: processors 1-4 (bits 1-4 = 0x1E)
    DataflowBufferConfig dfb_cfg;
    dfb_cfg.entry_size            = ENTRY_BYTES;
    dfb_cfg.num_entries           = NUM_ENTRIES;
    dfb_cfg.producer_risc_mask    = 0x01;
    dfb_cfg.num_producers         = 1;
    dfb_cfg.consumer_risc_mask    = 0x1E;
    dfb_cfg.num_consumers         = NUM_CONSUMERS;
    dfb_cfg.producer_access_pattern = AccessPattern::STRIDED;
    dfb_cfg.consumer_access_pattern = AccessPattern::STRIDED;
    CreateDataflowBuffer(prog, dfb_cfg);

    // Producer kernel: processor_id=0
    uint32_t prod_id = CreateKernel(prog, dfb_mc_producer_kernel_main, {0, 0},
                                    QuasarDataMovementConfig{}, /*processor_id=*/0);
    SetRuntimeArgs(prog, prod_id, {0, 0},
                   {0, NUM_ENTRIES, ENTRY_WORDS, SEED});

    // Consumer kernels: processor_id=1..4
    uint32_t cons_ids[NUM_CONSUMERS];
    for (uint32_t c = 0; c < NUM_CONSUMERS; ++c) {
        cons_ids[c] = CreateKernel(prog, dfb_mc_consumer_kernel_main, {0, 0},
                                   QuasarDataMovementConfig{},
                                   /*processor_id=*/static_cast<uint8_t>(1 + c));
        SetRuntimeArgs(prog, cons_ids[c], {0, 0},
                       {0, ENTRIES_PER_CONSUMER, ENTRY_WORDS, output_addrs[c]});
    }

    // Run
    EnqueueProgram(*device, prog);
    Finish(*device);

    // Verify: with STRIDED 1P-4C, the producer round-robins across 4 TC slots.
    // Entry i (global producer index) goes to TC slot i%4 = consumer i%4.
    // Consumer c receives entries at global indices c, c+4, c+8, ...
    // Producer wrote: seed + global_i * ENTRY_WORDS + w
    // So consumer c, local entry 0 has data for global_i = c
    //             consumer c, local entry 1 has data for global_i = c+4

    bool pass = true;
    for (uint32_t c = 0; c < NUM_CONSUMERS && pass; ++c) {
        const uint32_t* result = reinterpret_cast<const uint32_t*>(
            static_cast<uintptr_t>(output_addrs[c]));

        for (uint32_t local = 0; local < ENTRIES_PER_CONSUMER && pass; ++local) {
            uint32_t global_i = c + local * NUM_CONSUMERS;
            for (uint32_t w = 0; w < ENTRY_WORDS; ++w) {
                uint32_t expected = SEED + global_i * ENTRY_WORDS + w;
                uint32_t actual   = result[local * ENTRY_WORDS + w];
                if (actual != expected) {
                    std::fprintf(stderr,
                        "FAIL: consumer[%u] local_entry[%u] (global=%u) word[%u] = %u (expected %u)\n",
                        c, local, global_i, w, actual, expected);
                    pass = false;
                    break;
                }
            }
        }
    }

    CloseDevice(device);

    if (pass) {
        std::printf("PASSED: DFB multi-consumer 1P-%uC, %u total entries x %u words\n",
                    NUM_CONSUMERS, NUM_ENTRIES, ENTRY_WORDS);
        return 0;
    }
    return 1;
}
