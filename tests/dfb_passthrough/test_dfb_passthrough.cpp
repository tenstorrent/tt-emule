#include "tt_emule/host_api.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

extern void dfb_producer_kernel_main();
extern void dfb_consumer_kernel_main();

using namespace tt_emule;

static constexpr uint32_t NUM_ENTRIES  = 8;
static constexpr uint32_t ENTRY_WORDS  = 256;  // 1 KB entries
static constexpr uint32_t ENTRY_BYTES  = ENTRY_WORDS * sizeof(uint32_t);
static constexpr uint32_t SEED         = 42;

int main() {
    Device* device = CreateDevice(0);

    // Allocate output region in L1
    uint32_t output_bytes = NUM_ENTRIES * ENTRY_BYTES;
    uint32_t output_addr  = device->l1_alloc(output_bytes);

    // Build program
    Program prog = CreateProgram();

    // DFB 0: 1 DM producer (processor 0), 1 DM consumer (processor 1)
    DataflowBufferConfig dfb_cfg;
    dfb_cfg.entry_size    = ENTRY_BYTES;
    dfb_cfg.num_entries   = NUM_ENTRIES;
    dfb_cfg.producer_risc_mask = 0x01;  // bit 0 = DM0
    dfb_cfg.num_producers = 1;
    dfb_cfg.consumer_risc_mask = 0x02;  // bit 1 = DM1
    dfb_cfg.num_consumers = 1;
    dfb_cfg.producer_access_pattern = AccessPattern::STRIDED;
    dfb_cfg.consumer_access_pattern = AccessPattern::STRIDED;
    CreateDataflowBuffer(prog, dfb_cfg);

    // Producer: QuasarDM, processor_id=0
    uint32_t prod_id = CreateKernel(prog, dfb_producer_kernel_main, {0, 0},
                                    QuasarDataMovementConfig{}, /*processor_id=*/0);
    SetRuntimeArgs(prog, prod_id, {0, 0},
                   {0, NUM_ENTRIES, ENTRY_WORDS, SEED});

    // Consumer: QuasarDM, processor_id=1
    uint32_t cons_id = CreateKernel(prog, dfb_consumer_kernel_main, {0, 0},
                                    QuasarDataMovementConfig{}, /*processor_id=*/1);
    SetRuntimeArgs(prog, cons_id, {0, 0},
                   {0, NUM_ENTRIES, ENTRY_WORDS, output_addr});

    // Run
    EnqueueProgram(*device, prog);
    Finish(*device);

    // Verify
    const uint32_t* result = reinterpret_cast<const uint32_t*>(
        static_cast<uintptr_t>(output_addr));

    bool pass = true;
    for (uint32_t i = 0; i < NUM_ENTRIES; ++i) {
        for (uint32_t w = 0; w < ENTRY_WORDS; ++w) {
            uint32_t expected = SEED + i * ENTRY_WORDS + w;
            uint32_t actual   = result[i * ENTRY_WORDS + w];
            if (actual != expected) {
                std::fprintf(stderr,
                    "FAIL: entry[%u] word[%u] = %u (expected %u)\n",
                    i, w, actual, expected);
                pass = false;
                goto done;
            }
        }
    }

done:
    CloseDevice(device);

    if (pass) {
        std::printf("PASSED: DFB passthrough %u entries x %u words\n",
                    NUM_ENTRIES, ENTRY_WORDS);
        return 0;
    }
    return 1;
}
