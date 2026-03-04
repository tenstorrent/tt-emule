#include "tt_emule/host_api.hpp"
#include "tt_emule/tile.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

// Kernel entry points defined in tests/matmul/*.cpp
extern void reader_kernel_main();
extern void writer_kernel_main();
extern void compute_kernel_main();

using namespace tt_emule;

static constexpr uint32_t NUM_TILES = 1;

int main() {
    // ---- Setup device ----
    Device* device = CreateDevice(0);

    // ---- Allocate DRAM buffers ----
    constexpr size_t BUF_SIZE = NUM_TILES * Tile::SIZE_BYTES;
    auto src0_buf = CreateBuffer(*device, BUF_SIZE, Tile::SIZE_BYTES);
    auto src1_buf = CreateBuffer(*device, BUF_SIZE, Tile::SIZE_BYTES);
    auto dst_buf  = CreateBuffer(*device, BUF_SIZE, Tile::SIZE_BYTES);

    // ---- Fill source data: A = all 1.0f, B = all 1.0f ----
    // Result: each element = dot(row_of_A, col_of_B) = 32 * (1.0 * 1.0) = 32.0f
    std::vector<float> src0_data(NUM_TILES * Tile::NUM_ELEMENTS, 1.0f);
    std::vector<float> src1_data(NUM_TILES * Tile::NUM_ELEMENTS, 1.0f);
    EnqueueWriteBuffer(*device, *src0_buf, src0_data.data());
    EnqueueWriteBuffer(*device, *src1_buf, src1_data.data());

    // ---- Build program ----
    Program prog = CreateProgram();

    CreateCircularBuffer(prog, {0, 0}, {0,  2, Tile::SIZE_BYTES}); // cb_in0
    CreateCircularBuffer(prog, {0, 0}, {1,  2, Tile::SIZE_BYTES}); // cb_in1
    CreateCircularBuffer(prog, {0, 0}, {16, 2, Tile::SIZE_BYTES}); // cb_out0

    constexpr uint32_t NOC_X = 1, NOC_Y = 0;

    uint32_t reader_id = CreateKernel(prog, reader_kernel_main, {0, 0},
                                      DataMovementConfig{KernelType::DataMovement0});
    SetRuntimeArgs(prog, reader_id, {0, 0}, {
        static_cast<uint32_t>(src0_buf->dram_offset()),
        static_cast<uint32_t>(src1_buf->dram_offset()),
        NUM_TILES,
        static_cast<uint32_t>(Tile::SIZE_BYTES),
        NOC_X, NOC_Y
    });

    uint32_t writer_id = CreateKernel(prog, writer_kernel_main, {0, 0},
                                      DataMovementConfig{KernelType::DataMovement1});
    SetRuntimeArgs(prog, writer_id, {0, 0}, {
        static_cast<uint32_t>(dst_buf->dram_offset()),
        NUM_TILES,
        static_cast<uint32_t>(Tile::SIZE_BYTES),
        NOC_X, NOC_Y
    });

    uint32_t compute_id = CreateKernel(prog, compute_kernel_main, {0, 0},
                                       ComputeConfig{});
    SetRuntimeArgs(prog, compute_id, {0, 0}, {NUM_TILES});

    // ---- Run ----
    EnqueueProgram(*device, prog);
    Finish(*device);

    // ---- Verify ----
    std::vector<float> result(NUM_TILES * Tile::NUM_ELEMENTS, 0.0f);
    EnqueueReadBuffer(*device, *dst_buf, result.data());

    bool pass = true;
    for (size_t i = 0; i < result.size(); ++i) {
        if (result[i] != 32.0f) {
            std::fprintf(stderr, "FAIL: result[%zu] = %f (expected 32.0)\n", i, result[i]);
            pass = false;
            break;
        }
    }

    CloseDevice(device);

    if (pass) {
        std::printf("PASSED: all %u tiles == 32.0f\n", NUM_TILES);
        return 0;
    }
    return 1;
}
