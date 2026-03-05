// Smoke test: exercises the tt::tt_metal compatibility interface.
// All host API calls go through tt_metal_compat.hpp; kernel source files are the
// same objects compiled for the eltwise_add test (no source changes needed).
#include "tt_emule/tt_metal_compat.hpp"
#include "tt_emule/tile.hpp"
#include <cstdio>
#include <vector>

extern void reader_kernel_main();
extern void writer_kernel_main();
extern void compute_kernel_main();

namespace ttm = tt::tt_metal;

static constexpr uint32_t NUM_TILES = 4;

int main() {
    // ---- Device + CommandQueue ----
    ttm::IDevice* device = ttm::CreateDevice(0);
    ttm::CommandQueue cq = ttm::CreateCommandQueue(*device);

    // ---- Allocate DRAM buffers ----
    constexpr size_t BUF_SIZE = NUM_TILES * tt_emule::Tile::SIZE_BYTES;
    auto src0_buf = ttm::CreateBuffer(*device, BUF_SIZE, tt_emule::Tile::SIZE_BYTES);
    auto src1_buf = ttm::CreateBuffer(*device, BUF_SIZE, tt_emule::Tile::SIZE_BYTES);
    auto dst_buf  = ttm::CreateBuffer(*device, BUF_SIZE, tt_emule::Tile::SIZE_BYTES);

    // ---- Fill source data via CommandQueue overloads ----
    std::vector<float> src0_data(NUM_TILES * tt_emule::Tile::NUM_ELEMENTS, 1.0f);
    std::vector<float> src1_data(NUM_TILES * tt_emule::Tile::NUM_ELEMENTS, 2.0f);
    ttm::EnqueueWriteBuffer(cq, *src0_buf, src0_data.data());
    ttm::EnqueueWriteBuffer(cq, *src1_buf, src1_data.data());

    // ---- Build program ----
    ttm::Program prog = ttm::CreateProgram();
    ttm::CreateCircularBuffer(prog, {0, 0}, {0,  2, tt_emule::Tile::SIZE_BYTES}); // cb_in0
    ttm::CreateCircularBuffer(prog, {0, 0}, {1,  2, tt_emule::Tile::SIZE_BYTES}); // cb_in1
    ttm::CreateCircularBuffer(prog, {0, 0}, {16, 2, tt_emule::Tile::SIZE_BYTES}); // cb_out0

    constexpr uint32_t NOC_X = 1, NOC_Y = 0;

    ttm::KernelHandle reader_id = ttm::CreateKernel(
        prog, reader_kernel_main, {0, 0},
        ttm::DataMovementConfig{ttm::KernelType::DataMovement0});
    ttm::SetRuntimeArgs(prog, reader_id, {0, 0}, {
        static_cast<uint32_t>(src0_buf->dram_offset()),
        static_cast<uint32_t>(src1_buf->dram_offset()),
        NUM_TILES,
        static_cast<uint32_t>(tt_emule::Tile::SIZE_BYTES),
        NOC_X, NOC_Y
    });

    ttm::KernelHandle writer_id = ttm::CreateKernel(
        prog, writer_kernel_main, {0, 0},
        ttm::DataMovementConfig{ttm::KernelType::DataMovement1});
    ttm::SetRuntimeArgs(prog, writer_id, {0, 0}, {
        static_cast<uint32_t>(dst_buf->dram_offset()),
        NUM_TILES,
        static_cast<uint32_t>(tt_emule::Tile::SIZE_BYTES),
        NOC_X, NOC_Y
    });

    ttm::KernelHandle compute_id = ttm::CreateKernel(
        prog, compute_kernel_main, {0, 0}, ttm::ComputeConfig{});
    ttm::SetRuntimeArgs(prog, compute_id, {0, 0}, {NUM_TILES});

    // ---- Run via CommandQueue ----
    ttm::EnqueueProgram(cq, prog);
    ttm::Finish(cq);

    // ---- Verify ----
    std::vector<float> result(NUM_TILES * tt_emule::Tile::NUM_ELEMENTS, 0.0f);
    ttm::EnqueueReadBuffer(cq, *dst_buf, result.data());

    bool pass = true;
    for (size_t i = 0; i < result.size(); ++i) {
        if (result[i] != 3.0f) {
            std::fprintf(stderr, "FAIL: result[%zu] = %f (expected 3.0)\n", i, result[i]);
            pass = false;
            break;
        }
    }

    ttm::CloseDevice(device);

    if (pass) {
        std::printf("PASSED: all %u tiles == 3.0f (via tt::tt_metal compat)\n", NUM_TILES);
        return 0;
    }
    return 1;
}
