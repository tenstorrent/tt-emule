// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/mman.h>

#define TRISC_MATH 1
#define NUM_DRAM_BANKS 12
#define NUM_L1_BANKS 64
#define EMULE_CB_DATA_FORMATS \
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, \
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5
#define EMULE_TILE_R_DIM \
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, \
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32
#define EMULE_TILE_C_DIM \
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, \
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32

#include "jit_hw/api/compute/tilize.h"

thread_local ThreadCommonCtx* __emule_self = nullptr;
thread_local uint8_t my_x[2] = {};
thread_local uint8_t my_y[2] = {};
thread_local uint32_t __emule_pending_noc_reads = 0;

extern "C" uint8_t* __emule_dram_ptr(uint64_t) { return nullptr; }
extern "C" uint8_t* __emule_noc_resolve(uint32_t, uint32_t, uint64_t) { return nullptr; }
uint16_t dram_bank_to_noc_xy[2][NUM_DRAM_BANKS] = {};
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] = {};
uint16_t l1_bank_to_noc_xy[2][NUM_L1_BANKS] = {};
int32_t bank_to_l1_offset[NUM_L1_BANKS] = {};

namespace {

constexpr uint32_t kInputCb = 3;
constexpr uint32_t kOutputCb = 2;
constexpr uint32_t kBlockTiles = 9;
constexpr uint32_t kTileElems = 32 * 32;
constexpr uint32_t kTileBytes = kTileElems * sizeof(uint16_t);
constexpr uint32_t kInputBytes = kBlockTiles * kTileBytes;
constexpr uint16_t kSentinel = 0xdead;

struct Harness {
    ComputeThreadCtx ctx;
    std::array<tt_emule::CBSyncState, 32> cbs{};
    uint16_t* input = nullptr;
    uint16_t* output = nullptr;

    Harness() {
        ctx.cbs = cbs.data();
        __emule_self = &ctx;

        input = static_cast<uint16_t*>(alloc_low(kInputBytes));
        output = static_cast<uint16_t*>(alloc_low(kBlockTiles * 3 * kTileBytes));

        cbs[kInputCb].base = reinterpret_cast<uint8_t*>(input);
        cbs[kInputCb].page_size = kInputBytes;
        cbs[kInputCb].num_pages = 1;

        cbs[kOutputCb].base = reinterpret_cast<uint8_t*>(output);
        cbs[kOutputCb].page_size = kTileBytes;
        cbs[kOutputCb].num_pages = kBlockTiles * 3;

        fill_output(kSentinel);
    }

    ~Harness() {
        if (input != nullptr) {
            munmap(input, kInputBytes);
        }
        if (output != nullptr) {
            munmap(output, kBlockTiles * 3 * kTileBytes);
        }
    }

    static void* alloc_low(size_t bytes) {
        void* ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
        if (ptr == MAP_FAILED) {
            std::perror("mmap");
            std::abort();
        }
        std::memset(ptr, 0, bytes);
        return ptr;
    }

    void fill_input(uint16_t base) {
        for (uint32_t t = 0; t < kBlockTiles; ++t) {
            const uint16_t value = static_cast<uint16_t>(base + t);
            for (uint32_t r = 0; r < 32; ++r) {
                for (uint32_t c = 0; c < 32; ++c) {
                    input[r * kBlockTiles * 32 + t * 32 + c] = value;
                }
            }
        }
    }

    void fill_output(uint16_t value) {
        std::fill(output, output + kBlockTiles * 3 * kTileElems, value);
    }

    uint16_t output_tile_head(uint32_t tile) const {
        return output[tile * kTileElems + __emule_nfaces::tile_rm_to_nfaces(0, 32)];
    }
};

bool tilize_resets_pack_offset_for_each_reserved_block() {
    Harness h;

    h.fill_input(0x1000);
    tilize_block(kInputCb, kBlockTiles, kOutputCb);

    __emule_cb_advance_wr(kOutputCb, kBlockTiles);
    h.fill_input(0x2000);
    tilize_block(kInputCb, kBlockTiles, kOutputCb);

    for (uint32_t t = 0; t < kBlockTiles; ++t) {
        const uint16_t got = h.output_tile_head(kBlockTiles + t);
        const uint16_t expected = static_cast<uint16_t>(0x2000 + t);
        if (got != expected) {
            std::cerr << "second tilize block wrote tile " << t << " to the wrong slot: got 0x"
                      << std::hex << got << " expected 0x" << expected << std::dec << '\n';
            return false;
        }
    }

    for (uint32_t t = kBlockTiles * 2; t < kBlockTiles * 3; ++t) {
        if (h.output_tile_head(t) != kSentinel) {
            std::cerr << "tilize_block leaked into tile " << t << '\n';
            return false;
        }
    }

    return true;
}

}  // namespace

int main() {
    return tilize_resets_pack_offset_for_each_reserved_block() ? EXIT_SUCCESS : EXIT_FAILURE;
}
