// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/mman.h>

#define TRISC_MATH 1
#define NUM_DRAM_BANKS 12
#define NUM_L1_BANKS 64
#define EMULE_CB_DATA_FORMATS \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#define EMULE_TILE_R_DIM \
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, \
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32
#define EMULE_TILE_C_DIM \
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, \
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32

#include "jit_hw/api/compute/cb_api.h"
#include "jit_hw/api/compute/eltwise_unary/rounding.h"
#include "jit_hw/api/dataflow/circular_buffer.h"
#include "jit_hw/llk_math_eltwise_binary.h"
#include "jit_hw/llk_unpack_a.h"

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

constexpr uint32_t kPageBytes = 32 * 32 * sizeof(float);

struct Harness {
    ComputeThreadCtx ctx;
    std::array<tt_emule::CBSyncState, 32> cbs{};
    std::array<float*, 3> tiles{};
    std::array<size_t, 3> tile_bytes{};

    Harness() {
        ctx.cbs = cbs.data();
        __emule_self = &ctx;
    }

    ~Harness() {
        for (uint32_t i = 0; i < tiles.size(); ++i) {
            if (tiles[i] != nullptr) {
                munmap(tiles[i], tile_bytes[i]);
            }
        }
    }

    float* alloc_tile(uint32_t slot, uint32_t pages = 1) {
        const size_t bytes = pages * kPageBytes;
        void* ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
        if (ptr == MAP_FAILED) {
            std::perror("mmap");
            std::abort();
        }
        std::memset(ptr, 0, bytes);
        tiles[slot] = static_cast<float*>(ptr);
        tile_bytes[slot] = bytes;
        return tiles[slot];
    }

    void bind_cb(uint32_t cb_id, float* tile, uint32_t pages = 1) {
        cbs[cb_id].base = reinterpret_cast<uint8_t*>(tile);
        cbs[cb_id].page_size = kPageBytes;
        cbs[cb_id].num_pages = pages;
        cbs[cb_id].page_mask = pages - 1;
    }
};

bool require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

void fill_tile(float* tile, float value) {
    for (uint32_t i = 0; i < 32 * 32; ++i) {
        tile[i] = value;
    }
}

bool experimental_circular_buffer_alias_compiles() {
    experimental::CircularBuffer cb(7);
    return require(cb.get_cb_id() == 7, "experimental::CircularBuffer alias used the wrong CB id");
}

bool stochastic_round_symbol_is_available() {
    Harness h;
    h.ctx.dst[2][0] = 1.6f;
    ckernel::stochastic_round_tile(2);
    return require(h.ctx.dst[2][0] == std::nearbyint(1.6f), "stochastic_round_tile did not round DST");
}

bool llk_binary_uses_unpack_tile_indices() {
    Harness h;
    float* a = h.alloc_tile(0, 2);
    float* b = h.alloc_tile(1);
    h.bind_cb(0, a, 2);
    h.bind_cb(1, b);

    fill_tile(a, 100.0f);
    fill_tile(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(a) + kPageBytes), 3.0f);
    fill_tile(b, 4.0f);

    llk_unpack_AB<ckernel::BroadcastType::NONE>(0, 1, 1, 0);
    llk_math_eltwise_binary<
        ckernel::EltwiseBinaryType::ELWADD,
        ckernel::BroadcastType::NONE,
        DST_ACCUM_MODE,
        ckernel::MathFidelity::LoFi>(0, 1, 0, true);

    return require(h.ctx.dst[0][0] == 7.0f, "llk binary path ignored llk_unpack_AB tile indices");
}

}  // namespace

int main() {
    const bool ok = experimental_circular_buffer_alias_compiles() &&
                    stochastic_round_symbol_is_available() &&
                    llk_binary_uses_unpack_tile_indices();
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
