// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
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

#include "jit_hw/api/compute/matmul.h"
#include "jit_hw/api/compute/reduce.h"
#include "jit_hw/api/compute/reg_api.h"

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
    std::array<float*, 4> tiles{};

    Harness() {
        ctx.cbs = cbs.data();
        __emule_self = &ctx;
    }

    ~Harness() {
        for (float* tile : tiles) {
            if (tile != nullptr) {
                munmap(tile, kPageBytes);
            }
        }
    }

    float* alloc_tile(uint32_t slot) {
        void* ptr = mmap(nullptr, kPageBytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
        if (ptr == MAP_FAILED) {
            std::perror("mmap");
            std::abort();
        }
        std::memset(ptr, 0, kPageBytes);
        tiles[slot] = static_cast<float*>(ptr);
        return tiles[slot];
    }

    void bind_cb(uint32_t cb_id, float* tile) {
        cbs[cb_id].base = reinterpret_cast<uint8_t*>(tile);
        cbs[cb_id].page_size = kPageBytes;
        cbs[cb_id].num_pages = 1;
        cbs[cb_id].page_mask = 0;
    }

    void reset_dst() {
        std::memset(ctx.dst, 0, sizeof(ctx.dst));
        for (bool& fresh : ctx.dst_fresh) {
            fresh = true;
        }
    }
};

void set_rm(float* tile, uint32_t r, uint32_t c, float value) {
    tile[__emule_nfaces::rowmajor_to_nfaces[r * 32 + c]] = value;
}

bool require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

bool reduce_col_skips_zero_scaler_inf_lane() {
    Harness h;
    float* src = h.alloc_tile(0);
    float* scaler = h.alloc_tile(1);
    h.bind_cb(0, src);
    h.bind_cb(1, scaler);

    set_rm(src, 0, 0, 3.0f);
    set_rm(src, 1, 0, std::numeric_limits<float>::infinity());
    set_rm(scaler, 0, 0, 1.0f);
    set_rm(scaler, 1, 0, 0.0f);

    h.reset_dst();
    tile_regs_acquire();
    ckernel::reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(0, 1, 0, 0, 0);

    const float out = h.ctx.dst[0][0];
    return require(!std::isnan(out), "REDUCE_COL produced NaN from masked inf lane") &&
           require(out == 3.0f, "REDUCE_COL did not preserve the active lane value");
}

bool reduce_row_skips_zero_scaler_inf_lane() {
    Harness h;
    float* src = h.alloc_tile(0);
    float* scaler = h.alloc_tile(1);
    h.bind_cb(0, src);
    h.bind_cb(1, scaler);

    set_rm(src, 0, 0, 5.0f);
    set_rm(src, 0, 1, std::numeric_limits<float>::infinity());
    set_rm(scaler, 0, 0, 1.0f);
    set_rm(scaler, 0, 1, 0.0f);

    h.reset_dst();
    tile_regs_acquire();
    ckernel::reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(0, 1, 0, 0, 0);

    const float out = h.ctx.dst[0][0];
    return require(!std::isnan(out), "REDUCE_ROW produced NaN from masked inf lane") &&
           require(out == 5.0f, "REDUCE_ROW did not preserve the active lane value");
}

bool matmul_skips_zero_multiplier_inf_lane() {
    Harness h;
    float* a = h.alloc_tile(0);
    float* b = h.alloc_tile(1);
    h.bind_cb(0, a);
    h.bind_cb(1, b);

    set_rm(a, 0, 0, 2.0f);
    set_rm(a, 0, 1, std::numeric_limits<float>::infinity());
    set_rm(b, 0, 0, 7.0f);
    set_rm(b, 1, 0, 0.0f);

    h.reset_dst();
    tile_regs_acquire();
    ckernel::matmul_tiles(0, 1, 0, 0, 0);

    const float out = h.ctx.dst[0][0];
    return require(!std::isnan(out), "matmul_tiles produced NaN from zero B lane") &&
           require(out == 14.0f, "matmul_tiles did not preserve active contribution");
}

}  // namespace

int main() {
    const bool ok = reduce_col_skips_zero_scaler_inf_lane() &&
                    reduce_row_skips_zero_scaler_inf_lane() &&
                    matmul_skips_zero_multiplier_inf_lane();
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
