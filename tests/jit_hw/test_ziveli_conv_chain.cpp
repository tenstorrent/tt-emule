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
#include <vector>

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

#include "jit_hw/api/bfloat16.h"
#include "jit_hw/api/compute/bcast.h"
#include "jit_hw/api/compute/matmul.h"
#include "jit_hw/api/compute/tilize.h"
#include "jit_hw/api/compute/untilize.h"
#include "jit_hw/llk_pack.h"

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

constexpr uint32_t kTileDim = 32;
constexpr uint32_t kTileElems = kTileDim * kTileDim;
constexpr uint32_t kBf16TileBytes = kTileElems * sizeof(uint16_t);

struct Harness {
    ComputeThreadCtx ctx;
    std::array<tt_emule::CBSyncState, 32> cbs{};
    std::vector<std::pair<void*, size_t>> allocations;

    Harness() {
        ctx.cbs = cbs.data();
        __emule_self = &ctx;
    }

    ~Harness() {
        for (auto [ptr, bytes] : allocations) {
            munmap(ptr, bytes);
        }
    }

    uint8_t* alloc_bytes(size_t bytes) {
        void* ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
        if (ptr == MAP_FAILED) {
            std::perror("mmap");
            std::abort();
        }
        std::memset(ptr, 0, bytes);
        allocations.push_back({ptr, bytes});
        return static_cast<uint8_t*>(ptr);
    }

    uint8_t* bind_cb(uint32_t cb_id, uint32_t page_size, uint32_t pages) {
        uint8_t* ptr = alloc_bytes(static_cast<size_t>(page_size) * pages);
        cbs[cb_id].base = ptr;
        cbs[cb_id].page_size = page_size;
        cbs[cb_id].num_pages = pages;
        cbs[cb_id].page_mask = pages - 1;
        return ptr;
    }
};

float bf16_round(float value) {
    return __emule_bf16::to_f32(__emule_bf16::from_f32(value));
}

void write_bf16(uint8_t* base, uint32_t elem, float value) {
    reinterpret_cast<uint16_t*>(base)[elem] = __emule_bf16::from_f32(value);
}

float read_bf16(const uint8_t* base, uint32_t elem) {
    return __emule_bf16::to_f32(reinterpret_cast<const uint16_t*>(base)[elem]);
}

void write_tile_nf(uint8_t* base, uint32_t tile, uint32_t row, uint32_t col, float value) {
    const uint32_t nf = __emule_nfaces::tile_rc_to_nfaces(row, col, kTileDim, kTileDim);
    write_bf16(base + tile * kBf16TileBytes, nf, value);
}

float read_dense_bf16(const uint8_t* base, uint32_t row, uint32_t col, uint32_t cols) {
    return read_bf16(base, row * cols + col);
}

bool require_close(float got, float expected, const char* label, uint32_t row, uint32_t col) {
    const float diff = std::fabs(got - expected);
    const float tol = 0.020f + 0.015f * std::fabs(expected);
    if (!(diff <= tol)) {
        std::cerr << label << " mismatch at (" << row << ", " << col << "): got "
                  << got << " expected " << expected << " diff " << diff << " tol " << tol << '\n';
        return false;
    }
    return true;
}

bool ziveli_conv_tilize_matmul_add_untilize_matches_cpu() {
    constexpr uint32_t cb_im2col_rm = 3;
    constexpr uint32_t cb_im2col = 2;
    constexpr uint32_t cb_bmm = 5;
    constexpr uint32_t cb_add = 6;
    constexpr uint32_t cb_weight = 7;
    constexpr uint32_t cb_bias = 8;
    constexpr uint32_t cb_untilized = 10;
    constexpr uint32_t kt = 9;
    constexpr uint32_t nt = 2;
    constexpr uint32_t out_cols = nt * kTileDim;

    Harness h;
    uint8_t* rm = h.bind_cb(cb_im2col_rm, kt * kBf16TileBytes, 1);
    h.bind_cb(cb_im2col, kBf16TileBytes, kt);
    h.bind_cb(cb_bmm, kBf16TileBytes, nt);
    h.bind_cb(cb_add, kBf16TileBytes, nt);
    uint8_t* weight = h.bind_cb(cb_weight, kBf16TileBytes, kt * nt);
    uint8_t* bias = h.bind_cb(cb_bias, kBf16TileBytes, nt);
    uint8_t* untilized = h.bind_cb(cb_untilized, out_cols * kTileDim * sizeof(uint16_t), 1);

    std::array<float, kTileDim * kt * kTileDim> a{};
    std::array<float, kt * kTileDim * out_cols> b{};
    std::array<float, out_cols> bias_vals{};

    for (uint32_t r = 0; r < kTileDim; ++r) {
        for (uint32_t k = 0; k < kt * kTileDim; ++k) {
            const float value = bf16_round(((static_cast<int>(r * 17 + k * 13) % 29) - 14) / 19.0f);
            a[r * kt * kTileDim + k] = value;
            write_bf16(rm, r * kt * kTileDim + k, value);
        }
    }
    for (uint32_t k = 0; k < kt * kTileDim; ++k) {
        for (uint32_t n = 0; n < out_cols; ++n) {
            const float value = bf16_round(((static_cast<int>(k * 7 + n * 5) % 23) - 11) / 31.0f);
            b[k * out_cols + n] = value;
            const uint32_t tile = (k / kTileDim) * nt + (n / kTileDim);
            write_tile_nf(weight, tile, k % kTileDim, n % kTileDim, value);
        }
    }
    for (uint32_t n = 0; n < out_cols; ++n) {
        const float value = bf16_round(((static_cast<int>(n * 3) % 17) - 8) / 13.0f);
        bias_vals[n] = value;
        write_tile_nf(bias, n / kTileDim, 0, n % kTileDim, value);
    }

    tilize_block(cb_im2col_rm, kt, cb_im2col);
    tile_regs_acquire();
    for (uint32_t k = 0; k < kt; ++k) {
        ckernel::matmul_block(cb_im2col, cb_weight, k, k * nt, 0, 0, nt, 1, kt);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile_block(0, cb_bmm, nt);
    tile_regs_release();

    tile_regs_acquire();
    ckernel::add_tiles_bcast_rows(cb_bmm, cb_bias, 0, 0, 0);
    ckernel::add_tiles_bcast_rows(cb_bmm, cb_bias, 1, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile_block(0, cb_add, nt);
    tile_regs_release();

    untilize_block(cb_add, nt, cb_untilized);

    for (uint32_t r = 0; r < kTileDim; ++r) {
        for (uint32_t n = 0; n < out_cols; ++n) {
            float expected = 0.0f;
            for (uint32_t k = 0; k < kt * kTileDim; ++k) {
                expected += a[r * kt * kTileDim + k] * b[k * out_cols + n];
            }
            expected = bf16_round(expected + bias_vals[n]);
            const float got = read_dense_bf16(untilized, r, n, out_cols);
            if (!require_close(got, expected, "conv chain", r, n)) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    return ziveli_conv_tilize_matmul_add_untilize_matches_cpu() ? EXIT_SUCCESS : EXIT_FAILURE;
}
