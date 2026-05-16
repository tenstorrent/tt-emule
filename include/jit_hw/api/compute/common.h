// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT compute API — common definitions.
// Self-contained: provides all compute macros, enums, DST register ops,
// pack/copy tile functions, CB forwarding, and no-op reconfig stubs.
//
// On the real device these map to LLK/ckernel calls on TRISC cores.
// In emulation, DST is a thread-local array (4 bytes per element) and compute
// ops handle both bfloat16 and INT32 tile formats via page_size dispatch.

#include "jit_hw/jit_kernel_stubs.hpp"
#include "jit_hw/api/cb_api.h"
#include "jit_hw/api/compute/common_globals.h"
#include "jit_hw/api/compute/nfaces.h"
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ---- TRISC execution macros ----
// On device, PACK/MATH/UNPACK select which TRISC core runs the code.
// In emulation, all three share one thread — execute everything.
#define PACK(x) x
#define MATH(x) x
#define UNPACK(x) x

#define ALWI FORCE_INLINE

// ---- Enums ----

namespace ckernel {

enum class EltwiseBinaryType { ELWADD, ELWSUB, ELWMUL };
enum class BroadcastType { NONE, COL, ROW, SCALAR };
enum class EltwiseBinaryReuseDestType { NONE, DEST_TO_SRCA, DEST_TO_SRCB };

} // namespace ckernel

// Note: MathFidelity may also be defined in llk_defs.h — guard against redefinition.
// Values must match tt-metal's enum: LoFi=0, HiFi2=2, HiFi3=3, HiFi4=4.
#ifndef __EMULE_MATH_FIDELITY_DEFINED
#define __EMULE_MATH_FIDELITY_DEFINED
enum class MathFidelity : uint8_t { LoFi = 0, HiFi2 = 2, HiFi3 = 3, HiFi4 = 4 };
#endif

enum class ReluType { NO_RELU, ZERO_RELU, MIN_THRESHOLD_RELU, MAX_THRESHOLD_RELU };

// DST_ACCUM_MODE: On real device, this is a compile-time integer define.
// In emulation, provide it as a constexpr if not already defined as a macro.
#ifndef DST_ACCUM_MODE
#define DST_ACCUM_MODE 0
#endif

// ---- bfloat16 conversion helpers ----
#include "jit_hw/api/bfloat16.h"

// ---- Thread-local DST register file ----
// Physical size: 16 tile slots × 1024 elements × 4 bytes = 64 KB (same on WH/BH/Quasar).
// Active slots depend on DST_ACCUM_MODE: bf16 mode → 16 slots, fp32 mode → 8 slots.
// Stores float32 for bfloat16 ops or raw int32 bit patterns for INT32 ops.

static constexpr uint32_t __EMULE_DST_TILES = 16;     // bf16 half-dest
static constexpr uint32_t __EMULE_DST_TILES_FP32 = 8; // f32 half-dest (2x element size)
static constexpr uint32_t __EMULE_TILE_ELEMS = 1024;
static constexpr uint32_t __EMULE_DST_BYTES = __EMULE_TILE_ELEMS * sizeof(float);
static thread_local float __emule_dst[__EMULE_DST_TILES][__EMULE_TILE_ELEMS];
static thread_local bool __emule_l1_acc_enabled = false;

// Assert FULL DEST is not used
#ifdef FULL_DEST
#error "FULL DEST mode is not supported in emulation"
#endif

// Active DST tile count depends on accumulation mode, not architecture.
// bf16 mode (DST_ACCUM_MODE==0): 16 half-dest slots.
// fp32 mode (DST_ACCUM_MODE!=0): 8 half-dest slots (elements are 2x size).
inline constexpr uint32_t __emule_dst_active_tiles() {
    return (DST_ACCUM_MODE != 0) ? __EMULE_DST_TILES_FP32 : __EMULE_DST_TILES;
}

// DST bounds guard — call before any DST[slot] access
inline void __emule_dst_check(uint32_t slot, const char* caller) {
    uint32_t limit = __emule_dst_active_tiles();
    if (slot >= limit) {
        fprintf(stderr, "[EMULE] DST out-of-bounds: %s accessed slot %u (max %u, DST_ACCUM_MODE=%d)\n",
                caller, slot, limit, DST_ACCUM_MODE);
        std::abort();
    }
}

// Helper: access DST slot as int32_t* (type-pun via memcpy in SFPU ops).
inline int32_t __emule_dst_load_i32(uint32_t slot, uint32_t idx) {
    __emule_dst_check(slot, "__emule_dst_load_i32");
    int32_t v;
    std::memcpy(&v, &__emule_dst[slot][idx], sizeof(int32_t));
    return v;
}
inline void __emule_dst_store_i32(uint32_t slot, uint32_t idx, int32_t v) {
    __emule_dst_check(slot, "__emule_dst_store_i32");
    std::memcpy(&__emule_dst[slot][idx], &v, sizeof(int32_t));
}

// ---- DST state machine ----
// tile_regs_acquire / tile_regs_commit / tile_regs_wait / tile_regs_release
// are now owned by api/compute/reg_api.h (per PR #21 review feedback).
// Including it transitively here preserves back-compat: existing callers
// that `#include "api/compute/common.h"` still see the symbols.
#include "jit_hw/api/compute/reg_api.h"

// ---- Core logical coordinates (for D2M compute kernels) ----
// Guarded to avoid conflict with dataflow_api.h if both are included.
// Return uint32_t to match the dataflow API signature (uint8_t is sufficient
// for real coordinates but uint32_t avoids narrowing surprises).
#ifndef __EMULE_GET_LOGICAL_COORDS_DEFINED
#define __EMULE_GET_LOGICAL_COORDS_DEFINED
inline uint32_t get_absolute_logical_x() { return __emule_logical_x; }
inline uint32_t get_absolute_logical_y() { return __emule_logical_y; }
#endif

// ---- CB helpers (read/write via shared CBSyncState) ----

namespace __emule_compute {

inline uint8_t* cb_read_ptr_at(uint32_t cb_id, uint32_t tile_offset) {
    return const_cast<uint8_t*>(
        tt_emule::cb_sync_read_ptr_at(__emule_cbs[cb_id], tile_offset));
}

inline uint8_t* cb_write_ptr(uint32_t cb_id) {
    return tt_emule::cb_sync_write_ptr(__emule_cbs[cb_id]);
}

inline uint8_t* cb_write_ptr_at(uint32_t cb_id, uint32_t tile_offset) {
    return tt_emule::cb_sync_write_ptr_at(__emule_cbs[cb_id], tile_offset);
}

inline uint32_t cb_page_size(uint32_t cb_id) {
    return __emule_cbs[cb_id].page_size;
}

// Number of bfloat16 elements in a page (only valid for bf16 format).
inline uint32_t cb_tile_elems(uint32_t cb_id) {
    return __emule_cbs[cb_id].page_size / sizeof(uint16_t);
}

// Is this CB using a 32-bit data format (INT32, Float32)?
// Heuristic: bf16 tiles = 2048 bytes (1024 × 2), 32-bit tiles > 2048.
inline bool cb_is_32bit_format(uint32_t cb_id) {
    return __emule_cbs[cb_id].page_size > 2048;
}

// pack_dst_to_buf: PACK row-major DST → nfaces CB with L1 accumulation support.
// When __emule_l1_acc_enabled, adds DST to existing CB contents instead of overwriting.
inline void pack_dst_to_buf(uint8_t* buf, uint32_t dst_slot, uint32_t ocb) {
    if (cb_is_32bit_format(ocb)) {
        uint32_t n = cb_page_size(ocb) / sizeof(uint32_t);
        if (n > __EMULE_TILE_ELEMS) n = __EMULE_TILE_ELEMS;
        if (__emule_l1_acc_enabled) {
            float* out = reinterpret_cast<float*>(buf);
            for (uint32_t i = 0; i < n; i++) {
                uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
                out[ni] += __emule_dst[dst_slot][i];
            }
        } else {
            // Bit-exact copy via memcpy — preserves INT32 bit patterns that are
            // denormalized floats (would be flushed to zero by float assignment
            // when x86 DAZ/FTZ is set).
            uint32_t* out = reinterpret_cast<uint32_t*>(buf);
            for (uint32_t i = 0; i < n; i++) {
                uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
                std::memcpy(&out[ni], &__emule_dst[dst_slot][i], sizeof(uint32_t));
            }
        }
    } else {
        uint16_t* bf = reinterpret_cast<uint16_t*>(buf);
        uint32_t n = cb_tile_elems(ocb);
        if (__emule_l1_acc_enabled) {
            for (uint32_t i = 0; i < n; i++) {
                uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
                bf[ni] = __emule_bf16::from_f32(
                    __emule_bf16::to_f32(bf[ni]) + __emule_dst[dst_slot][i]);
            }
        } else {
            for (uint32_t i = 0; i < n; i++) {
                uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
                bf[ni] = __emule_bf16::from_f32(__emule_dst[dst_slot][i]);
            }
        }
    }
}

} // namespace __emule_compute

// ---- Compute operations ----

namespace ckernel {

// binary_op_init_common — no-op (hardware pipeline init)
ALWI void binary_op_init_common(uint32_t, uint32_t, uint32_t) {}
ALWI void binary_op_init_common(uint32_t, uint32_t, uint32_t, uint32_t) {}

// binary_tiles_init — no-op (per-op hardware init)
template<bool FullInit = true, EltwiseBinaryType BinaryType = EltwiseBinaryType::ELWADD>
ALWI void binary_tiles_init(uint32_t, uint32_t, bool = false) {}

// add_tiles: UNPACK + MATH — DST[idst] = CB[icb0][itile0] + CB[icb1][itile1]
ALWI void add_tiles(uint32_t icb0, uint32_t icb1,
                    uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_dst_check(idst, "add_tiles");
    __emule_dst_mark_dirty(idst);
    if (__emule_compute::cb_is_32bit_format(icb0)) {
        const float* buf0 = reinterpret_cast<const float*>(__emule_compute::cb_read_ptr_at(icb0, itile0));
        const float* buf1 = reinterpret_cast<const float*>(__emule_compute::cb_read_ptr_at(icb1, itile1));
        uint32_t n = __emule_compute::cb_page_size(icb0) / sizeof(float);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            __emule_dst[idst][i] = buf0[ni] + buf1[ni];
        }
    } else {
        uint16_t* buf0 = reinterpret_cast<uint16_t*>(__emule_compute::cb_read_ptr_at(icb0, itile0));
        uint16_t* buf1 = reinterpret_cast<uint16_t*>(__emule_compute::cb_read_ptr_at(icb1, itile1));
        uint32_t n = __emule_compute::cb_tile_elems(icb0);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            __emule_dst[idst][i] = __emule_bf16::to_f32(buf0[ni]) + __emule_bf16::to_f32(buf1[ni]);
        }
    }
}

// sub_tiles: UNPACK + MATH — DST[idst] = CB[icb0][itile0] - CB[icb1][itile1]
ALWI void sub_tiles(uint32_t icb0, uint32_t icb1,
                    uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_dst_check(idst, "sub_tiles");
    __emule_dst_mark_dirty(idst);
    if (__emule_compute::cb_is_32bit_format(icb0)) {
        const float* buf0 = reinterpret_cast<const float*>(__emule_compute::cb_read_ptr_at(icb0, itile0));
        const float* buf1 = reinterpret_cast<const float*>(__emule_compute::cb_read_ptr_at(icb1, itile1));
        uint32_t n = __emule_compute::cb_page_size(icb0) / sizeof(float);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            __emule_dst[idst][i] = buf0[ni] - buf1[ni];
        }
    } else {
        uint16_t* buf0 = reinterpret_cast<uint16_t*>(__emule_compute::cb_read_ptr_at(icb0, itile0));
        uint16_t* buf1 = reinterpret_cast<uint16_t*>(__emule_compute::cb_read_ptr_at(icb1, itile1));
        uint32_t n = __emule_compute::cb_tile_elems(icb0);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            __emule_dst[idst][i] = __emule_bf16::to_f32(buf0[ni]) - __emule_bf16::to_f32(buf1[ni]);
        }
    }
}

// mul_tiles: UNPACK + MATH — DST[idst] = CB[icb0][itile0] * CB[icb1][itile1]
ALWI void mul_tiles(uint32_t icb0, uint32_t icb1,
                    uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_dst_check(idst, "mul_tiles");
    __emule_dst_mark_dirty(idst);
    if (__emule_compute::cb_is_32bit_format(icb0)) {
        const float* buf0 = reinterpret_cast<const float*>(__emule_compute::cb_read_ptr_at(icb0, itile0));
        const float* buf1 = reinterpret_cast<const float*>(__emule_compute::cb_read_ptr_at(icb1, itile1));
        uint32_t n = __emule_compute::cb_page_size(icb0) / sizeof(float);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            __emule_dst[idst][i] = buf0[ni] * buf1[ni];
        }
    } else {
        uint16_t* buf0 = reinterpret_cast<uint16_t*>(__emule_compute::cb_read_ptr_at(icb0, itile0));
        uint16_t* buf1 = reinterpret_cast<uint16_t*>(__emule_compute::cb_read_ptr_at(icb1, itile1));
        uint32_t n = __emule_compute::cb_tile_elems(icb0);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            __emule_dst[idst][i] = __emule_bf16::to_f32(buf0[ni]) * __emule_bf16::to_f32(buf1[ni]);
        }
    }
}

// pack_tile: write DST[idst] → CB[ocb] write slot.
// Format-aware: bf16 (page_size ≤ 2048) or raw 32-bit (page_size > 2048).
// When L1 acc enabled, accumulates into existing CB data instead of overwriting.
ALWI void pack_tile(uint32_t idst, uint32_t ocb) {
    __emule_dst_check(idst, "pack_tile");
    // PACK engine auto-advance: write to current offset, then advance.
    __emule_compute::pack_dst_to_buf(
        __emule_compute::cb_write_ptr_at(ocb, __emule_pack_offset[ocb]++), idst, ocb);
}

// pack_tile (templated): used by D2M-generated code.
// Template param <true> means "use output_offset as the write slot index".
template <bool UseOutputOffset>
ALWI void pack_tile(uint32_t idst, uint32_t ocb, uint32_t output_offset = 0) {
    __emule_dst_check(idst, "pack_tile<templated>");
    if constexpr (UseOutputOffset) {
        __emule_compute::pack_dst_to_buf(__emule_compute::cb_write_ptr_at(ocb, output_offset), idst, ocb);
    } else {
        pack_tile(idst, ocb);
    }
}

// pack_tile_block: write DST[ifrom_dst .. ifrom_dst+ntiles-1] → CB[ocb] consecutive write slots.
ALWI void pack_tile_block(uint32_t ifrom_dst, uint32_t ocb, uint32_t ntiles) {
    if (ntiles > 0)
        __emule_dst_check(ifrom_dst + ntiles - 1, "pack_tile_block");
    for (uint32_t i = 0; i < ntiles; i++) {
        __emule_compute::pack_dst_to_buf(__emule_compute::cb_write_ptr_at(ocb, i), ifrom_dst + i, ocb);
    }
}

// copy_tile: UNPACK CB[icb][itile] → DST[idst] with nfaces→row-major conversion.
// Format-aware: bf16 (page_size ≤ 2048) or raw 32-bit (page_size > 2048).
ALWI void copy_tile(uint32_t icb, uint32_t itile, uint32_t idst) {
    __emule_dst_check(idst, "copy_tile");
    __emule_dst_mark_dirty(idst);
    uint8_t* buf = __emule_compute::cb_read_ptr_at(icb, itile);
    if (__emule_compute::cb_is_32bit_format(icb)) {
        // 32-bit format: UNPACK nfaces→row-major.
        // Use memcpy per element to preserve INT32 bit patterns (small positive
        // ints are denormalized floats that x86 DAZ/FTZ would flush to zero).
        const uint32_t* ubuf = reinterpret_cast<const uint32_t*>(buf);
        uint32_t n = __emule_compute::cb_page_size(icb) / sizeof(uint32_t);
        if (n > __EMULE_TILE_ELEMS) n = __EMULE_TILE_ELEMS;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            std::memcpy(&__emule_dst[idst][i], &ubuf[ni], sizeof(uint32_t));
        }
    } else {
        // bfloat16: UNPACK nfaces→row-major + bf16→f32 conversion
        uint16_t* bf = reinterpret_cast<uint16_t*>(buf);
        uint32_t n = __emule_compute::cb_tile_elems(icb);
        for (uint32_t i = 0; i < n; i++)
            __emule_dst[idst][i] = __emule_bf16::to_f32(bf[__emule_nfaces::rowmajor_to_nfaces[i]]);
    }
}

// copy_block_matmul_partials: reload a block of tiles from CB into DST.
ALWI void copy_block_matmul_partials(
    uint32_t in_cb_id, uint32_t start_in_tile_index,
    uint32_t start_dst_tile_index, uint32_t ntiles) {
    if (ntiles > 0)
        __emule_dst_check(start_dst_tile_index + ntiles - 1, "copy_block_matmul_partials");
    for (uint32_t i = 0; i < ntiles; i++) {
        copy_tile(in_cb_id, start_in_tile_index + i, start_dst_tile_index + i);
    }
}

// copy_tile_to_dst_init_short — no-op (hardware reconfiguration)
ALWI void copy_tile_to_dst_init_short(uint32_t) {}
ALWI void copy_tile_to_dst_init_short(uint32_t, uint32_t) {}

// copy_tile_init — alias used by D2M-generated code
ALWI void copy_tile_init(uint32_t = 0) {}

// copy_tile_to_dst_init_short_with_dt — no-op (hardware SrcA reconfig)
ALWI void copy_tile_to_dst_init_short_with_dt(uint32_t, uint32_t, uint32_t = 0) {}

// ---- Reconfig operations (no-ops) ----
ALWI void reconfig_data_format(uint32_t) {}
ALWI void reconfig_data_format(uint32_t, uint32_t) {}
template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_srca(uint32_t) {}
template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_srca(uint32_t, uint32_t) {}
template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_srcb(uint32_t) {}
template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_srcb(uint32_t, uint32_t) {}
ALWI void pack_reconfig_data_format(uint32_t) {}
ALWI void pack_reconfig_data_format(uint32_t, uint32_t) {}

// ---- Pack configuration (no-ops) ----
ALWI void llk_pack_relu_config(ReluType) {}
ALWI void pack_set_relu_threshold(float) {}

// binary_dest_reuse stubs
template<EltwiseBinaryReuseDestType ReuseType = EltwiseBinaryReuseDestType::NONE>
ALWI void binary_dest_reuse_tiles_init(uint32_t = 0, uint32_t = 0, bool = false) {}

template<EltwiseBinaryReuseDestType ReuseType, EltwiseBinaryType BinaryType>
ALWI void binary_dest_reuse_tiles(uint32_t icb0, uint32_t icb1,
                                  uint32_t itile0, uint32_t itile1, uint32_t idst) {
    // Fallback to regular binary op
    if constexpr (BinaryType == EltwiseBinaryType::ELWADD)
        add_tiles(icb0, icb1, itile0, itile1, idst);
    else if constexpr (BinaryType == EltwiseBinaryType::ELWSUB)
        sub_tiles(icb0, icb1, itile0, itile1, idst);
    else
        mul_tiles(icb0, icb1, itile0, itile1, idst);
}

// state_configure — no-op
ALWI void state_configure(uint32_t = 0) {}

// llk_pack_reconfig_l1_acc — L1 accumulation toggle for pack operations.
// enable=1: pack_tile adds DST to existing CB data; enable=0: pack_tile overwrites.
ALWI void llk_pack_reconfig_l1_acc(uint32_t enable) {
    __emule_l1_acc_enabled = (enable != 0);
}

} // namespace ckernel

// ---- Deprecated DST lock wrappers (used by bmm.cpp) ----
// These are global-scope functions matching the real device API (reg_api.h).
ALWI void acquire_dst() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void release_dst() {
    tile_regs_commit();
    tile_regs_release();
}

// CB operations provided by jit_hw/api/cb_api.h (included above).

// Bring ckernel functions into the global namespace (matches real device behavior,
// where "using namespace ckernel" is pulled in via ckernel.h / risc_common.h).
using namespace ckernel;
