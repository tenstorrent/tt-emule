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
#include "jit_hw/api/compute/bfp4.h"
#include "jit_hw/api/compute/bfp8.h"
#include "jit_hw/internal/cb_interface.h"
#include "jit_hw/ckernel.h"
// llk_types.h provides ckernel::DataCopyType, ckernel::DstSync,
// ckernel::PoolType, ckernel::ReduceDim, ckernel::MathFidelity (redefinition-
// guarded), and global UnpackToDestEn. Safe to include here — types-only,
// no globals or templates that would conflict with SFPU INT32 paths.
#include "jit_hw/llk_types.h"
// llk_state.h: tilize/untilize per-thread trackers, inline format arrays
// (unpack_src_format/unpack_dst_format), and operand-id helpers
// (get_operand_id, get_operand_face_r_dim, etc.). Loaded here so upstream kernels
// compute kernels see these symbols without including the heavier
// compute_kernel_hw_startup.h entrypoint.
#include "jit_hw/internal/llk_state.h"
// Note: llk_pack.h / llk_unpack_a.h / llk_math_eltwise_unary_datacopy.h /
// llk_sync_stubs.h are pulled in at the END of this header (after
// __emule_compute::, __emule_dst[][], __emule_bf16:: are defined — they
// transitively reference these).
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

// ---- VectorMode ----
// Silicon: selects which SFPU lanes are active for an SFPU op. Emule has no
// SFPU vector hardware — provide the enum so kernels parse; ops that take
// it as an arg ignore the value. Values match the existing definition in
// api/compute/eltwise_unary/exp.h (RC=0 is the default lane mask), guarded
// against double-definition. Canonical definition lives in
// jit_hw/api/compute/vector_mode.h (ckernel::VectorMode); we pull it in here
// and `using namespace ckernel;` at the bottom of common.h re-exports it.
#include "jit_hw/api/compute/vector_mode.h"

// ---- Enums ----

namespace ckernel {

enum class EltwiseBinaryType { ELWADD, ELWSUB, ELWMUL };
enum class BroadcastType { NONE, COL, ROW, SCALAR };
enum class EltwiseBinaryReuseDestType { NONE, DEST_TO_SRCA, DEST_TO_SRCB };

// Tile/face dimension constants. Upstream exposes these in ckernel via
// `tt_llk_wormhole_b0/common/inc/ckernel_defs.h:89`; emule already has
// the values in `tt::constants` (include/jit_hw/tt-metalium/constants.hpp)
// but compute kernels reference the bare names (e.g.
// `transpose_wh_rm.cpp` does `last_output_row_num_datums < TILE_WIDTH`).
// `using namespace ckernel;` at the bottom of this file pulls them to
// global scope for those bare references.
constexpr uint32_t TILE_WIDTH  = 32;
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_HW     = TILE_WIDTH * TILE_HEIGHT;
constexpr uint32_t FACE_WIDTH  = 16;
constexpr uint32_t FACE_HEIGHT = 16;
constexpr uint32_t FACE_HW     = FACE_WIDTH * FACE_HEIGHT;
// TILE_C_DIM lives in include/jit_hw/llk_types.h as a `#define` (because the
// upstream llk_api headers expect it as a macro). Don't redefine it here as a
// constexpr — the macro would expand inside the declaration and break the
// parse on any kernel that includes both this header and llk_types.h.

} // namespace ckernel

// D2M emits `binary_dest_reuse_tiles<ELWADD, …>(…)` with `ELWADD` as a bare
// (unqualified) name — apparently inconsistent with its sibling
// `EltwiseBinaryReuseDestType::DEST_TO_SRCA` which IS qualified. Expose the
// enum-class values at global scope so the unqualified form resolves.
inline constexpr ckernel::EltwiseBinaryType ELWADD = ckernel::EltwiseBinaryType::ELWADD;
inline constexpr ckernel::EltwiseBinaryType ELWSUB = ckernel::EltwiseBinaryType::ELWSUB;
inline constexpr ckernel::EltwiseBinaryType ELWMUL = ckernel::EltwiseBinaryType::ELWMUL;

// Note: MathFidelity may also be defined in llk_defs.h — guard against redefinition.
// Values must match tt-metal's enum: LoFi=0, HiFi2=2, HiFi3=3, HiFi4=4.
#ifndef __EMULE_MATH_FIDELITY_DEFINED
#define __EMULE_MATH_FIDELITY_DEFINED
enum class MathFidelity : uint8_t { LoFi = 0, HiFi2 = 2, HiFi3 = 3, HiFi4 = 4 };
#endif

enum class ReluType { NO_RELU, ZERO_RELU, MIN_THRESHOLD_RELU, MAX_THRESHOLD_RELU };

// p_dim_stride_target: reconfig behaviour for dim and stride. Defined upstream
// at tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_common.h:25. Used
// as a template arg on llk_unpack_reconfig_data_format_* (LLK functions emule
// stubs as no-ops). reduce_helpers_compute.inl in upstream references
// `p_dim_stride_target::IGNORE` directly, so the enum must be in scope when
// that .inl is parsed by any kernel that pulls it in (softmax, moreh_dot,
// etc.). Placed at global scope (not inside `ckernel`) to match upstream's
// usage `p_dim_stride_target::IGNORE`.
enum class p_dim_stride_target {
    IGNORE,         // do not modify dim/stride
    FACE_ROW_MAJOR  // set dim/stride for unpacking face in row-major format
};

// DST_ACCUM_MODE: On real device, this is a compile-time integer define.
// In emulation, provide it as a constexpr if not already defined as a macro.
#ifndef DST_ACCUM_MODE
#define DST_ACCUM_MODE 0
#endif

// ---- bfloat16 conversion helpers ----
#include "jit_hw/api/bfloat16.h"
#include "jit_hw/api/bfp8.h"

// ---- Thread-local DST register file ----
// Physical Dst on both WH-B0 and BH is 64 KB (two banks of 32 KB; one bank
// is 1024 rows × 16 cols × 2 bytes = 32 KB).  Kernel-visible tile-slot
// capacity is identical on both arches:
//
//   SyncFull + bf16: 16 tiles   SyncFull + fp32: 8 tiles
//   SyncHalf + bf16: 8 tiles    SyncHalf + fp32: 4 tiles
//
// Production kernels run in SyncHalf (double-buffered: math writes one bank
// while pack reads the other).  emule allows SyncFull capacity (16 / 8) as
// the kernel-visible ceiling — strictly stricter than SyncHalf, generous
// enough that no in-tree LLK errors on a legitimate slot index.  Stores
// float32 for bfloat16 ops or raw int32 bit patterns for INT32 ops.

static constexpr uint32_t __EMULE_DST_TILES = 16;     // bf16 SyncFull ceiling (arch-independent)
static constexpr uint32_t __EMULE_DST_TILES_FP32 = 8; // f32 SyncFull ceiling (arch-independent)
static constexpr uint32_t __EMULE_TILE_ELEMS = 1024;
static constexpr uint32_t __EMULE_DST_BYTES = __EMULE_TILE_ELEMS * sizeof(float);
static thread_local float __emule_dst[__EMULE_DST_TILES][__EMULE_TILE_ELEMS];
static thread_local bool __emule_l1_acc_enabled = false;

// Pack-fused ReLU state — silicon STACC_RELU is a single packer CFG reg, so
// thread-global like __emule_l1_acc_enabled.  `llk_pack_relu_config(ReluType)`
// writes the mode; `pack_set_relu_threshold(float)` writes the threshold.
// `pack_dst_to_buf` / `__llk_pack_tiled` / `__llk_pack_untilize` apply the
// clamp before format conversion.
static thread_local ReluType __emule_pack_relu_mode = ReluType::NO_RELU;
static thread_local float    __emule_pack_relu_threshold = 0.0f;

inline float __emule_apply_pack_relu(float v) {
    switch (__emule_pack_relu_mode) {
        case ReluType::NO_RELU:            return v;
        case ReluType::ZERO_RELU:          return v < 0.0f ? 0.0f : v;
        case ReluType::MIN_THRESHOLD_RELU: return v < __emule_pack_relu_threshold ? __emule_pack_relu_threshold : v;
        case ReluType::MAX_THRESHOLD_RELU: return v > __emule_pack_relu_threshold ? __emule_pack_relu_threshold : v;
    }
    return v;
}

// Emule model of one SRC register bank. Real silicon's UNPACK path
// routes CB tiles into SRCA/SRCB; for the DEST_TO_SRC{A,B} reuse path
// (binary_dest_reuse_tiles) we need a working buffer that is NOT a DST
// slot — otherwise we steal kernel-addressable DST space and go
// out-of-bounds in fp32 mode. Not addressable by kernels.
static thread_local float __emule_src_scratch[__EMULE_TILE_ELEMS];

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
// are owned by api/compute/reg_api.h. Include it transitively so callers
// that `#include "api/compute/common.h"` still see the symbols.
#include "jit_hw/api/compute/reg_api.h"

// ---- LLK sync primitives ----
// `t6_semaphore_*` / `semaphore::*` / `p_stall::*` are referenced by the
// verbatim-inlined body of `experimental::unpack_stall_on_pack` that D2M
// emits into compute kernels. Compute kernels generally include common.h
// but not compute_kernel_hw_startup.h, so route the sync stubs here.
#include "jit_hw/llk_sync_stubs.h"

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

// Real per-CB data format (tt::DataFormat enum value) from the compile-time
// unpack_src_format[] array (cb_api.h, fed by EMULE_CB_DATA_FORMATS). 255 == Invalid
// (e.g. standalone tt_emule_lib builds with no define), in which case callers fall
// back to the page_size heuristic. Block-float dispatch is enum-driven because
// Bfp8_b (~1088B) and Bfp4_b (~576B) both fall in the same sub-2048 page range.
inline uint8_t cb_data_format(uint32_t cb_id) { return ::unpack_src_format[cb_id]; }

// Is this CB using a 32-bit data format (INT32, Float32)?
// Heuristic: bf16 tiles = 2048 bytes (1024 × 2), 32-bit tiles > 2048.
inline bool cb_is_32bit_format(uint32_t cb_id) {
    return __emule_cbs[cb_id].page_size > 2048;
}

// Is this CB using Bfp8_b? Enum-driven (Bfp8_b vs Bfp4_b are indistinguishable
// by page size alone). Falls back to the page-size heuristic only when the
// format is unset (Invalid) — standalone builds, which don't exercise Bfp4_b.
inline bool cb_is_bfp8_b_format(uint32_t cb_id) {
    const uint8_t fmt = cb_data_format(cb_id);
    if (fmt != static_cast<uint8_t>(DataFormat::Invalid)) {
        return fmt == static_cast<uint8_t>(DataFormat::Bfp8_b);
    }
    const uint32_t ps = __emule_cbs[cb_id].page_size;
    return ps > 0 && ps < 2048;
}

// Is this CB using Bfp4_b? Enum-only — its page size overlaps Bfp8_b's range.
inline bool cb_is_bfp4_b_format(uint32_t cb_id) {
    return cb_data_format(cb_id) == static_cast<uint8_t>(DataFormat::Bfp4_b);
}

inline bool cb_is_uint16_format(uint32_t cb_id) {
    return cb_data_format(cb_id) == static_cast<uint8_t>(DataFormat::UInt16);
}

// Block-float formats emule does not encode/decode (only Bfp8_b / Bfp4_b are
// supported). Without this guard they fall through to the bf16 path and produce
// silent garbage; abort with a clear message instead.
inline void __emule_check_blockfloat_supported(uint32_t cb_id, const char* caller) {
    const uint8_t fmt = cb_data_format(cb_id);
    const bool unsupported = fmt == static_cast<uint8_t>(DataFormat::Bfp8)
                          || fmt == static_cast<uint8_t>(DataFormat::Bfp4)
                          || fmt == static_cast<uint8_t>(DataFormat::Bfp2)
                          || fmt == static_cast<uint8_t>(DataFormat::Bfp2_b);
    if (unsupported) {
        fprintf(stderr, "[EMULE] %s: unsupported block-float format %u on CB %u "
                        "(only Bfp8_b/Bfp4_b are emulated)\n", caller, fmt, cb_id);
        std::abort();
    }
}

// pack_dst_to_buf: PACK row-major DST → nfaces CB with L1 accumulation support.
// When __emule_l1_acc_enabled, adds DST to existing CB contents instead of overwriting.
//
// Tile-shape aware: silicon supports thin output tiles (M×32 with
// M ∈ {1,2,4,8,16}). Page size determines how many DST rows to pack and which
// nfaces layout to use. For rows<32, 2 column-faces of rows×16 instead of 4
// face-packed 16×16.
inline void pack_dst_to_buf(uint8_t* buf, uint32_t dst_slot, uint32_t ocb) {
    __emule_check_blockfloat_supported(ocb, "pack_dst_to_buf");
    if (cb_is_bfp4_b_format(ocb)) {
        // Bfp4_b: 64 face-row exponents + 512 mantissa bytes (two 4-bit elements
        // per byte). Symmetric with __emule_bfp4::to_f32.
        uint8_t* exp_base  = buf;
        uint8_t* mant_base = buf + 64;
        for (uint32_t fr = 0; fr < 64; ++fr) {
            float row16[16];
            for (uint32_t k = 0; k < 16; ++k) {
                const uint32_t ni = fr * 16 + k;
                const uint32_t rm = __emule_nfaces::nfaces_to_rowmajor[ni];
                row16[k] = __emule_apply_pack_relu(__emule_dst[dst_slot][rm]);
            }
            uint8_t packed[8];
            __emule_bfp4::encode_face_row(row16, exp_base[fr], packed);
            std::memcpy(&mant_base[fr * 8], packed, 8);
        }
        return;
    }
    if (cb_is_bfp8_b_format(ocb)) {
        // Bfp8_b: iterate in nfaces order so each face-row of 16 contiguous
        // mantissa bytes shares one exponent byte. DST is row-major fp32.
        // Mapping: nfaces index → row-major index uses the same LUT every
        // other compute primitive does, just inverted (here we *consume*
        // row-major DST and write nfaces output, so we look up the row-major
        // index for each nfaces position).
        uint8_t* exp_base  = buf;          // 64 face-row exponents
        uint8_t* mant_base = buf + 64;     // 1024 mantissa bytes (face-major)
        for (uint32_t fr = 0; fr < 64; ++fr) {
            float row16[16];
            for (uint32_t k = 0; k < 16; ++k) {
                const uint32_t ni = fr * 16 + k;
                const uint32_t rm = __emule_nfaces::nfaces_to_rowmajor[ni];
                row16[k] = __emule_apply_pack_relu(__emule_dst[dst_slot][rm]);
            }
            uint8_t mant_row[16];
            __emule_bfp8::encode_face_row(row16, exp_base[fr], mant_row);
            std::memcpy(&mant_base[fr * 16], mant_row, 16);
        }
        // L1 accumulation for Bfp8_b is rare and would require decoding the
        // existing tile, adding DST, then re-encoding — defer until a test
        // exposes the path (op-bring-up skill convention).
        return;
    }
    if (cb_is_32bit_format(ocb)) {
        const uint32_t page = cb_page_size(ocb);
        const uint32_t rows = __emule_nfaces::tile_rows_from_pagesize(page, sizeof(uint32_t));
        uint32_t n = rows * 32u;
        if (n > __EMULE_TILE_ELEMS) n = __EMULE_TILE_ELEMS;
        if (__emule_l1_acc_enabled) {
            float* out = reinterpret_cast<float*>(buf);
            for (uint32_t i = 0; i < n; i++) {
                uint32_t ni = __emule_nfaces::tile_rm_to_nfaces(i, rows);
                out[ni] += __emule_apply_pack_relu(__emule_dst[dst_slot][i]);
            }
        } else if (__emule_pack_relu_mode == ReluType::NO_RELU) {
            // Fast path: bit-exact copy via memcpy preserves INT32 bit patterns that
            // are denormalized floats (would be flushed to zero by float assignment
            // when x86 DAZ/FTZ is set).
            uint32_t* out = reinterpret_cast<uint32_t*>(buf);
            for (uint32_t i = 0; i < n; i++) {
                uint32_t ni = __emule_nfaces::tile_rm_to_nfaces(i, rows);
                std::memcpy(&out[ni], &__emule_dst[dst_slot][i], sizeof(uint32_t));
            }
        } else {
            // ReLU clamp path: must reinterpret as float, clamp, write back.
            // Loses INT32 bit-exactness, but ReLU+INT32 isn't a coherent silicon
            // configuration anyway.
            float* out = reinterpret_cast<float*>(buf);
            for (uint32_t i = 0; i < n; i++) {
                uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
                out[ni] = __emule_apply_pack_relu(__emule_dst[dst_slot][i]);
            }
        }
    } else if (cb_is_uint16_format(ocb)) {
        // 16-bit integer output: write the low 16 bits of the DST int32 bit pattern.
        // (DST holds int values bit-punned into its fp32 storage — e.g. comparison
        // results after typecast<*,UInt16>; no bf16 float conversion.)  ReLU is
        // skipped on int output (not a coherent silicon config).
        uint16_t* out = reinterpret_cast<uint16_t*>(buf);
        uint32_t n = cb_tile_elems(ocb);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            uint32_t bits;
            std::memcpy(&bits, &__emule_dst[dst_slot][i], sizeof(uint32_t));
            out[ni] = static_cast<uint16_t>(bits);
        }
    } else {
        uint16_t* bf = reinterpret_cast<uint16_t*>(buf);
        const uint32_t page = cb_page_size(ocb);
        const uint32_t rows = __emule_nfaces::tile_rows_from_pagesize(page, sizeof(uint16_t));
        uint32_t n = rows * 32u;
        if (n > __EMULE_TILE_ELEMS) n = __EMULE_TILE_ELEMS;
        if (__emule_l1_acc_enabled) {
            for (uint32_t i = 0; i < n; i++) {
                uint32_t ni = __emule_nfaces::tile_rm_to_nfaces(i, rows);
                bf[ni] = __emule_bf16::from_f32(
                    __emule_bf16::to_f32(bf[ni])
                    + __emule_apply_pack_relu(__emule_dst[dst_slot][i]));
            }
        } else {
            for (uint32_t i = 0; i < n; i++) {
                uint32_t ni = __emule_nfaces::tile_rm_to_nfaces(i, rows);
                bf[ni] = __emule_bf16::from_f32(
                    __emule_apply_pack_relu(__emule_dst[dst_slot][i]));
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

// add_tiles_init / sub_tiles_init / mul_tiles_init are declared in
// `api/compute/eltwise_binary.h` (canonical home). Only the `_nof`/`_f`
// variants live here since they have no analog in that header.
ALWI void add_tiles_init_nof() {}
ALWI void sub_tiles_init_nof() {}
ALWI void mul_tiles_init_f() {}

// Thin-tile broadcast helper: when icb1 has a smaller page_size than icb0
// (e.g. mask is a Tile([1, W]) thin tile but operand 0 is a full 32x32 tile),
// the operand-1 buffer holds only `n_b1` row-major bf16/fp32 elements stored
// contiguously. We treat that as a per-column mask broadcast across all rows,
// matching how the softmax_k kernel uses `add_tiles(scores, after_k_mask)` with a
// [1, W] mask tile. For row-major position i = r*32 + c, we return buf1[c].
//
// Returns true if icb1 is a thin-tile broadcast (page_size mismatch).
inline bool __emule_thin_broadcast_b1(uint32_t icb0, uint32_t icb1) {
    return __emule_compute::cb_page_size(icb1) < __emule_compute::cb_page_size(icb0);
}

// Tile-shape-aware binary-op helper. The {add,sub,mul}_tiles primitives all
// iterate row-major over operand 0 and project into the nfaces layout via
// `rowmajor_to_nfaces`.  That LUT assumes a 32×32 4-face tile; thin tiles
// (rows < 32) use a 2-column-face layout and require `tile_rm_to_nfaces(i,
// rows)` instead, otherwise the second column-face of the thin tile reads
// from offsets beyond the tile bound (returns garbage for cols 16..31).
//
// Op = ELWADD / ELWSUB / ELWMUL.
template <EltwiseBinaryType Op>
inline void __emule_eltwise_binary_tile(uint32_t icb0, uint32_t icb1,
                                        uint32_t itile0, uint32_t itile1,
                                        uint32_t idst) {
    const bool thin_b1 = __emule_thin_broadcast_b1(icb0, icb1);
    const bool is_32b = __emule_compute::cb_is_32bit_format(icb0);
    const uint32_t elem_a = is_32b ? 4u : 2u;
    const uint32_t elem_b = __emule_compute::cb_is_32bit_format(icb1) ? 4u : 2u;
    const uint32_t page_a = __emule_compute::cb_page_size(icb0);
    const uint32_t page_b = __emule_compute::cb_page_size(icb1);
    const uint32_t rows_a = __emule_nfaces::tile_rows_from_pagesize(page_a, elem_a);
    const uint32_t rows_b = __emule_nfaces::tile_rows_from_pagesize(page_b, elem_b);
    const uint32_t n = rows_a * 32u;
    const uint32_t n_b1 = (thin_b1 ? (page_b / elem_b) : (rows_b * 32u));
    const uint8_t* p0 = __emule_compute::cb_read_ptr_at(icb0, itile0);
    const uint8_t* p1 = __emule_compute::cb_read_ptr_at(icb1, itile1);
    auto apply = [](float a, float b) -> float {
        if constexpr (Op == EltwiseBinaryType::ELWADD) return a + b;
        if constexpr (Op == EltwiseBinaryType::ELWSUB) return a - b;
        return a * b;  // ELWMUL
    };
    if (is_32b) {
        const float* buf0 = reinterpret_cast<const float*>(p0);
        const float* buf1 = reinterpret_cast<const float*>(p1);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni_a = __emule_nfaces::tile_rm_to_nfaces(i, rows_a);
            float v1;
            if (thin_b1) {
                v1 = buf1[i % n_b1];
            } else {
                uint32_t ni_b = __emule_nfaces::tile_rm_to_nfaces(i, rows_b);
                v1 = buf1[ni_b];
            }
            __emule_dst[idst][i] = apply(buf0[ni_a], v1);
        }
    } else {
        const uint16_t* buf0 = reinterpret_cast<const uint16_t*>(p0);
        const uint16_t* buf1 = reinterpret_cast<const uint16_t*>(p1);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni_a = __emule_nfaces::tile_rm_to_nfaces(i, rows_a);
            float v1;
            if (thin_b1) {
                v1 = __emule_bf16::to_f32(buf1[i % n_b1]);
            } else {
                uint32_t ni_b = __emule_nfaces::tile_rm_to_nfaces(i, rows_b);
                v1 = __emule_bf16::to_f32(buf1[ni_b]);
            }
            __emule_dst[idst][i] = apply(__emule_bf16::to_f32(buf0[ni_a]), v1);
        }
    }
}

// Forward decl: format-aware CB→row-major-float reader (defined below). Lets the
// binary primitives decode block-float (Bfp8_b/Bfp4_b) inputs via the one central
// reader instead of re-implementing every format.
inline void __emule_unpack_cb_tile_to(uint32_t icb, uint32_t itile, float* out);

// add/sub/mul_tiles dispatch: when operand 1 is a thin-tile broadcast (smaller
// page than operand 0, e.g. a [1,W] mask) use the tile-shape-aware helper that
// broadcasts operand 1 across rows. Otherwise UNPACK both operands via the
// central format-aware reader, which handles block-float (Bfp8_b/Bfp4_b) and
// mixed-format inputs (e.g. bf16 + Bfp4_b for MoE bias-add).
ALWI void add_tiles(uint32_t icb0, uint32_t icb1,
                    uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_dst_check(idst, "add_tiles");
    __emule_dst_mark_dirty(idst);
    if (__emule_thin_broadcast_b1(icb0, icb1)) {
        __emule_eltwise_binary_tile<EltwiseBinaryType::ELWADD>(icb0, icb1, itile0, itile1, idst);
        return;
    }
    float a[__EMULE_TILE_ELEMS], b[__EMULE_TILE_ELEMS];
    __emule_unpack_cb_tile_to(icb0, itile0, a);
    __emule_unpack_cb_tile_to(icb1, itile1, b);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) __emule_dst[idst][i] = a[i] + b[i];
}

ALWI void sub_tiles(uint32_t icb0, uint32_t icb1,
                    uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_dst_check(idst, "sub_tiles");
    __emule_dst_mark_dirty(idst);
    if (__emule_thin_broadcast_b1(icb0, icb1)) {
        __emule_eltwise_binary_tile<EltwiseBinaryType::ELWSUB>(icb0, icb1, itile0, itile1, idst);
        return;
    }
    float a[__EMULE_TILE_ELEMS], b[__EMULE_TILE_ELEMS];
    __emule_unpack_cb_tile_to(icb0, itile0, a);
    __emule_unpack_cb_tile_to(icb1, itile1, b);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) __emule_dst[idst][i] = a[i] - b[i];
}

ALWI void mul_tiles(uint32_t icb0, uint32_t icb1,
                    uint32_t itile0, uint32_t itile1, uint32_t idst) {
    __emule_dst_check(idst, "mul_tiles");
    __emule_dst_mark_dirty(idst);
    if (__emule_thin_broadcast_b1(icb0, icb1)) {
        __emule_eltwise_binary_tile<EltwiseBinaryType::ELWMUL>(icb0, icb1, itile0, itile1, idst);
        return;
    }
    float a[__EMULE_TILE_ELEMS], b[__EMULE_TILE_ELEMS];
    __emule_unpack_cb_tile_to(icb0, itile0, a);
    __emule_unpack_cb_tile_to(icb1, itile1, b);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) __emule_dst[idst][i] = a[i] * b[i];
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

// pack_tile (templated 3-arg form). Default `false` matches upstream's
// `out_of_order_output = false` (llk_pack_common_api.h:70-74): the explicit
// offset is ignored, slot auto-advances. `<true>` honours the offset.
//
// pack_offset semantics: the explicit-slot path bypasses the
// auto-advance `__emule_pack_offset[ocb]++` that the no-arg path uses. To
// keep `cb_push_back(ocb, n)` consistent, advance pack_offset past
// `output_offset`. Mirrors silicon's single PACK pointer.
template <bool UseOutputOffset = false>
ALWI void pack_tile(uint32_t idst, uint32_t ocb, uint32_t output_offset = 0) {
    __emule_dst_check(idst, "pack_tile<templated>");
    if constexpr (UseOutputOffset) {
        __emule_compute::pack_dst_to_buf(__emule_compute::cb_write_ptr_at(ocb, output_offset), idst, ocb);
        if (output_offset >= __emule_pack_offset[ocb]) {
            __emule_pack_offset[ocb] = output_offset + 1;
        }
    } else {
        pack_tile(idst, ocb);
    }
}

// pack_tile (non-templated 3-arg): some upstream ops (matmul_fused_act) call this
// with an explicit output_offset without specifying the template parameter.
// Treat the offset as the write slot index (i.e. UseOutputOffset = true).
ALWI void pack_tile(uint32_t idst, uint32_t ocb, uint32_t output_offset) {
    pack_tile<true>(idst, ocb, output_offset);
}

// pack_tile_block: write DST[ifrom_dst .. ifrom_dst+ntiles-1] → CB[ocb] consecutive write slots.
ALWI void pack_tile_block(uint32_t ifrom_dst, uint32_t ocb, uint32_t ntiles) {
    if (ntiles > 0)
        __emule_dst_check(ifrom_dst + ntiles - 1, "pack_tile_block");
    for (uint32_t i = 0; i < ntiles; i++) {
        __emule_compute::pack_dst_to_buf(__emule_compute::cb_write_ptr_at(ocb, i), ifrom_dst + i, ocb);
    }
}

// __emule_unpack_cb_tile_to: read CB[icb][itile] into a caller-supplied float
// buffer, with nfaces→row-major conversion. The destination can be either a
// DST slot (regular copy_tile path) or __emule_src_scratch (binary_dest_reuse_tiles
// path); both are layout-identical 1024-element float tiles. Format-aware:
// bf16 (page_size ≤ 2048) or raw 32-bit (page_size > 2048).
inline void __emule_unpack_cb_tile_to(uint32_t icb, uint32_t itile, float* out) {
    __emule_compute::__emule_check_blockfloat_supported(icb, "__emule_unpack_cb_tile_to");
    uint8_t* buf = __emule_compute::cb_read_ptr_at(icb, itile);
    if (__emule_compute::cb_is_bfp4_b_format(icb)) {
        // Bfp4_b: UNPACK nfaces→row-major + decode shared exponent + 3-bit
        // raw mantissa (two elements per byte). Symmetric with pack_dst_to_buf.
        const uint32_t n = __EMULE_TILE_ELEMS;
        for (uint32_t i = 0; i < n; i++) {
            out[i] = __emule_bfp4::to_f32(buf, __emule_nfaces::rowmajor_to_nfaces[i]);
        }
        return;
    }
    if (__emule_compute::cb_is_bfp8_b_format(icb)) {
        // Bfp8_b: UNPACK nfaces→row-major + decode shared exponent + 7-bit
        // raw mantissa to fp32. Decoder is symmetric with pack_dst_to_buf's
        // Bfp8_b branch.
        const uint32_t n = __EMULE_TILE_ELEMS;
        for (uint32_t i = 0; i < n; i++) {
            out[i] = __emule_bfp8::to_f32(buf, __emule_nfaces::rowmajor_to_nfaces[i]);
        }
        return;
    }
    if (__emule_compute::cb_is_32bit_format(icb)) {
        // 32-bit format: UNPACK nfaces→row-major.
        // Use memcpy per element to preserve INT32 bit patterns (small positive
        // ints are denormalized floats that x86 DAZ/FTZ would flush to zero).
        const uint32_t* ubuf = reinterpret_cast<const uint32_t*>(buf);
        uint32_t n = __emule_compute::cb_page_size(icb) / sizeof(uint32_t);
        if (n > __EMULE_TILE_ELEMS) n = __EMULE_TILE_ELEMS;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            std::memcpy(&out[i], &ubuf[ni], sizeof(uint32_t));
        }
    } else if (__emule_compute::cb_is_uint16_format(icb)) {
        // 16-bit integer input: widen each uint16 to int32 and store the bit pattern
        // into the fp32 DST slot (bit-preserving, like the 32-bit branch) so int
        // comparison shims read the value via __emule_dst_load_i32.
        const uint16_t* ubuf = reinterpret_cast<const uint16_t*>(buf);
        uint32_t n = __emule_compute::cb_tile_elems(icb);
        for (uint32_t i = 0; i < n; i++) {
            int32_t v = static_cast<int32_t>(ubuf[__emule_nfaces::rowmajor_to_nfaces[i]]);
            std::memcpy(&out[i], &v, sizeof(uint32_t));
        }
    } else {
        // bfloat16: UNPACK nfaces→row-major + bf16→f32 conversion.
        // Thin tiles (rows ∈ {1,2,4,8,16}) use a 2-column-face layout where
        // each face is `rows × 16`; need tile_rm_to_nfaces(i, rows) not the
        // 32×32 LUT (rowmajor_to_nfaces) which assumes 4 faces of 16×16.
        uint16_t* bf = reinterpret_cast<uint16_t*>(buf);
        const uint32_t page = __emule_compute::cb_page_size(icb);
        const uint32_t rows = __emule_nfaces::tile_rows_from_pagesize(page, sizeof(uint16_t));
        const uint32_t n    = rows * 32u;
        for (uint32_t i = 0; i < n; i++)
            out[i] = __emule_bf16::to_f32(bf[__emule_nfaces::tile_rm_to_nfaces(i, rows)]);
    }
}

// copy_tile: UNPACK CB[icb][itile] → DST[idst].
ALWI void copy_tile(uint32_t icb, uint32_t itile, uint32_t idst) {
    __emule_dst_check(idst, "copy_tile");
    __emule_dst_mark_dirty(idst);
    __emule_unpack_cb_tile_to(icb, itile, &__emule_dst[idst][0]);
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
// 4-arg form: (srca_old, srca_new, srcb_old, srcb_new). No-op.
ALWI void reconfig_data_format(uint32_t, uint32_t, uint32_t, uint32_t) {}
template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format(uint32_t) {}
template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
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
template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
ALWI void pack_reconfig_data_format() {}
template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
ALWI void pack_reconfig_data_format(uint32_t) {}
template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
ALWI void pack_reconfig_data_format(uint32_t, uint32_t) {}

// ---- Pack-fused ReLU configuration ----
// Silicon STACC_RELU is a single packer CFG reg that clamps PACK output
// per-element before format conversion.  emule mirrors with thread-global
// state applied at every pack site.  Mode is set by `llk_pack_relu_config`,
// threshold by `pack_set_relu_threshold`; `pack_relu_config(uint32_t)`
// decodes the silicon-style packed config.
ALWI void llk_pack_relu_config(ReluType type) { __emule_pack_relu_mode = type; }
ALWI void pack_set_relu_threshold(float threshold) { __emule_pack_relu_threshold = threshold; }

// Public-API forwarders for handwritten compute kernels that use silicon's
// public pack_* names (rather than the llk_-prefixed forms).  Forward
// declaration of `llk_pack_reconfig_l1_acc` (defined later in this file)
// lets the alias compile regardless of order.
ALWI void llk_pack_reconfig_l1_acc(uint32_t enable);
ALWI void pack_init(uint32_t = 0, uint32_t = 0) {}
ALWI void pack_dest_init(uint32_t = 0) {}
ALWI void pack_reconfig_l1_acc(uint32_t enable) { llk_pack_reconfig_l1_acc(enable); }
ALWI void pack_relu_config(ReluType t) { llk_pack_relu_config(t); }
// silicon BH/WH integer-config overload: low 4 bits = mode (0=NO_RELU,
// 3=MAX_THRESHOLD_RELU, else MIN_THRESHOLD_RELU); upper 16 bits = threshold
// (uint16, reinterpreted to float by silicon — emule matches by casting the
// 16-bit value to float).
ALWI void pack_relu_config(uint32_t config) {
    const ReluType mode = (config & 0xf) == 0 ? ReluType::NO_RELU
                        : (config & 0xf) == 3 ? ReluType::MAX_THRESHOLD_RELU
                                              : ReluType::MIN_THRESHOLD_RELU;
    llk_pack_relu_config(mode);
    pack_set_relu_threshold(static_cast<float>(config >> 16));
}

// llk_pack_hw_configure: BH (#ifdef ARCH_BLACKHOLE) binary_ng SFPU bcast kernels call
// the templated form to configure the pack HW engine (the WH path skips it). emule packs
// in software, so it's a no-op — matches all the other llk pack/unpack HW-config stubs.
template <bool is_fp32_dest_acc_en>
ALWI void llk_pack_hw_configure(uint32_t /*pack_output*/) {}

// binary_dest_reuse stubs.
// D2M emits these as `binary_dest_reuse_tiles{,_init}<BinaryType, ReuseType>(...)`
// — note the template param ORDER is (BinaryType first, ReuseType second). Older
// signatures here had only one template param. The init takes a single `cb_id`.
template<EltwiseBinaryType BinaryType = EltwiseBinaryType::ELWADD,
         EltwiseBinaryReuseDestType ReuseType = EltwiseBinaryReuseDestType::NONE>
ALWI void binary_dest_reuse_tiles_init(uint32_t = 0, uint32_t = 0, bool = false) {}

template<EltwiseBinaryType BinaryType, EltwiseBinaryReuseDestType ReuseType>
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

// 3-arg overload for DEST_TO_SRC{A,B} reuse: read in_tile from icb, combine
// with DST[idst] via BinaryType, write back to DST[idst]. D2M emits the call
// as `binary_dest_reuse_tiles<BinaryType, ReuseType>(icb, in_tile, idst)`.
//
// Real silicon (tt_metal/hw/inc/api/compute/eltwise_binary.h):
//   llk_unpack_A<…DEST_TO_SRC{A,B}>(in_cb, in_tile)  // CB → one SRC bank,
//                                                    // DST[idst] → the other
//   llk_math_eltwise_binary<…DEST_TO_SRC…>(...)      // SRCA op SRCB → DST[idst]
// DST is never used as scratch — both operands cross SRC registers.
//
// Emule mirrors this with __emule_src_scratch standing in for the
// CB-side SRC bank; DST[idst] stays in place and holds the result.
template<EltwiseBinaryType BinaryType, EltwiseBinaryReuseDestType ReuseType>
ALWI void binary_dest_reuse_tiles(uint32_t icb, uint32_t in_tile, uint32_t idst) {
    __emule_dst_check(idst, "binary_dest_reuse_tiles");
    __emule_dst_mark_dirty(idst);
    __emule_unpack_cb_tile_to(icb, in_tile, __emule_src_scratch);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float a = __emule_dst[idst][i];
        float b = __emule_src_scratch[i];
        if constexpr (BinaryType == EltwiseBinaryType::ELWADD)
            __emule_dst[idst][i] = a + b;
        else if constexpr (BinaryType == EltwiseBinaryType::ELWSUB)
            __emule_dst[idst][i] = (ReuseType == EltwiseBinaryReuseDestType::DEST_TO_SRCB) ? (b - a) : (a - b);
        else
            __emule_dst[idst][i] = a * b;
    }
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

// Some D2M-emitted binary-int kernels call `mul_int_tile_init` (and similar
// add_int / sub_int variants) without including the per-op
// `api/compute/{mul,add,sub}_int_sfpu.h` header — tt-mlir's emit chain doesn't
// always pick the per-op include for these. Make them transitively available
// via common.h so any kernel that includes common.h (which the D2M wrapper
// always does) can resolve the symbols. Placed at the END of common.h so
// `ALWI` / `__EMULE_TILE_ELEMS` / DST helpers are already defined.
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"

// ---- rt_args::get<> template ----
// Lives in api/rt_arg.h (emule shadow) so both compute and dataflow kernels
// see it. Include here to keep compute/common.h consumers self-sufficient.
#include "api/rt_arg.h"

// ---- LLK function templates ----
// Pulled in AFTER __emule_compute::, __emule_dst[][], __emule_bf16::, and
// __EMULE_TILE_ELEMS are defined above — these LLK headers reference all
// of them. Templates here are inline + datacopy-routed; no SFPU INT32
// conflict (common.h's copy_tile remains the canonical CB→DST path).
#include "jit_hw/llk_pack.h"
#include "jit_hw/llk_unpack_a.h"
#include "jit_hw/llk_math_eltwise_unary_datacopy.h"
#include "jit_hw/llk_math_unary_sfpu.h"
#include "jit_hw/llk_sync_stubs.h"

// (ELWADD / ELWSUB / ELWMUL unscoped imports live earlier in this file
//  alongside the EltwiseBinaryType enum, near line 79.)
