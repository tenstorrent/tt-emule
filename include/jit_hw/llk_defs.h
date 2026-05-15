#pragma once
// LLK definitions stub for emulated mode
// Core enums come from api/compute/common.h; this adds LLK-level stubs.
#include "api/compute/common.h"
#include "jit_hw/api/compute/nfaces.h"
#include "internal/firmware_common.h"

// Data copy type — moved from legacy `#define A2D 0` to the enum class
// declared below in namespace ckernel. tt-metal PR #43035 converted this
// from a plain enum to enum class; D2M-emitted kernels now use
// `DataCopyType::A2D` qualified form.

// Tile dimension constants (used by experimental LLK headers)
#ifndef FACE_R_DIM
#define FACE_R_DIM 16
#endif

#ifndef TILE_C_DIM
#define TILE_C_DIM 32
#endif

// Pool type for reduce operations
enum class PoolType : uint8_t {
    SUM = 0,
    AVG = 1,
    MAX = 2,
};

// Reduce dimension
enum class ReduceDim : uint8_t {
    REDUCE_ROW = 0,
    REDUCE_COL = 1,
    REDUCE_SCALAR = 2,
};

// Upstream ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp uses
// ckernel::PoolType / ckernel::ReduceDim. Real tt-metal defines them in the
// ckernel namespace; the JIT here keeps them at global scope, so re-export.
namespace ckernel {
using ::PoolType;
using ::ReduceDim;

// MathFidelity / DstSync — referenced by ttnn/cpp/ttnn/kernel_lib/
// reduce_helpers_compute.inl matmul wrappers and by dest_helpers.hpp's
// SyncFull/SyncHalf comparisons.  Mirrored from upstream
// tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_defs.h.
enum class MathFidelity : std::uint8_t {
    LoFi  = 0,
    HiFi2 = 2,
    HiFi3 = 3,
    HiFi4 = 4,
};
enum class DstSync : std::uint8_t {
    SyncHalf,
    SyncFull,
};

// DataCopyType — refactored to an enum class by tt-metal PR #43035
// "refactor(llk): convert DataCopyType from plain enum to enum class".
// D2M-emitted kernels reference it as `DataCopyType::A2D` etc. inside the
// embedded `experimental::tilize_block` / similar bodies. Values mirror
// upstream tt-llk/llk_defs.h.
enum class DataCopyType : std::uint8_t {
    A2D = 0,
    B2D = 1,
};
}  // namespace ckernel

// UnpackToDestEn — bool flag used as a template parameter in D2M-emitted
// `llk_math_eltwise_unary_datacopy<…, UnpackToDestEn>(…)` calls. On the
// emulator we don't dispatch on it; default to false.
inline constexpr bool UnpackToDestEn = false;

// Operand CB interface stub
struct CbInterface {
    uint32_t fifo_page_size = 0;
    uint32_t fifo_rd_ptr = 0;
    uint32_t fifo_wr_ptr = 0;
    uint32_t fifo_size = 0;
    uint32_t fifo_limit = 0;
    uint32_t fifo_num_pages = 0;
};

// Global operand interface stubs
// Guarded: dataflow_api.h provides a CB-backed version; if both headers are
// included, the dataflow version (which reads real CB state) should win.
#ifndef __EMULE_GET_LOCAL_CB_INTERFACE_DEFINED
#define __EMULE_GET_LOCAL_CB_INTERFACE_DEFINED
inline CbInterface& get_local_cb_interface(uint32_t) {
    static CbInterface dummy;
    return dummy;
}
#endif

inline uint32_t get_operand_id(uint32_t operand) { return operand; }
inline uint32_t get_output_id(uint32_t output) { return output; }
inline uint32_t get_operand_face_r_dim(uint32_t) { return 16; }
inline uint32_t get_operand_num_faces(uint32_t) { return 4; }
inline bool get_operand_narrow_tile(uint32_t) { return false; }
inline bool get_output_partial_face(uint32_t) { return false; }
inline bool get_output_narrow_tile(uint32_t) { return false; }
inline uint32_t get_output_face_r_dim(uint32_t) { return 16; }
inline uint32_t get_output_num_faces(uint32_t) { return 4; }

// Format arrays (stub)
inline uint32_t unpack_src_format[32] = {};
inline uint32_t unpack_dst_format[32] = {};

// ---- Tilize/Untilize state ----
// Unpack state: tracks source CB and tile position for datacopy calls
static thread_local uint32_t __llk_unpack_src_cb = 0;
static thread_local uint32_t __llk_unpack_start_tile_idx = 0;
static thread_local uint32_t __llk_unpack_block_c = 0;
static thread_local uint32_t __llk_unpack_current_tile = 0;
static thread_local bool __llk_unpack_is_tilize = false;

// Pack state: tracks output position and layout mode
static thread_local uint32_t __llk_pack_offset = 0;
static thread_local bool __llk_pack_is_untilize = false;
static thread_local uint32_t __llk_pack_block_c = 0;

// ---- Hardware-level stubs (not called in emulation) ----
template <typename... Args>
inline void _llk_unpack_tilize_(Args&&...) {}

// ---- Unpack tilize/untilize: set source CB and tile position ----

inline void llk_unpack_tilize(uint32_t, uint32_t, uint32_t, uint32_t) {}

inline void llk_unpack_tilize_block(uint32_t icb, uint32_t block_c, uint32_t start_tile_idx) {
    __llk_unpack_src_cb = icb;
    __llk_unpack_start_tile_idx = start_tile_idx;
    __llk_unpack_block_c = block_c;
    __llk_unpack_current_tile = 0;
}

inline void llk_unpack_untilize(uint32_t icb, uint32_t block_c, uint32_t start_tile_idx) {
    __llk_unpack_src_cb = icb;
    __llk_unpack_start_tile_idx = start_tile_idx;
    __llk_unpack_block_c = block_c;
    __llk_unpack_current_tile = 0;
    __llk_pack_block_c = block_c;
}

// ---- llk_unpack_A: general-purpose tile unpack (no-op in emulation) ----
// In emulation, actual data movement happens in copy_tile / datacopy helpers.
// This stub exists so that code paths referencing llk_unpack_A compile.
template <ckernel::BroadcastType BType = ckernel::BroadcastType::NONE,
          bool acc_to_dest = false,
          ckernel::EltwiseBinaryReuseDestType binary_reuse_dest = ckernel::EltwiseBinaryReuseDestType::NONE,
          bool unpack_to_dest = false>
inline void llk_unpack_A(uint32_t /*operand*/, uint32_t /*tile_index*/) {}

// ---- LLK CB stubs (delegate to cb_api.h functions) ----
// The real tt-metal cb_api.h inlines these as calls to llk_wait_tiles etc.
// In emulation, the CB functions are the real implementation; llk_* wrappers delegate.
inline void llk_wait_tiles(int operand, std::int32_t num_tiles) {
    cb_wait_front(static_cast<uint32_t>(operand), static_cast<uint32_t>(num_tiles));
}
inline void llk_pop_tiles(std::int32_t operand, std::int32_t num_tiles, std::int32_t = 0) {
    cb_pop_front(static_cast<uint32_t>(operand), static_cast<uint32_t>(num_tiles));
}
template <bool = false, bool = false, bool = false>
inline void llk_wait_for_free_tiles(std::int32_t operand, std::int32_t num_tiles) {
    cb_reserve_back(static_cast<uint32_t>(operand), static_cast<uint32_t>(num_tiles));
}
template <bool = false, bool = false>
inline void llk_push_tiles(std::int32_t operand, std::int32_t num_tiles) {
    cb_push_back(static_cast<uint32_t>(operand), static_cast<uint32_t>(num_tiles));
}

// ---- Sync stubs (no-op in single-threaded compute) ----
inline void llk_math_wait_for_dest_available() {}
inline void llk_packer_wait_for_math_done() {}

using ckernel::BroadcastType;

// ---- Datacopy helpers ----

// Tilize mode: extract 32x32 tile from row-major input CB into DST
inline void __llk_tilize_datacopy(uint32_t dst_idx) {
    uint32_t cb = __llk_unpack_src_cb;
    uint32_t ps = __emule_compute::cb_page_size(cb);
    uint32_t elem_size = ps / 1024;  // 2 for bf16, 4 for f32
    uint32_t tile_col = __llk_unpack_current_tile;
    uint32_t block_c = __llk_unpack_block_c;

    uint8_t* base = __emule_compute::cb_read_ptr_at(cb, 0);
    uint32_t tile_row_offset = __llk_unpack_start_tile_idx * ps;
    uint32_t tile_col_offset = tile_col * 32 * elem_size;
    uint32_t row_stride = block_c * 32 * elem_size;

    if (elem_size == 2) {  // bfloat16
        for (uint32_t r = 0; r < 32; r++) {
            uint16_t* src = reinterpret_cast<uint16_t*>(
                base + tile_row_offset + r * row_stride + tile_col_offset);
            for (uint32_t c = 0; c < 32; c++)
                __emule_dst[dst_idx][r * 32 + c] = __emule_bf16::to_f32(src[c]);
        }
    } else {  // float32 / int32
        for (uint32_t r = 0; r < 32; r++) {
            float* src = reinterpret_cast<float*>(
                base + tile_row_offset + r * row_stride + tile_col_offset);
            for (uint32_t c = 0; c < 32; c++)
                __emule_dst[dst_idx][r * 32 + c] = src[c];
        }
    }
    __llk_unpack_current_tile++;
}

// Untilize mode: copy individual tile page from input CB into DST
inline void __llk_untilize_datacopy(uint32_t dst_idx) {
    uint32_t tile_idx = __llk_unpack_start_tile_idx + __llk_unpack_current_tile;
    copy_tile(__llk_unpack_src_cb, tile_idx, dst_idx);
    __llk_unpack_current_tile++;
}

// 4-param version (used by tilize). PR #43035 changed CopyType from plain
// enum (int) to enum class DataCopyType.
template <ckernel::DataCopyType CopyType, int AccumMode, BroadcastType Bcast, bool UnpackToDest>
inline void llk_math_eltwise_unary_datacopy(uint32_t dst_idx) {
    if (__llk_unpack_is_tilize) __llk_tilize_datacopy(dst_idx);
    else __llk_untilize_datacopy(dst_idx);
}

// 3-param version (used by untilize)
template <ckernel::DataCopyType CopyType, int AccumMode, BroadcastType Bcast>
inline void llk_math_eltwise_unary_datacopy(uint32_t dst_idx) {
    if (__llk_unpack_is_tilize) __llk_tilize_datacopy(dst_idx);
    else __llk_untilize_datacopy(dst_idx);
}

// ---- Pack helpers ----

// Tilize pack: PACK row-major DST → nfaces CB at pack_offset
inline void __llk_pack_tiled(uint32_t tile_idx, uint32_t ocb) {
    uint8_t* buf = __emule_compute::cb_write_ptr_at(ocb, __llk_pack_offset);
    if (__emule_compute::cb_is_32bit_format(ocb)) {
        uint32_t n = __emule_compute::cb_page_size(ocb) / sizeof(uint32_t);
        if (n > __EMULE_TILE_ELEMS) n = __EMULE_TILE_ELEMS;
        uint32_t* out = reinterpret_cast<uint32_t*>(buf);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            std::memcpy(&out[ni], &__emule_dst[tile_idx][i], sizeof(uint32_t));
        }
    } else {
        uint16_t* bf = reinterpret_cast<uint16_t*>(buf);
        uint32_t n = __emule_compute::cb_tile_elems(ocb);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            bf[ni] = __emule_bf16::from_f32(__emule_dst[tile_idx][i]);
        }
    }
}

// Untilize pack: scatter DST tile into row-major output CB
inline void __llk_pack_untilize(uint32_t tile_idx, uint32_t ocb) {
    uint32_t ps = __emule_compute::cb_page_size(ocb);
    uint32_t elem_size = ps / 1024;
    uint32_t block_c = __llk_pack_block_c;
    uint32_t tile_row = __llk_pack_offset / block_c;
    uint32_t tile_col = __llk_pack_offset % block_c;

    uint8_t* base = __emule_compute::cb_write_ptr_at(ocb, 0);
    uint32_t row_stride = block_c * 32 * elem_size;
    uint32_t tile_row_offset = tile_row * 32 * row_stride;
    uint32_t tile_col_offset = tile_col * 32 * elem_size;

    if (elem_size == 2) {  // bfloat16
        for (uint32_t r = 0; r < 32; r++) {
            uint16_t* dst = reinterpret_cast<uint16_t*>(
                base + tile_row_offset + r * row_stride + tile_col_offset);
            for (uint32_t c = 0; c < 32; c++)
                dst[c] = __emule_bf16::from_f32(__emule_dst[tile_idx][r * 32 + c]);
        }
    } else {  // float32 / int32
        for (uint32_t r = 0; r < 32; r++) {
            float* dst = reinterpret_cast<float*>(
                base + tile_row_offset + r * row_stride + tile_col_offset);
            for (uint32_t c = 0; c < 32; c++)
                dst[c] = __emule_dst[tile_idx][r * 32 + c];
        }
    }
}

// 2-param version (used by tilize)
template <bool Untilize, bool IsApprox>
inline void llk_pack(uint32_t tile_idx, uint32_t ocb) {
    if (__llk_pack_is_untilize) __llk_pack_untilize(tile_idx, ocb);
    else __llk_pack_tiled(tile_idx, ocb);
}

// 3-param version (used by untilize)
template <int AccumMode, bool Untilize, bool IsApprox>
inline void llk_pack(uint32_t tile_idx, uint32_t ocb) {
    if (__llk_pack_is_untilize) __llk_pack_untilize(tile_idx, ocb);
    else __llk_pack_tiled(tile_idx, ocb);
}

template <int AccumMode>
inline void llk_math_dest_section_done() {}

template <int AccumMode>
inline void llk_pack_dest_section_done() {
    __llk_pack_offset++;
}

// ---- Compute startup and init ----
inline void compute_kernel_hw_startup(uint32_t, uint32_t) {
    __llk_pack_offset = 0;
    __llk_pack_is_untilize = false;
    __llk_unpack_is_tilize = false;
    // Reset PACK engine auto-advance offsets and L1 acc flag to prevent
    // stale state from prior kernel invocations in the same thread.
    std::memset(__emule_pack_offset, 0, sizeof(__emule_pack_offset));
    __emule_l1_acc_enabled = false;
}
inline void compute_kernel_hw_startup(uint32_t a, uint32_t b, uint32_t) {
    compute_kernel_hw_startup(a, b);
}
inline void tilize_init(uint32_t, uint32_t, uint32_t) {
    __llk_unpack_is_tilize = true;
    __llk_pack_is_untilize = false;
}
inline void tilize_init_short(uint32_t, uint32_t) {
    __llk_unpack_is_tilize = true;
    __llk_pack_is_untilize = false;
}
inline void untilize_init(uint32_t, uint32_t = 0) {
    __llk_unpack_is_tilize = false;
    __llk_pack_is_untilize = true;
}
inline void untilize_init_short(uint32_t) {
    __llk_unpack_is_tilize = false;
    __llk_pack_is_untilize = true;
}

// ---- Experimental pack_untilize_block (used by D2M-generated untilize kernels) ----
// Uses copy_tile (CB→DST) + __llk_pack_untilize (DST→CB row-major).
// __llk_pack_untilize needs __llk_pack_block_c (row stride in tiles) and
// __llk_pack_offset (linear tile position) to compute scatter coordinates.
namespace experimental {

template <uint32_t cols_per_dst_pass, uint32_t total_col_tiles>
inline void pack_untilize_block(uint32_t icb, uint32_t ocb,
                                uint32_t block_row_tiles,
                                uint32_t block_col_tiles) {
    __llk_pack_block_c = total_col_tiles;
    __llk_pack_offset = 0;

    const uint32_t num_col_blocks = block_col_tiles / cols_per_dst_pass;
    for (uint32_t r = 0; r < block_row_tiles; ++r) {
        for (uint32_t b = 0; b < num_col_blocks; ++b) {
            for (uint32_t c = 0; c < cols_per_dst_pass; ++c) {
                uint32_t src_tile = r * block_col_tiles + b * cols_per_dst_pass + c;
                copy_tile(icb, src_tile, c);
            }

            for (uint32_t c = 0; c < cols_per_dst_pass; ++c) {
                __llk_pack_untilize(c, ocb);
                __llk_pack_offset++;
            }
        }
    }
}

} // namespace experimental
