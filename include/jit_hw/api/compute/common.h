#pragma once
// JIT compute API — common definitions.
// Self-contained: provides all compute macros, enums, DST register ops,
// pack/copy tile functions, CB forwarding, and no-op reconfig stubs.
//
// On the real device these map to LLK/ckernel calls on TRISC cores.
// In emulation, DST is a thread-local float32 array and compute ops
// read bfloat16 from CBs, operate in float32, and pack back to bfloat16.

#include "jit_hw/jit_kernel_stubs.hpp"
#include <cstring>
#include <cstdint>
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

enum class MathFidelity { LoFi, HiFi2, HiFi3, HiFi4 };
enum class ReluType { NO_RELU, ZERO_RELU, MIN_THRESHOLD_RELU, MAX_THRESHOLD_RELU };
enum class DST_ACCUM_MODE { None, Half, Full };

// ---- bfloat16 conversion helpers ----

namespace __emule_bf16 {

inline float to_f32(uint16_t bf16) {
    uint32_t f32 = static_cast<uint32_t>(bf16) << 16;
    float val;
    std::memcpy(&val, &f32, sizeof(float));
    return val;
}

inline uint16_t from_f32(float val) {
    uint32_t f32;
    std::memcpy(&f32, &val, sizeof(float));
    // Round to nearest even (add 0x7FFF + bit 16 for tie-breaking)
    f32 += 0x7FFF + ((f32 >> 16) & 1);
    return static_cast<uint16_t>(f32 >> 16);
}

} // namespace __emule_bf16

// ---- Thread-local DST register file ----
// 8 tile slots, each 1024 float32 values (32x32).

static constexpr uint32_t __EMULE_DST_TILES = 8;
static constexpr uint32_t __EMULE_TILE_ELEMS = 1024;
static thread_local float __emule_dst[__EMULE_DST_TILES][__EMULE_TILE_ELEMS];

// ---- DST state machine (no-ops in single-thread-per-compute emulation) ----

ALWI void tile_regs_acquire() {
    // Zero DST on acquire (matches device behavior: acquire gives clean regs)
    for (uint32_t s = 0; s < __EMULE_DST_TILES; s++)
        std::memset(__emule_dst[s], 0, sizeof(__emule_dst[s]));
}
ALWI void tile_regs_commit()  {}
ALWI void tile_regs_wait()    {}
ALWI void tile_regs_release() {}

// ---- CB helpers (read/write via __emule_cbs) ----

namespace __emule_compute {

inline uint8_t* cb_read_ptr_at(uint32_t cb_id, uint32_t tile_offset) {
    auto& cb = __emule_cbs[cb_id];
    return cb.base + ((cb.read_idx + tile_offset) % cb.num_pages) * cb.page_size;
}

inline uint8_t* cb_write_ptr(uint32_t cb_id) {
    auto& cb = __emule_cbs[cb_id];
    return cb.base + (cb.write_idx % cb.num_pages) * cb.page_size;
}

inline uint32_t cb_tile_elems(uint32_t cb_id) {
    return __emule_cbs[cb_id].page_size / sizeof(uint16_t);
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

// add_tiles: DST[idst] = CB[icb0][itile0] + CB[icb1][itile1]
ALWI void add_tiles(uint32_t icb0, uint32_t icb1,
                    uint32_t itile0, uint32_t itile1, uint32_t idst) {
    uint16_t* buf0 = reinterpret_cast<uint16_t*>(__emule_compute::cb_read_ptr_at(icb0, itile0));
    uint16_t* buf1 = reinterpret_cast<uint16_t*>(__emule_compute::cb_read_ptr_at(icb1, itile1));
    uint32_t n = __emule_compute::cb_tile_elems(icb0);
    for (uint32_t i = 0; i < n; i++)
        __emule_dst[idst][i] = __emule_bf16::to_f32(buf0[i]) + __emule_bf16::to_f32(buf1[i]);
}

// sub_tiles: DST[idst] = CB[icb0][itile0] - CB[icb1][itile1]
ALWI void sub_tiles(uint32_t icb0, uint32_t icb1,
                    uint32_t itile0, uint32_t itile1, uint32_t idst) {
    uint16_t* buf0 = reinterpret_cast<uint16_t*>(__emule_compute::cb_read_ptr_at(icb0, itile0));
    uint16_t* buf1 = reinterpret_cast<uint16_t*>(__emule_compute::cb_read_ptr_at(icb1, itile1));
    uint32_t n = __emule_compute::cb_tile_elems(icb0);
    for (uint32_t i = 0; i < n; i++)
        __emule_dst[idst][i] = __emule_bf16::to_f32(buf0[i]) - __emule_bf16::to_f32(buf1[i]);
}

// mul_tiles: DST[idst] = CB[icb0][itile0] * CB[icb1][itile1]
ALWI void mul_tiles(uint32_t icb0, uint32_t icb1,
                    uint32_t itile0, uint32_t itile1, uint32_t idst) {
    uint16_t* buf0 = reinterpret_cast<uint16_t*>(__emule_compute::cb_read_ptr_at(icb0, itile0));
    uint16_t* buf1 = reinterpret_cast<uint16_t*>(__emule_compute::cb_read_ptr_at(icb1, itile1));
    uint32_t n = __emule_compute::cb_tile_elems(icb0);
    for (uint32_t i = 0; i < n; i++)
        __emule_dst[idst][i] = __emule_bf16::to_f32(buf0[i]) * __emule_bf16::to_f32(buf1[i]);
}

// pack_tile: write DST[idst] → CB[ocb] write slot as bfloat16
ALWI void pack_tile(uint32_t idst, uint32_t ocb) {
    uint16_t* buf = reinterpret_cast<uint16_t*>(__emule_compute::cb_write_ptr(ocb));
    uint32_t n = __emule_compute::cb_tile_elems(ocb);
    for (uint32_t i = 0; i < n; i++)
        buf[i] = __emule_bf16::from_f32(__emule_dst[idst][i]);
}

// copy_tile: DST[idst] = CB[icb][itile]
ALWI void copy_tile(uint32_t icb, uint32_t itile, uint32_t idst) {
    uint16_t* buf = reinterpret_cast<uint16_t*>(__emule_compute::cb_read_ptr_at(icb, itile));
    uint32_t n = __emule_compute::cb_tile_elems(icb);
    for (uint32_t i = 0; i < n; i++)
        __emule_dst[idst][i] = __emule_bf16::to_f32(buf[i]);
}

// copy_tile_to_dst_init_short — no-op
ALWI void copy_tile_to_dst_init_short(uint32_t) {}
ALWI void copy_tile_to_dst_init_short(uint32_t, uint32_t) {}

// ---- Reconfig operations (no-ops) ----
ALWI void reconfig_data_format(uint32_t, uint32_t) {}
ALWI void reconfig_data_format_srca(uint32_t, uint32_t) {}
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

} // namespace ckernel

// CB operations forwarded (already defined in dataflow_api.h but compute
// kernels include common.h, not dataflow_api.h — duplicate inline is OK).
inline void cb_reserve_back(uint32_t cb_id, uint32_t n) {
    auto& cb = __emule_cbs[cb_id];
    std::unique_lock<std::mutex> lk(cb.mu);
    cb.space_cv.wait(lk, [&]{ return (cb.num_pages - cb.occupied) >= n; });
}

inline void cb_push_back(uint32_t cb_id, uint32_t n) {
    auto& cb = __emule_cbs[cb_id];
    std::unique_lock<std::mutex> lk(cb.mu);
    cb.write_idx = (cb.write_idx + n) % cb.num_pages;
    cb.occupied += n;
    cb.data_cv.notify_all();
}

inline void cb_wait_front(uint32_t cb_id, uint32_t n) {
    auto& cb = __emule_cbs[cb_id];
    std::unique_lock<std::mutex> lk(cb.mu);
    cb.data_cv.wait(lk, [&]{ return cb.occupied >= n; });
}

inline void cb_pop_front(uint32_t cb_id, uint32_t n) {
    auto& cb = __emule_cbs[cb_id];
    std::unique_lock<std::mutex> lk(cb.mu);
    cb.read_idx = (cb.read_idx + n) % cb.num_pages;
    cb.occupied -= n;
    cb.space_cv.notify_all();
}
