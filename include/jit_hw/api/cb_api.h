#pragma once
// CB operations for JIT-compiled kernels.
// Delegates to shared tt_emule::cb_sync_* functions — single source of truth.

#include "jit_hw/emule_cb_state.h"
#include "jit_hw/api/compute/common_globals.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>

// Forward declarations for hang diagnostics (defined in jit_kernel_stubs.hpp).
extern thread_local uint8_t my_x[2];
extern thread_local uint8_t my_y[2];
extern thread_local uint32_t __emule_logical_x;
extern thread_local uint32_t __emule_logical_y;

// ---- Constexpr tile metadata arrays (populated by JIT defines) ----
// EMULE_TILE_SIZES is defined by the JIT compiler as a comma-separated list of
// 32 page sizes (one per CB index), matching the real device's unpack_tile_size[].

#ifdef EMULE_TILE_SIZES
constexpr uint16_t unpack_tile_size[32] = { EMULE_TILE_SIZES };
#else
constexpr uint16_t unpack_tile_size[32] = {};
#endif

// Standard 32×32 tiles: 2 faces in each dimension
constexpr uint8_t unpack_tile_r_dim[32] = {
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
};
constexpr uint8_t unpack_tile_c_dim[32] = {
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
};
constexpr uint8_t unpack_num_faces_r_dim[32] = {
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
};
constexpr uint8_t unpack_num_faces_c_dim[32] = {
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
};

// ---- Circular Buffer sync operations ----

inline void cb_reserve_back(uint32_t cb_id, uint32_t n) {
    auto& cb = __emule_cbs[cb_id];
    std::unique_lock<std::mutex> lk(cb.mu);
    if (!cb.space_cv.wait_for(lk, std::chrono::seconds(5),
            [&]{ return (cb.num_pages - cb.occupied) >= n; })) {
        fprintf(stderr, "EMULE HANG: cb_reserve_back(cb_id=%u, n=%u) stuck "
                "(occupied=%u, num_pages=%u, page_size=%u) "
                "[phys (%u,%u) logical (%u,%u)]\n",
                cb_id, n, cb.occupied, cb.num_pages, cb.page_size,
                my_x[0], my_y[0], __emule_logical_x, __emule_logical_y);
        std::abort();
    }
}

inline void cb_push_back(uint32_t cb_id, uint32_t n) {
    tt_emule::cb_sync_push(__emule_cbs[cb_id], n);
}

inline void cb_wait_front(uint32_t cb_id, uint32_t n) {
    auto& cb = __emule_cbs[cb_id];
    std::unique_lock<std::mutex> lk(cb.mu);
    if (!cb.data_cv.wait_for(lk, std::chrono::seconds(5),
            [&]{ return cb.occupied >= n; })) {
        fprintf(stderr, "EMULE HANG: cb_wait_front(cb_id=%u, n=%u) stuck "
                "(occupied=%u, num_pages=%u, page_size=%u) "
                "[phys (%u,%u) logical (%u,%u)]\n",
                cb_id, n, cb.occupied, cb.num_pages, cb.page_size,
                my_x[0], my_y[0], __emule_logical_x, __emule_logical_y);
        std::abort();
    }
}

inline void cb_pop_front(uint32_t cb_id, uint32_t n) {
    tt_emule::cb_sync_pop(__emule_cbs[cb_id], n);
}

// ---- Pointer accessors ----

// Return uint32_t (truncated host pointer). CB memory is mmap'd below 4 GB.
inline uint32_t get_write_ptr(uint32_t cb_id) {
    uint8_t* ptr = tt_emule::cb_sync_write_ptr(__emule_cbs[cb_id]);
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr));
}

inline uint32_t get_read_ptr(uint32_t cb_id) {
    uint8_t* ptr = tt_emule::cb_sync_read_ptr(__emule_cbs[cb_id]);
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr));
}

// get_tile_size — return page size (bytes) for a CB.
// Uses constexpr unpack_tile_size[] array (populated by JIT EMULE_TILE_SIZES define).
constexpr inline uint32_t get_tile_size(uint32_t cb_id) {
    return static_cast<uint32_t>(unpack_tile_size[cb_id]);
}

// get_tile_hw — return tile height × width (elements).
constexpr inline uint32_t get_tile_hw(uint32_t cb_id) {
    return static_cast<uint32_t>(unpack_tile_r_dim[cb_id]) *
           static_cast<uint32_t>(unpack_tile_c_dim[cb_id]);
}

// get_tile_num_faces — return number of faces per tile.
constexpr inline uint32_t get_tile_num_faces(uint32_t cb_id) {
    return static_cast<uint32_t>(unpack_num_faces_r_dim[cb_id]) *
           static_cast<uint32_t>(unpack_num_faces_c_dim[cb_id]);
}

// get_dataformat — infer DataFormat from tile size.
// bf16 = 2048 bytes (Float16_b), 32-bit = 4096 bytes (Float32), else Float16_b.
constexpr inline DataFormat get_dataformat(uint32_t cb_id) {
    uint32_t sz = unpack_tile_size[cb_id];
    if (sz > 2048) return DataFormat::Float32;
    return DataFormat::Float16_b;
}
