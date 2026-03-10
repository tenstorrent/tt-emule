#pragma once
// CB operations for JIT-compiled kernels.
// Delegates to shared tt_emule::cb_sync_* functions — single source of truth.

#include "jit_hw/emule_cb_state.h"
#include <cstdint>

// ---- Circular Buffer sync operations ----

inline void cb_reserve_back(uint32_t cb_id, uint32_t n) {
    tt_emule::cb_sync_reserve(__emule_cbs[cb_id], n);
}

inline void cb_push_back(uint32_t cb_id, uint32_t n) {
    tt_emule::cb_sync_push(__emule_cbs[cb_id], n);
}

inline void cb_wait_front(uint32_t cb_id, uint32_t n) {
    tt_emule::cb_sync_wait(__emule_cbs[cb_id], n);
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
inline uint32_t get_tile_size(uint32_t cb_id) {
    return __emule_cbs[cb_id].page_size;
}
