// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "cb_sync_state.hpp"
#include <vector>
#include <cstdint>
#include <cstring>

namespace tt_emule {

// Thread-safe bounded FIFO backed by raw memory.
// Owns storage and exposes a CBSyncState for shared sync operations.
class CircularBuffer {
public:
    static constexpr uint32_t DEFAULT_PAGE_SIZE = 32 * 32 * sizeof(float); // 4096

    explicit CircularBuffer(size_t capacity, uint32_t page_size = DEFAULT_PAGE_SIZE)
        : storage_(capacity * page_size, 0) {
        state_.base      = storage_.data();
        state_.page_size = page_size;
        state_.num_pages = static_cast<uint32_t>(capacity);
        state_.page_mask = (capacity > 0 && (capacity & (capacity - 1)) == 0)
                           ? static_cast<uint32_t>(capacity) - 1 : 0;
    }

    size_t   capacity()  const { return state_.num_pages; }
    uint32_t page_size() const { return state_.page_size; }

    // Access underlying sync state (for shared CB operations).
    CBSyncState&       sync_state()       { return state_; }
    const CBSyncState& sync_state() const { return state_; }

    // ---- Sync operations ----
    //
    // This is a single-threaded host helper (tilize / dram_model tests), so the
    // read/write indices are plain members here — unlike the JIT path, where the
    // per-RISC pointers are thread_local (jit_hw/internal/emule_cb_ptr.h). The
    // shared CBSyncState now owns only the occupied semaphore.

    void reserve_back(size_t n) { cb_sync_reserve(state_, static_cast<uint32_t>(n)); }
    void push_back(size_t n)    { advance(wr_idx_, n); cb_sync_push(state_, static_cast<uint32_t>(n)); }
    void wait_front(size_t n)   { cb_sync_wait(state_, static_cast<uint32_t>(n)); }
    void pop_front(size_t n)    { advance(rd_idx_, n); cb_sync_pop(state_, static_cast<uint32_t>(n)); }

    // ---- Pointer accessors ----

    uint8_t*       get_write_ptr()                      { return addr(wr_idx_); }
    const uint8_t* get_read_ptr() const                 { return addr(rd_idx_); }
    uint8_t*       get_read_ptr_mut()                    { return addr(rd_idx_); }
    uint8_t*       get_write_ptr_at(size_t offset)       { return addr_at(wr_idx_, offset); }
    const uint8_t* get_read_ptr_at(size_t offset) const  { return addr_at(rd_idx_, offset); }

private:
    uint32_t wrap(uint32_t idx) const {
        return state_.page_mask ? (idx & state_.page_mask)
                                : (state_.num_pages ? (idx % state_.num_pages) : 0);
    }
    void advance(uint32_t& idx, size_t n) {
        idx = wrap(idx + static_cast<uint32_t>(n));
    }
    uint8_t* addr(uint32_t idx) const {
        return state_.base + wrap(idx) * state_.page_size;
    }
    uint8_t* addr_at(uint32_t idx, size_t offset) const {
        return state_.base + wrap(idx + static_cast<uint32_t>(offset)) * state_.page_size;
    }

    CBSyncState state_;
    std::vector<uint8_t> storage_;  // Owns the memory that state_.base points to
    uint32_t wr_idx_ = 0;
    uint32_t rd_idx_ = 0;
};

} // namespace tt_emule
