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

    // ---- Sync operations (delegate to shared logic) ----

    void reserve_back(size_t n) { cb_sync_reserve(state_, static_cast<uint32_t>(n)); }
    void push_back(size_t n)    { cb_sync_push(state_, static_cast<uint32_t>(n)); }
    void wait_front(size_t n)   { cb_sync_wait(state_, static_cast<uint32_t>(n)); }
    void pop_front(size_t n)    { cb_sync_pop(state_, static_cast<uint32_t>(n)); }

    // ---- Pointer accessors (delegate to shared logic) ----

    uint8_t*       get_write_ptr()                    { return cb_sync_write_ptr(state_); }
    const uint8_t* get_read_ptr() const               { return cb_sync_read_ptr(const_cast<CBSyncState&>(state_)); }
    uint8_t*       get_read_ptr_mut()                  { return cb_sync_read_ptr(state_); }
    uint8_t*       get_write_ptr_at(size_t offset)     { return cb_sync_write_ptr_at(state_, static_cast<uint32_t>(offset)); }
    const uint8_t* get_read_ptr_at(size_t offset) const { return cb_sync_read_ptr_at(state_, static_cast<uint32_t>(offset)); }

private:
    CBSyncState state_;
    std::vector<uint8_t> storage_;  // Owns the memory that state_.base points to
};

} // namespace tt_emule
