#pragma once
// Shared circular buffer synchronization state and operations.
// Source of truth for CB FIFO logic, consumed by JIT-compiled kernels via
// jit_hw/emule_cb_state.h (__emule_cb_state / __emule_cbs) and by the host
// CircularBuffer wrapper used by the tilize host-only test.

#include <atomic>
#include <cstdint>
#include <mutex>
#include <condition_variable>

namespace tt_emule {

struct CBSyncState {
    uint8_t*  base      = nullptr;  // Host pointer to start of CB memory region
    uint32_t  page_size = 0;        // Bytes per page (tile size)
    uint32_t  num_pages = 0;        // Capacity
    uint32_t  page_mask = 0;        // num_pages - 1 (for bitmask modulo; 0 if non-power-of-2)
    uint32_t  write_idx = 0;        // Current write index
    uint32_t  read_idx  = 0;        // Current read index
    std::atomic<uint32_t> occupied{0};  // Number of occupied pages (atomic for lock-free fast path)
    std::mutex              mu;
    std::condition_variable space_cv;
    std::condition_variable data_cv;
};

// ---- Sync operations on CBSyncState ----

inline void cb_sync_reserve(CBSyncState& cb, uint32_t n) {
    // Fast path: lock-free check (safe for SPSC — only consumer decrements occupied)
    if ((cb.num_pages - cb.occupied.load(std::memory_order_acquire)) >= n) return;
    // Slow path: wait under lock
    std::unique_lock<std::mutex> lk(cb.mu);
    cb.space_cv.wait(lk, [&]{ return (cb.num_pages - cb.occupied.load(std::memory_order_relaxed)) >= n; });
}

inline void cb_sync_push(CBSyncState& cb, uint32_t n) {
    std::unique_lock<std::mutex> lk(cb.mu);
    cb.write_idx = cb.page_mask ? (cb.write_idx + n) & cb.page_mask
                                : (cb.write_idx + n) % cb.num_pages;
    cb.occupied.fetch_add(n, std::memory_order_release);
    cb.data_cv.notify_one();
}

inline void cb_sync_wait(CBSyncState& cb, uint32_t n) {
    // Fast path: lock-free check (safe for SPSC — only producer increments occupied)
    if (cb.occupied.load(std::memory_order_acquire) >= n) return;
    // Slow path: wait under lock
    std::unique_lock<std::mutex> lk(cb.mu);
    cb.data_cv.wait(lk, [&]{ return cb.occupied.load(std::memory_order_relaxed) >= n; });
}

inline void cb_sync_pop(CBSyncState& cb, uint32_t n) {
    std::unique_lock<std::mutex> lk(cb.mu);
    cb.read_idx = cb.page_mask ? (cb.read_idx + n) & cb.page_mask
                               : (cb.read_idx + n) % cb.num_pages;
    cb.occupied.fetch_sub(n, std::memory_order_release);
    cb.space_cv.notify_one();
}

// ---- Pointer helpers ----

inline uint8_t* cb_sync_write_ptr(CBSyncState& cb) {
    uint32_t idx = cb.page_mask ? cb.write_idx & cb.page_mask
                                : cb.write_idx % cb.num_pages;
    return cb.base + idx * cb.page_size;
}

inline uint8_t* cb_sync_read_ptr(CBSyncState& cb) {
    uint32_t idx = cb.page_mask ? cb.read_idx & cb.page_mask
                                : cb.read_idx % cb.num_pages;
    return cb.base + idx * cb.page_size;
}

inline uint8_t* cb_sync_write_ptr_at(CBSyncState& cb, uint32_t offset) {
    uint32_t idx = cb.page_mask ? (cb.write_idx + offset) & cb.page_mask
                                : (cb.write_idx + offset) % cb.num_pages;
    return cb.base + idx * cb.page_size;
}

inline const uint8_t* cb_sync_read_ptr_at(const CBSyncState& cb, uint32_t offset) {
    uint32_t idx = cb.page_mask ? (cb.read_idx + offset) & cb.page_mask
                                : (cb.read_idx + offset) % cb.num_pages;
    return cb.base + idx * cb.page_size;
}

} // namespace tt_emule
