// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Tile counter synchronization primitives for Quasar DFB emulation.
// Each TileCounter tracks posted/acked counts for one hardware tile counter.
// TileCounterArray manages num_neos * 32 tile counters (data-driven, not hard-coded).

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <stdexcept>

namespace tt_emule {

static constexpr uint32_t TILE_COUNTERS_PER_NEO = 32;

struct TileCounter {
    std::atomic<uint32_t> posted{0};
    std::atomic<uint32_t> acked{0};
    uint32_t capacity{0};
    std::mutex mu;
    std::condition_variable space_cv;  // producer waits here
    std::condition_variable data_cv;   // consumer waits here

    uint32_t occupancy() const {
        return posted.load(std::memory_order_acquire)
             - acked.load(std::memory_order_acquire);
    }

    uint32_t free_space() const {
        return capacity - occupancy();
    }

    void reset() {
        posted.store(0, std::memory_order_relaxed);
        acked.store(0, std::memory_order_relaxed);
        capacity = 0;
    }
};

class TileCounterArray {
public:
    explicit TileCounterArray(uint32_t num_neos)
        : num_neos_(num_neos),
          counters_(std::make_unique<TileCounter[]>(
              static_cast<size_t>(num_neos) * TILE_COUNTERS_PER_NEO)) {}

    TileCounter& get(uint8_t neo_id, uint8_t counter_id) {
        return counters_[neo_id * TILE_COUNTERS_PER_NEO + counter_id];
    }

    uint32_t num_neos() const { return num_neos_; }

    void inc_posted(uint8_t neo_id, uint8_t counter_id, uint32_t n) {
        auto& tc = get(neo_id, counter_id);
        {
            std::lock_guard<std::mutex> lk(tc.mu);
            tc.posted.fetch_add(n, std::memory_order_release);
        }
        tc.data_cv.notify_all();
    }

    void inc_acked(uint8_t neo_id, uint8_t counter_id, uint32_t n) {
        auto& tc = get(neo_id, counter_id);
        {
            std::lock_guard<std::mutex> lk(tc.mu);
            tc.acked.fetch_add(n, std::memory_order_release);
        }
        tc.space_cv.notify_all();
    }

    void wait_free_space(uint8_t neo_id, uint8_t counter_id, uint32_t n) {
        auto& tc = get(neo_id, counter_id);
        // Always take the mutex — lockless fast path removed because
        // occupancy()/free_space() read two independent atomics non-atomically,
        // risking unsigned underflow in MPMC scenarios.
        std::unique_lock<std::mutex> lk(tc.mu);
        tc.space_cv.wait(lk, [&] { return tc.free_space() >= n; });
    }

    void wait_occupancy(uint8_t neo_id, uint8_t counter_id, uint32_t n) {
        auto& tc = get(neo_id, counter_id);
        std::unique_lock<std::mutex> lk(tc.mu);
        tc.data_cv.wait(lk, [&] { return tc.occupancy() >= n; });
    }

    void reset_all() {
        for (uint32_t i = 0; i < num_neos_ * TILE_COUNTERS_PER_NEO; ++i)
            counters_[i].reset();
    }

private:
    uint32_t num_neos_;
    std::unique_ptr<TileCounter[]> counters_;
};

} // namespace tt_emule
