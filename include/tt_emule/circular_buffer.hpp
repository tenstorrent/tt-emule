#pragma once
#include "tile.hpp"
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cstdint>

namespace tt_emule {

// Thread-safe bounded FIFO of Tile slots.
// Two condition variables: space_avail_ (signaled on pop_front) and data_avail_ (signaled on push_back).
class CircularBuffer {
public:
    explicit CircularBuffer(size_t capacity)
        : capacity_(capacity), occupied_(0), write_idx_(0), read_idx_(0),
          storage_(capacity) {}

    size_t capacity() const { return capacity_; }

    // Producer: wait until capacity - occupied >= n, then advance write_idx
    void reserve_back(size_t n) {
        std::unique_lock<std::mutex> lk(mu_);
        space_avail_.wait(lk, [&]{ return (capacity_ - occupied_) >= n; });
    }

    // Producer: push n tiles (writer already wrote into storage via get_write_ptr)
    void push_back(size_t n) {
        std::unique_lock<std::mutex> lk(mu_);
        write_idx_ = (write_idx_ + n) % capacity_;
        occupied_ += n;
        data_avail_.notify_all();
    }

    // Consumer: wait until occupied >= n
    void wait_front(size_t n) {
        std::unique_lock<std::mutex> lk(mu_);
        data_avail_.wait(lk, [&]{ return occupied_ >= n; });
    }

    // Consumer: pop n tiles (reader already consumed via get_read_ptr)
    void pop_front(size_t n) {
        std::unique_lock<std::mutex> lk(mu_);
        read_idx_ = (read_idx_ + n) % capacity_;
        occupied_ -= n;
        space_avail_.notify_all();
    }

    // Raw pointer into write slot (call after reserve_back)
    uint8_t* get_write_ptr() {
        return storage_[write_idx_].bytes();
    }

    // Raw pointer into read slot (call after wait_front)
    const uint8_t* get_read_ptr() const {
        return storage_[read_idx_].bytes();
    }

    // Non-const version for compute thread DST write-back
    uint8_t* get_read_ptr_mut() {
        return storage_[read_idx_].bytes();
    }

    Tile& write_tile() { return storage_[write_idx_]; }
    const Tile& read_tile() const { return storage_[read_idx_]; }

private:
    size_t capacity_;
    size_t occupied_;
    size_t write_idx_;
    size_t read_idx_;
    std::vector<Tile> storage_;

    mutable std::mutex mu_;
    std::condition_variable space_avail_; // signaled when a slot is freed
    std::condition_variable data_avail_;  // signaled when a tile is pushed
};

} // namespace tt_emule
