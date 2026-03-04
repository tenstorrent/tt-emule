#pragma once
#include "tile.hpp"
#include <array>
#include <mutex>
#include <condition_variable>

namespace tt_emule {

// DST register file: 8 tile slots.
// State machine: IDLE -> ACQUIRED -> COMMITTED -> PACKING -> IDLE
class DstRegisterFile {
public:
    static constexpr size_t NUM_SLOTS = 8;

    enum class State { IDLE, ACQUIRED, COMMITTED, PACKING };

    DstRegisterFile() : state_(State::IDLE) {}

    // Compute thread: acquire DST; blocks if not IDLE
    void acquire() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]{ return state_ == State::IDLE; });
        state_ = State::ACQUIRED;
    }

    // Compute thread: signal that tiles are ready for packing
    void commit() {
        std::unique_lock<std::mutex> lk(mu_);
        state_ = State::COMMITTED;
        cv_.notify_all();
    }

    // Packer (compute thread after commit): wait until COMMITTED
    void wait() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]{ return state_ == State::COMMITTED; });
        state_ = State::PACKING;
    }

    // Release DST back to IDLE
    void release() {
        std::unique_lock<std::mutex> lk(mu_);
        state_ = State::IDLE;
        cv_.notify_all();
    }

    Tile& operator[](size_t idx) { return slots_[idx]; }
    const Tile& operator[](size_t idx) const { return slots_[idx]; }

    State state() const {
        std::unique_lock<std::mutex> lk(mu_);
        return state_;
    }

private:
    std::array<Tile, NUM_SLOTS> slots_;
    State state_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
};

} // namespace tt_emule
