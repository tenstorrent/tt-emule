// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "tile.hpp"
#include <array>
#include <mutex>
#include <condition_variable>

namespace tt_emule {

// DST register file: fixed physical size of 16 tile slots (64 KB).
// Active slot count depends on accumulation mode:
//   BF16 (DST_ACCUM_MODE==0): 16 active slots (2 bytes/element)
//   FP32 (DST_ACCUM_MODE!=0):  8 active slots (4 bytes/element, 2x per slot)
// State machine: IDLE -> ACQUIRED -> COMMITTED -> PACKING -> IDLE
class DstRegisterFile {
public:
    static constexpr size_t TOTAL_SLOTS = 16;  // physical size, always 16
    static constexpr size_t BF16_SLOTS  = 16;  // half-dest, bf16 mode
    static constexpr size_t FP32_SLOTS  =  8;  // half-dest, fp32 mode

    enum class State { IDLE, ACQUIRED, COMMITTED, PACKING };

    DstRegisterFile() : state_(State::IDLE), fp32_mode_(false) {}

    void set_fp32_mode(bool fp32) { fp32_mode_ = fp32; }
    bool fp32_mode() const { return fp32_mode_; }
    size_t active_slots() const { return fp32_mode_ ? FP32_SLOTS : BF16_SLOTS; }

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
    std::array<Tile, TOTAL_SLOTS> slots_;
    State state_;
    bool fp32_mode_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
};

} // namespace tt_emule
