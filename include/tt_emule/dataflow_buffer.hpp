// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Quasar DataflowBuffer emulation -- kernel-facing API.
// Each thread constructs a DataflowBuffer from its per-thread EmuleDFBInterface
// and the shared TileCounterArray. Implements reserve_back/push_back/wait_front/
// pop_front/finish plus pointer accessors.

#include "dfb_sync_state.hpp"
#include "tile_counter.hpp"
#include <cstdint>

namespace tt_emule {

class DataflowBuffer {
public:
    DataflowBuffer(EmuleDFBInterface& iface, TileCounterArray& tc_array, uint16_t id)
        : iface_(iface), tc_array_(tc_array), id_(id) {}

    void reserve_back(uint16_t n) {
        if (iface_.broadcast_tc) {
            for (uint8_t i = 0; i < iface_.num_tcs_to_rr; ++i) {
                auto& slot = iface_.tc_slots[i];
                tc_array_.wait_free_space(slot.neo_id, slot.counter_id, n);
            }
        } else {
            auto& slot = iface_.tc_slots[iface_.tc_idx];
            tc_array_.wait_free_space(slot.neo_id, slot.counter_id, n);
        }
    }

    void push_back(uint16_t n) {
        if (iface_.broadcast_tc) {
            for (uint8_t i = 0; i < iface_.num_tcs_to_rr; ++i) {
                auto& slot = iface_.tc_slots[i];
                tc_array_.inc_posted(slot.neo_id, slot.counter_id, n);
            }
            auto& slot0 = iface_.tc_slots[0];
            slot0.wr_ptr = advance_ptr(slot0.wr_ptr, slot0.base_addr,
                                       slot0.limit, n);
        } else {
            auto& slot = iface_.tc_slots[iface_.tc_idx];
            tc_array_.inc_posted(slot.neo_id, slot.counter_id, n);
            slot.wr_ptr = advance_ptr(slot.wr_ptr, slot.base_addr,
                                      slot.limit, n);
            iface_.tc_idx = (iface_.tc_idx + 1) % iface_.num_tcs_to_rr;
        }
        iface_.wr_entry_idx += n;
    }

    void wait_front(uint16_t n) {
        auto& slot = iface_.tc_slots[iface_.tc_idx];
        tc_array_.wait_occupancy(slot.neo_id, slot.counter_id, n);
    }

    void pop_front(uint16_t n) {
        auto& slot = iface_.tc_slots[iface_.tc_idx];
        tc_array_.inc_acked(slot.neo_id, slot.counter_id, n);
        slot.rd_ptr = advance_ptr(slot.rd_ptr, slot.base_addr,
                                  slot.limit, n);
        iface_.rd_entry_idx += n;
        if (iface_.drain_per_tc) {
            // BLOCKED consumer: advance to next TC only when current slot is drained.
            if (slot.rd_ptr == slot.base_addr)
                iface_.tc_idx = (iface_.tc_idx + 1) % iface_.num_tcs_to_rr;
        } else {
            iface_.tc_idx = (iface_.tc_idx + 1) % iface_.num_tcs_to_rr;
        }
    }

    void finish() {
        for (uint8_t i = 0; i < iface_.num_tcs_to_rr; ++i) {
            auto& slot = iface_.tc_slots[i];
            auto& tc = tc_array_.get(slot.neo_id, slot.counter_id);
            if (tc.posted.load(std::memory_order_acquire) == 0 &&
                tc.acked.load(std::memory_order_acquire) == 0)
                continue;
            std::unique_lock<std::mutex> lk(tc.mu);
            auto done = [&] {
                return tc.posted.load(std::memory_order_relaxed) ==
                       tc.acked.load(std::memory_order_relaxed);
            };
            tc.space_cv.wait(lk, done);
        }
    }

    uint32_t get_write_ptr() const {
        auto& slot = iface_.tc_slots[iface_.broadcast_tc ? 0 : iface_.tc_idx];
        return slot.wr_ptr;
    }

    uint32_t get_read_ptr() const {
        auto& slot = iface_.tc_slots[iface_.tc_idx];
        return slot.rd_ptr;
    }

    uint32_t get_entry_size() const { return iface_.entry_size; }
    uint32_t get_stride_size() const { return iface_.stride_size; }
    uint16_t get_id() const { return id_; }

private:
    uint32_t advance_ptr(uint32_t ptr, uint32_t base, uint32_t limit,
                         uint16_t n) const {
        ptr += static_cast<uint32_t>(n) * iface_.stride_size;
        if (ptr >= limit)
            ptr = base + (ptr - limit);
        return ptr;
    }

    EmuleDFBInterface& iface_;
    TileCounterArray& tc_array_;
    uint16_t id_;
};

} // namespace tt_emule
