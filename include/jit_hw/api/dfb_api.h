#pragma once
// DFB operations for JIT-compiled Quasar kernels (parallel to cb_api.h).
// Delegates to EmuleDFBInterface + TileCounterArray via thread-local pointers.

#include "jit_hw/emule_dfb_state.h"
#include "tt_emule/tile_counter.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

inline int __emule_dfb_timeout_sec() {
    static int val = []() {
        const char* env = std::getenv("TT_EMULE_DFB_TIMEOUT");
        return env ? std::atoi(env) : 120;
    }();
    return val;
}

inline void dfb_reserve_back(uint32_t dfb_id, uint16_t n) {
    auto& iface = __emule_dfbs[dfb_id];
    if (!iface.active) return;
    if (iface.broadcast_tc) {
        for (uint8_t i = 0; i < iface.num_tcs_to_rr; ++i) {
            auto& slot = iface.tc_slots[i];
            auto& tc = __emule_tc_array->get(slot.neo_id, slot.counter_id);
            if (tc.free_space() >= n) continue;
            std::unique_lock<std::mutex> lk(tc.mu);
            if (!tc.space_cv.wait_for(lk,
                    std::chrono::seconds(__emule_dfb_timeout_sec()),
                    [&]{ return tc.free_space() >= n; })) {
                fprintf(stderr, "EMULE HANG: dfb_reserve_back(dfb=%u, n=%u) timed out "
                        "on TC(%u,%u) after %ds\n",
                        dfb_id, n, slot.neo_id, slot.counter_id,
                        __emule_dfb_timeout_sec());
                std::abort();
            }
        }
    } else {
        auto& slot = iface.tc_slots[iface.tc_idx];
        auto& tc = __emule_tc_array->get(slot.neo_id, slot.counter_id);
        if (tc.free_space() >= n) return;
        std::unique_lock<std::mutex> lk(tc.mu);
        if (!tc.space_cv.wait_for(lk,
                std::chrono::seconds(__emule_dfb_timeout_sec()),
                [&]{ return tc.free_space() >= n; })) {
            fprintf(stderr, "EMULE HANG: dfb_reserve_back(dfb=%u, n=%u) timed out "
                    "on TC(%u,%u) after %ds\n",
                    dfb_id, n, slot.neo_id, slot.counter_id,
                    __emule_dfb_timeout_sec());
            std::abort();
        }
    }
}

inline void dfb_push_back(uint32_t dfb_id, uint16_t n) {
    auto& iface = __emule_dfbs[dfb_id];
    if (!iface.active) return;
    if (iface.broadcast_tc) {
        for (uint8_t i = 0; i < iface.num_tcs_to_rr; ++i) {
            auto& slot = iface.tc_slots[i];
            __emule_tc_array->inc_posted(slot.neo_id, slot.counter_id, n);
        }
        auto& slot0 = iface.tc_slots[0];
        slot0.wr_ptr += static_cast<uint32_t>(n) * iface.stride_size;
        if (slot0.wr_ptr >= slot0.limit)
            slot0.wr_ptr = slot0.base_addr + (slot0.wr_ptr - slot0.limit);
    } else {
        auto& slot = iface.tc_slots[iface.tc_idx];
        __emule_tc_array->inc_posted(slot.neo_id, slot.counter_id, n);
        slot.wr_ptr += static_cast<uint32_t>(n) * iface.stride_size;
        if (slot.wr_ptr >= slot.limit)
            slot.wr_ptr = slot.base_addr + (slot.wr_ptr - slot.limit);
        iface.tc_idx = (iface.tc_idx + 1) % iface.num_tcs_to_rr;
    }
    iface.wr_entry_idx += n;
}

inline void dfb_wait_front(uint32_t dfb_id, uint16_t n) {
    auto& iface = __emule_dfbs[dfb_id];
    if (!iface.active) return;
    auto& slot = iface.tc_slots[iface.tc_idx];
    auto& tc = __emule_tc_array->get(slot.neo_id, slot.counter_id);
    if (tc.occupancy() >= n) return;
    std::unique_lock<std::mutex> lk(tc.mu);
    if (!tc.data_cv.wait_for(lk,
            std::chrono::seconds(__emule_dfb_timeout_sec()),
            [&]{ return tc.occupancy() >= n; })) {
        fprintf(stderr, "EMULE HANG: dfb_wait_front(dfb=%u, n=%u) timed out "
                "on TC(%u,%u) after %ds\n",
                dfb_id, n, slot.neo_id, slot.counter_id,
                __emule_dfb_timeout_sec());
        std::abort();
    }
}

inline void dfb_pop_front(uint32_t dfb_id, uint16_t n) {
    auto& iface = __emule_dfbs[dfb_id];
    if (!iface.active) return;
    auto& slot = iface.tc_slots[iface.tc_idx];
    __emule_tc_array->inc_acked(slot.neo_id, slot.counter_id, n);
    slot.rd_ptr += static_cast<uint32_t>(n) * iface.stride_size;
    if (slot.rd_ptr >= slot.limit)
        slot.rd_ptr = slot.base_addr + (slot.rd_ptr - slot.limit);
    iface.rd_entry_idx += n;
    iface.tc_idx = (iface.tc_idx + 1) % iface.num_tcs_to_rr;
}

inline void dfb_finish(uint32_t dfb_id) {
    auto& iface = __emule_dfbs[dfb_id];
    if (!iface.active) return;
    for (uint8_t i = 0; i < iface.num_tcs_to_rr; ++i) {
        auto& slot = iface.tc_slots[i];
        auto& tc = __emule_tc_array->get(slot.neo_id, slot.counter_id);
        std::unique_lock<std::mutex> lk(tc.mu);
        tc.space_cv.wait(lk, [&] {
            return tc.posted.load(std::memory_order_relaxed) ==
                   tc.acked.load(std::memory_order_relaxed);
        });
    }
}

inline uint32_t dfb_get_write_ptr(uint32_t dfb_id) {
    auto& iface = __emule_dfbs[dfb_id];
    auto& slot = iface.tc_slots[iface.broadcast_tc ? 0 : iface.tc_idx];
    return slot.wr_ptr;
}

inline uint32_t dfb_get_read_ptr(uint32_t dfb_id) {
    auto& iface = __emule_dfbs[dfb_id];
    auto& slot = iface.tc_slots[iface.tc_idx];
    return slot.rd_ptr;
}

inline uint32_t dfb_get_entry_size(uint32_t dfb_id) {
    return __emule_dfbs[dfb_id].entry_size;
}
