// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// DFB operations for JIT-compiled Quasar kernels (parallel to cb_api.h).
// Delegates to EmuleDFBInterface + TileCounterArray via thread-local pointers.

#include "jit_hw/emule_dfb_state.h"
#include "jit_hw/emule_cb_state.h"
#include "jit_hw/internal/emule_cb_ptr.h"   // per-RISC CB pointers (advance helpers)
#include "jit_hw/api/compute/common_globals.h"
#include "tt_emule/tile_counter.hpp"
#include "jit_hw/emule_wait.h"
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

inline void __emule_dfb_check_id(uint32_t dfb_id, const char* caller) {
    if (dfb_id >= tt_emule::MAX_DFBS) {
        fprintf(stderr, "[EMULE] %s: dfb_id %u out of bounds (max %u)\n",
                caller, dfb_id, tt_emule::MAX_DFBS);
        std::abort();
    }
}

inline void dfb_reserve_back(uint32_t dfb_id, uint16_t n) {
    __emule_dfb_check_id(dfb_id, "dfb_reserve_back");
    auto& iface = __emule_dfbs[dfb_id];
    if (!iface.active) return;
    // No lockless fast path — occupancy()/free_space() read two independent
    // atomics non-atomically, risking unsigned underflow in MPMC scenarios.
    if (iface.broadcast_tc) {
        for (uint8_t i = 0; i < iface.num_tcs_to_rr; ++i) {
            auto& slot = iface.tc_slots[i];
            auto& tc = __emule_tc_array->get(slot.neo_id, slot.counter_id);
            std::unique_lock<std::mutex> lk(tc.mu);
            if (!__emule_cv_wait(tc.space_cv, lk, __emule_dfb_timeout_sec(),
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
        std::unique_lock<std::mutex> lk(tc.mu);
        if (!__emule_cv_wait(tc.space_cv, lk, __emule_dfb_timeout_sec(),
                [&]{ return tc.free_space() >= n; })) {
            fprintf(stderr, "EMULE HANG: dfb_reserve_back(dfb=%u, n=%u) timed out "
                    "on TC(%u,%u) after %ds\n",
                    dfb_id, n, slot.neo_id, slot.counter_id,
                    __emule_dfb_timeout_sec());
            std::abort();
        }
    }
    // Reset PACK engine auto-advance offset for this new batch.
    __emule_pack_offset[dfb_id] = 0;
}

inline void dfb_push_back(uint32_t dfb_id, uint16_t n) {
    __emule_dfb_check_id(dfb_id, "dfb_push_back");
    auto& iface = __emule_dfbs[dfb_id];
    if (!iface.active) return;
    // Bridge DFB→CB: bump the occupied semaphore so compute's cb_wait_front sees
    // the pushed tiles, and advance THIS thread's per-RISC CB write pointer so a
    // consumer reading DFB tiles via cb_read_ptr_at / get_read_ptr (e.g. reduce)
    // tracks the producer's progress (mirrors the old shared-write_idx coupling).
    if (__emule_cbs && __emule_cbs[dfb_id].num_pages > 0) {
        __emule_cb_advance_wr(dfb_id, n);
        tt_emule::cb_sync_push(__emule_cbs[dfb_id], n);
    }
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
    __emule_dfb_check_id(dfb_id, "dfb_wait_front");
    auto& iface = __emule_dfbs[dfb_id];
    if (!iface.active) return;
    auto& slot = iface.tc_slots[iface.tc_idx];
    auto& tc = __emule_tc_array->get(slot.neo_id, slot.counter_id);
    std::unique_lock<std::mutex> lk(tc.mu);
    if (!__emule_cv_wait(tc.data_cv, lk, __emule_dfb_timeout_sec(),
            [&]{ return tc.occupancy() >= n; })) {
        fprintf(stderr, "EMULE HANG: dfb_wait_front(dfb=%u, n=%u) timed out "
                "on TC(%u,%u) after %ds\n",
                dfb_id, n, slot.neo_id, slot.counter_id,
                __emule_dfb_timeout_sec());
        std::abort();
    }
}

inline void dfb_pop_front(uint32_t dfb_id, uint16_t n) {
    __emule_dfb_check_id(dfb_id, "dfb_pop_front");
    auto& iface = __emule_dfbs[dfb_id];
    if (!iface.active) return;
    // Bridge DFB→CB: drop the occupied semaphore so compute's cb_reserve_back
    // sees freed space, and advance THIS thread's per-RISC CB read pointer so a
    // consumer reading DFB tiles via cb_read_ptr_at / get_read_ptr (e.g. reduce's
    // WaitAndPopPerTile) advances to the next tile (mirrors the old shared
    // read_idx coupling).
    if (__emule_cbs && __emule_cbs[dfb_id].num_pages > 0) {
        __emule_cb_advance_rd(dfb_id, n);
        tt_emule::cb_sync_pop(__emule_cbs[dfb_id], n);
    }
    auto& slot = iface.tc_slots[iface.tc_idx];
    __emule_tc_array->inc_acked(slot.neo_id, slot.counter_id, n);
    slot.rd_ptr += static_cast<uint32_t>(n) * iface.stride_size;
    if (slot.rd_ptr >= slot.limit)
        slot.rd_ptr = slot.base_addr + (slot.rd_ptr - slot.limit);
    iface.rd_entry_idx += n;
    if (iface.drain_per_tc) {
        // BLOCKED consumer: advance to next TC only when current slot is drained.
        // The hardware reads all capacity entries from TC0 before moving to TC1.
        if (slot.rd_ptr == slot.base_addr)
            iface.tc_idx = (iface.tc_idx + 1) % iface.num_tcs_to_rr;
    } else {
        iface.tc_idx = (iface.tc_idx + 1) % iface.num_tcs_to_rr;
    }
}

inline void dfb_finish(uint32_t dfb_id) {
    __emule_dfb_check_id(dfb_id, "dfb_finish");
    auto& iface = __emule_dfbs[dfb_id];
    if (!iface.active) return;
    for (uint8_t i = 0; i < iface.num_tcs_to_rr; ++i) {
        auto& slot = iface.tc_slots[i];
        auto& tc = __emule_tc_array->get(slot.neo_id, slot.counter_id);
        std::unique_lock<std::mutex> lk(tc.mu);
        tc.space_cv.wait(lk, [&] {
            return tc.posted.load(std::memory_order_acquire) ==
                   tc.acked.load(std::memory_order_acquire);
        });
    }
}

inline uint32_t dfb_get_write_ptr(uint32_t dfb_id) {
    __emule_dfb_check_id(dfb_id, "dfb_get_write_ptr");
    auto& iface = __emule_dfbs[dfb_id];
    auto& slot = iface.tc_slots[iface.broadcast_tc ? 0 : iface.tc_idx];
    return slot.wr_ptr;
}

inline uint32_t dfb_get_read_ptr(uint32_t dfb_id) {
    __emule_dfb_check_id(dfb_id, "dfb_get_read_ptr");
    auto& iface = __emule_dfbs[dfb_id];
    auto& slot = iface.tc_slots[iface.tc_idx];
    return slot.rd_ptr;
}

inline uint32_t dfb_get_entry_size(uint32_t dfb_id) {
    __emule_dfb_check_id(dfb_id, "dfb_get_entry_size");
    return __emule_dfbs[dfb_id].entry_size;
}
