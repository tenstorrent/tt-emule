#pragma once
// Standalone DFB operations for Quasar kernels (parallel to dataflow_api.hpp).
// Uses __core thread-local for tile counter and DFB sync state access.

#include "common.hpp"
#include "tt_emule/device.hpp"
#include "tt_emule/dataflow_buffer.hpp"
#include <cstdint>

// Thread-local per-DFB interface array, set by kernel_runner before each thread.
extern thread_local tt_emule::EmuleDFBInterface* __dfb_ifaces;

inline void dfb_reserve_back(uint32_t dfb_id, uint16_t n) {
    auto* tc = __core->tile_counters();
    if (!tc) return;
    auto& iface = __dfb_ifaces[dfb_id];
    if (!iface.active) return;
    tt_emule::DataflowBuffer dfb(iface, *tc, static_cast<uint16_t>(dfb_id));
    dfb.reserve_back(n);
}

inline void dfb_push_back(uint32_t dfb_id, uint16_t n) {
    auto* tc = __core->tile_counters();
    if (!tc) return;
    auto& iface = __dfb_ifaces[dfb_id];
    if (!iface.active) return;
    tt_emule::DataflowBuffer dfb(iface, *tc, static_cast<uint16_t>(dfb_id));
    dfb.push_back(n);
}

inline void dfb_wait_front(uint32_t dfb_id, uint16_t n) {
    auto* tc = __core->tile_counters();
    if (!tc) return;
    auto& iface = __dfb_ifaces[dfb_id];
    if (!iface.active) return;
    tt_emule::DataflowBuffer dfb(iface, *tc, static_cast<uint16_t>(dfb_id));
    dfb.wait_front(n);
}

inline void dfb_pop_front(uint32_t dfb_id, uint16_t n) {
    auto* tc = __core->tile_counters();
    if (!tc) return;
    auto& iface = __dfb_ifaces[dfb_id];
    if (!iface.active) return;
    tt_emule::DataflowBuffer dfb(iface, *tc, static_cast<uint16_t>(dfb_id));
    dfb.pop_front(n);
}

inline void dfb_finish(uint32_t dfb_id) {
    auto* tc = __core->tile_counters();
    if (!tc) return;
    auto& iface = __dfb_ifaces[dfb_id];
    if (!iface.active) return;
    tt_emule::DataflowBuffer dfb(iface, *tc, static_cast<uint16_t>(dfb_id));
    dfb.finish();
}

inline uint32_t dfb_get_write_ptr(uint32_t dfb_id) {
    auto& iface = __dfb_ifaces[dfb_id];
    auto& slot = iface.tc_slots[iface.broadcast_tc ? 0 : iface.tc_idx];
    return slot.wr_ptr;
}

inline uint32_t dfb_get_read_ptr(uint32_t dfb_id) {
    auto& iface = __dfb_ifaces[dfb_id];
    auto& slot = iface.tc_slots[iface.tc_idx];
    return slot.rd_ptr;
}

inline uint32_t dfb_get_entry_size(uint32_t dfb_id) {
    return __dfb_ifaces[dfb_id].entry_size;
}
