#pragma once
// DFB synchronization state structures for Quasar emulation.
// DFBTCSlot: per-tile-counter-slot state (pointer tracking).
// EmuleDFBInterface: per-thread per-DFB view (round-robin TC slots).
// DFBSyncState: per-DFB metadata (buffer geometry, shared across threads).

#include <cstdint>

namespace tt_emule {

static constexpr uint32_t MAX_DFBS = 32;
static constexpr uint32_t MAX_TC_SLOTS_PER_DFB = 4;

struct DFBTCSlot {
    uint32_t rd_ptr    = 0;
    uint32_t wr_ptr    = 0;
    uint32_t base_addr = 0;   // start of this slot's region in L1
    uint32_t limit     = 0;   // base_addr + capacity * entry_size
    uint8_t  neo_id    = 0;
    uint8_t  counter_id = 0;
};

struct EmuleDFBInterface {
    DFBTCSlot tc_slots[MAX_TC_SLOTS_PER_DFB] = {};
    uint8_t  num_tcs_to_rr  = 0;   // how many TC slots to round-robin
    uint8_t  tc_idx         = 0;   // current round-robin index
    uint32_t entry_size     = 0;   // bytes per entry
    uint32_t stride_size    = 0;   // bytes between consecutive entries for this RISC
    uint32_t rd_entry_idx   = 0;   // logical read entry index
    uint32_t wr_entry_idx   = 0;   // logical write entry index
    uint32_t num_entries    = 0;   // total entries in the DFB
    bool     broadcast_tc   = false; // BLOCKED mode: post to all TCs
    bool     active         = false; // whether this thread participates in this DFB
};

struct DFBSyncState {
    uint8_t* base           = nullptr;  // L1 backing memory
    uint32_t entry_size     = 0;
    uint32_t num_entries    = 0;
    uint32_t capacity       = 0;        // per-RISC capacity (num_entries / max(producers, consumers))
    uint32_t stride_in_entries = 1;     // spacing between entries for one RISC
};

} // namespace tt_emule
