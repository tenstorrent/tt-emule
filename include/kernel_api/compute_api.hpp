#pragma once
#include "common.hpp"
#include "tt_emule/device.hpp"
#include "tt_emule/circular_buffer.hpp"
#include "tt_emule/tile.hpp"
#include <cstring>
#include <cstdint>

// ---- DST register file state machine ----

inline void tile_regs_acquire() {
    __core->dst().acquire();
}

inline void tile_regs_commit() {
    __core->dst().commit();
}

inline void tile_regs_wait() {
    __core->dst().wait();
}

inline void tile_regs_release() {
    __core->dst().release();
}

// ---- Init functions (no-ops in prototype) ----

inline void binary_op_init_common(uint32_t /*icb0*/, uint32_t /*icb1*/, uint32_t /*ocb*/) {}
inline void add_tiles_init(uint32_t /*icb0*/, uint32_t /*icb1*/) {}
inline void copy_tile_init(uint32_t /*icb0*/) {}

// ---- Compute operations ----

// Element-wise add: DST[idst] = CB[icb0][itile0] + CB[icb1][itile1]
// Reads from the current read slots of both CBs.
inline void add_tiles(uint32_t icb0, uint32_t icb1,
                      uint32_t itile0, uint32_t itile1,
                      uint32_t idst) {
    // For the prototype, itile0/itile1 are relative offsets from read_idx.
    // Since reserve/wait always operate on n=1, we treat offset 0 as current.
    (void)itile0; (void)itile1;

    auto& cb0 = *__core->cb(icb0);
    auto& cb1 = *__core->cb(icb1);

    // Read raw bytes from CB read slots into temporary Tiles
    tt_emule::Tile t0, t1;
    std::memcpy(t0.bytes(), cb0.get_read_ptr(), tt_emule::Tile::SIZE_BYTES);
    std::memcpy(t1.bytes(), cb1.get_read_ptr(), tt_emule::Tile::SIZE_BYTES);

    __core->dst()[idst] = t0 + t1;
}

// Copy tile: DST[idst] = CB[icb0][itile]
inline void copy_tile(uint32_t icb0, uint32_t itile, uint32_t idst) {
    (void)itile;
    tt_emule::Tile t;
    std::memcpy(t.bytes(), __core->cb(icb0)->get_read_ptr(), tt_emule::Tile::SIZE_BYTES);
    __core->dst()[idst] = t;
}

// Pack tile: copy DST[idst] into CB write slot
inline void pack_tile(uint32_t idst, uint32_t ocb) {
    std::memcpy(__core->cb(ocb)->get_write_ptr(),
                __core->dst()[idst].bytes(),
                tt_emule::Tile::SIZE_BYTES);
}
