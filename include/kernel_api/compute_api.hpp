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
inline void matmul_tiles_init(uint32_t /*icb0*/, uint32_t /*icb1*/) {}
inline void tilize_init(uint32_t /*icb*/, uint32_t /*num_tiles*/, uint32_t /*ocb*/) {}
inline void untilize_init(uint32_t /*icb*/, uint32_t /*ocb*/) {}

// ---- Compute operations ----

// Element-wise add: DST[idst] = CB[icb0][itile0] + CB[icb1][itile1]
inline void add_tiles(uint32_t icb0, uint32_t icb1,
                      uint32_t itile0, uint32_t itile1,
                      uint32_t idst) {
    auto& cb0 = *__core->cb(icb0);
    auto& cb1 = *__core->cb(icb1);

    tt_emule::Tile t0, t1;
    std::memcpy(t0.bytes(), cb0.get_read_ptr_at(itile0), tt_emule::Tile::SIZE_BYTES);
    std::memcpy(t1.bytes(), cb1.get_read_ptr_at(itile1), tt_emule::Tile::SIZE_BYTES);

    __core->dst()[idst] = t0 + t1;
}

// Copy tile: DST[idst] = CB[icb0][itile]
inline void copy_tile(uint32_t icb0, uint32_t itile, uint32_t idst) {
    tt_emule::Tile t;
    std::memcpy(t.bytes(), __core->cb(icb0)->get_read_ptr_at(itile), tt_emule::Tile::SIZE_BYTES);
    __core->dst()[idst] = t;
}

// Pack tile: copy DST[idst] into CB write slot
inline void pack_tile(uint32_t idst, uint32_t ocb) {
    std::memcpy(__core->cb(ocb)->get_write_ptr(),
                __core->dst()[idst].bytes(),
                tt_emule::Tile::SIZE_BYTES);
}

// Matrix multiply: DST[idst] += CB[icb0][itile0] x CB[icb1][itile1]
// Accumulates into DST (caller must zero DST via tile_regs_acquire before first call).
inline void matmul_tiles(uint32_t icb0, uint32_t icb1,
                         uint32_t itile0, uint32_t itile1,
                         uint32_t idst, bool transpose) {
    tt_emule::Tile t0, t1;
    std::memcpy(t0.bytes(), __core->cb(icb0)->get_read_ptr_at(itile0), tt_emule::Tile::SIZE_BYTES);
    std::memcpy(t1.bytes(), __core->cb(icb1)->get_read_ptr_at(itile1), tt_emule::Tile::SIZE_BYTES);

    for (uint32_t r = 0; r < tt_emule::Tile::ROWS; ++r)
        for (uint32_t c = 0; c < tt_emule::Tile::COLS; ++c) {
            float acc = 0.f;
            for (uint32_t k = 0; k < tt_emule::Tile::COLS; ++k)
                acc += t0(r, k) * (transpose ? t1(c, k) : t1(k, c));
            __core->dst()[idst](r, c) += acc;
        }
}

// Tilize: copy num_tiles from icb to ocb (layout is identical in prototype)
inline void tilize_block(uint32_t icb, uint32_t num_tiles, uint32_t ocb) {
    for (uint32_t i = 0; i < num_tiles; ++i)
        std::memcpy(__core->cb(ocb)->get_write_ptr_at(i),
                    __core->cb(icb)->get_read_ptr_at(i),
                    tt_emule::Tile::SIZE_BYTES);
}

// Untilize: copy num_tiles from icb to ocb (layout is identical in prototype)
inline void untilize_block(uint32_t icb, uint32_t num_tiles, uint32_t ocb) {
    for (uint32_t i = 0; i < num_tiles; ++i)
        std::memcpy(__core->cb(ocb)->get_write_ptr_at(i),
                    __core->cb(icb)->get_read_ptr_at(i),
                    tt_emule::Tile::SIZE_BYTES);
}
