// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single-threaded tilize/untilize round-trip test.
// Directly uses CircularBuffer and Tile; no kernel threads needed.
// Verifies that tilize_block followed by untilize_block is an identity transform.
#include "tt_emule/tile.hpp"
#include "tt_emule/circular_buffer.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace tt_emule;

static constexpr uint32_t NUM_TILES = 4;

int main() {
    // cb_in0: source, cb_mid: tilized, cb_out0: untilized
    CircularBuffer cb_in(NUM_TILES);
    CircularBuffer cb_mid(NUM_TILES);
    CircularBuffer cb_out(NUM_TILES);

    // Fill cb_in with a known pattern: tile i has all values = (float)(i + 1)
    cb_in.reserve_back(NUM_TILES);
    for (uint32_t i = 0; i < NUM_TILES; ++i) {
        Tile t(static_cast<float>(i + 1));
        std::memcpy(cb_in.get_write_ptr_at(i), t.bytes(), Tile::SIZE_BYTES);
    }
    cb_in.push_back(NUM_TILES);

    // tilize_block: cb_in -> cb_mid
    cb_in.wait_front(NUM_TILES);
    cb_mid.reserve_back(NUM_TILES);
    for (uint32_t i = 0; i < NUM_TILES; ++i)
        std::memcpy(cb_mid.get_write_ptr_at(i),
                    cb_in.get_read_ptr_at(i),
                    Tile::SIZE_BYTES);
    cb_mid.push_back(NUM_TILES);
    cb_in.pop_front(NUM_TILES);

    // untilize_block: cb_mid -> cb_out
    cb_mid.wait_front(NUM_TILES);
    cb_out.reserve_back(NUM_TILES);
    for (uint32_t i = 0; i < NUM_TILES; ++i)
        std::memcpy(cb_out.get_write_ptr_at(i),
                    cb_mid.get_read_ptr_at(i),
                    Tile::SIZE_BYTES);
    cb_out.push_back(NUM_TILES);
    cb_mid.pop_front(NUM_TILES);

    // Verify round-trip identity
    cb_out.wait_front(NUM_TILES);
    bool pass = true;
    for (uint32_t i = 0; i < NUM_TILES && pass; ++i) {
        float expected = static_cast<float>(i + 1);
        Tile result;
        std::memcpy(result.bytes(), cb_out.get_read_ptr_at(i), Tile::SIZE_BYTES);
        for (size_t e = 0; e < Tile::NUM_ELEMENTS; ++e) {
            if (result.raw()[e] != expected) {
                std::fprintf(stderr, "FAIL: tile[%u] element[%zu] = %f (expected %f)\n",
                             i, e, result.raw()[e], expected);
                pass = false;
                break;
            }
        }
    }
    cb_out.pop_front(NUM_TILES);

    if (pass) {
        std::printf("PASSED: tilize->untilize round-trip identity for %u tiles\n", NUM_TILES);
        return 0;
    }
    return 1;
}
