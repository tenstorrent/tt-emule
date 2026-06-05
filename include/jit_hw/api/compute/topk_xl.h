// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Emule Layer-1 (PoR) shadow for the silicon distributed top-K compute kernel.
//
// Silicon delivers bitonic local-sort + merge + rebuild via the SFPU on the
// MATH thread. On emule we run the same algorithm as scalar C++ against
// __emule_dst[].
//
// Data layout (matches silicon kernel-author contract):
//   Each "K-element sequence" occupies as many consecutive DST tiles as needed
//   to hold K values:
//     K = 512  -> 1 DST tile (only the top half-tile populated; bottom half
//                 padded with -inf so inactive lanes sort last)
//     K = 1024 -> 1 DST tile
//     K = 2048 -> 2 DST tiles
//
//   Fused mode (default): each FP32 word in DST is (bf16 value | u16 index).
//   Unfused mode: values and indices live in separate consecutive DST tiles.
//
// Limitations: this shadow implements the high-level kernel-author contract
// (sort, merge two sequences, rebuild bitonic property); it does NOT emulate
// the silicon SFPU's specific lane-mask / face-pack / addr_mod orchestration.
// PCC validation against silicon golden is best-effort.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <limits>

#include "api/compute/common.h"        // __emule_dst, __emule_dst_check, TILE_ELEMS
#include "api/compute/tile_move_copy.h"

namespace ckernel {

namespace __emule_topk_xl {

constexpr uint32_t TILE_ELEMS = __EMULE_TILE_ELEMS;

// Decode a fused (bf16-value | u16-index) FP32 lane to a comparable (value, index)
// pair. The bf16 occupies bits 31..16; the u16 index occupies bits 15..0.
struct VI { float v; uint32_t idx; };

inline VI decode_fused(float fp32_word) {
    uint32_t bits;
    std::memcpy(&bits, &fp32_word, sizeof(bits));
    uint32_t bf16 = (bits >> 16) & 0xFFFFu;
    uint16_t idx = bits & 0xFFFFu;
    // bf16 -> f32: place into high half of f32.
    uint32_t fbits = bf16 << 16;
    float v;
    std::memcpy(&v, &fbits, sizeof(v));
    return {v, idx};
}

inline float encode_fused(float value, uint32_t index) {
    // bf16 = high 16 bits of float32 (truncation, not round-to-nearest).
    uint32_t fbits;
    std::memcpy(&fbits, &value, sizeof(fbits));
    uint32_t bf16 = (fbits >> 16) & 0xFFFFu;
    uint32_t out = (bf16 << 16) | (index & 0xFFFFu);
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

// Number of DST tiles that a K-element sequence occupies in fused mode.
template <uint32_t K>
constexpr uint32_t tiles_per_seq_fused() {
    return (K <= 1024) ? 1 : 2;
}

}  // namespace __emule_topk_xl

// ── init / no-op shims ───────────────────────────────────────────────────

template <uint32_t K, bool fused = true>
ALWI void topk_xl_init() {}

ALWI void topk_xl_copy_tile_init(uint32_t /*cbid*/) {}

template <bool fused = true>
ALWI void topk_xl_reinit_mop_after_copy() {}

ALWI void topk_xl_add_lsb_indices_init() {}

ALWI void topk_xl_remove_msb_values_init() {}

ALWI void topk_xl_separate_indices_init() {}

// ── topk_xl_copy_tile ────────────────────────────────────────────────────
// Unpacks `num_elements` values from CB (starting at in_tile_index_base) into
// DST. Values beyond num_elements are filled with -inf so they sort last.
//
// Fused mode is the default — values arriving from the CB are plain bf16
// (or fp32) values and we encode them as (value | index=0) into DST. The
// per-core index LSBs are appended by `topk_xl_add_lsb_indices` afterwards.

template <uint32_t K>
ALWI void topk_xl_copy_tile(
    uint32_t in_cb_id,
    uint32_t dst_start_tile_index,
    uint32_t in_tile_index_base,
    uint32_t num_elements) {
    using namespace __emule_topk_xl;
    constexpr uint32_t SEQ_TILES = tiles_per_seq_fused<K>();

    // Determine source format from CB page size.
    const uint32_t page_sz = __emule_compute::cb_page_size(in_cb_id);
    const bool is_32bit = (page_sz >= 4096);

    uint32_t copied = 0;
    for (uint32_t t = 0; t < SEQ_TILES; ++t) {
        __emule_dst_check(dst_start_tile_index + t, "topk_xl_copy_tile");
        float* dst = __emule_dst[dst_start_tile_index + t];
        const uint8_t* src = __emule_compute::cb_read_ptr_at(in_cb_id, in_tile_index_base + t);

        for (uint32_t i = 0; i < TILE_ELEMS; ++i) {
            float v;
            if (copied < num_elements) {
                if (is_32bit) {
                    std::memcpy(&v, src + i * sizeof(float), sizeof(float));
                } else {
                    uint16_t bf;
                    std::memcpy(&bf, src + i * sizeof(uint16_t), sizeof(uint16_t));
                    uint32_t fbits = static_cast<uint32_t>(bf) << 16;
                    std::memcpy(&v, &fbits, sizeof(v));
                }
                ++copied;
            } else {
                v = -std::numeric_limits<float>::infinity();
            }
            dst[i] = encode_fused(v, 0);
        }
    }
}

// ── topk_xl_add_lsb_indices ──────────────────────────────────────────────
// Adds an LSB index (lane position within the K sequence | core_id<<11) to
// each fused word.

template <uint32_t K, uint32_t core_id>
ALWI void topk_xl_add_lsb_indices(uint32_t idst) {
    using namespace __emule_topk_xl;
    constexpr uint32_t SEQ_TILES = tiles_per_seq_fused<K>();
    uint32_t lane = 0;
    for (uint32_t t = 0; t < SEQ_TILES; ++t) {
        __emule_dst_check(idst + t, "topk_xl_add_lsb_indices");
        float* dst = __emule_dst[idst + t];
        for (uint32_t i = 0; i < TILE_ELEMS; ++i) {
            auto vi = decode_fused(dst[i]);
            // Index encoding: lane in low 11 bits, core_id in bits 11..15.
            uint32_t new_idx = (lane & 0x7FF) | ((core_id & 0x1F) << 11);
            dst[i] = encode_fused(vi.v, new_idx);
            ++lane;
        }
    }
}

// ── topk_xl_local_sort ──────────────────────────────────────────────────
// Sort K values across (up to) 2 tiles in DST. Bitonic in spirit but emule
// just runs std::sort over the flat sequence — the consumer only cares about
// the final sorted order (top-K). For K=2048 the values span tiles
// [idst..idst+1].

template <uint32_t K>
ALWI void topk_xl_local_sort(uint32_t idst, bool ascending) {
    using namespace __emule_topk_xl;
    constexpr uint32_t SEQ_TILES = tiles_per_seq_fused<K>();
    constexpr uint32_t N = SEQ_TILES * TILE_ELEMS;
    __emule_dst_check(idst + SEQ_TILES - 1, "topk_xl_local_sort");

    std::vector<float> buf(N);
    for (uint32_t t = 0; t < SEQ_TILES; ++t) {
        std::memcpy(buf.data() + t * TILE_ELEMS, __emule_dst[idst + t],
                    TILE_ELEMS * sizeof(float));
    }
    if (ascending) {
        std::sort(buf.begin(), buf.end(),
                  [](float a, float b) { return decode_fused(a).v < decode_fused(b).v; });
    } else {
        std::sort(buf.begin(), buf.end(),
                  [](float a, float b) { return decode_fused(a).v > decode_fused(b).v; });
    }
    for (uint32_t t = 0; t < SEQ_TILES; ++t) {
        std::memcpy(__emule_dst[idst + t], buf.data() + t * TILE_ELEMS,
                    TILE_ELEMS * sizeof(float));
    }
}

// ── topk_xl_merge ───────────────────────────────────────────────────────
// Merge two sorted sequences in DST [idst, idst+SEQ_TILES) and
// [idst+SEQ_TILES, idst+2*SEQ_TILES); top K stay in the first half.
// We approximate by collecting both halves, taking the top-K by abs value
// magnitude (use descending compare since silicon merge keeps the largest).

template <uint32_t K, bool fused = true>
ALWI void topk_xl_merge(uint32_t idst) {
    using namespace __emule_topk_xl;
    constexpr uint32_t SEQ_TILES = fused ? tiles_per_seq_fused<K>() : (2 * tiles_per_seq_fused<K>());
    constexpr uint32_t N = SEQ_TILES * TILE_ELEMS;
    __emule_dst_check(idst + 2 * SEQ_TILES - 1, "topk_xl_merge");

    std::vector<float> all(2 * N);
    for (uint32_t t = 0; t < 2 * SEQ_TILES; ++t) {
        std::memcpy(all.data() + t * TILE_ELEMS, __emule_dst[idst + t],
                    TILE_ELEMS * sizeof(float));
    }
    // Keep the top-N by value (descending sort, take first N).
    std::partial_sort(all.begin(), all.begin() + N, all.end(),
                      [](float a, float b) { return decode_fused(a).v > decode_fused(b).v; });
    for (uint32_t t = 0; t < SEQ_TILES; ++t) {
        std::memcpy(__emule_dst[idst + t], all.data() + t * TILE_ELEMS,
                    TILE_ELEMS * sizeof(float));
    }
}

// ── topk_xl_rebuild ─────────────────────────────────────────────────────
// Reorder the K already-top values into the bitonic direction expected by
// the next stage. We just sort by direction.

template <uint32_t K, bool fused = true>
ALWI void topk_xl_rebuild(uint32_t idst, bool ascending) {
    topk_xl_local_sort<K>(idst, ascending);
}

// ── topk_xl_separate_indices ─────────────────────────────────────────────
// In unfused mode the value/index halves split across consecutive tiles.
// Emule keeps fused representation so this is a no-op for correctness as
// long as remove_msb / pack tile uses the same fused layout.

template <uint32_t K, uint32_t group_id>
ALWI void topk_xl_separate_indices(uint32_t /*idst*/) {
    (void)group_id;
}

// ── topk_xl_remove_msb_values ────────────────────────────────────────────
// Strip the value half (upper 16 bits) leaving only the index in the
// low 16 bits. Result is FP32-but-meaningful-as-uint32.

template <uint32_t K>
ALWI void topk_xl_remove_msb_values(uint32_t idst) {
    using namespace __emule_topk_xl;
    constexpr uint32_t SEQ_TILES = tiles_per_seq_fused<K>();
    for (uint32_t t = 0; t < SEQ_TILES; ++t) {
        __emule_dst_check(idst + t, "topk_xl_remove_msb_values");
        float* dst = __emule_dst[idst + t];
        for (uint32_t i = 0; i < TILE_ELEMS; ++i) {
            auto vi = decode_fused(dst[i]);
            uint32_t bits = vi.idx & 0xFFFFu;  // only LSB index half
            float v;
            std::memcpy(&v, &bits, sizeof(v));
            dst[i] = v;
        }
    }
}

}  // namespace ckernel
