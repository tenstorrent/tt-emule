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
    // bf16 = round-to-nearest-even of float32, matching __emule_bf16::from_f32
    // (and silicon's bf16 round-trip). Truncating to the high 16 bits instead
    // would shift top-k ordering around ties and cause PCC mismatches.
    uint32_t bf16 = __emule_bf16::from_f32(value);
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

ALWI void topk_xl_separate_indices_init(uint32_t /*group_id_bit_shift*/ = 0) {}

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

    // Source format is enum-driven (cb_is_32bit_format), not page-size-derived:
    // a small-page int32/uint32 CB or a thin fp32 tile is <4096B yet still 32-bit.
    const bool is_32bit = __emule_compute::cb_is_32bit_format(in_cb_id);

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
    // Silicon's SFPU encoder packs each element's position as a face/tile
    // coordinate, NOT a linear lane:
    //   [ core_id (15:11) | col (10:6) | tile_idx (bit 5) | row (4:0) ]
    // (test_distributed_topk decodes exactly this; for epc=2048 the within
    // field col*64 + tile_idx*32 + row is the face-permuted linear offset).
    // topk_xl_copy_tile loaded this DST tile in raw nfaces (face) order, so
    // element i is a compact face offset — map it back to the 32x32 (row,col)
    // via the shared nfaces inverse before encoding. A linear lane is WRONG:
    // it sets bit 5 and uses a contiguous range, so the host gather lands on
    // the wrong cells whenever a proper subset of the values is selected
    // (pos not a multiple of epc, or >1 core's data merged).
    for (uint32_t t = 0; t < SEQ_TILES; ++t) {
        __emule_dst_check(idst + t, "topk_xl_add_lsb_indices");
        float* dst = __emule_dst[idst + t];
        for (uint32_t i = 0; i < TILE_ELEMS; ++i) {
            auto vi = decode_fused(dst[i]);
            const uint32_t rm  = __emule_nfaces::tile_nfaces_to_rm(i, 32u, 32u);  // row*32 + col
            const uint32_t row = rm >> 5;    // 0..31
            const uint32_t col = rm & 0x1F;  // 0..31
            uint32_t new_idx = ((core_id & 0x1F) << 11) | ((col & 0x1F) << 6)
                             | ((t & 0x1u) << 5) | (row & 0x1F);
            dst[i] = encode_fused(vi.v, new_idx);
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
// We approximate by collecting both halves and taking the top-N by raw decoded
// value (descending compare, since silicon merge keeps the largest values).

template <uint32_t K, bool fused = true>
ALWI void topk_xl_merge(uint32_t idst) {
    using namespace __emule_topk_xl;
    constexpr uint32_t SEQ_V = tiles_per_seq_fused<K>();
    constexpr uint32_t N = SEQ_V * TILE_ELEMS;

    if constexpr (fused) {
        // Two fused sequences at [idst, idst+SEQ_V) and [idst+SEQ_V, idst+2*SEQ_V).
        __emule_dst_check(idst + 2 * SEQ_V - 1, "topk_xl_merge");
        std::vector<float> all(2 * N);
        for (uint32_t t = 0; t < 2 * SEQ_V; ++t) {
            std::memcpy(all.data() + t * TILE_ELEMS, __emule_dst[idst + t],
                        TILE_ELEMS * sizeof(float));
        }
        std::partial_sort(all.begin(), all.begin() + N, all.end(),
                          [](float a, float b) { return decode_fused(a).v > decode_fused(b).v; });
        for (uint32_t t = 0; t < SEQ_V; ++t) {
            std::memcpy(__emule_dst[idst + t], all.data() + t * TILE_ELEMS,
                        TILE_ELEMS * sizeof(float));
        }
    } else {
        // Non-fused: each sequence is SEQ_V value-tiles followed by SEQ_V
        // index-tiles. Sequence A (acc) at [idst, idst+2*SEQ_V), sequence B
        // (recv) at [idst+2*SEQ_V, idst+4*SEQ_V). Sort (value, full-index)
        // pairs by value and keep the top N, then split back into the value
        // half and the index half at [idst, idst+2*SEQ_V).
        __emule_dst_check(idst + 4 * SEQ_V - 1, "topk_xl_merge");
        std::vector<std::pair<float, uint32_t>> all(2 * N);
        auto gather = [&](uint32_t seq_base, uint32_t out_off) {
            for (uint32_t t = 0; t < SEQ_V; ++t) {
                const float* vt = __emule_dst[idst + seq_base + t];
                const float* it = __emule_dst[idst + seq_base + SEQ_V + t];
                for (uint32_t i = 0; i < TILE_ELEMS; ++i) {
                    uint32_t ibits;
                    std::memcpy(&ibits, &it[i], sizeof(uint32_t));
                    all[out_off + t * TILE_ELEMS + i] = {decode_fused(vt[i]).v, ibits};
                }
            }
        };
        gather(0, 0);
        gather(2 * SEQ_V, N);
        std::partial_sort(all.begin(), all.begin() + N, all.end(),
                          [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
                              return a.first > b.first;
                          });
        for (uint32_t t = 0; t < SEQ_V; ++t) {
            float* vt = __emule_dst[idst + t];
            float* it = __emule_dst[idst + SEQ_V + t];
            for (uint32_t i = 0; i < TILE_ELEMS; ++i) {
                const auto& p = all[t * TILE_ELEMS + i];
                vt[i] = encode_fused(p.first, 0);  // value slot: low16 cleared (see separate)
                std::memcpy(&it[i], &p.second, sizeof(float));
            }
        }
    }
}

// ── topk_xl_rebuild ─────────────────────────────────────────────────────
// Reorder the K already-top values into the bitonic direction expected by
// the next stage. We just sort by direction.

template <uint32_t K, bool fused = true>
ALWI void topk_xl_rebuild(uint32_t idst, bool ascending) {
    using namespace __emule_topk_xl;
    if constexpr (fused) {
        topk_xl_local_sort<K>(idst, ascending);
    } else {
        // Non-fused: sort the single sequence's value-tiles [idst, idst+SEQ_V)
        // together with its parallel index-tiles [idst+SEQ_V, idst+2*SEQ_V).
        constexpr uint32_t SEQ_V = tiles_per_seq_fused<K>();
        constexpr uint32_t N = SEQ_V * TILE_ELEMS;
        __emule_dst_check(idst + 2 * SEQ_V - 1, "topk_xl_rebuild");
        std::vector<std::pair<float, uint32_t>> all(N);
        for (uint32_t t = 0; t < SEQ_V; ++t) {
            const float* vt = __emule_dst[idst + t];
            const float* it = __emule_dst[idst + SEQ_V + t];
            for (uint32_t i = 0; i < TILE_ELEMS; ++i) {
                uint32_t ibits;
                std::memcpy(&ibits, &it[i], sizeof(uint32_t));
                all[t * TILE_ELEMS + i] = {decode_fused(vt[i]).v, ibits};
            }
        }
        if (ascending) {
            std::sort(all.begin(), all.end(),
                      [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) { return a.first < b.first; });
        } else {
            std::sort(all.begin(), all.end(),
                      [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) { return a.first > b.first; });
        }
        for (uint32_t t = 0; t < SEQ_V; ++t) {
            float* vt = __emule_dst[idst + t];
            float* it = __emule_dst[idst + SEQ_V + t];
            for (uint32_t i = 0; i < TILE_ELEMS; ++i) {
                const auto& p = all[t * TILE_ELEMS + i];
                vt[i] = encode_fused(p.first, 0);  // value slot: low16 cleared (see separate)
                std::memcpy(&it[i], &p.second, sizeof(float));
            }
        }
    }
}

// ── topk_xl_separate_indices ─────────────────────────────────────────────
// Split the fused [bf16 value | u16 index] sequence into a value half and a
// full-width index half (non-fused layout): value-tiles stay at
// [idst, idst+SEQ_V) (kept fused so the sort still decodes the value), and the
// index-tiles at [idst+SEQ_V, idst+2*SEQ_V) receive the FULL 32-bit index with
// the group/device bits OR'd in at bit 16 (group_id = core_id>>5 | device<<2).
// This is why >32-core reductions go non-fused: core_id no longer fits the
// 5-bit fused field (bits 11..15), so the high core bits + device move to bits
// 16..20, which only a separate 32-bit index word can carry.
template <uint32_t K, uint32_t group_id>
ALWI void topk_xl_separate_indices(uint32_t idst) {
    using namespace __emule_topk_xl;
    constexpr uint32_t SEQ_V = tiles_per_seq_fused<K>();
    for (uint32_t t = 0; t < SEQ_V; ++t) {
        __emule_dst_check(idst + SEQ_V + t, "topk_xl_separate_indices");
        float* vt = __emule_dst[idst + t];
        float* it = __emule_dst[idst + SEQ_V + t];
        for (uint32_t i = 0; i < TILE_ELEMS; ++i) {
            auto vi = decode_fused(vt[i]);
            uint32_t full = (vi.idx & 0xFFFFu) | (group_id << 16);
            std::memcpy(&it[i], &full, sizeof(float));
            // Clear the value slot's low 16 bits (silicon's separate pass does):
            // keep_values reads the value as fp32 then rounds to bf16, so a
            // non-zero index payload in the mantissa would round the value off.
            vt[i] = encode_fused(vi.v, 0);
        }
    }
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
