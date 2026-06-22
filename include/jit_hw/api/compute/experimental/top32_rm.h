// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Emule model for blaze's row-major top-32 LLK pipeline
// (`blaze::local_top_k_detail::run_top32_llk` and the `_presorted_1024_opt`
// variant in blaze/ops/local_top_k/kernels/run_top32_llk.hpp).
//
// Silicon implements this as a chain of SFPU LLKs (llk_unpack_A_top32_rm /
// llk_math_top32_rm / llk_math_deepseek_top32_rm_{local_sort,merge,rebuild})
// that emule does not model. The kernel-author contract is simply: read
// `row_elements` (score, index) pairs from two parallel ROW_MAJOR input CBs,
// keep the top-32 by score (descending; ties broken by lower original index,
// matching the torch.topk golden), and write them contiguously to the two
// ROW_MAJOR output CBs (the host reads the first K). The `presorted` /
// `_1024_opt` variants only change the silicon sort schedule — the net result
// is identical, so all variants route here.
//
// Layout notes: inputs are ROW_MAJOR (not tilized), so each CB page is a flat
// run of values; the per-page element count comes from the CB page_size. The
// index CB carries the original positions (the kernel reads them rather than
// deriving from lane), so the (score, index) pairing is layout-independent.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "api/compute/common.h"  // cb ops, __emule_cbs, __emule_bf16, __emule_compute::cb_*_ptr

namespace blaze {
namespace local_top_k_detail {

namespace __emule_top32 {
constexpr uint32_t TOPK = 32;

// Read `count` contiguous values from CB pages, honouring the per-page element
// capacity (page_size / elem_size). Returns (score, index) pairs.
inline std::vector<std::pair<float, uint32_t>> gather(
    uint32_t in_scores_cb, uint32_t in_indices_cb, uint32_t count, uint32_t num_input_tiles) {
    const uint32_t score_per_page = ::__emule_cbs[in_scores_cb].page_size / sizeof(uint16_t);
    const uint32_t index_per_page = ::__emule_cbs[in_indices_cb].page_size / sizeof(uint32_t);
    std::vector<std::pair<float, uint32_t>> pairs;
    pairs.reserve(count);
    uint32_t got = 0;
    for (uint32_t t = 0; t < num_input_tiles && got < count; ++t) {
        const uint16_t* sc = reinterpret_cast<const uint16_t*>(__emule_compute::cb_read_ptr_at(in_scores_cb, t));
        const uint32_t* ix = reinterpret_cast<const uint32_t*>(__emule_compute::cb_read_ptr_at(in_indices_cb, t));
        const uint32_t n = std::min({count - got, score_per_page, index_per_page});
        for (uint32_t i = 0; i < n; ++i) {
            pairs.emplace_back(::__emule_bf16::to_f32(sc[i]), ix[i]);
        }
        got += n;
    }
    return pairs;
}

template <uint32_t in_scores_cb, uint32_t in_indices_cb, uint32_t out_scores_cb, uint32_t out_indices_cb>
inline void run(uint32_t row_elements, uint32_t num_input_tiles) {
    cb_wait_front(in_scores_cb, num_input_tiles);
    cb_wait_front(in_indices_cb, num_input_tiles);
    cb_reserve_back(out_scores_cb, 1);
    cb_reserve_back(out_indices_cb, 1);

    auto pairs = gather(in_scores_cb, in_indices_cb, row_elements, num_input_tiles);
    const uint32_t keep = std::min<uint32_t>(pairs.size(), TOPK);
    std::partial_sort(
        pairs.begin(), pairs.begin() + keep, pairs.end(),
        [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
            if (a.first != b.first) return a.first > b.first;  // larger score first
            return a.second < b.second;                        // tie: lower index (golden)
        });

    // Output shards are ROW_MAJOR [1, K]; the page_size bounds how many cells we
    // may write (K <= 32). Host reads the first K descending.
    uint16_t* osc = reinterpret_cast<uint16_t*>(__emule_compute::cb_write_ptr(out_scores_cb));
    uint32_t* oix = reinterpret_cast<uint32_t*>(__emule_compute::cb_write_ptr(out_indices_cb));
    const uint32_t out_scores_n = ::__emule_cbs[out_scores_cb].page_size / sizeof(uint16_t);
    const uint32_t out_indices_n = ::__emule_cbs[out_indices_cb].page_size / sizeof(uint32_t);
    for (uint32_t i = 0; i < out_scores_n; ++i)
        osc[i] = (i < keep) ? ::__emule_bf16::from_f32(pairs[i].first) : 0;
    for (uint32_t i = 0; i < out_indices_n; ++i)
        oix[i] = (i < keep) ? pairs[i].second : 0;

    cb_pop_front(in_scores_cb, num_input_tiles);
    cb_pop_front(in_indices_cb, num_input_tiles);
    cb_push_back(out_scores_cb, 1);
    cb_push_back(out_indices_cb, 1);
}
}  // namespace __emule_top32

template <uint32_t in_scores_cb, uint32_t in_indices_cb, uint32_t out_scores_cb, uint32_t out_indices_cb,
          bool presorted = false, bool fp32_dest_acc_en = false>
inline void run_top32_llk(uint32_t row_elements, uint32_t num_input_tiles, uint32_t /*phase_number*/) {
    __emule_top32::run<in_scores_cb, in_indices_cb, out_scores_cb, out_indices_cb>(row_elements, num_input_tiles);
}

template <uint32_t in_scores_cb, uint32_t in_indices_cb, uint32_t out_scores_cb, uint32_t out_indices_cb,
          bool fp32_dest_acc_en = false>
inline void run_top32_llk_presorted_1024_opt(uint32_t row_elements, uint32_t num_input_tiles, uint32_t /*phase_number*/) {
    __emule_top32::run<in_scores_cb, in_indices_cb, out_scores_cb, out_indices_cb>(row_elements, num_input_tiles);
}

}  // namespace local_top_k_detail
}  // namespace blaze
