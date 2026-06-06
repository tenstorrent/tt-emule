// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// emule shadow of the silicon alignment API. Same semantics as silicon —
// emule's chain triggers the silicon version's `__attribute__((always_inline))`
// before silicon's includes have set up the right context, causing
// 'attribute declaration must precede definition' warnings + downstream
// redefinitions. Provide the same constexpr helpers here.

#include <cstdint>

constexpr uint32_t align_power_of_2(uint32_t addr, uint32_t alignment) {
    return ((addr - 1) | (alignment - 1)) + 1;
}

constexpr uint32_t align(uint32_t addr, uint32_t alignment) {
    return ((addr + alignment - 1) / alignment) * alignment;
}

constexpr bool is_power_of_2(uint32_t n) { return (n != 0) && (n & (n - 1)) == 0; }
