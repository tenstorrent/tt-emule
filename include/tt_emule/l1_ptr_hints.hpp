// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstdio>

namespace tt_emule {

// Optional caller context for Core::l1_ptr diagnostics. The sentinels keep
// valid address/core values, including noc_addr=0 and self=(0,0), observable.
inline constexpr uint64_t L1_PTR_HINT_NOC_ADDR_UNSET = UINT64_MAX;
inline constexpr uint32_t L1_PTR_HINT_COORD_UNSET = UINT32_MAX;
inline thread_local uint64_t l1_ptr_hint_noc_addr = L1_PTR_HINT_NOC_ADDR_UNSET;
inline thread_local uint32_t l1_ptr_hint_self_x = L1_PTR_HINT_COORD_UNSET;
inline thread_local uint32_t l1_ptr_hint_self_y = L1_PTR_HINT_COORD_UNSET;
inline thread_local const char* l1_ptr_hint_tag = nullptr;

// Append only complete hints. A coordinate is meaningful only when both
// components are set; L1PtrHintScope always updates the pair atomically from a
// caller's perspective.
inline void format_l1_ptr_hints(FILE* stream) {
    if (l1_ptr_hint_noc_addr != L1_PTR_HINT_NOC_ADDR_UNSET) {
        std::fprintf(stream, " noc_addr=0x%llx", static_cast<unsigned long long>(l1_ptr_hint_noc_addr));
    }
    if (l1_ptr_hint_self_x != L1_PTR_HINT_COORD_UNSET && l1_ptr_hint_self_y != L1_PTR_HINT_COORD_UNSET) {
        std::fprintf(stream, " self=(%u,%u)", l1_ptr_hint_self_x, l1_ptr_hint_self_y);
    }
    if (l1_ptr_hint_tag != nullptr) {
        std::fprintf(stream, " tag=%s", l1_ptr_hint_tag);
    }
}

// Sets caller context for one lexical scope and restores the exact previous
// context on every exit path. Restoring prior state, rather than sentinels,
// makes nested resolver/helper scopes safe.
class L1PtrHintScope {
public:
    L1PtrHintScope(uint64_t noc_addr, uint32_t self_x, uint32_t self_y, const char* tag) noexcept :
        previous_noc_addr_(l1_ptr_hint_noc_addr),
        previous_self_x_(l1_ptr_hint_self_x),
        previous_self_y_(l1_ptr_hint_self_y),
        previous_tag_(l1_ptr_hint_tag) {
        l1_ptr_hint_noc_addr = noc_addr;
        l1_ptr_hint_self_x = self_x;
        l1_ptr_hint_self_y = self_y;
        l1_ptr_hint_tag = tag;
    }

    ~L1PtrHintScope() noexcept {
        l1_ptr_hint_noc_addr = previous_noc_addr_;
        l1_ptr_hint_self_x = previous_self_x_;
        l1_ptr_hint_self_y = previous_self_y_;
        l1_ptr_hint_tag = previous_tag_;
    }

    L1PtrHintScope(const L1PtrHintScope&) = delete;
    L1PtrHintScope& operator=(const L1PtrHintScope&) = delete;
    L1PtrHintScope(L1PtrHintScope&&) = delete;
    L1PtrHintScope& operator=(L1PtrHintScope&&) = delete;

private:
    uint64_t previous_noc_addr_;
    uint32_t previous_self_x_;
    uint32_t previous_self_y_;
    const char* previous_tag_;
};

}  // namespace tt_emule
