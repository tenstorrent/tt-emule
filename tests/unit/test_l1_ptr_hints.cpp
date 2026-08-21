// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_emule/l1_ptr_hints.hpp"

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

std::string formatted_hints() {
    FILE* stream = std::tmpfile();
    assert(stream != nullptr);
    tt_emule::format_l1_ptr_hints(stream);
    std::fflush(stream);
    assert(std::fseek(stream, 0, SEEK_END) == 0);
    const long length = std::ftell(stream);
    assert(length >= 0);
    assert(std::fseek(stream, 0, SEEK_SET) == 0);
    std::string output(static_cast<size_t>(length), '\0');
    if (!output.empty()) {
        assert(std::fread(output.data(), 1, output.size(), stream) == output.size());
    }
    std::fclose(stream);
    return output;
}

void verify_unset() {
    assert(tt_emule::l1_ptr_hint_noc_addr == tt_emule::L1_PTR_HINT_NOC_ADDR_UNSET);
    assert(tt_emule::l1_ptr_hint_self_x == tt_emule::L1_PTR_HINT_COORD_UNSET);
    assert(tt_emule::l1_ptr_hint_self_y == tt_emule::L1_PTR_HINT_COORD_UNSET);
    assert(tt_emule::l1_ptr_hint_tag == nullptr);
    assert(formatted_hints().empty());
}

void return_early() {
    tt_emule::L1PtrHintScope scope(7, 8, 9, "early");
    return;
}

}  // namespace

int main() {
    static_assert(!std::is_copy_constructible_v<tt_emule::L1PtrHintScope>);
    static_assert(!std::is_move_constructible_v<tt_emule::L1PtrHintScope>);

    verify_unset();

    // Zero is valid caller context, not the unset representation.
    {
        tt_emule::L1PtrHintScope zero(0, 0, 0, "origin");
        assert(formatted_hints() == " noc_addr=0x0 self=(0,0) tag=origin");
    }
    verify_unset();

    // An inner scope restores the outer scope, not the global sentinels.
    {
        tt_emule::L1PtrHintScope outer(0x123, 4, 5, "outer");
        assert(formatted_hints() == " noc_addr=0x123 self=(4,5) tag=outer");
        {
            tt_emule::L1PtrHintScope inner(0x456, 0, 0, "inner");
            assert(formatted_hints() == " noc_addr=0x456 self=(0,0) tag=inner");
        }
        assert(formatted_hints() == " noc_addr=0x123 self=(4,5) tag=outer");
    }
    verify_unset();

    return_early();
    verify_unset();

    try {
        tt_emule::L1PtrHintScope scope(10, 11, 12, "exception");
        throw std::runtime_error("leave scope");
    } catch (const std::runtime_error&) {
    }
    verify_unset();
}
