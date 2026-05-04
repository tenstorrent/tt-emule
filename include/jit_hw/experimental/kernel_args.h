// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Mirrors tt_metal/hw/inc/experimental/kernel_args.h.
// Emule resolves get_arg_addr / get_common_arg_addr / tt_l1_ptr through the
// existing jit_kernel_stubs, so the upstream body works unmodified.

#include <cstdint>

namespace experimental {

template <typename T>
struct RtaArg {
    uint32_t byte_offset;
};

template <typename T>
struct CrtaArg {
    uint32_t byte_offset;
};

template <typename T>
struct CtaVal {
    T value;
};

template <typename T>
FORCE_INLINE T get_arg(RtaArg<T> arg) {
    static_assert(sizeof(T) == 4, "Only uint32_t args are currently supported.");
    return *((tt_l1_ptr T*)(get_arg_addr(arg.byte_offset / sizeof(uint32_t))));
}

template <typename T>
FORCE_INLINE T get_arg(CrtaArg<T> arg) {
    static_assert(sizeof(T) == 4, "Only uint32_t args are currently supported.");
    return *((tt_l1_ptr T*)(get_common_arg_addr(arg.byte_offset / sizeof(uint32_t))));
}

template <typename T>
FORCE_INLINE constexpr T get_arg(CtaVal<T> arg) {
    return arg.value;
}

}  // namespace experimental
