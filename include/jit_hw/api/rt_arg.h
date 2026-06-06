// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// emule shadow of the upstream runtime-arg API. Adds the
// `rt_args::get<>()` template so dataflow kernels (which don't pull in
// api/compute/common.h) still see it. Types (Arg, ArrayArg, Dispatch)
// stay identical to silicon for ABI compatibility.

#include <cstdint>

namespace rt_args {

enum class Dispatch : uint8_t { COMMON, PER_CORE };

struct Arg {
    uint32_t index;
    Dispatch dispatch;
};

struct ArrayArg {
    uint32_t index;
    uint32_t length;
    Dispatch dispatch;
};

}  // namespace rt_args

// rt_args::get<arg, T>(i) — silicon-mirrored template (originally in
// the silicon compute common API). Dispatches on the compile-time `arg`
// descriptor's `dispatch` field to emule's existing global get_arg_val<T> /
// get_common_arg_val<T> (defined in jit_kernel_stubs.hpp). Available to
// both compute and dataflow kernels via this single rt_arg.h shadow.
// The two global getters are forward-declared here so this header can be
// included before jit_kernel_stubs.hpp; the actual definitions resolve at
// JIT compile time.
template <typename T>
T get_arg_val(int arg_idx);
template <typename T>
T get_common_arg_val(int arg_idx);

namespace rt_args {
template <auto arg, typename T = uint32_t>
inline T get(uint32_t i = 0) {
    if constexpr (arg.dispatch == Dispatch::COMMON) {
        return ::get_common_arg_val<T>(arg.index + i);
    } else {
        return ::get_arg_val<T>(arg.index + i);
    }
}
}  // namespace rt_args
