// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <string_view>
#include <utility>

// Compile-time kernel arguments.  The JIT compiler passes
//   -DKERNEL_COMPILE_TIME_ARGS=val0,val1,...
// on the command line.  If not provided, a single zero placeholder is used so
// that kernels which never call get_compile_time_arg_val still compile cleanly.

#ifndef KERNEL_COMPILE_TIME_ARGS
#define KERNEL_COMPILE_TIME_ARGS 0
#endif

namespace {
constexpr uint32_t kernel_compile_time_args_arr[] = {KERNEL_COMPILE_TIME_ARGS};
// Upstream-name alias. Real `tt_metal/hw/inc/api/compile_time_args.h:23` exposes
// the array under the name `kernel_compile_time_args` (no `_arr` suffix).
// Mesh tensor accessor + concat reader kernels reference the unsuffixed name.
constexpr auto& kernel_compile_time_args = kernel_compile_time_args_arr;
template<int N>
constexpr uint32_t get_ct_arg() { return kernel_compile_time_args_arr[N]; }
} // anonymous namespace

#define get_compile_time_arg_val(N) get_ct_arg<N>()

// Named compile-time args.  The JIT compiler passes
//   -DKERNEL_COMPILE_TIME_ARG_MAP={"name1",val1},{"name2",val2},...
// on the command line.  get_named_compile_time_arg_val("name") does a
// constexpr linear search through the map.

#ifdef KERNEL_COMPILE_TIME_ARG_MAP
namespace {
constexpr std::pair<std::string_view, uint32_t> __emule_named_ct_args[] = {KERNEL_COMPILE_TIME_ARG_MAP};
} // anonymous namespace

constexpr uint32_t get_named_ct_arg(std::string_view name) {
    for (const auto& [arg_name, arg_value] : __emule_named_ct_args) {
        if (name == arg_name) return arg_value;
    }
    return 0; // unreachable for valid args
}

constexpr uint32_t get_named_compile_time_arg_val(std::string_view name) {
    return get_named_ct_arg(name);
}
#endif
