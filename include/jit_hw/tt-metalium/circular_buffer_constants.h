// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Local mirror of tt_metal/api/tt-metalium/circular_buffer_constants.h.
// This file exists only because the JIT include path does not (today) reach
// tt_metal/api/, so kernels referencing <tt-metalium/...> would otherwise fail
// to resolve.  It is intentionally kept minimal — only the constants actually
// consumed by JIT-compiled kernels are mirrored here.  Drift risk is real but
// low: the constants are public-API values and rarely change.
//
// TODO: drop this file once emulated_program_runner.cpp's JIT command line
// adds tt_metal/api as a `-I` path, or once jit_hw is restructured to pull
// shared headers from a single canonical location.

#include <cstdint>

#if defined(ARCH_WORMHOLE)
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
#else
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 64;
#endif
constexpr static std::uint32_t UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG  = 4;
constexpr static std::uint32_t UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG = 2;
constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_WORD_SIZE = 16;
constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT = 4;
