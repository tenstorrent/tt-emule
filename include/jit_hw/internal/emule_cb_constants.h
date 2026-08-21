// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Maximum circular-buffer capacity modeled by emule. Blackhole has 64 CBs;
// Wormhole uses the first 32 slots. This dependency-free header is shared by
// the host Core model and JIT/runtime state so their array extents cannot drift.
inline constexpr std::uint32_t EMULE_MAX_CBS = 64;

// Compatibility for current external tt-metal runner code. New tt-emule code
// should use EMULE_MAX_CBS directly.
inline constexpr std::uint32_t __EMULE_CTX_MAX_CBS = EMULE_MAX_CBS;
