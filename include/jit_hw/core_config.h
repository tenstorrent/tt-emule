// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Stub of the silicon per-arch core type definitions. Real silicon defines per-arch core type
// counts (PROCESSOR_COUNT, MaxProcessorsPerCoreType, MAX_NUM_NOCS_PER_CORE,
// etc.). Provide minimal values matching Wormhole/Blackhole single-chip
// expectations.

#include <cstdint>

constexpr uint32_t PROCESSOR_COUNT = 5;              // BRISC + NCRISC + 3 TRISC
constexpr uint32_t MaxProcessorsPerCoreType = 5;
constexpr uint32_t MaxProcessorsForThreadingVariables = 5;
constexpr uint32_t MAX_NUM_NOCS_PER_CORE = 2;

enum class ProgrammableCoreType : uint8_t {
    TENSIX = 0,
    ACTIVE_ETH = 1,
    IDLE_ETH = 2,
};
