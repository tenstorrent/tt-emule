// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Deep-path shadow of the per-arch firmware `eth_chan_noc_mapping.h`. The bare-include resolution
// (tt_fabric_utils.h does `#include "eth_chan_noc_mapping.h"`) lands on the jit_hw root shadow; this
// deep-path copy exists for full-path includes. Both share one definition of `eth_chan_to_noc_xy` via
// the guard below, so they never double-define if both land in one translation unit. See the root
// shadow (include/jit_hw/eth_chan_noc_mapping.h) for the rationale.
#ifndef __EMULE_ETH_CHAN_NOC_MAPPING_DEFINED
#define __EMULE_ETH_CHAN_NOC_MAPPING_DEFINED
#include <cstdint>
inline uint16_t eth_chan_to_noc_xy[2][16] __attribute__((used)) = {};
#endif
