// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
// emule shadow of the per-arch firmware `eth_chan_noc_mapping.h`. Reached by the BARE include
// `#include "eth_chan_noc_mapping.h"` from tt_fabric_utils.h (resolved via the jit_hw -I root).
//
// Silicon defines `eth_chan_to_noc_xy[2][16]` — the NOC (x,y) of each ethernet channel's router, used
// by tt_fabric_utils.h's router-address helpers. emule has no ethernet/router model (the teleport
// resolves the destination chip + core via the cluster neighbor table), so the values are inert; we
// only need the array DECLARED with the right [2][16] shape so those helpers compile (they index it and
// take sizeof to count routers). Guarded (not just #pragma once) so the deep-path shadow and this
// root-path shadow never double-define the array if both land in one translation unit.
#ifndef __EMULE_ETH_CHAN_NOC_MAPPING_DEFINED
#define __EMULE_ETH_CHAN_NOC_MAPPING_DEFINED
#include <cstdint>
inline uint16_t eth_chan_to_noc_xy[2][16] __attribute__((used)) = {};
#endif
