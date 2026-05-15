// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Empty shim. Shadows upstream tt_metal/hw/inc/api/compute/pack.h to avoid a
// pack_tile redefinition; emule's pack_tile lives in jit_hw/api/compute/common.h.
//
// The shadow only works because the JIT command line lists `-I .../jit_hw`
// before any tt-metal include path. If that ordering ever changes, the
// upstream pack.h would resolve instead, ODR-conflicting with common.h's
// pack_tile.  Upstream does not expose a sentinel macro we can probe, so this
// invariant is documented rather than asserted.
