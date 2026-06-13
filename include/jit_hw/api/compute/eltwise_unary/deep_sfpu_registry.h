// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Per-op deep-SFPU override registry.
//
// Policy (matches the deep-SFPU plan):
//   * Default: an op that HAS an emule layer-1 shadow (eltwise_unary/<op>.h)
//     keeps using that shadow — the shadowed baseline stays byte-identical.
//   * Auto-engage: an op with NO shadow routes to the deep path automatically
//     (real silicon ckernel_sfpu_<op>.h on the sfpi backend) via the deep arm
//     of sfpu_split_includes.h.
//   * Explicit override: define EMULE_DEEP_SFPU_<OP> (typically via a JIT -D /
//     the runner's defines map, so it lands in the JIT cache key) to promote a
//     shadowed op to the deep path. Its eltwise_unary/<op>.h carries a guarded
//     #ifdef branch whose else-branch is the untouched layer-1 default.
//
// This header intentionally defines NOTHING by default; overrides come from the
// build/JIT define set. It exists as the single documented home for the policy
// and as a stable include point for the shimmed ops' guarded branches.
//
// Current state:
//   * sqrt           — DEEP-DEFAULT (migrated; no libm shadow, no toggle —
//                      eltwise_unary/sqrt.h runs the real silicon SQRT_23). This
//                      is the shape an unshadowed op takes via the deep bridge.
//   * silu/sigmoid/tanh — toggle-gated overrides (EMULE_DEEP_SFPU_<OP> via the
//                      TT_EMULE_DEEP_SFPU env var); default stays the libm shadow.
