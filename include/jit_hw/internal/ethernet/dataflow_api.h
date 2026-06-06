// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Stub of the silicon ethernet dataflow API. Real silicon
// provides ethernet-specific NOC + erisc helpers. emule has no ethernet
// model — short-circuit the include chain here so upstream ops that pull in
// fabric headers (which transitively include this) parse cleanly.
//
// Add real symbols only when a upstream kernel actually consumes ethernet
// primitives (none of the dsa/cb_reconfig/argmax seed-test family do).

#include <cstdint>
