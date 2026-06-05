// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Stub of the silicon non-blocking NOC API. Real silicon defines non-blocking
// NOC primitives. emule's NOC ops are all synchronous memcpy via the
// __emule_resolve_noc_addr bridge — no non-blocking surface needed.
// Provide minimal stubs only if a upstream kernel references them by name.

#include <cstdint>
