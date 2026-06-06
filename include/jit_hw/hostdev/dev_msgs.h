// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Stub of the silicon host-device message headers. The real file declares host-device
// mailbox layouts (launch_msg_t, watcher_msg_t, profiler_msg_buffer_t, etc.)
// gated behind #ifdef KERNEL_BUILD || FW_BUILD || HAL_BUILD, and #errors if
// none are defined. emule kernels reach this file transitively via
// silicon fabric/ethernet headers but don't actually consume its types —
// shadow with this empty stub so the silicon version is never parsed.
//
// Add struct/enum stubs only if a upstream kernel references them directly.

#include <cstdint>
