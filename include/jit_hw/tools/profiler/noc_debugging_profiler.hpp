// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT emulation stub for tools/profiler/noc_debugging_profiler.hpp.
//
// The real header only defines RECORD_SCOPED_LOCK_EVENT when DEVICE_DEBUG_DUMP
// is set (hardware profiling path).  In emulation, always use the no-op.

#ifndef RECORD_SCOPED_LOCK_EVENT
#define RECORD_SCOPED_LOCK_EVENT(event_type, locked_address_base, num_bytes) ((void)0)
#endif
