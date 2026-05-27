// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT emulation stub for tools/profiler/noc_debugging_metadata.hpp.
//
// The real header is at tt_metal/tools/profiler/ which is not on the JIT
// include path.  RECORD_SCOPED_LOCK_EVENT is a no-op in emulation (see
// noc_debugging_profiler.hpp stub), so NocDebuggingEventMetadata is never
// referenced in compiled code — this empty stub satisfies the #include.
