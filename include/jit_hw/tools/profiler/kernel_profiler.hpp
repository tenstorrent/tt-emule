// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// No-op kernel profiler stub for emulated mode
#define DeviceZoneScopedN(...)
#define DeviceZoneScoped(...)
#define DeviceZoneScopedMainN(...)
#define DeviceZoneScopedMainChildN(...)
#define DeviceTimestampedData(...)
namespace kernel_profiler {
inline void mark_time(unsigned = 0) {}
inline void mark_padding() {}
inline void set_host_counter(unsigned) {}
inline void init_profiler(unsigned = 0, unsigned = 0) {}
inline void finish_profiler() {}
} // namespace kernel_profiler
