// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

// TLS variables set by the kernel runner before kernel launch.
extern thread_local uint32_t __emule_num_threads;
extern thread_local uint32_t __emule_my_thread_id;

inline uint32_t get_num_threads() { return __emule_num_threads; }
inline uint32_t get_my_thread_id() { return __emule_my_thread_id; }
