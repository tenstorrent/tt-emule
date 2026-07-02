// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Header-defined JIT state shared by CB operation checks and L1 boundary checks.
inline thread_local const char* __emule_cb_reserve_file[32] = {};
inline thread_local uint32_t __emule_cb_reserve_line[32] = {};
inline thread_local const char* __emule_cb_wait_file[32] = {};
inline thread_local uint32_t __emule_cb_wait_line[32] = {};
inline thread_local uint32_t __emule_cb_reserved_pages[32] = {};
inline thread_local uint32_t __emule_cb_waited_pages[32] = {};
inline thread_local bool __emule_cb_reserve_dangling[32] = {};
inline thread_local bool __emule_cb_wait_dangling[32] = {};
inline thread_local bool __emule_cb_boundary_strict = false;
