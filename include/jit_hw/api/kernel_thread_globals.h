// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "jit_hw/internal/emule_thread_ctx.h"

inline uint32_t get_num_threads() { return __emule_self->num_threads; }
inline uint32_t get_my_thread_id() { return __emule_self->my_thread_id; }
