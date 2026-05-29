// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Redirect shim for upstream `api/compute/cb_api.h`, which references TRISC-only
// LLK primitives. Pull in emule's host-side CB ops + `ALWI` macro instead.
#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/cb_api.h"
