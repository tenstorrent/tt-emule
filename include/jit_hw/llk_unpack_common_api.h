// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Stub of the silicon LLK unpack common API. Real silicon defines unpack
// helpers used across TRISCs. emule's llk_unpack_a.h provides the function
// surface; this file exists so silicon include paths resolve. Add helpers
// only if a upstream kernel references them directly.

#include "jit_hw/llk_unpack_a.h"
