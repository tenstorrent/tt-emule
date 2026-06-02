// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Emule shim for upstream tt_metal/hw/inc/api/compute/softmax.h
// Pure re-export header: softmax composition happens elsewhere; this
// header just gathers exp + recip declarations.

#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
