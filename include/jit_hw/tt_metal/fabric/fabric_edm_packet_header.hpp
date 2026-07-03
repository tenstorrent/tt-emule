// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Shadow the tt_metal/-prefixed packet-header include path too (the unprefixed
// fabric/ path is already shadowed). Both must resolve to the emule fabric stub so
// the real packet header (with its HW deps + host-path TT_THROW) is never pulled in
// and the emule decodable-layout PacketHeader is the single definition everywhere.
#include "jit_hw/__emule_fabric_stubs.h"
