// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
// emule shadow: the real fabric mux/EDM-channel headers model the ERISC data-mover
// (eth-core hardware) which emule does not run. The fabric client API is shimmed in
// __emule_fabric_stubs.h (teleport), so these forward there.
#include "jit_hw/__emule_fabric_stubs.h"
