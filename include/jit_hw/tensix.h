// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Stub of the silicon TENSIX core header. Real silicon defines TENSIX hardware
// register layout, RISC core IDs, semaphore indices, etc. emule's
// compute model is single-thread per core (no separate Tensix RISCs);
// add constants here only as upstream kernels require them.
