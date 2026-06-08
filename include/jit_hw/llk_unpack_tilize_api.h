// SPDX-License-Identifier: Apache-2.0
// Emule shadow for the silicon LLK unpack tilize API.
// Included by upstream custom_tilize.h under #ifdef TRISC_UNPACK (set by emule's
// program runner for non-Quasar compute kernels). Forwards to the existing
// `llk_unpack_a.h` shim which provides `llk_unpack_tilize_init` and
// `llk_unpack_tilize_block` stubs.
#pragma once

#include "llk_unpack_a.h"
