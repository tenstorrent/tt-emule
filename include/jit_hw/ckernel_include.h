#pragma once
// Stub for upstream tt_metal/tt-llk/<arch>/common/inc/ckernel_include.h.
// Provides the firmware-msg enum referenced by some kernels; LLK headers it
// chains into upstream are not needed in emule's JIT path.

#include "ckernel_defs.h"

namespace ckernel {

enum firmware_msg_e {
    FLIP_STATE_ID        = 1,
    RUN_INSTRUCTIONS     = 2,
    RESET_DEST_OFFSET_ID = 3,
};

}  // namespace ckernel
