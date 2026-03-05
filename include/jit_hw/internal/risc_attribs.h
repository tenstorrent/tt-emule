#pragma once
// On the real RISC-V target these control memory qualifiers and inlining.
// In the JIT emulation host build, they are no-ops.
#define tt_l1_ptr
#define FORCE_INLINE inline __attribute__((always_inline))
