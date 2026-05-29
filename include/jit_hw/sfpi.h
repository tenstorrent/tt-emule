// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for sfpi.h — RISC-V SFPU vector-ops DSL header.
//
// Real sfpi.h is RISC-V-only (uses inline asm + ckernel::instrn_buffer).
// SFPU primitive headers (ckernel_sfpu_*.h in tt-llk) chain through it to
// emit SFPU instructions. In emule we don't execute SFPU ops via sfpi —
// we override at the SFPU-op level (ckernel::sfpu_*) with C++ math in
// jit_hw/api/compute/eltwise_unary/*.h.
//
// Providing this empty shim lets the include chain succeed without
// pulling in the RISC-V-specific bits. The downstream ckernel_sfpu_*.h
// headers may still reference symbols sfpi defines — those are caught
// at use site (kernels needing real SFPU semantics must route through
// emule's existing SFPU shim layer).

#include <cstdint>

// Minimal sfpi namespace skeleton — empty types so includes that reference
// `sfpi::vFloat` etc parse, even though emule never executes sfpi code.
namespace sfpi {
struct vFloat {};
struct vInt {};
struct vUInt {};
struct vCond {};

template <typename T>
struct vector { T v; };

// Common sfpi helpers — declared but never called in emule path.
inline void v_endif() {}
inline void v_else() {}
}  // namespace sfpi

// Macros sfpi.h normally defines for kernel-side use; provide empty
// expansions so kernels that reference them parse.
#define v_if(...) if (false)
#define v_elseif(...) else if (false)
#define v_else else
#define v_endif
