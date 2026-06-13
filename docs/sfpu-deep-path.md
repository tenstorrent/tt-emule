# Deep-SFPU path

emule's default SFPU strategy is **layer-1**: each `<op>_tile` is a hand-written
scalar shim (libm) under `include/jit_hw/api/compute/eltwise_unary/`. The
**deep-SFPU path** is an optional, deeper alternative that compiles and runs the
**real silicon** `ckernel_sfpu_<op>.h` math on emule's faithful `sfpi` backend
(`include/jit_hw/sfpi.h`), giving silicon-representative numerics and covering
sfpi-written ops that have no layer-1 shadow.

## Why it works without a runner change

The JIT compile (`tt_metal/impl/emulation/emulated_program_runner.cpp`) sets the
include search order:

1. `include/jit_hw` (first)
2. `include/`
3. the kernel's own directory
4. extras: `ttnn/cpp`, **the tt-metal source root**, `tt_metal/hw/inc`, `hostdevcommon/api`

Two consequences make the deep path pure-header:

- Because `jit_hw` is first, a real ckernel header's `#include "sfpi.h"` still
  resolves to **emule's** `sfpi.h` (the real `runtime/sfpi/include` is *not* on
  the path). So the real math runs on emule's scalar-per-lane backend.
- Because the tt-metal **source root** is on the path, a deep shim reaches the
  real op via a root-relative include, e.g.
  `#include "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sqrt.h"`.

## The sfpi backend (`include/jit_hw/sfpi.h`)

A real `_calculate_<op>_` iterates `for (d<iters) { v=dst_reg[0]; …; dst_reg[0]=…; dst_reg++; }`
over 32-lane windows of the DST tile. emule models each `vFloat`/`vInt`/`vUInt`
as a 32-lane `std::array` and steps a thread-local cursor.

Load-bearing semantics (validated against silicon behavior):

- **Lane-masked register assignment** — `vFloat`/`vInt`/`vUInt` `operator=` blends
  against the active `v_if` lane mask (`__emule_sfpi_mask`). On silicon an sfpi
  local *is* a register, so a write inside `v_if(...)` only updates active lanes
  (e.g. sqrt's final `v_if(x<0){ y=NaN; }` must not touch positive lanes). This is
  the substance of the old `TODO(#102)`.
- **Programmable constants** — `vConstFloatPrgm0..2` / `vConstIntPrgm0..2` alias the
  same three slots (`__emule_prgm_creg[3]`); set in an op's `_init_`, read in its
  `_calculate_`.
- **`int32_to_float` is sign-magnitude** (SFPCAST IntFloat), not two's-complement —
  callers pass sign-magnitude (e.g. `setsgn(~exp+1, 1)` in log).
- Field ops: `exexp`/`setexp`/`addexp`, `sFloat16a`/`sFloat16b` + `convert<vFloat16b>`
  (bf16 narrowing on the store boundary).

## Engaging the deep path

Default = layer-1 shadow (byte-identical baseline). To promote a shadowed op,
define `EMULE_DEEP_SFPU_<OP>` (e.g. via the JIT define set, so it lands in the
cache key). The op's `eltwise_unary/<op>.h` carries a guarded branch whose
*else*-branch is the untouched layer-1 default; the deep branch delegates to the
real silicon calculate via `__emule_deep::run_unary_sfpu` (`internal/deep_sfpu.h`),
which points the sfpi cursor at `__emule_dst[idst]` for the call. Policy lives in
`api/compute/eltwise_unary/deep_sfpu_registry.h`. Ops with **no** shadow engage the
deep path automatically (planned: deep arm of `sfpu_split_includes.h`).

`sqrt`, `silu`, `sigmoid`, and `tanh` are wired as reference overrides
(`EMULE_DEEP_SFPU_SQRT` / `_SILU` / `_SIGMOID` / `_TANH`).

## LUT-based ops (sigmoid / tanh)

`tanh` and `sigmoid` are piecewise-linear LUT ops. Their `_init_` loads packed
coefficient bit-patterns into LRegs via `SFPLOADI` (through
`ckernel_sfpu_load_config.h` → `ckernel_ops.h`), then `_calculate_` calls
`lut()` (3-entry `SFPLUT`) / `lut2()` (6-entry `SFPLUTFP32`) which read those
LRegs. Emule provides:

- `sfpi.h`: `l_reg[LRegN]` backed by 32-lane `vUInt __emule_lreg[16]`, the
  `__emule_sfploadi` writer, and faithful `lut`/`lut2` evaluators
  (`__lut8_to_fp32` / `__lut16_to_fp32` decoders, `Abs(x)` range buckets, and
  `VD = a·Abs(x) + c` with sign-retain) — exact per tt-isa-documentation.
- `jit_hw/ckernel_ops.h`: a shim defining the `TT_SFPLOADI`/`TTI_SFPLOADI`
  (→ `__emule_sfploadi`) and `TTI_SFPCONFIG` (no-op) macros; the real
  `ckernel_ops.h` emits raw Tensix instructions and can't compile on x86.

Note: the silicon `tanh` LUT is coarse (3 pieces; ~0.14 abs error near |x|=1) —
the deep path reproduces **silicon**, not torch, so PCC-vs-torch reflects that.

## Status / verification

- Real `ckernel_sfpu_{sqrt,log,silu,sigmoid,tanh}.h` compile on clang-20 against
  emule `sfpi.h` and run correctly (PCC vs torch): **sqrt → exact**; **log →
  0.999999** (silicon 3rd-order Chebyshev, incl. sub-1 inputs via the
  sign-magnitude exponent path); **silu → 0.999992**; **sigmoid → 0.999940**
  (6-piece SFPLUTFP32); **tanh → 0.997499** (coarse 3-piece SFPLUT — silicon's
  own accuracy, faithfully reproduced).
- Blackhole **19/0** and Wormhole **39/0** regressions are unchanged from baseline
  after the `sfpi.h` model change + deep-path wiring (the masked-assignment change
  did not regress the existing sfpi ops: clamped_silu, topk, …).

## Arch note

Tier-1 is Blackhole-first. Wormhole reuses the same mechanism via
`tt_llk_wormhole_b0` paths; Quasar later. Tier-2 (LUT transcendentals — exp/gelu/
sigmoid via `SFPLUTFP32`) and centralized fidelity knobs (partially-fused FMA,
DEST-format-aware store truncation) are the next deep-path increments.
