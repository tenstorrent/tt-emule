---
name: llk-bringup
description: Use when bringing up an emule LLK shim — write a new mock implementation of an upstream tt-metal compute-kernel API under `include/jit_hw/api/compute/`, wire it into the regression suite, and address PCC/divergence failures. Distilled from rounds 1–5 of the bring-up campaign on `arminale/mass-llk-bringup`.
---

# LLK bring-up — emule shim authoring skill

## When to invoke

User says one of:
- "bring up an LLK" / "bring up the X op"
- "write a shim for X" / "add an emule shim for X"
- "address the PCC failures on X"
- "implement the missing X SFPU op"

Or you encounter a JIT compile error of the form
`jit_compile_kernel: compiler failed (exit 256) for kernel: ...`
where the underlying clang error is a missing `<name>_tile` symbol or an
`api/compute/<name>.h` not-found.

## The bring-up loop

For each target op:

1. **Triage: does the shim already exist?**
2. **Identify the target path.**
3. **Read upstream — lock signatures and semantics.**
4. **Write the shim using the standard pattern.**
5. **Wire it in (if needed).**
6. **Rebuild + run targeted tests.**
7. **Promote passes into `scripts/run_ttnn_pytests.sh`. Commit.**
8. **PCC failures: investigate up to 3 iterations, then defer.**

## Step 1 — Triage

Before writing anything, check whether the op is already covered. Many "missing"
ops are in shared shims:

- **`include/jit_hw/api/compute/eltwise_unary/activations.h`** — abs, celu,
  hardshrink, softshrink, hardsigmoid, softsign. If the op is here, the
  per-op file is redundant and will cause an ODR conflict — don't write it.
- **`include/jit_hw/api/compute/compute_kernel_api.h`** — sign, signbit,
  heaviside, sigmoid, silu, exp2, expm1, log, square, power, topk (stub).
- **`include/jit_hw/api/compute/eltwise_unary/relu.h`** — relu, relu_max,
  relu_min, leaky_relu (the RELU_FAMILY).
- **`include/jit_hw/api/compute/eltwise_unary/sfpu_split_includes.h`** — the
  wiring file: lists which `SFPU_OP_<NAME>_INCLUDE` branches are active.
  As of round 5, 23 of upstream's 51 branches are wired.

If your op is already there, stop. You'd be duplicating.

## Step 2 — Target path

Shims **mirror upstream paths** under `tt_metal/hw/inc/api/compute/`:

| Upstream | Emule target |
|---|---|
| `tt_metal/hw/inc/api/compute/eltwise_unary/<name>.h` | `include/jit_hw/api/compute/eltwise_unary/<name>.h` |
| `tt_metal/hw/inc/api/compute/<name>.h` (fused / multi-tile) | `include/jit_hw/api/compute/<name>.h` |
| Functions in upstream `activations.h` | Add to emule `eltwise_unary/activations.h` in-place |
| Functions in upstream `compute_kernel_api.h` | Add to emule `compute_kernel_api.h` in-place |

**Don't invent paths.** If upstream has no header for the op (e.g. hardswish —
composed in-kernel from primitives), no shim is needed. Verify the ttnn op
handler decomposes it via `SFPU_OP_CHAIN` rather than calling a standalone
`<name>_tile`.

## Step 3 — Read upstream

Three references per op, in order of priority:

1. **Upstream header** at the matching path. Lock the signature(s) verbatim —
   parameter names, types, defaulted args.
2. **LLK implementation** at `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_<name>.h`
   (or, for the actual numerics, the SFPU code at
   `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_<name>.h`).
   This is the source of truth for the **math formula** — polynomial
   coefficients, Cody-Waite reductions, saturation branches.
3. **A reference emule shim** for the boilerplate. Good references:
   - `eltwise_unary/relu.h` — simplest per-element shim
   - `eltwise_unary/dropout.h` — pattern with thread_local state
   - `eltwise_unary/bitwise_and.h` — int32 DST via `__emule_dst_load_i32`
   - `transpose_wh.h` — temp-buffer pattern for in-place transforms

**Hard input cap: ~3 files.** If you find yourself reading 4+ host-side files
or wandering into `tt_metal/llrt/`, `tt_metal/impl/dispatch/`, or
`tt_metal/soc_descriptors/` — stop and reclassify the task.

## Step 4 — The shim template

```cpp
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/<path>.h`. Brief one-line description.
// Real LLK reference:
//   tt_metal/hw/inc/api/compute/<path>.h
//   tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_<name>.h

#include <cstdint>
#include <cmath>     // if you use std::exp, std::log, etc.
#include <cstring>   // if you std::memcpy uint32 fp32 bit-patterns

#include "jit_hw/api/compute/common.h"  // provides ALWI, __emule_dst, __EMULE_TILE_ELEMS, __emule_dst_check

namespace ckernel {

ALWI void <name>_tile_init(/* upstream args */) {}

ALWI void <name>_tile(uint32_t idst, /* upstream args */) {
    __emule_dst_check(idst, "<name>_tile");
    // Decode fp32 bit-pattern params if any:
    //   float param; std::memcpy(&param, &uint_param, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_dst[idst][i] = /* math on __emule_dst[idst][i] */;
    }
}

}  // namespace ckernel
```

### Variants

- **int32 DST ops** (bitwise, shift, quantization int paths):
  ```cpp
  for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
      int32_t v = __emule_dst_load_i32(idst, i);
      v = /* int32 op */;
      __emule_dst_store_i32(idst, i, v);
  }
  ```
  See `bitwise_and.h` or `relu_tile_int32` in `relu.h`.

- **Stateful (cumsum, dropout)**:
  ```cpp
  static thread_local <type> __emule_<name>_state = ...;
  ALWI void <name>_tile_init(uint32_t seed) {
      __emule_<name>_state = ...;
  }
  ```
  See `dropout.h` (xorshift32 PRNG) and `cumsum.h` (per-column accumulator).

- **Multi-tile (mask, welford)**:
  ```cpp
  ALWI void <name>_tile(uint32_t idst_a, uint32_t idst_b, ...) {
      for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
          // Read both __emule_dst[idst_a][i] and __emule_dst[idst_b][i].
      }
  }
  ```

- **In-place mutation requiring temp buffer (reshuffle, transpose)**:
  ```cpp
  alignas(16) float tmp[__EMULE_TILE_ELEMS];
  std::memcpy(tmp, __emule_dst[idst], sizeof(tmp));
  // ...compute from tmp into __emule_dst[idst]...
  ```
  See `transpose_wh.h`.

## Step 5 — Wire-up

After writing the shim, check if upstream has a corresponding `SFPU_OP_<NAME>_INCLUDE`
in `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`. If yes,
add the same branch to emule's `sfpu_split_includes.h`:

```cpp
#if SFPU_OP_<NAME>_INCLUDE
#include "api/compute/eltwise_unary/<name>.h"  // or compute/<name>.h
#endif
```

If the upstream path is direct (no SFPU_OP gate) — e.g. `api/compute/softmax.h`,
`api/compute/cumsum.h`, `api/compute/welford.h` — no wiring needed; the JIT
include path resolves the emule version automatically when the kernel does
`#include "api/compute/<name>.h"`.

## Step 6 — Build + test

```bash
TT_EMULE_DIR=/localdev/arminale/tt-emule TT_METAL_DIR=/localdev/arminale/tt-metal \
    bash /localdev/arminale/tt-emule/.github/scripts/ci-build.sh
```

Then run the targeted pytest. Map ops to test files:

| Op family | Test file |
|---|---|
| activations (hardtanh, mish, threshold, swish, gelu, …) | `tests/ttnn/unit_tests/operations/eltwise/test_activation.py::test_<name>` |
| math (cbrt, i0, i1, erfinv, digamma, polygamma, lgamma) | `tests/ttnn/unit_tests/operations/eltwise/test_math.py::test_<name>` |
| elu, celu | `test_elu.py`, `test_celu_21f.py` |
| binary scalar (fmod, remainder, rsub) | `test_binary_composite.py`, `test_binaryng_fp32.py` |
| bitwise | `test_binary_int32.py` |
| reduce (cumsum, cumprod, mean, min, sum) | `tests/ttnn/unit_tests/operations/reduce/test_<name>.py` |
| fused (softmax, layernorm) | `tests/ttnn/unit_tests/operations/fused/test_<name>.py` |

Run with:
```bash
cd /localdev/arminale/tt-metal && rm -rf /tmp/tt_emule_jit_cache_$(id -u) && \
    PYTHONPATH=$PWD/ttnn:$PWD/tools:$PWD/build_emule/lib:$PWD \
    LD_LIBRARY_PATH=$PWD/build_emule/lib \
    TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD \
    TT_METAL_EMULE_MODE=1 TT_METAL_SLOW_DISPATCH_MODE=1 \
    TT_METAL_MOCK_CLUSTER_DESC_PATH=$PWD/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml \
    MESH_DEVICE=N150 \
    /opt/ttmlir-toolchain/venv/bin/pytest <test_path> --forked --tb=line -q
```

## Step 7 — Promote into the regression script

Edit `scripts/run_ttnn_pytests.sh`. Add one `run_pytest` line per cleanly-passing
function. Place near the relevant family of existing entries.

```bash
run_pytest "elt_test_<name>" "$ELT_TEST_DIR/test_<group>.py::test_<name>"
```

Or with a parametrize filter:

```bash
run_pytest "elt_test_<name>" "$ELT_TEST_DIR/test_<group>.py::test_<name>" -k 'not sharded'
```

Caveats on `pytest -k`:
- Does NOT tokenize `=` or `::` — `K=128` won't parse as a single identifier.
- DOES tokenize plain alphanumeric+underscore substrings.
- For `--deselect`, use **rootdir-relative** node IDs (`tests/...`), not
  absolute paths. Absolute paths silently fail to deselect.

Commit per-bring-up:
```bash
git add include/jit_hw/<files> scripts/run_ttnn_pytests.sh
git commit -m "llk-bringup: <one-line summary> (N/N)"
git push origin arminale/mass-llk-bringup
```

## Step 8 — PCC failure triage

Hard cap: **3 iterations per shim**. Past that, defer with a written note.

When PCC fails, check in order:

0. **Verify upstream signatures literally before anything else.** Open the
   upstream header (`tt_metal/hw/inc/api/compute/eltwise_unary/<name>.h`) and
   compare the `<name>_tile` signature byte-for-byte against the emule shim.
   Sub-agents' inline notes about parameter ordering can be wrong; the
   committed upstream code is the source of truth.

1. **Polynomial approximation gap**. `<cmath>` ≠ upstream LLK. Read
   `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_<name>.h`
   for the actual coefficients. Port them.
2. **Parameter ordering**. Compare your shim's args against the upstream
   `ALWI void <name>_tile(...)` declaration in
   `tt_metal/hw/inc/api/compute/eltwise_unary/<name>.h`. The agent's prompt
   hint may be wrong — upstream is the source of truth.
3. **Stateful accumulator direction**. For cumsum/cumprod/welford, check
   upstream's tile-row order convention (N×W×H per
   `compute/cumsum.h:18-19`). The `first` flag in cumsum applies per
   (N, W) column-of-tiles, not per-tile.
4. **Layout-specific issue**. ROW_MAJOR vs TILE vs bf16 vs fp32 can hit
   different reader kernel paths. If only some dtypes fail, look at the
   reader kernel for that dtype's branch.

If none of those: defer with a one-paragraph note in the test entry's comment
in `scripts/run_ttnn_pytests.sh` and move on. The PCC-fix work goes into a
dedicated round of its own.

## Parallel sub-agent dispatch (proven in round 5)

When bringing up many shims at once (≥4), use the `Workflow` tool with parallel
`agent()` calls. Each agent writes one file in isolation.

### Worker prompt skeleton

```
You are implementing an emule LLK shim for tt-emule. Write exactly ONE header file.

TARGET FILE: /localdev/arminale/tt-emule/include/jit_hw/api/compute/<path>.h
UPSTREAM REFERENCE (read-only): /localdev/arminale/tt-metal/tt_metal/hw/inc/api/compute/<path>.h
LLK REF (read-only, only if upstream signatures unclear): <ckernel_sfpu path>
REFERENCE EMULE SHIMS (read for boilerplate):
  /localdev/arminale/tt-emule/include/jit_hw/api/compute/eltwise_unary/relu.h
  /localdev/arminale/tt-emule/include/jit_hw/api/compute/eltwise_unary/dropout.h

REQUIRED SIGNATURES (mirror upstream — if upstream differs, follow upstream):
<list>

SEMANTICS (per element):
<formula>

HARD CONSTRAINTS:
- Do NOT read tt_metal/llrt/, tt_metal/impl/dispatch/, tt_metal/soc_descriptors/.
- Do NOT modify any file other than the TARGET FILE.
- Do NOT touch sfpu_split_includes.h.
- Do NOT run pytest, build, or commit.
- Read at most 3 files: upstream header, LLK impl (optional), one reference shim.

OUTPUT: write the file, return status=DONE with file=<relative path> and lines=<count>.
If signature is ambiguous or upstream missing, return status=STUCK with one-sentence reason.
Do NOT invent signatures.
```

### Result schema

```js
const SHIM_RESULT_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['DONE', 'STUCK'] },
    file: { type: 'string' },
    lines: { type: 'integer' },
    reason: { type: 'string' },
  },
  required: ['status', 'file'],
  additionalProperties: false,
}
```

### Orchestration

```js
phase('Write shims')
const results = await parallel(SHIMS.map(s => () =>
  agent(makePrompt(s), {
    label: `shim:${s.name}`,
    phase: 'Write shims',
    schema: SHIM_RESULT_SCHEMA,
  }).then(r => ({ shim: s.name, ...(r || {status: 'NULL'}) }))
))
const done = results.filter(r => r.status === 'DONE')
const stuck = results.filter(r => r.status !== 'DONE')
log(`${done.length}/${SHIMS.length} DONE; ${stuck.length} STUCK`)
return { done, stuck }
```

After agents return, **orchestrator** (you) handles centrally:
1. Visual diff each new file.
2. Wire SFPU_OP_*_INCLUDE branches.
3. Rebuild.
4. Run targeted tests for each op (also parallel-able with one pytest call per op).
5. Promote 100%-pass functions into the regression script.
6. For partial passes, decide: pytest `-k` filter, or defer.

## Composed ops (no standalone upstream `<name>_tile`)

Some activation-style ops are not standalone SFPU primitives — they're
composed in-kernel from underlying primitives. If a sub-agent returns STUCK
because "upstream has no `<name>_tile`," that's correct behavior. The
upstream op chain typically looks like:

| Op | Composition | Kernel source |
|---|---|---|
| `hardswish` | `hardsigmoid_tile + mul_binary_tile` | `ttnn/.../unary/device/kernels/compute/hardswish_kernel.cpp` |
| `tanhshrink` | `tanh_tile + sub_binary_tile` | `tanhshrink_kernel.cpp` |
| `swish` | alias for `silu_tile` (primitive, not a composition) | n/a |

**Diagnosing a composed-op failure:** when the JIT compile fails on
`<name>_kernel.cpp`, check whether the kernel calls into primitives we
already have. If yes, the gap is in the primitive (or in
`compute_kernel_api.h`'s coverage of it) — not a missing `<name>_tile`.

**Worked example (round 6 Cat F):** `tanhshrink_kernel.cpp` called
`tanh_tile()`. emule's `eltwise_unary/trigonometry.h` had `tanh_tile`, but the
kernel only included `compute_kernel_api.h` (not trigonometry.h), and emule's
`compute_kernel_api.h` was missing `tanh_tile`. Fix: add `tanh_tile` to
`compute_kernel_api.h` (upstream defines it there as a catch-all).

## Porting upstream polynomials (PCC triage)

When `<cmath>` numerics drift past atol/ULP tolerance, port the exact polynomial
form from `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_<name>.h`.

**Process:**
1. Open the upstream `ckernel_sfpu_<name>.h`. Identify regions (e.g. range
   split, saturation thresholds), coefficient arrays (Horner-form polynomial),
   and boundary branches.
2. Copy the coefficient constants **verbatim** as `constexpr float`. Don't
   round, don't reformat.
3. Translate the SFPI vector eval (`sfpi::vFloat`, `v_if/v_endif`) to scalar
   per-element math: `if (cond) {…} else {…}` inside the `__EMULE_TILE_ELEMS`
   loop.
4. Use only `<cmath>` builtins (std::exp, std::log, std::sqrt, std::fabs,
   std::nearbyint, std::ldexp, std::copysign). No platform intrinsics.

**Worked example (round 6 Cat B, i1):** replaced `std::cyl_bessel_i(1, x)` with
two-region rational `p(t)/q(t)` (|x|≤10) plus asymptotic `exp(|x|)/sqrt(|x|)·P(1/|x|)`
(|x|>10). 7-coefficient numerator + 8-coefficient denominator + 6-coefficient
asymptotic minimax, all from `ckernel_sfpu_i1.h:62-126`. Result: 0/12 fail → 14/14 PASS.

**Caveat (round 6 Cat B, celu):** I attempted to port the Cody-Waite expm1 from
`ckernel_sfpu_expm1_cw.h`. The implementation was numerically off enough to
regress test_celu_allclose / test_celu_arange — the simpler `std::exp(x*ar)-1`
form had been passing those. Reverted. Lessons:

- **A polynomial port is not guaranteed to pass PCC.** If your port regresses
  cases that were passing, the new form has a bug or differs from upstream
  somewhere subtle (e.g. wrong reconstruction step). Revert before chasing.
- **Don't replace a working simple form unless tests are demonstrably failing
  on it.** celu PCC was already at the boundary; the new form moved it the
  wrong way.

## Stateful ops & `first` flag — confirm the shim is even used

A shim file's existence doesn't imply ttnn uses it. The host op may
preprocess inputs (permute, reshape) so a *different* kernel chain runs.

**Worked example (round 6 Cat D):** `cumsum.h` defines `cumsum_tile(idst,
first)` matching upstream. The actual `ttnn.cumsum` op:

1. Host-side: permutes the cumsum axis to dim 0
   (`accumulation_common.cpp:43-50`), reshapes to 4D, then sends to the
   compute kernel.
2. Compute kernel `accumulation_compute.cpp` doesn't call `cumsum_tile` —
   it uses `add_binary_tile(DST_IN, DST_ACC, DST_ACC)` per tile in a
   cross-tile loop. The shim file was never reached.
3. PCC failures came from the wider chain (likely permute or tile-padding
   handling on small shapes), not from `cumsum_tile`.

**Diagnostic:** before debugging a stateful-op shim's math, dump the JIT
wrapper (`TT_EMULE_KEEP_JIT_SRC=1`), grep the kept `patched_kernel.cpp`
for the shim function name. If it's not there, the failure is upstream
of your shim — chase the actual primitives the kernel does call.

For ops that DO use stateful per-tile primitives (like `dropout_tile` —
PRNG state survives across calls), the `thread_local` accumulator pattern
from `dropout.h` is correct. But verify before assuming.

## Anti-patterns (consolidated)

1. **Don't read host code.** `tt_metal/llrt/`, `tt_metal/impl/dispatch/`,
   `tt_metal/soc_descriptors/` — none of these are relevant to a shim. If you
   find yourself there, the failure isn't a shim gap.
2. **Don't mock TensorAccessor internals** from inside a per-op shim. Sharded
   support is a cross-cutting effort (separate round, see plan files).
3. **Don't accumulate JIT cache + restart** as a substitute for understanding
   the failure. Each `rm -rf /tmp/tt_emule_jit_cache_$(id -u)` should be
   deliberate (e.g. after a shim edit), not a cargo-cult retry.
4. **Don't duplicate shims** that `activations.h` or `compute_kernel_api.h`
   already define. ODR conflicts result.
5. **Don't invent upstream signatures.** If upstream has no header, return
   STUCK or ask. Hardswish/swish/tanhshrink have no standalone upstream API —
   they're composed in-kernel from primitives.
6. **Don't iterate PCC failures more than a few times.** 3 attempts max per
   shim, then document and move on.
7. **Don't have parallel sub-agents touch shared files.** Each agent gets a
   unique path. The orchestrator handles `sfpu_split_includes.h` and other
   centralized wiring after agents return.

## References

- Plan: `/home/arminale/.claude/plans/vivid-foraging-nebula.md` — current
  campaign status, target manifest, failure catalog.
- Failure categories: see plan's "Failure catalog" section. A/B/C/D/E/F are
  the actionable ones; G/H/I/J/K are deferred.
- Round 5 outcome: 30 new functional LLK shims via parallel agents, ~162 new
  pytest cases promoted across 27 functions.
