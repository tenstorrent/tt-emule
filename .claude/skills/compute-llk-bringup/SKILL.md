---
name: compute-llk-bringup
description: Specialization of `/implement-mock` for compute-kernel LLK shims (`<op>_tile` / `<op>_tile_init` under `include/jit_hw/api/compute/`). Use when bringing up a missing compute op or addressing its PCC failures.
---

# Compute LLK bring-up

This skill is a **specialization** of `/implement-mock` — specifically,
Strategy A (stub in `jit_hw/`) applied to compute-kernel ops. The
mechanics here all assume the silicon API lives at
`tt_metal/hw/inc/api/compute/<path>.h` and produces a header-only shim
under `include/jit_hw/api/compute/`.

For mock work outside compute (NOC, dataflow, dispatch, fabric, tensor),
use `/implement-mock` for the broader strategy-picking flow. For batch
bring-up (≥4 shims at once), see `/parallel-mock-implementation`.

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
8. **PCC failures: investigate up to ~3 iterations, then defer.**

## Step 1 — Triage

Before writing anything, check whether the op is already covered. Many "missing"
ops are in shared shims. **`STRUCTURE.md` is the canonical file-level index**
— grep it for the op name to see whether a file already defines `<op>_tile`.

Common shared homes (not exhaustive):

- **`include/jit_hw/api/compute/eltwise_unary/activations.h`** — small,
  closely-related per-element ops grouped together. If the op is here, a
  separate per-op file is redundant and will cause an ODR conflict.
- **`include/jit_hw/api/compute/compute_kernel_api.h`** — single-tile fused
  ops that upstream's compute_kernel_api.h exposes as catch-alls (sign,
  sigmoid, silu, exp2, expm1, log, square, power, topk stub, etc.).
- **`include/jit_hw/api/compute/eltwise_unary/relu.h`** — the RELU_FAMILY
  (relu, relu_max, relu_min, leaky_relu) shares one file.
- **`include/jit_hw/api/compute/eltwise_unary/sfpu_split_includes.h`** —
  the wiring file: lists which `SFPU_OP_<NAME>_INCLUDE` branches are active.

If your op is already covered, stop. You'd be duplicating.

## Step 2 — Target path

Shims **mirror upstream paths** under `tt_metal/hw/inc/api/compute/`:

| Upstream | Emule target |
|---|---|
| `tt_metal/hw/inc/api/compute/eltwise_unary/<name>.h` | `include/jit_hw/api/compute/eltwise_unary/<name>.h` |
| `tt_metal/hw/inc/api/compute/<name>.h` (fused / multi-tile) | `include/jit_hw/api/compute/<name>.h` |
| Functions in upstream `activations.h` | Add to emule `eltwise_unary/activations.h` in place |
| Functions in upstream `compute_kernel_api.h` | Add to emule `compute_kernel_api.h` in place |

**Don't invent paths.** If upstream has no header for the op (e.g. some
activations are composed in-kernel from primitives, no standalone `<op>_tile`),
no shim is needed. Verify the ttnn op handler decomposes it via `SFPU_OP_CHAIN`
rather than calling a standalone `<name>_tile`.

## Step 3 — Read upstream

Three references per op, in order of priority:

1. **Upstream header** at the matching path. Lock the signature(s) verbatim —
   parameter names, types, defaulted args.
2. **LLK implementation** at `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_<name>.h`
   (or, for actual numerics, `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_<name>.h`).
   Source of truth for the **math formula** — polynomial coefficients,
   Cody-Waite reductions, saturation branches.
3. **A reference emule shim** for the boilerplate. Pick by category if possible — see
   `STRUCTURE.md` for the catalog. Useful representatives:
   - per-element stateless: `eltwise_unary/relu.h`
   - thread_local state: `eltwise_unary/dropout.h`
   - int32 DST: any `eltwise_unary/bitwise_*.h`
   - in-place temp-buffer: `transpose_wh.h`

**Hard input cap: ~3 files.** If you find yourself reading 4+ host-side files
or wandering into `tt_metal/llrt/`, `tt_metal/impl/dispatch/`, or
`tt_metal/soc_descriptors/` — stop and reclassify the task. Host code is out
of shim scope.

## Step 4 — The shim template

Don't write a shim from scratch — pick an existing one whose pattern matches
and copy its skeleton. The standard pieces (`#pragma once`, SPDX header,
`#include "jit_hw/api/compute/common.h"`, `namespace ckernel { ... }`,
`ALWI void <name>_tile_init(...) {}`, `__emule_dst_check(idst, "<name>_tile")`,
`for (i = 0; i < __EMULE_TILE_ELEMS; i++)` over `__emule_dst[idst][i]`) are
the same across all of them — they're already correct in the existing files
and stay correct as `common.h` evolves.

Pick by pattern if possible. It might not always be possible. `STRUCTURE.md` 
catalogs every shim under `include/jit_hw/api/compute/` with its top-level 
symbols — grep there to find the closest match to your target op's shape, 
then open that file as the boilerplate template. 

Some common patterns and representative shims:

| Pattern | Representative shim(s) |
|---|---|
| Per-element stateless float math | `eltwise_unary/relu.h`, `eltwise_unary/cbrt.h`, `eltwise_unary/mish.h` |
| Per-element with `fp32` bit-pattern params (decode via `std::memcpy`) | `eltwise_unary/threshold.h`, `eltwise_unary/elu.h`, `eltwise_unary/hardtanh.h` |
| Int32 DST (via `__emule_dst_load_i32` / `__emule_dst_store_i32`) | `eltwise_unary/bitwise_and.h`, `eltwise_unary/left_shift.h` |
| Thread-local PRNG / per-call state | `eltwise_unary/dropout.h` (xorshift32) |
| Per-tile accumulator across calls | `cumsum.h`, `cumprod.h` |
| Multi-tile inputs (one call reads several DST slots) | `mask.h`, `welford.h`, `logsigmoid.h` |
| In-place mutation requiring a temp buffer | `transpose_wh.h`, `reshuffle.h` |
| Composed re-export (no own logic) | `softmax.h` |
| Ported upstream polynomial (region split + Horner form) | `eltwise_unary/i1.h`, `eltwise_unary/lgamma.h` |

**Keep comments brief.** Per project convention (see `CLAUDE.md`): default to
no comments, write one only when the *why* is non-obvious. Concretely for
shims:

- The file path already tells the reader what's being shimmed — no need to
  spell out "Intercepts the upstream include path which pulls in
  `<llk_header>.h` (an LLK-only header that references SFPU intrinsics)".
- Don't restate the math the loop body already does (no `// p(t) =
  (((c2*t + c1)*t + c0)` running commentary on a Horner expansion).
- Don't restate accessor semantics (no `// Spill mean to DST[mean_dst_idx]`
  next to a `std::memcpy` that obviously spills).
- DO keep: a one-line "what this op computes" or "what the encoded params
  mean" if the function signature alone is ambiguous; a single `Real LLK:`
  pointer to the silicon source-of-truth for math-heavy ports.

Target header docblock for a typical shim is **2–4 lines**: one-line op
summary, one-line encoded-param note if any, one `Real LLK:` line.

## Step 5 — Wire-up

If upstream has a corresponding `SFPU_OP_<NAME>_INCLUDE` guard in
`tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`, add the
same branch to emule's `sfpu_split_includes.h`:

```cpp
#if SFPU_OP_<NAME>_INCLUDE
#include "api/compute/eltwise_unary/<name>.h"  // or compute/<name>.h
#endif
```

If the upstream path is direct (no SFPU_OP gate) — e.g. `api/compute/softmax.h`,
`api/compute/cumsum.h`, `api/compute/welford.h` — no wiring needed; the JIT
include path resolves the emule version automatically when the kernel does
`#include "api/compute/<name>.h"`.

Per the project rule in `CLAUDE.md`: when you add a source file or a
top-level symbol, update its entry in `STRUCTURE.md` in the same change.

## Step 6 — Build + test

Build + base smoke test command lines live in `BUILD_GUIDE.md`.
Run the standard build, then a targeted pytest invocation for the op's test file.

Op → test-file mapping (typical):

| Op family | Test file under `tests/ttnn/unit_tests/operations/` |
|---|---|
| activations (hardtanh, mish, threshold, swish, gelu, …) | `eltwise/test_activation.py::test_<name>` |
| math (cbrt, i0, i1, erfinv, digamma, polygamma, lgamma) | `eltwise/test_math.py::test_<name>` |
| elu, celu | `eltwise/test_elu.py`, `eltwise/test_celu_21f.py` |
| binary scalar (fmod, remainder, rsub) | `eltwise/test_binary_composite.py`, `eltwise/test_binaryng_fp32.py` |
| bitwise | `eltwise/test_binary_int32.py` |
| reduce / accumulation (cumsum, cumprod, mean, min, sum) | `reduce/test_<name>.py` |
| fused (softmax, layernorm) | `fused/test_<name>.py` |

Project conventions (from `CLAUDE.md`): wormhole N150, slow dispatch always.
Standard env vars and pytest invocation are in BUILD_GUIDE.md.

When debugging a failing kernel, set `TT_EMULE_KEEP_JIT_SRC=1` and inspect
the kept `patched_kernel.cpp` / `wrapper.cpp` under `/tmp/tt_emule_jit_*/`.

## Step 7 — Promote into the regression script

Edit `scripts/run_ttnn_pytests.sh`. Add one `run_pytest` line per
cleanly-passing function, placed near the relevant family of existing
entries:

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

Commit per bring-up with a concise message:
```bash
git add include/jit_hw/<files> STRUCTURE.md scripts/run_ttnn_pytests.sh
git commit -m "<shim-name>: <one-line summary> (<n>/<n> pass)"
```

## Step 8 — PCC failure triage

Hard cap: **~3 iterations per shim**. Past that, defer with a written note.

When PCC fails, check in order:

0. **Verify upstream signatures literally before anything else.** Open the
   upstream header and compare the `<name>_tile` signature byte-for-byte
   against the emule shim. Inline notes about parameter ordering can be
   wrong; the committed upstream code is the source of truth.

1. **Polynomial approximation gap.** `<cmath>` ≠ upstream LLK. Read
   `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_<name>.h`
   for the actual coefficients. Port them — see "Porting upstream
   polynomials" below.
2. **Parameter ordering.** Compare your shim's args against the upstream
   `ALWI void <name>_tile(...)` declaration in the upstream header.
3. **Stateful accumulator direction.** For cumsum/cumprod/welford, check
   upstream's tile-row order convention (typically N×W×H). Flags like
   `first` may apply per (N, W) column-of-tiles rather than per-tile.
4. **Layout-specific issue.** ROW_MAJOR vs TILE vs bf16 vs fp32 can hit
   different reader kernel paths. If only some dtypes fail, look at the
   reader kernel for that dtype's branch.

If none of those: defer with a one-paragraph note in the test entry's
comment in `scripts/run_ttnn_pytests.sh` and move on. PCC-fix work
deserves its own focused round.

## Separate kernel chain not shimmed

Some ttnn ops dispatch to alternate kernel implementations (moreh, fused,
deepseek, etc.) that live outside the standard `eltwise_unary/` API surface.
These kernels often include headers under `ttnn/cpp/ttnn/kernel/`,
`ttnn/cpp/ttnn/kernel_lib/`, or arch-specific paths like
`noc/noc_parameters.h` (which on real silicon lives at
`tt_metal/hw/inc/internal/tt-Nxx/<arch>/noc/`).

**Diagnostic pattern:** the JIT compile fails not in `eltwise_unary/<name>.h`
but in some other kernel chain. Capture the failing kernel via
`TT_EMULE_KEEP_JIT_SRC=1`, look at the kept `patched_kernel.cpp`'s
`#include` list, identify the missing header. Add a thin re-export shim
under `include/jit_hw/` at the matching path.

## Composed ops (no standalone upstream `<name>_tile`)

Some activation-style ops are not standalone SFPU primitives — they're
composed in-kernel from underlying primitives. If a sub-agent returns
STUCK because "upstream has no `<name>_tile`," that's correct behavior.

**Diagnosing a composed-op failure:** when the JIT compile fails on
`<name>_kernel.cpp`, check whether the kernel calls into primitives we
already have. If yes, the gap is in the primitive (or in
`compute_kernel_api.h`'s coverage of it) — not a missing `<name>_tile`.
Typical fix: add the primitive name to `compute_kernel_api.h` so the
catch-all include surfaces it.

## Porting upstream polynomials (PCC triage)

When `<cmath>` numerics drift past atol/ULP tolerance, port the exact
polynomial form from
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_<name>.h`.

Process:
1. Open the upstream `ckernel_sfpu_<name>.h`. Identify regions (range splits,
   saturation thresholds), coefficient arrays (Horner-form polynomial), and
   boundary branches.
2. Copy the coefficient constants **verbatim** as `constexpr float`. Don't
   round, don't reformat.
3. Translate the SFPI vector eval (`sfpi::vFloat`, `v_if/v_endif`) to scalar
   per-element math: `if (cond) {…} else {…}` inside the `__EMULE_TILE_ELEMS`
   loop.
4. Use only `<cmath>` builtins (`std::exp`, `std::log`, `std::sqrt`,
   `std::fabs`, `std::nearbyint`, `std::ldexp`, `std::copysign`). No platform
   intrinsics.

**A polynomial port is not guaranteed to pass PCC.** If a new port regresses
cases that were passing under a simpler `<cmath>` formulation, the new form
has a bug (often in the reconstruction step) or differs from upstream
somewhere subtle. Revert before chasing. Don't replace a working simple
form unless tests are demonstrably failing on it.

## Stateful ops & `first` flag — confirm the shim is even used

A shim file's existence doesn't imply ttnn uses it. The host op may
preprocess inputs (permute, reshape) so a *different* kernel chain runs.

**Diagnostic:** before debugging a stateful-op shim's math, dump the JIT
wrapper (`TT_EMULE_KEEP_JIT_SRC=1`), grep the kept `patched_kernel.cpp`
for the shim function name. If it's not there, the failure is upstream
of your shim — chase the actual primitives the kernel does call.

For ops that DO use stateful per-tile primitives (PRNGs, per-tile
accumulators), the `thread_local` accumulator pattern is correct. But
verify before assuming.

## Anti-patterns (consolidated)

1. **Don't read host code.** `tt_metal/llrt/`, `tt_metal/impl/dispatch/`,
   `tt_metal/soc_descriptors/` — none of these are relevant to a shim. If
   you find yourself there, the failure isn't a shim gap.
2. **Don't accumulate JIT cache + restart** as a substitute for understanding
   the failure. Each `rm -rf /tmp/tt_emule_jit_cache_$(id -u)` should be
   deliberate (e.g. after a shim edit), not a cargo-cult retry.
3. **Don't duplicate shims** that `activations.h` or `compute_kernel_api.h`
   already define. ODR conflicts result. Grep `STRUCTURE.md` first.
4. **Don't invent upstream signatures.** If upstream has no header, return
   STUCK or ask. Some activation-family ops are composed in-kernel and
   have no standalone API.
5. **Don't iterate PCC failures more than a few times.** ~3 attempts max
   per shim, then document and move on.
## Batch mode

For sweeps that bring up ≥4 shims at once, dispatch one sub-agent per
shim via `/parallel-mock-implementation`. The orchestrator (you)
handles `sfpu_split_includes.h` wiring, `STRUCTURE.md` updates, build,
and per-op test runs centrally after the workers return.

## References

- `/implement-mock` — the broader strategy-picking flow this skill
  specializes (Strategy A for compute shims).
- `/parallel-mock-implementation` — Workflow-tool dispatch pattern for
  batch shim authoring.
- `STRUCTURE.md` — file-level index of `src/` + `include/` with top-level
  symbols. Grep first when triaging.
- `IMPLEMENTATION_REPORT.md` §JIT Kernel API Coverage — what's currently
  shimmed.
- See `/implement-mock` References for the broader project conventions
  (`CLAUDE.md`, `BUILD_GUIDE.md`).
