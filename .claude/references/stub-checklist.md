# Stub checklist — verify your mock before declaring done

When you've implemented a mock for a silicon API in tt-emule, walk
this checklist before considering it complete. Skipping any of these
is how regressions creep in.

## Pre-implementation

- [ ] Ran `/arch-lookup "<silicon_function>"` — have the authoritative
  HW spec
- [ ] Picked a strategy (A/B/C per `emule-mapping.md` §7)
- [ ] If the spec is arch-specific (WH vs BH), confirmed which arch
  the test you're targeting uses (`TT_EMULE_ARCH=blackhole|wormhole`)
- [ ] Checked there isn't already an existing mock at the place you
  were going to add one (avoid duplicates / conflicts)

## Implementation

- [ ] Mock signature **exactly** matches silicon's signature
  (template params, defaults, return type). Mismatches cause silent
  ADL surprises.
- [ ] Real math (not no-op) if the test will check PCC against torch.
  No-op stubs are only OK for HW pipeline state (UNPACK/MATH/PACK
  config calls).
- [ ] Format-aware path for bf16 vs fp32 if the API reads/writes a CB
  (use `__emule_compute::cb_is_32bit_format(cb_id)`).
- [ ] Uses `__emule_nfaces::rowmajor_to_nfaces[]` on the
  reads/writes — emule DST is row-major, CB is face-packed.
- [ ] If touching DST: call `__emule_dst_check(idst, "name")` and
  `__emule_dst_mark_dirty(idst)` early.
- [ ] If the silicon impl has multiple template variants (e.g.
  `<BroadcastType bcast_type>`), all variants supported (use `if
  constexpr` to dispatch).
- [ ] Comment cites silicon source (DeepWiki page, Confluence page,
  or `tt_llk_*/...:line`) for non-obvious choices.

## Build + sanity

- [ ] Build passes (if you touched `emulated_program_runner.cpp`):
  ```bash
  cmake --build ${TT_METAL_DIR}/build_emule -j$(nproc)
  ```
- [ ] Wipe JIT cache before testing:
  ```bash
  rm -rf /tmp/tt_emule_jit_* /tmp/tt_emule_src_*
  ```
  (The persistent disk cache at `/tmp/tt_emule_jit_cache_*` will
  self-invalidate on op.hpp content change — wave-2 A1. But per-test
  temp dirs at `/tmp/tt_emule_jit_*/` need explicit wipe.)

## Verification

- [ ] **Sentinel test passes** (your project's smallest end-to-end
  emule test that exercises the JIT compile + dispatch path)
- [ ] Target test reaches the new mock (verify with
  `TT_EMULE_KEEP_JIT_TEMP=1` + grep the patched_kernel.cpp)
- [ ] Target test passes (or improves) — PCC > threshold
- [ ] **tt-metal regression matches the recorded baseline**:
  ```bash
  TT_METAL_DIR=<tt-metal-checkout> bash run_regression.sh
  ```
  (Compare totals to your last known good run; any regression in
  previously-passing tests blocks ship.)

## Documentation

- [ ] If you added a new strategy / pattern, append to
  `.claude/references/emule-mapping.md`
- [ ] If you discovered a HW-spec-vs-emule-mock divergence the test
  doesn't catch, note it somewhere persistent (commit message, design
  note, etc.) so future readers know.

## Common gotchas

- **No-op stub compiles but produces zero/garbage at runtime.** Add
  real math even if "obvious" — `recip_tile` returning 0 is harder to
  debug than `recip_tile` undefined.
- **Template parameter mismatch with silicon signature.** Causes
  "no matching function" errors that look like the mock isn't found.
  Always copy the silicon signature exactly, then maybe add defaults.
- **`#ifdef __EMULE_JIT_MODE` placed inside a `#if defined(COMPILE_FOR_*)`
  block.** Both run; the inner one wins. Make sure the gate is at the
  right level — usually OUTSIDE the per-RISC guard.
- **Allowlist-add poisons sentinel.** If the op's op.hpp doesn't
  compile under TRISC (even if your test only uses BRISC), the whole
  allowlist breaks. Test that the op.hpp parses cleanly for all 3
  processors before declaring done. The sentinel run catches this.
- **`tt_l1_ptr`, `VALID`/`INVALID` undefined.** These come from
  `dataflow_api.h` (`tt_l1_ptr`) and `hostdevcommon/common_values.hpp`
  (`VALID`/`INVALID`). Both must be `#include`d for any code path
  that uses them — silicon path may pull them transitively, emule may
  not.
- **`compute_kernel_hw_startup` redefinition.** Both
  `jit_kernel_stubs.hpp` and `api/compute/compute_kernel_hw_startup.h`
  define overloads. Use the `__EMULE_COMPUTE_KERNEL_HW_STARTUP_DEFINED`
  guard pattern (first-included-wins).

## Anti-checklist

These are NOT done criteria — don't get hung up:

- ❌ Bit-exact match with silicon output. PCC > test's threshold
  (~0.998) is the bar.
- ❌ Real `sfpi::` SIMD math (the shim provides types only).
- ❌ Fabric / multichip semantics (out of scope).

## Reference paths

- Build: `cmake --build ${TT_METAL_DIR}/build_emule -j$(nproc)`
- tt-metal regression: `TT_METAL_DIR=<tt-metal-checkout> bash run_regression.sh`
- Inspect generated JIT: `TT_EMULE_KEEP_JIT_TEMP=1 <run-test>` + `ls /tmp/tt_emule_jit_*/wrapper.cpp`
