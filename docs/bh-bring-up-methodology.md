# BH bring-up: methodology + case studies

How to drive a WH-pass / BH-fail ttnn pytest entry to ground in
emule, validated on real WH/BH-divergence cases before mechanizing
the parallel sweep. The workflow itself is encoded in the skills
under `.claude/skills/`; this doc is the *applied* form, with worked
examples that the parallel agents can pattern-match against.

## Workflow

```
For a WH-pass / BH-fail test:

  Step A: rerun on BH against current tt-metal HEAD.
          PASS → resolved by upstream; remove from baseline.
          FAIL → continue.

  Step A.5: rerun on WH too.
          PASS on WH, FAIL on BH → genuine WH/BH divergence; continue.
          FAIL on WH too → it's an emule regression, not a parity
                            issue.  Reclassify, fix the regression, and
                            pick a different exemplar.

  Step B: characterize the failure mode and choose the next skill.
          JIT compile fails (missing symbol)      → /compute-llk-bringup
          Kernel runs, wrong output (PCC / ATOL)  → /memory-debug discriminator
          Crash / hang / timeout                  → /memory-debug §Crashes
          Process aborts before pytest summary    → gdb on the test process

  Step C: surface the actual error (pytest hides the JIT compiler
          stderr by default).  For JIT compile failures, rerun with
            -v --tb=long -s --exitfirst
          and grep the output for `error:` / `undeclared identifier`.

  Step D: /arch-lookup on the failing op or symbol — only if Step C
          didn't already make the cause obvious.  /arch-lookup answers
          "is BH HW different at this op", which is what's load-bearing
          for picking a strategy.

  Step E: pick a strategy from references/emule-mapping.md:
          A: enhanced jit_hw stub (generic API, arch-conditional code)
          B: per-op `#ifdef __EMULE_JIT_MODE` patch in consumer header
          C: semantic rewrite under `#ifdef __EMULE_JIT_MODE`
          (BH LLK shim gaps almost always land in A.)

  Step F: implement minimal fix; clear JIT cache; rerun target +
          WH sentinel (a WH passing test that shouldn't regress).

  Step G: capture the case here; identify likely siblings for the
          parallel sweep.
```

## Case study: `dm_test_repeat_interleave`

Genuine WH/BH parity case — missing BH-only LLK shim. The clearest
exemplar of the dominant failure mode the parallel sweep will face.

### Step A: rerun on BH

```
======== 24 failed, 8 passed, 1 skipped ========
E   RuntimeError: jit_compile_kernel: compiler failed (exit 256)
    for kernel: …/ttnn/cpp/ttnn/kernel/compute/tilize.cpp
```

### Step A.5: rerun on WH

```
======== 32 passed, 1 skipped ========
```

→ Confirmed WH/BH divergence. Proceed.

### Step B + C: surface the compile error

```bash
pytest <test> -v --tb=long -s --exitfirst 2>&1 | grep -E "error:|undeclared"
```

```
ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl:134:17:
  error: use of undeclared identifier 'fast_tilize_init_skip_remap'
```

The call site is gated by `#ifdef ARCH_BLACKHOLE`:

```cpp
if constexpr (use_fast) {
#ifdef ARCH_BLACKHOLE
    if constexpr (remap_mode == tilize_config::RemapMode::AssumeConfigured) {
        fast_tilize_init_skip_remap(input_dfb, block_width_tiles, output_dfb);
    } else
#endif
    { fast_tilize_init(input_dfb, block_width_tiles, output_dfb); }
}
```

So `fast_tilize_init_skip_remap` is a BH-only LLK that emule's
`include/jit_hw/api/compute/tilize.h` didn't shim. WH never compiles
that branch.

### Step D: /arch-lookup (skipped)

The failure mode + the explicit `#ifdef ARCH_BLACKHOLE` gating make
the cause obvious without round-tripping through `/arch-lookup`:
this is a BH-only LLK we're missing a shim for. Run /arch-lookup if
you're uncertain whether the BH function is semantically different
from its non-BH counterpart, OR if the kernel is failing at runtime
rather than at compile time.

### Step E: strategy A (enhanced jit_hw stub)

`fast_tilize_init_skip_remap` is the BH-only variant of
`fast_tilize_init` that skips the LLK math tile-remap reconfig step.
Emule already shims `fast_tilize_init` to route to plain `tilize_init`
— there's no "fast path" / "slow path" distinction in host C++ — so
the skip_remap variant routes the same way. The "skipped remap state"
also doesn't exist in emule, so the variant is semantically identical
in this context.

### Step F: the fix

```cpp
// include/jit_hw/api/compute/tilize.h
inline void fast_tilize_init_skip_remap(uint32_t icb, uint32_t /*full_dim*/,
                                        uint32_t ocb, uint32_t /*call_line*/ = 0) {
    tilize_init(icb, 0, ocb);
}
```

### Verification

```
BH dm_test_repeat_interleave:        24/8/1 → 32/0/1 (PASS/FAIL/SKIP)
WH dm_test_repeat_interleave:        32 passed (no regression)
```

### Likely siblings (parallel sweep targets sharing this root cause)

Any baseline entry whose BH failure signature is "JIT compile failure
on a kernel that includes `tilize_helpers.hpp`" and whose -k subset
hits `RemapMode::AssumeConfigured`. Other call sites of
`fast_tilize_init_skip_remap` and adjacent BH-only LLKs are also good
candidates — `grep -rn fast_tilize_init_skip_remap` in `ttnn/cpp/`
identifies the kernel set that needs the shim.

More broadly: any "undeclared identifier" error in a file under
`ckernels/blackhole/metal/llk_api/experimental/` is a BH-only LLK
that emule's `jit_hw/api/compute/` needs to mirror as a host-side
shim.

## Case study (false positive): `reduce_test_mean_scaling`

Recorded for the lesson, not the methodology.

### Step A revealed an emule regression, not parity divergence

`reduce_test_mean_scaling` was in the BH baseline as 28/28 FAIL.
Rerun on BH: still 28/28 FAIL. Rerun on **WH** (Step A.5): **also
28/28 FAIL** with the same JIT compile error
(`undeclared identifier 'MEM_ZEROS_SIZE'`).

The "WH passes" entry in the baseline was a stale relic from a sweep
captured *before* commit `cc964c2` (which replaced `dev_mem_map.h`
with `emule_l1_bridge.h` and dropped the JIT-side MEM_ZEROS constants).
After cc964c2, both archs broke equally.

### Fix

Reverted cc964c2's structural change (restored
`include/jit_hw/dev_mem_map.h`) and added per-arch `MEM_ZEROS_BASE`
dispatch via the `ARCH_*` JIT defines.

### Lesson for the parallel sweep

**Step A.5 is load-bearing.** Without rerunning on WH, this would
have looked like a WH/BH parity issue and the methodology would have
chased a ghost. The baseline file is a snapshot at a point in time —
treat its "WH passes" labels as stale until re-verified.

## Open: the SIGABRT cluster

519 subtests / 43 entries in the baseline file share the signature
`:-1: running the test CRASHED with signal 6`. None of these were
investigated in this round. They almost certainly share root causes
(probably small in number), but the SIGABRT signature is too generic
to drive methodology design from — a parallel sweep run AFTER this
fix lands will surface the new signatures for those entries (since
many were probably the MEM_ZEROS / tilize cases unblocked here).
The right next exemplar is one SIGABRT entry whose root cause is
**not** already covered by this round's fixes, identified by
re-baselining first.
