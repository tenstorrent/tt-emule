---
name: workarounds
description: Registry of deliberate, non-ideal workarounds currently live in tt-emule (hacks we accepted to make something pass, NOT normal development practice). Read this before adding any new workaround, and when touching code that carries one — each entry records what the hack is, why it is not the right fix, the real root cause, and the path to removing it.
---

# Workarounds registry

This file is the **single, explicit list of deliberate workarounds** in tt-emule.

A workaround here is a change we knowingly accepted that is **not** the correct
fix — it papers over a problem to make a test pass while the real fix lives
elsewhere (usually an upstream kernel, a tt-metal runtime bug, or a missing emule
model we did not have time to build properly).

**These entries are not a template for how to develop.** They exist precisely so
that a hack does **not** silently become precedent. The default tt-emule rules
still hold (see `.claude/CLAUDE.md`): be faithful to silicon, keep a single code
path, and **surface caller bugs rather than mask them**. Every entry below
violates one of those on purpose, is documented as such, and has a tracked path
to removal.

## Rules for this registry

- **Adding a workaround is a last resort.** First try the faithful fix (model the
  real behavior; fix the actual bug upstream). Only fall back to a workaround when
  that is blocked, and say so explicitly in review.
- **Every workaround must be registered here** with: what it is, why it is a hack
  (how it can fail / what rule it bends), the real root cause (file + line), the
  proper fix, and a tracking link (issue/PR).
- **Link from the code to this file.** The workaround's code site must carry a
  comment pointing here (e.g. `// WORKAROUND: see .claude/skills/workarounds`), so
  anyone editing it knows it is a known hack, not load-bearing design.
- **Workarounds are removed, not grown.** When the real fix lands, delete the
  workaround and its entry. Do not extend one hack to cover a new case — add the
  faithful fix instead.

---

## WA-1 — first-read latency in `noc_async_read_barrier` (argmax multi-core)

**Code site:** `include/jit_hw/api/dataflow/dataflow_api.h`,
`__emule_model_first_read_latency()` (called from `noc_async_read_barrier` /
`noc_async_read_barrier_with_trid`).

**What it does:** the first `noc_async_read_barrier` on each DM thread sleeps a
fixed ~200 µs (`usleep(200)`, once per thread). It is a stand-in for the
NOC/DRAM read latency that emule's instant-memcpy reads collapse to zero.

**Why it is a workaround (not robust, bends a rule):**
- It is a **timing heuristic**, not a guarantee. It works because 200 µs greatly
  exceeds a reducer core's prologue time and because sleeping threads cede their
  cores; under pathological host scheduling (a reducer thread starved past the
  window) it could still race. It has held over repeated runs on a 64-core host,
  but it is not a correctness invariant.
- It **masks a latent kernel race rather than surfacing it**, which is the exact
  thing `.claude/CLAUDE.md` says emule should *not* do. emule was correctly
  exposing a real kernel fragility; this hack hides it.

**Real root cause (kernel, not emule):**
`tt-metal/ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved_multicore.cpp`,
`kernel_main`, the `k == 0` iteration:
- reducer resets `done_sem` (`done_sem.set(0)`, ~L353 pre-loop and ~L359 in-loop),
- workers increment it via `done_sem.up(reducer)` (~L417),
- but the `start_sem` gate is `if (k > 0) { start_sem.wait(k+1); }` (~L378), so at
  `k == 0` **nothing orders the reducer's reset against the workers' increments**.
For `k >= 1` the reducer multicasts `start_sem` after its reset and workers wait on
it (explicit happens-before, race-free); `k == 0` is special-cased out of that.
The kernel relies on workers being slow (their DRAM read) so their increment can't
land before the reducer's local reset — a timing assumption, not a guarantee. On
silicon it is safe by a huge margin (a ~1-cycle L1 write vs a hundreds-of-cycle
DRAM round-trip), and the reset is in fact **redundant** because the dispatcher
zero-initializes the semaphore. emule's instant reads + sequentially-spawned core
threads remove that margin, so the increment can land first and be clobbered,
hanging the exact-match `done_sem.wait(num_cores)`.

**Proper fix (removes this workaround):** fix the kernel so `k == 0` does not rely
on timing — either drop/guard the redundant `k == 0` `done_sem` reset (the
semaphore is already zero-initialized by the dispatcher), or extend the
`start_sem` gate to `k == 0` so the reset → release → increment ordering is
explicit. Once the kernel is timing-independent, delete
`__emule_model_first_read_latency()` and revert `noc_async_read_barrier` to a
no-op.

**Related (not itself a workaround):** the startup barrier in
`tt-metal …/emulated_program_runner.cpp::launch_cores` (merged as metal #47346,
pinned via `tt-metal-pin.txt`). It models silicon's simultaneous multi-core
dispatch and is a defensible faithfulness improvement; it was introduced alongside
this fix but does not mask a bug. If the kernel is fixed it can be re-evaluated but
need not be removed.

**Tracking:** 
https://github.com/tenstorrent/tt-emule/issues/184; 
emule: ttnn.argmax support PR #181; 
metal startup-barrier PR #47346.
