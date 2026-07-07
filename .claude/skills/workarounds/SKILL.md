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

## WA-1 — fabric mux kernel replaced with a no-op (n300/p300 CCL)

**Code site:** `tt-metal …/impl/emulation/emulated_program_runner.cpp`,
`jit_compile_kernel` preprocess step — when the kernel source basename is
`tt_fabric_mux.cpp`, emule writes a no-op `kernel_main` instead of compiling the
real source (search "emule no-op stub for tt_fabric_mux.cpp").

**What it does:** the fabric MUX kernel (a persistent worker-side kernel that
aggregates worker packets onto ethernet links and forwards them) is not compiled
or run. The CCL workers' `WorkerToFabricMuxSender` teleports each packet straight
to its final destination instead (the same direct path the no-mux
`num_workers_per_link=1` configuration uses).

**Why it is a workaround (bends a rule):**
- It special-cases a specific kernel **by filename** and substitutes a no-op,
  rather than modeling the kernel or providing a general "transport kernel" hook.
- It is functionally faithful (the data + semaphore effects are identical — the
  teleport delivers to the same destination the mux would have forwarded to), but
  it does **not** model the mux's flow-control / buffer back-pressure. emule is a
  functional (not cycle-accurate) emulator, so this is acceptable for correctness,
  but it is a divergence from the silicon code path.

**Real root cause (emule scope, not a bug):** the mux is part of the fabric
*transport* layer (worker → mux → ethernet → router → dest), which emule
deliberately collapses into the synchronous teleport (see
`docs/fabric-ccl-emulation.md`). The real mux additionally pulls in
erisc firmware headers emule does not model and is a persistent loop that would
never terminate under emule's run-to-completion join.

**Proper fix (removes this workaround):** if mux fidelity is ever required (e.g.
modeling link contention), model the mux faithfully — workers write packets into
the mux core's L1 channels and a mux execution context drains + teleports them —
behind a general transport-kernel mechanism rather than a filename special-case.
Until then the no-op stub is the correct functional model.

**Tracking:** [#227](https://github.com/tenstorrent/tt-emule/issues/227) (WA-1 faithful-transport
removal path) and [#231](https://github.com/tenstorrent/tt-emule/issues/231) (reduce the fabric shim
surface: real packet header + faithful mux); also documented in `docs/fabric-ccl-emulation.md`.

---

## WA-2 — reduce/matmul skip zero-multiplier lanes to suppress `inf * 0 → NaN`

**Code site:** `include/jit_hw/api/compute/reduce.h` (the SUM/AVG accumulation loops in
`reduce_tile`, both REDUCE_COL and REDUCE_ROW) and `include/jit_hw/api/compute/matmul.h`
(`matmul_tiles`, both the AVX2 blend path and the scalar path). Each skips a lane whose
scaler / B-multiplier is exactly `0.0f`.

**What it does:** when a masked/padded lane holds `inf` (or `nan`) and its scaler/multiplier is
`0`, emule skips the term instead of computing `inf * 0`. This stops masked-reduction outputs
(e.g. `prod_last`, log/reciprocal reductions over padded regions) from turning into `NaN`.

**Why it is a workaround (bends a rule):**
- On IEEE-754 — and the WH/BH FPU — `0 * inf = NaN`. Real silicon, given the same `inf` in a
  masked lane, would also produce `NaN`. Skipping the lane is therefore a **behavioral divergence
  from silicon**, not a faithful model.
- For all finite inputs the skip is a mathematical no-op (`x * 0 == 0`), so it changes results
  only in the `inf/nan * 0` corner — but it changes that corner for **every** `matmul_tiles`
  call, not just reductions.
- It is unconditional (no runtime gate/log): gating a hot matmul inner loop is not viable, and the
  skip is inert for the common finite case.

**Real root cause:** emule leaves `inf`/`nan` in padded lanes that silicon fills with a finite pad
value (or the kernel only computes `log`/`recip` on valid lanes). The NaN is a *symptom*; the
actual defect is upstream in the emule im2col/tilize/pad path for the NHWC reduction case. This
change masks the symptom — the underlying numeric divergence on that path is still open (the
`conv2d_nhwc` point still fails at PCC ~0.92, [#247]).

**Proper fix (removes this workaround):** make emule pad the reduced/matmul input lanes with the
same finite value silicon uses (so `finite * 0 = 0` naturally), then revert the reduce/matmul skip
so the core ops reproduce silicon FP exactly. Until the padding path is fixed, the skip keeps
masked reductions usable.

**Tracking:** [#248](https://github.com/tenstorrent/tt-emule/issues/248) (masked reductions
propagate NaNs from zero-multiplier lanes) and [#247](https://github.com/tenstorrent/tt-emule/issues/247)
(the residual `conv2d_nhwc` numeric divergence this sits under).

---

# Design divergences (faithful mechanisms — NOT workarounds)

These are deliberate emule mechanisms that *diverge from how silicon addresses memory* but are
**faithful in effect** and are the **correct** handling given emule's chosen model — not hacks
papering over a bug. They are listed here for discoverability (so anyone touching the addressing
path knows the divergence exists), but they have no "removal path": removing them would break
correctness, and the only alternative is a different emule addressing model.

## DM-1 — `__emule_chip_relative_l1`: cross-chip global-semaphore pointer remap

**Code site:** `tt-metal …/impl/emulation/emulated_program_runner.cpp::__emule_chip_relative_l1`;
called from the semaphore chokepoint `__emule_sem_atomic` in
`tt-emule/include/jit_hw/api/dataflow/dataflow_api.h`.

**What it does:** given a host pointer a kernel is about to dereference, if that pointer lands in
some chip's L1 mmap, it returns the SAME (core, offset) on the CURRENT chip (`__emule_chip_id`).
Single-chip runs short-circuit (one map-size check).

**Why it exists (the divergence):** emule's faithful addressing model gives each chip its OWN
`MAP_32BIT` L1 mmap, so the same L1 offset is a DIFFERENT host pointer per chip. A cross-chip-shared
object — notably a global semaphore created once and shared across the mesh — reaches kernels as a
single absolute host pointer valid for only ONE chip's mmap. A worker on a different chip spinning
on that global semaphore (which a peer increments over fabric) must hit ITS OWN chip's copy, so the
pointer is remapped to the current chip's (core, offset). On silicon this is a non-issue: every
chip sees the same L1 *offset* backed by its own physical memory, so no remap is needed.

**Why it is faithful, not a hack:** the effect is identical to silicon (the worker reads/increments
the correct per-chip semaphore copy); it does not mask a bug or add a divergent op path — it is the
correct translation between emule's per-chip-mmap reality and the silicon-offset semantics kernels
assume.

**Why it cannot be avoided:** it is an inherent consequence of the MAP_32BIT per-chip-aliasing model
(chosen so a uint32 L1 address truncates to a valid host pointer). The only way to remove it is to
change the addressing model so all chips share one offset space — which would defeat the truncation
model. Tracked (for visibility, not as a fix-it):
[#230](https://github.com/tenstorrent/tt-emule/issues/230).

**Related fabric/CCL follow-up issues:**
[#228](https://github.com/tenstorrent/tt-emule/issues/228) (scale beyond n300 → quietbox/galaxy),
[#229](https://github.com/tenstorrent/tt-emule/issues/229) (DRAM-resident CCL bank-view collapse),
[#232](https://github.com/tenstorrent/tt-emule/issues/232) (inert but value-divergent fabric stubs).
