# Pillar 0 — the fiber execution engine (decision record)

**Status: IMPLEMENTED.** The engine is live and validated bit/PCC-identical on the
single-chip WH N150 + BH regressions. This doc is the **design / decision record** —
why the engine is shaped the way it is, the alternatives weighed, and the multichip-
forward rationale. For the as-built mechanics (API, concurrency model, hang detection,
the K knob), see the first-class reference [`../fiber-engine.md`](../fiber-engine.md).

Pillar 0 is independently valuable — it removes the OS-thread-count ceiling and the
busy-spin waits of the old model on a **single chip** — and it is the load-bearing
prerequisite for multichip: every other `docs/multichip/` doc assumes it exists and
states only the API it needs. It introduced **no functionality change** (same PCC, far
fewer OS threads), which is what made it safe to land ahead of any multichip work.

---

## 1. Why Pillar 0

The old model spawned an OS thread per core **and** a nested thread per RISC, then
joined them per device. The thread count is the ceiling (a quad ≈ 46k RISC-threads —
impossible as OS threads), and a bounded threadpool that ran kernels to completion would
**deadlock**: emule kernels block at sync points (`cb_wait_front`, `Semaphore::down`)
and pin their worker, so with P workers, if P kernels block waiting on data from
still-queued kernels, no worker is free to produce it. Stackful fibers solve both: a
blocked fiber yields its worker and is re-queued when its condition holds; fibers are
cheap, so tens of thousands run on a few workers. See [`../fiber-engine.md`](../fiber-engine.md) §1.

It also unblocks multichip: all chips' fibers share one runnable pool, so a fiber blocked
on a cross-chip semaphore parks and is woken by delivery — no per-device join ordering
(the cross-chip CCL deadlock in [`scaling-architecture.md`](scaling-architecture.md) §9).

---

## 2. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Mechanism | **Full M:N stackful fibers** (`ucontext`) | The lighter "cooperative threads + run-permit governor" keeps native `thread_local` but stays bounded by OS thread count → can't reach quad scale. Full fibers remove the ceiling. |
| Backend | Custom `ucontext` | boost.context is not in the build; the SIGFPE handler does not conflict. `swapcontext`'s sigmask syscall is ~10× a `fcontext` switch — acceptable for correctness-first; boost.context is the documented later perf option. |
| Scheduler location | **tt-metal** (`tt_metal/impl/emulation/emule_fiber_scheduler.{hpp,cpp}`), its own file | The runner owns kernel launch; the blocking primitives are jit_hw headers in the dlopen'd `.so` and reach the scheduler through the existing extern-C + `-rdynamic` bridge. tt-emule stays header-only. Kept out of `emulated_program_runner.cpp` for separation. |
| Process model | **Process-global singleton above the IDevices**; single-process-multi-device primary | Shared ready/parked pool + host-address wake keys make multi-device additive. Multi-process is a measurement-gated fallback at scale (the low-2 GB L1-aliasing window bounds chips/process) — designed later via a chip-keyed wake. The **register/run split** is built from day one so multi-device needs no engine change (§4). |
| Worker count K | **Runtime** env `TT_EMULE_FIBER_WORKERS` (default 1); general K-worker pool | One design, not a K=1→K>1 phasing. K is not in the JIT cache key, so a warm cache is reused across K. |
| Fiber↔worker | **Pinned** (no migration) — see §3 | Forced by the JIT kernel's native `thread_local` access; M:N-with-migration is unsafe. |
| Per-fiber state | The single `thread_local __emule_self` ctx from [`../state-tiers.md`](../state-tiers.md) | Already done by the state-tiering refactor; a swap is one pointer store. This **supersedes** the original macro-migration plan (struck — see §5). |
| CB/DFB locks | **Dropped** from the fiber path; fields kept for ABI | Atomic occupancy + the scheduler lock carry correctness; a non-jit TU still embeds `CBSyncState`, so the size stays stable. |
| `kernel_start_barrier` | No-op stub | Fibers park rather than spin, so "start together" is unneeded; neutralizing avoids perturbing start-ordering. |
| argmax `first_read` `usleep` | `__emule_fiber_yield()` | Yielding the worker to the starved reducer is the workaround's actual intent; `usleep` would stall the whole worker. |
| Safety nets | Scheduler tier-1/tier-2 watchdog **+** retained wall-clock backstop | Replaces the 10M-spin aborts and the per-op 120 s CV timeout. See [`../fiber-engine.md`](../fiber-engine.md) §6. |

---

## 3. The pinning decision (the sharp one)

The original design assumed **M:N with migration** — a fiber could resume on any worker
("`ucontext` is thread-agnostic; per-worker TLS is set on swap-in"). That assumption is
**wrong** for native-TLS kernels and was corrected to **pinning**.

The JIT kernel reads `thread_local __emule_self` (and `my_x`/`my_y`) directly. An
optimizing compiler may compute the **address** of a `thread_local` once and reuse it for
a function's duration — the C++ object model guarantees a thread cannot change
mid-function, so the TLS slot is invariant. User-space fiber migration across OS threads
violates exactly that: a kernel that cached worker A's slot, parked, and resumed on worker
B dereferences A's stale slot (reset to `nullptr` when the fiber left) → a near-null
read / segfault. This manifested only at K>1 (K=1 has a single slot, no migration).

Resolution: keep M:N **multiplexing** (many fibers per worker) but **pin** each fiber to
one home worker (round-robin at run start; per-worker ready queues; wakes route home). We
cannot change the JIT codegen, and `thread_local` is required for K>1 concurrency, so
pinning is the only faithful fix. Pinning is orthogonal to device count, so the multichip
story (§4) is unaffected. Mechanics in [`../fiber-engine.md`](../fiber-engine.md) §2.4.

---

## 4. Multichip readiness (forward note, not scope)

Pillar 0 needs no multichip awareness but is designed not to block it:
- The **host-address wake key** generalizes — each chip's `CBSyncState` / semaphore L1 has
  a distinct host address in one process; the multi-process case extends the key with a
  chip field.
- The **register/run split** lets the mesh command queue register all chips' fibers, then
  drive them through one `run_until_idle` (the cross-chip concurrency the CCL ops need).

The chip-keyed core map, fabric teleport, and process model are the concern of
[`fabric-ccl-simulation.md`](fabric-ccl-simulation.md) /
[`implementation-plan.md`](implementation-plan.md), not Pillar 0.

---

## 5. Superseded: the `thread_local` → fiber-local macro migration

The original Pillar 0 design carried a large section on migrating ~47 per-RISC
`thread_local` variables to a `FiberCtx` via per-variable access macros. That work was
**done differently and earlier** by the state-tiering refactor
([`../state-tiers.md`](../state-tiers.md)): all per-RISC state lives in one
`thread_local ThreadCommonCtx* __emule_self` (with `ComputeThreadCtx` /
`DatamovementThreadCtx` derived), reached via typed accessors — **no macros**. A fiber
owns its ctx and a swap is a single pointer store. The macro plan is therefore obsolete
and struck; the migration surface it inventoried is fully covered by state-tiering.

---

## 6. Future phase: native data-race detection (TSAN-style)

**Out of scope for the engine as shipped — a documented later phase.** The other
poorly-written-op failure (beyond the hangs that §6 of the reference handles) is a **data
race**: two cores touch the same L1 with no synchronization ordering them. emule can detect
these **natively**, reusing ThreadSanitizer's *algorithm* without its machinery (no
external `-fsanitize=thread`).

**TSAN in brief (the algorithm we reuse).** A data race = two accesses to the same
location, from different threads, ≥1 a write, with no **happens-before** (HB) ordering. HB
is a partial order: program order within a thread; *across* threads, an edge exists only
through synchronization (a release-store observed by an acquire-load; here a
**semaphore-inc → semaphore-wait** or **`cb_push` → `cb_wait`**). TSAN tracks HB with
**vector clocks** plus **shadow memory** checked on each access, injected on every
load/store — hence 5–15× cost. emule reuses the algorithm but not the per-byte
instrumentation.

**Native design — reuse the algorithm, change the observation point.** emule owns the
scheduler and the sync layer:
- **Per-fiber vector clocks** — one VC per `(core, RISC)` fiber (already tracked).
- **HB edges from the silicon primitives** — `cb_push` / semaphore-inc = *release* (stamp
  the fiber's VC into the CB / semaphore shadow); a passing `cb_wait` / semaphore-wait =
  *acquire* (join that VC). The scheduler's park/wake is **not** an HB edge — so HB reflects
  the **silicon** ordering, not emule's cooperative interleaving.
- **Shadow per L1 region / CB / semaphore**, not per byte.
- **Observation = the hooks emule already mediates** — NOC transactions
  (`__emule_resolve_noc_addr` / `__emule_multicast_write`), CB/DFB pointer grants
  (`get_read_ptr`/`get_write_ptr`), and the sync ops. On each, a FastTrack-style check: a
  prior conflicting access from another fiber that does not happen-before the current one →
  **race**, thrown with a precise report.

**Worked example.** Core A writes payload into region R, then `noc_semaphore_inc`s sem S;
core B `noc_semaphore_wait`s S, then reads R. *Correct:* inc(release) → wait(acquire) puts
A's write before B's read in HB → no race. *Buggy (B omits the wait):* no HB edge → A's
write and B's read are concurrent on R → flagged — even though, under K=1, A happened to
run first and B read the right bytes. That is the silicon bug emule would otherwise hide.

**Catches vs misses.** Catches the cross-core surface: publish-before-write, multi-writer
with no arbitration, consume-before-release, CB/semaphore protocol misuse. Misses (vs real
TSAN): byte-precision within a granted region; races via raw pointer arithmetic that
escapes the CB/NOC API; purely-local accesses (which can't race cross-core anyway).

**Shared infrastructure.** Race detection and hang detection
([`../fiber-engine.md`](../fiber-engine.md) §6) run on the same scheduler-owned state
(per-fiber clocks + the sync edges at the same yield points), both deterministic and
reproducible — together turning a poorly-written op into a *diagnosed* error instead of a
silent wrong answer or a hang.
