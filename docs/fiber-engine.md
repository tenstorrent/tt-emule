# Fiber execution engine

The execution engine that runs emule kernels. Each `(core, RISC)` kernel runs on
a cooperatively-scheduled stackful **fiber**; the fibers are multiplexed onto a
runtime-sized pool of **K** worker OS threads. A fiber that blocks at a sync point
**parks** (yields its worker) and is **woken** when its predicate is satisfied — so
tens of thousands of fibers run on a handful of workers, with no OS-thread ceiling
and no busy-spin.

This is the home of emule's concurrency model. The CB / NOC / DFB subsystem docs
describe only their own park/wake handshake and link here for the mechanism.

- Scheduler: `tt-metal/tt_metal/impl/emulation/emule_fiber_scheduler.{hpp,cpp}`
  (built into `libtt_metal.so`).
- Bridge (kernel-facing): `include/jit_hw/internal/emule_fiber_bridge.h`.
- Per-fiber state: the `ThreadCommonCtx` from [`state-tiers.md`](state-tiers.md).

---

## 1. Why fibers

Kernels are native JIT-compiled x86. A kernel can be suspended only two ways: block
its OS thread, or run on a stackful fiber that can yield. The previous model spawned
**one OS thread per (core, RISC)** and joined them per device. That ceilings at the
OS thread count (a quad ≈ 46k threads — impossible) and forces a blocked kernel to
pin its worker, so a bounded threadpool that ran kernels to completion would
**deadlock**: if every worker is blocked waiting on data from a still-queued kernel,
no worker is free to produce it.

Stackful fibers remove both problems. A blocked fiber yields its worker to a runnable
one and is re-queued when its condition holds; fibers are cheap (parked when idle), so
the fiber count no longer maps to OS threads, and the spin-waits (the old semaphore
path burned up to 10M `sched_yield`/`usleep` iterations) become free parks.

Single-chip and changes **no functionality** — validated bit/PCC-identical on the WH
N150 + BH regressions (§7). It is the load-bearing prerequisite for multichip
([`multichip/pillar0-fiber-engine.md`](multichip/pillar0-fiber-engine.md)), but needs
no multichip awareness — only a forward-compatible API (§8).

---

## 2. Architecture

### 2.1 Backend — `ucontext`

`getcontext`/`makecontext`/`swapcontext`. Each fiber owns an `mmap`'d stack (default
1 MB, env `TT_EMULE_FIBER_STACK_BYTES`) with a low **guard page** (`mprotect
PROT_NONE`) to fault cleanly on overflow. No external dependency (boost.context is not
in the build); the existing SIGFPE handler only reads the interrupted context and does
not conflict. `swapcontext` saves/restores the signal mask via a syscall (~10× a
`fcontext` switch) — fine for correctness-first; boost.context is the documented later
perf option.

### 2.2 Process-global singleton

One `FiberScheduler` reached via `FiberScheduler::instance()`, sitting **above** the
`IDevice`s (not per-device). It exposes a **register / run split**:

- `spawn(entry, ctx, id)` — register a fiber; takes ownership of its per-RISC
  `ThreadCommonCtx`; does **not** run it.
- `run_until_idle()` — run all registered fibers to completion on K workers, then clear
  the registry.

`launch_cores` registers one fiber per `(core, RISC)`; `execute_program_emulated`
calls `run_until_idle()` and returns — preserving the synchronous `LaunchProgram`
contract. The split exists so a future multi-device mesh can register all chips' fibers
before one shared drain (§8).

### 2.3 The fiber-owned context — swap = one pointer store

The [state-tiering refactor](state-tiers.md) collapsed the ~47 per-RISC `thread_local`
globals into a single `thread_local ThreadCommonCtx* __emule_self`, with all mutable
state in `ComputeThreadCtx` / `DatamovementThreadCtx`. A fiber **owns** its ctx, so a
context switch is just:

```cpp
__emule_self = f->owned_ctx.get();   // the single thread_local repoint — nothing copied
my_x[0] = my_x[1] = f->id.phys_x;    // silicon-named coords (can't move into ctx)
my_y[0] = my_y[1] = f->id.phys_y;
```

`my_x`/`my_y` are read by unmodified upstream and stay separate `thread_local`s;
they are restored on every swap-in because one worker hosts many fibers (so the coords
must be reset to the incoming fiber's core). This `install_fiber` step runs on the
resuming worker immediately before `swapcontext`.

### 2.4 Worker pool — runtime K, and why fibers are **pinned**

K (worker OS threads) is set at runtime via `TT_EMULE_FIBER_WORKERS` (default 1), read
in `run_until_idle`. There is one general K-worker design — not a K=1→K>1 phasing.

**Fibers are pinned to a home worker.** Each fiber is assigned a `home` (round-robin
across the K workers at run start) and **always** runs on that worker; the scheduler
keeps a per-worker ready queue and a fiber never migrates. A wake routes the fiber back
to its home worker's queue.

This pinning is a hard correctness requirement, not a tuning choice. The JIT kernel
reads `thread_local __emule_self` (and `my_x`/`my_y`) directly, and an optimizing
compiler is entitled to compute the **address** of a `thread_local` once and reuse it
for a function's duration — the C++ object model guarantees a thread cannot change
mid-function, so the TLS slot is invariant. User-space fiber migration across OS
threads violates exactly that guarantee: a kernel that cached worker A's slot, parked,
and resumed on worker B would dereference A's now-stale slot (which the scheduler reset
to `nullptr` when the fiber left), reading a near-null pointer. Pinning keeps the
cached slot address valid for the fiber's whole life. We keep the M:N multiplexing
(many fibers per worker) and drop only cross-worker migration.

> This is the classic `ucontext` + `thread_local` incompatibility. We cannot change the
> JIT codegen, and `thread_local` is required for K>1 (each worker concurrently runs a
> different fiber → each needs its own current ctx), so pinning is the resolution.

---

## 3. The bridge

The blocking sync primitives (`cb_wait_front`, `noc_semaphore_wait`, `dfb_*`, …) are
jit_hw headers compiled **into** the dlopen'd kernel `.so`, but the scheduler lives in
`libtt_metal.so`. They reach the one scheduler instance through the same extern-C +
`-rdynamic` mechanism the runner already uses for `__emule_dram_ptr` /
`__emule_resolve_noc_addr` / `__emule_self`: the runner defines six thunks, exports
them, and the `.so` resolves them at dlopen. tt-emule stays header-only.

```cpp
// include/jit_hw/internal/emule_fiber_bridge.h
extern "C" {
  void __emule_fiber_lock(void);                  // acquire scheduler lock
  void __emule_fiber_unlock(void);                // release it
  void __emule_fiber_park_locked(const void* key);// pre: lock held; post: lock released
  void __emule_fiber_wake(const void* key);        // re-queue all fibers parked on key
  void __emule_fiber_yield(void);                  // voluntary reschedule (no park)
  void __emule_fiber_note_publish(unsigned pages); // tier-2 progress (data published)
}
```

`key` is the **host address of the sync object** (`CBSyncState*` / `TileCounter*` / the
semaphore atom). Unique per process, so no chip field is needed today; the multi-process
multichip case extends the key (§8).

---

## 4. Concurrency model

### 4.1 Fast path / slow path

Every blocking primitive keeps a lock-free **fast path** — check the predicate over the
atomic occupancy; if already satisfied, proceed with no lock and no park. Only on a
miss does it enter the slow path. The two paths are wrapped by one inline helper:

```cpp
template <class Pred>
inline void __emule_fiber_wait(const void* key, Pred pred) {
    if (pred()) return;                       // fast path
    for (;;) {
        __emule_fiber_lock();
        if (pred()) { __emule_fiber_unlock(); return; }  // re-check under the lock
        __emule_fiber_park_locked(key);       // park; returns unlocked when woken
    }
}
```

### 4.2 The lost-wakeup guard (register-then-recheck-under-lock)

At K>1 a naive "check predicate, then park" races a concurrent producer: the producer
can satisfy the predicate and emit its `wake` in the window *between* the waiter's check
and its park, so the waiter parks after the only wake it will ever get — a lost wakeup,
parked forever. The classic condvar fix is to **register the wait under the same lock
the waker takes, then re-check the predicate while holding it**. The predicate lives in
the `.so` (it owns the sync object's layout), so the scheduler lock is exported and the
re-check runs in `__emule_fiber_wait` above: take the lock → re-check → only then park.
A `wake` takes the same lock, so it cannot interleave between the re-check and the park.

At K=1 no wakeup can be lost (one worker, fully serialized — a producer only runs while
the consumer is parked), but the lock is still taken; it is simply uncontended.

### 4.3 The lock-across-swap protocol

`park_locked` is entered with the scheduler lock **held** (acquired by the waiter's
`__emule_fiber_lock`). It marks the fiber parked, links it into `parked_[key]`, and
`swapcontext`s back to the worker loop **without releasing the lock** — the lock is
handed across the same-thread switch to the worker, which continues lock-held and
releases it before its next `swapcontext` into a fiber. So a fiber resumes with the lock
**not** held, and `__emule_fiber_wait` re-acquires it on the next loop iteration. RAII
cannot track ownership across `swapcontext`, hence the manual lock/unlock discipline.

### 4.4 Re-check on resume

A wake means "look again," never "proceed." A woken fiber re-evaluates its predicate
(the `for(;;)` loop) and re-parks if still false — so a spurious or shared wake is
harmless.

---

## 5. Yield points

The conversion is uniform across all 12 sites: keep the lock-free fast path; replace the
old blocking slow path (CV wait / spin) with `__emule_fiber_wait(key, pred)`; have the
producer call `__emule_fiber_wake(key)` after its state-changing atomic store.

| Site | File | key | predicate |
|---|---|---|---|
| `cb_reserve_back`, `cb_wait_front` | `jit_hw/api/cb_api.h` | `&cbs[cb_id]` | free space / occupancy ≥ n |
| `dfb_reserve_back`, `dfb_wait_front`, `dfb_finish` | `jit_hw/api/dfb_api.h` | `&tc` | free / occupancy ≥ n; `posted==acked` |
| `Semaphore::wait/wait_min/down`, `noc_semaphore_wait[_min]`, compute `Semaphore::wait*` | `noc_semaphore.h`, `dataflow_api.h`, `compute/experimental/semaphore.h` | the atom ptr | `reached` |

Wakers: `cb_sync_push`/`cb_sync_pop` (`cb_sync_state.hpp`), `TileCounterArray::
inc_posted`/`inc_acked` (`tile_counter.hpp`), and `up`/`set`/`noc_semaphore_inc`/
`noc_semaphore_set` — each does its atomic update then `__emule_fiber_wake(key)`. The
CB/DFB `std::mutex` + condition variables are **dropped from the fiber path** (atomic
occupancy + the scheduler lock carry correctness); the struct fields are retained for
ABI/size stability where a non-jit TU embeds `CBSyncState`.

Two special cases:
- **argmax `first_read`** (`dataflow_api.h` `__emule_model_first_read_latency`): the
  `usleep(200)` that let the starved reducer run becomes `__emule_fiber_yield()` —
  yielding the worker is the workaround's actual intent (a `usleep` would stall the whole
  worker).
- **`kernel_start_barrier`**: a no-op stub — fibers park instead of spin, so "start
  together" is unneeded.

NOC barriers/flushes are already no-ops (emule NOC is synchronous memcpy) — unchanged.

**Key-identity requirement.** The waiter keys on its local L1 host pointer; the waker
(`noc_semaphore_inc`) keys on the resolved target host pointer. For a local handshake
these must resolve to the **same** host byte, or the wake lands on a different key and the
waiter hangs (→ tier-1, §6). This is the sharpest correctness check in the conversion.

---

## 6. Hang detection

Replaces the old 10M-spin aborts and the per-op 120 s CV timeout. A poorly-written op
must **error out with a diagnosis, not hang**.

**Tier 1 — quiescent deadlock (precise, instant).** When all K workers are idle, **no
fiber is runnable in any worker's ready queue**, nothing is executing, and the parked map
is non-empty: no fiber can run, and (by the progress-coupled wake invariant — a `wake` is
emitted only by a producer that actually moved state) no in-flight producer can wake
anyone → guaranteed deadlock. Throw with the dump. The explicit ready-queue check is
load-bearing at K>1: a worker counts itself idle the instant its *own* queue empties, but
a concurrent `wake()` may have just enqueued a runnable fiber into that worker's queue
before it re-acquired the lock — so an "all idle" count alone is a false positive. Testing
the ready queues under the scheduler lock closes that window, and the lock also guarantees
no producer is mid-`wake` (a worker mid-fiber keeps the executing count nonzero).

**Tier 2 — no-global-progress watchdog (livelock / wake-cycle backstop).** A buggy op can
ping-pong wakes forever so the ready queue never empties and tier 1 never fires. A single
global `progress_` counter is advanced by genuine forward progress — fiber completions
(primary) plus published pages (`__emule_fiber_note_publish`, secondary) — and is **never**
reset by an individual wakeup. If a bounded window of resumptions
(`TT_EMULE_FIBER_PROGRESS_WINDOW`, default 200000) advances it by zero, or a wall-clock
backstop (`TT_EMULE_FIBER_WATCHDOG_SEC`, default 120) elapses with no advance, the watchdog
aborts with the dump. Tier 2 is a heuristic (perfect livelock detection is undecidable): it
turns an infinite silent hang into a finite, diagnosed error.

**Diagnostic dump (both tiers):** every parked fiber — core (logical + phys x/y),
RISC/proc_id, kernel source, and the wait-key resolved to a human name (CB id if the key
lands in that fiber's `cbs[]`; else an L1 semaphore offset; else "sync object").

**Exceptions** are caught at the fiber trampoline into the fiber's `exception_ptr`;
`run_until_idle` rethrows the first **before** reporting any deadlock — a kernel exception
is the root cause and the resulting park is only a symptom.

---

## 7. The K knob + validation

The engine must be **bit/PCC-identical** across worker counts. The validation matrix runs
the full WH N150 + BH C++ regressions (sequentially — shared JIT cache) at three points:

- **K=1** — maximum multiplexing / synchronization stress (all fibers cooperative on one
  worker; the lost-wakeup guard and per-worker queue collapse to the single-worker case).
- **K = moderate** (e.g. 8) — multiple fibers per worker.
- **K = high** (≥ the program's fiber count) — approaching one worker per fiber.

All three must produce identical pass/fail tallies. Confirm the OS-thread count tracks K
(not the core/RISC count) via `/proc/self/status` `Threads` during a run. K is a runtime
env and is **not** part of the JIT cache key, so a warm cache is reused across K values.
`TT_EMULE_FIBER_LOG_N=1` logs each program's fiber count (the per-program N).

**Known K>1 limitation — argmax.** The multi-core argmax kernel's `k=0` path relies on NOC
read *latency* rather than a semaphore handshake (`dataflow_api.h` `__emule_model_first_read_latency`,
to be removed once the upstream argmax handshake lands). emule's NOC is zero-latency, so that
ordering cannot be faked consistently across worker counts: a `usleep` would stall the single
worker at K=1 (self-deadlock), so the latency model yields the fiber instead — which does not
reproduce the cross-core ordering at K>1, where argmax can deadlock. The default K=1 is
unaffected; this is an op-level fragility, not a scheduler defect (every handshake-synchronized
op is bit-identical across K).

---

## 8. Multichip readiness (forward note, not scope)

The engine is single-chip but designed not to block multichip
([`multichip/pillar0-fiber-engine.md`](multichip/pillar0-fiber-engine.md)):

- The **host-address wake key** already generalizes — each chip's `CBSyncState` /
  semaphore L1 has a distinct host address in one process; the multi-process case extends
  the key with a chip field.
- The **register/run split** lets a mesh command queue register all chips' fibers, then
  drive them through one `run_until_idle` — the cross-chip concurrency the CCL ops need.
- Pinning is orthogonal to device count: fibers from all devices are distributed
  round-robin across the K workers; cross-device wakes route to the home worker exactly as
  cross-core wakes do.

Native data-race detection (TSAN-style, on the same scheduler-owned sync edges) is a
documented future phase — see the design record — and is **out of scope** here.
