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
N150 + BH regressions (§7). It is the load-bearing prerequisite for multichip, but needs
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

One special case:
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

---

## 8. Multichip (mesh) execution — concurrent dispatch + cross-chip CCL

A dual-chip board modeled as a 1×2 mesh runs both **data-parallel eltwise (no inter-chip
communication)** and **fabric CCL (cross-chip — all_gather, point_to_point, …)** on n300 (2× Wormhole,
`wormhole_N300.yaml`) and p300 (2× Blackhole, `blackhole_P300_both_mmio.yaml`); set
`TT_METAL_MOCK_CLUSTER_DESC_PATH` to the board's descriptor (the 1×2 mesh auto-discovers).
Repro: `scripts/run_mesh_eltwise.sh <n300|p300>` (no-comm) and `scripts/run_ttnn_pytests_n300.sh` (CCL).
Each chip is a distinct in-process `SWEmuleChip`. No-comm ops host-scatter/gather
(`ShardTensorToMesh` / `ConcatMeshToTensor`); CCL ops exchange data device-side over the emulated
fabric — see [`fabric-ccl-emulation.md`](fabric-ccl-emulation.md) for the teleport transport. The fiber
engine is the substrate both rely on.

**What makes it work:**
- **Compile-once / dispatch-reuse — mirrors silicon.** `execute_program_emulated` splits into
  `prepare_program` — which resolves the program's kernels **once** (collect + JIT-compile +
  resolve → `core_kernels`), memoized by `program.impl().get_id()` (emule's analogue of
  silicon's `is_compiled`) — and `dispatch_to_device`, which runs per chip against the shared
  resolved kernels. Metal's mesh dispatch routes the first chip through `LaunchProgram` and the
  rest through `DispatchCompiledProgramToDevice` (the latter routed to `execute_program_emulated`
  for emule, since emule never sets `is_compiled`); `prepare_program` resolves on the first chip
  and is a no-op for the rest — **no per-device re-resolve**, the same compiled `.so` runs on
  every chip, and the resolved program is reused across program-cache invocations too, exactly
  like silicon's compile-once/dispatch-reuse.
- **Concurrent dispatch via the register/run split.** The slow-dispatch mesh CQ
  (`sd_mesh_command_queue.cpp`) brackets the **whole `enqueue_mesh_workload`** with
  `emule::begin_mesh_dispatch()` / `run_mesh_dispatch()`: **every program in the workload, across
  all chips,** *registers* its fibers (spawn, no run) with its borrowed per-device state kept alive;
  one final `run_until_idle` runs them **all concurrently in one scheduler generation**. (Single-device
  — no `begin_mesh_dispatch` — is unchanged: spawn + run per program.) The whole-workload bracket
  (vs per-program) is what lets a multi-program CCL workload — a `point_to_point` **sender** program on
  one chip + a **receiver** on another — co-run in one generation, so the sender's cross-chip teleport
  `wake(key)` lands on the already-parked receiver (a per-program bracket would split them into separate
  generations and the wake would miss). See [`fabric-ccl-emulation.md`](fabric-ccl-emulation.md).
- **Per-device state is concurrency-safe via the fiber context.** Each fiber's `__emule_self`
  carries its device's `core_map` / `bridge_dram` and its **`chip_id`** (set at spawn), and the runner
  bridges (`__emule_resolve_noc_addr`, the fabric teleport's `__emule_fabric_resolve_remote` /
  `__emule_fabric_neighbor`) resolve through it — so concurrent fibers from different chips map NOC
  addresses, and resolve the teleport's source/destination chip, correctly. The bank→NOC-xy tables
  remain process-global but are **topology-invariant across identical chips** (every WH, every BH), so
  they are correct shared, even concurrently. The **host-address wake key** likewise stays distinct
  per chip in one process.
- **Pinning is orthogonal to device count:** fibers from all devices distribute round-robin
  across the K workers; cross-device wakes route to the home worker exactly as cross-core wakes.

**Remaining for full multichip (not yet needed / out of scope):**
- **Scaling beyond 2 chips** (T3000 8-chip → quietbox/galaxy). Inter-chip CCL **is** emulated for the
  2-chip case (the concurrent dispatch here is the prerequisite; fabric transport is the teleport in
  [`fabric-ccl-emulation.md`](fabric-ccl-emulation.md)), but ≥3 chips needs an on-demand L1Pool (the 2 GB
  `MAP_32BIT` ceiling) + direction-aware multi-hop routing — see [#228](https://github.com/tenstorrent/tt-emule/issues/228).
- **Heterogeneous meshes** (mixed arch / differing bank topology) would need the bank tables
  made per-device (keyed via the fiber context, like `core_map`). No current Tenstorrent board
  is heterogeneous, so this is deferred.
- **Multi-process** meshes: the `uint32_t`-host-pointer identity can't span processes; the
  wake key would extend with a chip field.

Native data-race detection (TSAN-style, on the same scheduler-owned sync edges) is a
documented future phase — see the design record — and is **out of scope** here.

---

## 9. Performance: known overheads & future work

The engine is correctness-first. Its scheduling overhead is real and, on per-op kernel
execution, currently **larger** than the equivalent raw-OS-thread compute (measured 5–12×
slower per op). The engine's advantage over the legacy OS-thread executor is **not**
throughput — it is eliminating that executor's startup-barrier `sched_yield` storm (which
spends 87–99% of its kernel-exec wall spinning ~192 threads through a rendezvous on a 64-core
host). The fiber path has no global startup barrier (§4), so it skips that storm; what
remains are the overheads below, all of which sit **outside the kernels' own compute/sync**.

Each item notes which worker count `K` it bites at (the shipping default is **K=64**, activating
`min(K, fiber count)` workers per program — see §9.4/§9.7). Every fix that touches the
scheduler must be re-validated with the §7 WH+BH K-matrix — identical pass/fail at K=1 / moderate
/ high — before landing.

| Overhead | Bites at | Mechanism |
|---|---|---|
| `swapcontext` signal-mask syscall | all K (incl. default) | every park/wake (§2.1) |
| `cv_.notify_all()` thundering herd | active W>1 | single global `cv_`, pinned fibers (§2.4) |
| global `mu_` contention | active W>1 | one scheduler lock for all active workers |
| per-program fiber-stack mmap/munmap | all K | ucontext stacks allocated per program |
| per-program `setup_core_state` | all K (runner, both executors) | CB/sem/DFB rebuilt per program |

### 9.1 `swapcontext` signal-mask syscall — *all K*
glibc `swapcontext` saves/restores the signal mask via a `sigprocmask` syscall on every call
(§2.1), and a park/wake cycle is two `swapcontext` (fiber→scheduler, scheduler→next fiber) —
tens of thousands of syscalls per op (e.g. an eltwise op does ~1,920 CB waits, each a
park/wake). This is the one overhead that bites the **K=1 default**, so it is the highest-value
target. **Fix:** the boost.context `fcontext_t` switch named in §2.1, or — to avoid the
dependency — a ~20-line hand-rolled callee-saved-register switch that omits the sigmask
syscall. (emule installs only a SIGFPE handler, which reads but never alters the mask, so
dropping per-switch mask save/restore is safe.)

### 9.2 `cv_.notify_all()` thundering herd — *K>1*
Fibers are **pinned** to a home worker (§2.4): a woken fiber is enqueued on `ready_[home]` and
only that worker can run it. But `wake()`/`yield()` call `notify_all()` on a single global
`cv_`, waking **all K** idle workers — K−1 of them find nothing runnable and re-sleep. At
K=192 on 64 cores this is the dominant cost: ~814K voluntary (futex) context switches per
eltwise op, far above the ~3.8K CB operations. It is also the regime where the engine actually
parallelizes (best-K ≈ core count). **Fix:** per-worker condition variables — wake only the
home worker's `cv`.

### 9.3 Global `mu_` contention — *K>1*
A single `FiberSchedulerImpl::mu_` guards every park / wake / yield / queue manipulation across
all K workers. The lock is held only briefly (released before `swapcontext` into a fiber), but
at high K it serializes the scheduler. **Fix:** shard scheduler state per worker (naturally
paired with 9.2).

### 9.4 Per-program worker create/join — *ADDRESSED*
`run_until_idle` used to create K `std::thread` workers and join them every program. It now uses a
**persistent pool** of K threads, created once (lazily, first run) and reused across every program:
threads park on `start_cv_` between programs; `run_until_idle` bumps a generation counter +
`notify_all` to launch a run and waits on `done_cv_` for `workers_done_ == W`. Each program activates
only `W = min(K, fiber count)` workers (`home = i % W`), so surplus pool workers stay parked on
`start_cv_` and per-fiber `wake()`/`yield()` (which notify `cv_`) never wake them — a tiny program at
K=64 pays one start `notify_all`, not a per-op thread create/join nor a per-fiber herd. The pool is
drained + joined in `~FiberScheduler` (process exit). (Remaining minor: the tier-2 watchdog is still a
per-run thread — 1, not K.)

### 9.5 Per-program fiber-stack mmap/munmap — *all K*
Each fiber's ucontext stack (`TT_EMULE_FIBER_STACK_BYTES`, default 1 MB) is `mmap` + `mprotect`
(guard page) at spawn and `munmap` at `run_until_idle` teardown — one per (core, RISC), ~192
for a full 64-core program, i.e. hundreds of syscalls and ~100s of MB of virtual churn per op.
**Fix:** a free-list of fixed-size stacks reused across programs.

### 9.6 Per-program `setup_core_state` — *all K (runner, both executors)*
`setup_core_state` rebuilds per-core CB / semaphore / DFB state
(`init_core_cb_sync` / `init_core_semaphores` / `allocate_dfbs_on_core`) on every program. The
device core map is already memoized per device (`g_core_map_cache`), and the device-invariant
bank/coord-map setup is cheap (O(banks)/O(grid)); `setup_core_state` is the program-specific
remainder. **Fix:** a program-level cache keyed on the ttnn program-cache hit. (This is runner
setup, *before* `launch_cores` — outside the kernel-exec metric, but real per-op latency.)

### 9.7 The `K` knob — *config*
`TT_EMULE_FIBER_WORKERS` now defaults to **64** (the persistent pool size; §9.4) and is still
uncapped above that. Per program only `W = min(K, fiber count)` workers are activated, so a high K
never spawns more workers than a program has fibers, and — with the pool persistent — the old
"K = N regresses because of per-program thread create/join" failure mode (9.4) is gone. What remains
at high active W is genuine contention on the single global `cv_`/`mu_` (9.2/9.3): light /
data-movement ops still do best at low W and compute-heavy ops at **W ≈ core count**. K is not capped
to hardware concurrency; that and per-worker CVs (9.2) would let an arbitrarily high K stay flat.
