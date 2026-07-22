# Emule ASAN-Style Sanitizers — Maintainer Guide

This document explains how the emule sanitizers are implemented. It is meant
for future maintainers — people extending the sanitizer set, debugging a
false positive, or porting these checks to a new kernel API.

For a higher-level summary of what each check does and how it's wired into
tt-metal, see `tt-metal/tt_metal/impl/emulation/SANITIZERS.md`.

## Repository split

The implementation spans two repos, deliberately:

- **`tt-metal/tt_metal/impl/emulation/`** — host-side: the master switch,
  live-range registries that buffer alloc/dealloc updates, the runner that
  snapshots state before each kernel launch and runs post-launch comparison.
- **`tt-emule/include/jit_hw/`** — kernel-side: the inline checks that run
  inside JIT-compiled kernel code. These headers are translated and
  re-compiled per kernel by the emulator's JIT pipeline, so they must be
  self-contained and use only ABI-stable primitives.

The two sides communicate through a fixed set of `thread_local` symbols
declared `extern` in `jit_kernel_stubs.hpp` and defined in
`emulated_program_runner.cpp` (and mirrored in tt-emule's own
`kernel_runner.cpp` for the standalone build path).

## Master switch

A single env var, `TT_METAL_EMULE_ASAN`, gates every sanitizer:

- Host helper: `tt::tt_metal::emule::emule_asan_enabled()` in
  `tt-metal/tt_metal/impl/emulation/host_sanitizers.hpp`.
- Kernel helper: `__emule_asan_enabled()` in
  `tt-emule/include/jit_hw/asan/emule_asan.h` (alongside `__emule_asan_panic`).
  Every kernel translation unit pulls in `emule_asan.h`, so the one definition
  is visible to `cb_api.h`, `dataflow_api.h`, and the L1 chokepoint.

The **host** helper re-reads the environment on every call (no static
caching): a process-global `static bool` would be sticky to the first
observed value and break the combined gtest binary, where one test sets the
var via `setenv` and the next expects it unset.

The **kernel** helper (`__emule_asan_enabled()`) caches the result in a
`static thread_local int`. This is *not* a process-global cache: emule spawns
a fresh `std::thread` per kernel per launch (no thread pool), so the
thread_local is zero-initialised — and re-seeded from the env — at the start
of every launch. A test that toggles the var then launches a program gets the
new value, because the new value is read by the launch's brand-new threads.
The cache exists because every kernel memory access flows through
`__emule_local_l1_to_ptr`; a `getenv` per access would be far too costly, so
the cost is amortised to one `getenv` per kernel launch.

When the switch is off:

- The three host helpers in `host_sanitizers.hpp` return at the first line.
- The runner skips the snapshot/registration block in
  `execute_program_emulated` — every thread-local state pointer plumbed
  through `EmuleOobTensorState` stays null/zero.
- `__emule_local_l1_to_ptr` **early-outs at its first line** to the plain
  address→host-pointer translation, so the per-access semaphore / CB-scan /
  OOB / padding work is skipped entirely (not merely guarded item-by-item on
  null state). This makes the chokepoint a true no-op for normal emulation
  runs — the off-path translation is byte-identical to the tail every on-path
  branch already returns.
- The remaining kernel-side checks (NoC alignment, NoC-read-pending) are each
  guarded by `if (__emule_asan_enabled() && ...)`, and the
  `__emule_pending_noc_reads` counter is only incremented when the switch is
  on, so they too cost nothing when off.

The one exception is **CB Reservation Overflow** (in
`__emule_asan_cb_on_reserve`, `asan/asan_cb.h`), which is always on. It is
structurally load-bearing — gating it would let an over-reserve deadlock on the
space wait instead of reporting a clear error.

## Live-range registries

Three per-device registries in `tt-metal/tt_metal/impl/emulation/emule_live_ranges.hpp`:

| Class | What it tracks | Updated by |
|---|---|---|
| `LiveL1Ranges` | `(start, end)` of every allocated L1 buffer | `Buffer::allocate_impl` / `deallocate_impl` |
| `LiveDramRanges` | Same for DRAM buffers | Buffer alloc/dealloc |
| `LiveL1PaddingRanges` | `[logical_end, physical_end)` for buffers with declared logical size | `emule::register_logical_size` / `deallocate_impl` |

All three expose `snapshot(device_id)` returning `std::vector<uint64_t>`
of packed `(low << 32) | high` pairs. Packed uint64 is the on-wire
format consumed by kernel-side checks — the kernel-side scan is a flat
loop over uint64_t, no std::vector layout required in JIT translation
units.

The runner takes a snapshot once per program launch and threads the
resulting pointers through `EmuleOobTensorState` into each kernel
thread's thread-locals. Pointers are valid only for the duration of
`launch_cores`; they are nulled at kernel exit.

## Thread-local handshake

The host populates these thread-locals before each kernel invocation
(see `launch_cores` in `emulated_program_runner.cpp`, ~line 1690):

| Symbol | Purpose |
|---|---|
| `__emule_l1_unreserved_base` | First L1 address that can be a user buffer; below this is firmware/system, passes through |
| `__emule_l1_tensor_ranges` / `_count` | Packed live-L1 extents (OOB-L1 + Object Intent input) |
| `__emule_dram_unreserved_base` | DRAM equivalent of l1_unreserved_base |
| `__emule_dram_tensor_ranges` / `_count` | Packed live-DRAM extents (OOB-DRAM input) |
| `__emule_l1_padding_ranges` / `_count` | Packed padding regions (Padding Violation input) |
| `__emule_l1_host_ranges` / `_count` | Packed raw-L1 regions poked via `WriteToDeviceL1`/`ReadFromDeviceL1` — extra valid extents for the OOB check (§4), excluded from Object Intent |
| `__emule_sem_l1_range_start` / `_end` | Reserved Semaphore region bounds |
| `__emule_cb_boundary_strict` | Gate for CB Boundary Violation |
| `__emule_cb_reserved_pages[32]` | Per-CB write-window size (CB Boundary input); updated by `cb_reserve_back`/`cb_push_back` |
| `__emule_cb_waited_pages[32]` | Per-CB read-window size (CB Boundary input); updated by `cb_wait_front`/`cb_pop_front` |
| `__emule_cb_reserve_dangling[32]` | Dirty-CB leak flag: set by `cb_reserve_back`, cleared by any `cb_push_back` (decoupled from the window count above) |
| `__emule_cb_wait_dangling[32]` | Dirty-CB leak flag: set by `cb_wait_front`, cleared by any `cb_pop_front` |
| `__emule_self->san_resolved_log` / `_count` (fiber ctx, not a thread-local) | Object-intent log (kernel writes; runner reads at fiber exit) |
| `__emule_pending_noc_reads` | Outstanding noc_async_read count; consumed by NoC Barrier Missing check in cb_pop_front |

All these declarations live in `tt-emule/include/jit_hw/jit_kernel_stubs.hpp`
(extern) and are defined in `emulated_program_runner.cpp` and
`tt-emule/src/kernel_runner.cpp`. Adding a new sanitizer that needs new
state means adding to both files plus the populator in `launch_cores`.

## How each check is implemented

### Host-side checks

**Use-After-Free** — `Buffer::is_allocated()` returns false after
`DeallocateBuffer` even though the Buffer object is still live. The
helper just calls `is_allocated()`; the address cached on the Buffer
itself is what would otherwise be stomped through.

**L1/DRAM Alignment** — `address % 4 != 0` (L1) or `address % 32 != 0`
(DRAM, WH-specific). 4 is required by the L1 NOC transaction size; 32
matches the DRAM burst.

**Metadata Overflow** — Wraps `validate_circular_buffer_region` in
`ConfigureDeviceWithProgram`. The underlying ProgramImpl validator
throws when the static CB region grows past the lowest occupied L1
address; the emule wrapper translates that into an ASAN-style abort so
gtests can match on the abort message instead of catching the
exception.

**Dirty CB (per-kernel exit)** — A CB is "dirty" when the kernel exits with a
`cb_reserve_back` that **no** `cb_push_back` ever followed, or a `cb_wait_front`
that **no** `cb_pop_front` ever followed — the producer/consumer claimed the
handshake but never handed off (the consumer's matching `cb_wait_front` then hangs
on silicon). At each kernel's exit the runner reads the per-kernel *trailing-dangling*
flags `__emule_cb_reserve_dangling[cb]` (set by reserve, cleared by any push) and
`__emule_cb_wait_dangling[cb]` (set by wait_front, cleared by any pop); a flag still
set is the leak, and the reported page count is the window counter at exit.

These flags are **decoupled** from the window counters
`__emule_cb_reserved_pages`/`__emule_cb_waited_pages` (which the CB Boundary check
still uses cumulatively) on purpose: on silicon `cb_reserve_back` is a
non-cumulative free-space *wait* that creates no obligation to push exactly `n`, so
a net `reserved − pushed` count false-positives on **lookahead / double-buffer
producers** — the DRAM-sharded matmul in1 reader reserves 2 blocks of headroom but
pushes 1 per iteration (covering in-flight reads the free-space count can't see),
leaving a net residual of ≈ `num_blocks × block_tiles` that is pure headroom, not
stranded data. Because such producers always push after their last reserve, the
flag is clear and they are correctly not flagged. **Trade-off:** a
`reserve;reserve;push` (one *intermediate* push forgotten) clears the flag and is
missed here — it corrupts data in place and surfaces via the Object-Intent / OOB
checks or a PCC mismatch instead. This is a per-kernel property, checked inside the
kernel thread before the thread-locals are cleared — NOT a post-join occupancy
scan. Leftover occupancy alone is fine: a producer that reserves+pushes but is
never consumed ends occupied yet fully handed off (dangling flag clear).

This check has a dedicated per-check opt-out on top of the master switch:
setting `TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB` (non-empty, not `0`) makes the
runner's `sweep_per_kernel_dirty_cbs` return early, suppressing **only** the
Dirty CB check while every other sanitizer stays active. Its purpose is to let
a full regression run continue past a kernel with a known un-flushed-CB bug
without giving up OOB / Padding / Object-Intent / CB-Boundary coverage. The
gate helper is `dirty_cb_check_skipped()` in `host_sanitizers.hpp` (re-read
every call, like the master switch); the `test_cb_leak.cpp` death tests
`unsetenv` it so they still exercise the check when it is exported globally.

### Kernel-side checks (in `__emule_local_l1_to_ptr`)

`__emule_local_l1_to_ptr(uint32_t l1_addr)` is the single chokepoint for
every kernel L1 access — it converts a firmware-style L1 offset (or an
absolute host pointer) to a dereferenceable host pointer. The host's
regex-based JIT translation injects calls to it for `l1_arg_ptr`-style
expressions and for the explicit pointer casts that compute kernels
emit.

It has a single definition in `internal/emule_l1_to_ptr.h`, which both
`jit_kernel_stubs.hpp` and `dataflow_api.h` `#include` (so compute kernels —
which pull in only the stub via the regex rewriter — and dataflow kernels both
get it). It replaced two verbatim copies that were kept in sync by hand via an
`#ifndef __EMULE_LOCAL_L1_TO_PTR_DEFINED` guard and had begun to drift.

The chokepoint itself is thin: a master-switch early-out plus a dispatch to the
four check helpers, which live in `asan/asan_l1_checks.h` (`__emule_l1_translate`,
`__emule_asan_check_semaphore`, `__emule_asan_cb_resolve`,
`__emule_asan_check_oob_tensor`, `__emule_asan_check_padding`). That header also
owns the sanitizer thread-locals the checks consume.

Checks run in this order inside `__emule_local_l1_to_ptr`:

0. **Master-switch early-out** — `if (!__emule_asan_enabled())` return the
   plain address translation immediately. All the checks below only run with
   ASAN on; this is the first line so the off path is a true no-op.
1. **Illegal Semaphore Access** — `l1_addr` inside
   `[sem_l1_range_start, sem_l1_range_end)`. Sem accesses are supposed
   to go through `noc_semaphore_*` APIs; a raw pointer dereference
   bypasses the atomic NoC ops and races.

   **The semaphore API itself is exempt from the chokepoint.** The
   free-function API (`noc_semaphore_set/wait/wait_min`) reaches L1 via
   `__emule_sem_atomic` (`api/dataflow/dataflow_api.h`); it calls the plain
   `__emule_l1_translate` (offset-vs-absolute disambiguation only), **not**
   `__emule_local_l1_to_ptr`. This is required: the chokepoint cannot tell a
   valid `noc_semaphore_wait()` from a stray scalar write once both arrive as the
   same address, so routing the API through it false-positives. The
   remote/multicast semaphore-set APIs (`noc_semaphore_set_remote`,
   `noc_semaphore_set_multicast`, `noc_semaphore_set_multicast_loopback_src`) are
   exempt the same way: they read their LOCAL semaphore source via
   `__emule_l1_translate` (not `__emule_local_l1_to_ptr` / `noc_async_write`) before
   the remote resolver+memcpy, so pushing one's own semaphore to a remote core does
   not trip this check. Without that, a legitimate `noc_semaphore_set_remote(get_semaphore(id), ...)`
   — e.g. `minimal_matmul`'s `dm_in1_sender_out.cpp` in1-valid handshake — would
   false-positive on the local sem read. A firmware-offset
   semaphore (e.g. a constexpr receiver-semaphore offset, which `__emule_sem_atomic`
   explicitly supports) is *neither* a live tensor *nor*, in general, inside the
   sem range — so it would trip the **OOB** check (§4), not the semaphore check;
   verified by `atomic_semaphore_receiver.cpp` aborting with `noc_semaphore_wait`
   on the stack before this exemption. The exemption also makes the free-function
   path consistent with the `Semaphore` class path (`noc_semaphore.h`), which
   already operates on its cached pointer without the chokepoint. The check still
   fires on RAW derefs into the sem region (anything that bypasses the sem API and
   flows through `__emule_local_l1_to_ptr` directly — e.g. `l1_arg_ptr` arithmetic
   or an explicit pointer cast in a compute kernel), which is exactly what
   `test_semaphore_write.cpp` exercises.
2. **CB Boundary Violation** — `l1_addr` inside one of the
   `__emule_cbs[i].base + cb_size` ranges. If yes, check whether the
   access page is inside the active write window
   `[write_idx, write_idx + reserved)` modulo num_pages or the read
   window `[read_idx, read_idx + waited)`. If outside both, abort.
   **Two exemptions** keep this from false-positiving legitimate raw addressing:
   (a) **globally-allocated / sharded CBs** (`CBSyncState::globally_allocated`, set
   from `cb_impl->globally_allocated()`) — addressed across the whole backing buffer
   via computed offsets, reserved only nominally; (b) **reuse of produced data** — an
   access into the already-produced region `[read_idx, write_idx)` (`read_dist <
   produced`), which holds valid data a kernel may re-read/re-derive (conv
   activation-reuse writes back into earlier rows). Only an access outside the active
   window AND outside the produced region reaches not-yet-produced free space — the
   real over-reach. A write past the buffer is still caught by the OOB check.
   Returning early here is important — CB backing memory isn't
   registered in `LiveL1Ranges`, so a CB address that fell through to
   the OOB check below would always produce "Out-of-Bounds Write" even
   for legitimate CB accesses.
3. **Out-of-Bounds Write (L1)** — for addresses at-or-above
   `l1_unreserved_base` only, scan `__emule_l1_tensor_ranges` for a
   matching extent. If none, abort. When a match is found, also append
   the matched packed `(start, end)` to `__emule_self->san_resolved_log`
   (deduplicated by linear scan, capped at capacity) — this feeds the
   Object Intent check below.

   *Host-poked raw L1.* A tensor miss is not an immediate abort: the offset
   is first checked against `__emule_l1_host_ranges` — raw L1 the host
   designated via `WriteToDeviceL1`/`ReadFromDeviceL1` (the DM micro-benchmarks
   poke scratch at `DEFAULT_UNRESERVED` and hand the bare address to a kernel,
   so it is valid but absent from `LiveL1Ranges`). A hit returns; a miss
   aborts as before. These ranges are deliberately a **separate** array: a
   hit is **not** recorded into `__emule_self->san_resolved_log` and the regions
   are never snapshotted by Object Intent (§12), so a host-NOC write into a
   poked region can't be mis-flagged. The acceptance is precise — only the
   exact poked `[start, end)` — so an overrun past it still aborts. Host side:
   `LiveL1HostPokeRanges` in `[metal] emule_live_ranges.{hpp,cpp}`, populated from
   two places: (1) the poke APIs `WriteToDeviceL1`/`ReadFromDeviceL1` in
   `[metal] host_api/tt_metal.cpp` register the exact range they touch (precise,
   general); (2) the DM-test helper `get_l1_address_and_size()` registers the whole
   unreserved-L1 scratch extent it hands out, covering raw-L1 outputs the host only
   reads back *after* launch (which (1) can't see in time). See the metal-side
   SANITIZER_CHECKS.md "Host-poked L1 regions" for the full rationale.
4. **Tensor Padding Violation** — scan `__emule_l1_padding_ranges` for
   any entry whose `[logical_end, physical_end)` contains `l1_addr`.
   If matched, abort.

After all checks pass, translate `l1_addr` to a host pointer (subtract
`__emule_bridge_l1` base if it's a firmware offset; otherwise it's
already an absolute pointer, return as-is).

### Kernel-side checks elsewhere

**Out-of-Bounds Write (DRAM)** — Same shape as OOB-L1 but inside the
host-side `__emule_dram_ptr` (which is in `emulated_program_runner.cpp`
because it has to take the DRAM mmap base). Scans
`__emule_dram_tensor_ranges`.

**CB Reservation Overflow** — `cb_reserve_back(cb_id, n)`: if
`n > cb.num_pages`, abort. Always on (see master switch section).

**NoC Barrier Missing** — `cb_pop_front`: if
`__emule_pending_noc_reads > 0`, abort. The counter is incremented by
every `noc_async_read*` and zeroed by `noc_async_read_barrier`.
Detecting at `cb_pop_front` time is the load-bearing moment: the pop
frees the page for the producer to refill, so every read must have
completed first — a read still in flight when the page is released
races the refill and corrupts data. `cb_push_back` carries no such
requirement: only writes precede a push, so an unbarriered read there
is harmless.

**NOC Transfer Alignment** — `__emule_check_noc_{read,write}_alignment` in
`api/dataflow/asan/asan_dataflow.h`, called at the top of `noc_async_read`/
`noc_async_write`. Each endpoint is checked against its OWN memory-type
alignment (L1 = 16B; DRAM read = 32B WH / 64B BH; DRAM write = 16B), not a
relative "low bits of src and dst must match" rule — the two sides have
different requirements a shared mask can't express. See §10.

### Object Intent Violation — the complicated one

This is the check the reviewer explicitly called out. It detects
"unintended writes": a kernel that modifies an L1 buffer it never took
a pointer into via the public API. The classic instance is a kernel
that does pointer arithmetic past the end of buffer A and stomps on
buffer B's bytes — both buffers are legitimately allocated, so the OOB
and Padding checks both pass, but B was never the kernel's intended
target.

Detection has 6 stages, split between the runner (host) and the kernel:

**Fiber-engine realization.** Kernels run as cooperatively-scheduled
fibers on a shared worker pool, not one OS thread per (core, RISC). The
stages below apply unchanged, with three fiber-specific points: (a) the
per-core `ObjectIntentTracker` is owned by `launch_cores` so it outlives
the fiber run; (b) the resolved-range log is a per-fiber field of the
fiber context (`ThreadCommonCtx::san_resolved_*`, reached via
`__emule_self`), so it swaps in with the fiber — no thread-local, nothing
for the scheduler to restore, and a parked fiber's log is never confused
with a peer's on the same worker; (c) "after join" means at the
single-kernel core's fiber completion. Object Intent runs on the
non-deferred (single-device) dispatch path; deferred multi-chip mesh runs
skip it (ASAN is single-device-scoped).

**Stage 1: pre-launch snapshot (runner, in `launch_cores`).**
For each core: if `object_intent_strict && tensor_ranges != null`,
snapshot into the per-core `ObjectIntentTracker`. For every live L1
tensor extent, `memcpy` the current bytes of L1 in that range into the
snapshot, on the dispatch thread before the core's fiber runs.

**Stage 2: refuse multi-kernel launches.**
Exact attribution requires being able to tell, after join, which
kernel resolved which pointer. With >1 kernel per core, the per-kernel
resolved-range arrays would have to be merged somehow, and a write
mismatch couldn't be attributed back to a single kernel. We abort with
a friendly diagnostic when `cs.ki_list->size() != 1`, telling the user
to keep ASAN runs single-kernel-per-core.

**Stage 3: per-fiber resolved-range log.**
The log is a fixed `uint64_t san_resolved_log[64]` (+ `san_resolved_count`
and a `san_resolved_active` gate) inside the fiber context
(`ThreadCommonCtx`), reached via `__emule_self`. The runner just arms it
(`san_resolved_active = true`, count reset) before the kernel runs.
Fiber-local because:
- One log per fiber, no contention, reached through the same
  `__emule_self` pointer the kernel already uses — so it swaps in with the
  fiber (no thread-local, nothing to restore).
- Lifetime ties to the fiber exactly.
- Capacity 64 is plenty (even pathological kernels touch <10 distinct
  buffers); overflow drops the excess, which biases the post-launch
  comparison toward false positives, not false negatives.

**Stage 4: kernel records intent.**
Every successful resolution inside `__emule_local_l1_to_ptr` (i.e. the
access hit a live tensor extent in OOB check #3 above) appends the
matched `(start, end)` packed pair to `__emule_self->san_resolved_log`.
Linear-scan dedup keeps the array small. The append is the only
write — `__emule_local_l1_to_ptr` is the only place that writes the
log.

The semantics are: "this kernel intends to touch buffer
`[start, end)`." Crucially, the intent is recorded at *resolution*
time, not at *write* time. A kernel that resolves a pointer into B but
never actually writes through it is still considered to have intent to
touch B (the runner won't flag B even if B's bytes change — though
that wouldn't normally happen).

**Stage 5: accumulate at fiber exit.**
When the kernel fiber finishes, the runner passes its
`__emule_self->san_resolved_log` (and count) to
`ObjectIntentTracker::accumulate_resolved`, which appends into the
per-core resolved set (only one kernel fiber on single-kernel-per-core
launches, so no merge logic — direct copy).

**Stage 6: byte-diff snapshots whose extent was never resolved.**
Build an `unordered_set<uint64_t>` of resolved packed pairs. For each
`oi_snapshots[i]`: if its `packed` is in the set, skip (kernel had
intent). Otherwise `memcmp` `snap.bytes` vs `l1_data + start` for the
extent. Any mismatch is a stomp — abort with the "Object Intent
Violation" diagnostic naming the violated extent and the core
coordinates.

**Why this works.**
Consider a kernel that legitimately writes to buffer A and accidentally
overruns into buffer B:
- A is resolved (the kernel's `l1_arg_ptr` call for arg 0 goes through
  `__emule_local_l1_to_ptr`, hits A's range, appends A to resolved).
- B is *not* resolved — the overrun is computed pointer arithmetic
  from A's pointer; it never goes back through `__emule_local_l1_to_ptr`.
- B's snapshot is in `oi_snapshots`, B's `packed` is NOT in the
  resolved set, so we memcmp.
- The overrun wrote some of B's bytes, the memcmp differs, abort.

**Known limitations.**
- Single-kernel-per-core only (stage 2). Real programs often have
  producer/consumer kernels co-resident; for those, only the
  per-access checks fire (OOB, Padding, CB Boundary, etc.).
- A resolved range covers the *whole* tensor. A kernel that resolved A
  but stomps within A's padding is caught by Padding Violation, not by
  Object Intent — Object Intent excludes everything in the resolved
  set.
- Resolved-ranges log capacity (64) overflows silently; behavior
  degrades toward false positives, which is the safer direction.
- DRAM has no equivalent yet — DRAM is one large shared region and the
  snapshot cost would be prohibitive.

## Adding a new sanitizer

A repeatable recipe based on how the existing ones were built:

1. **Decide host or kernel.** If the bug needs the firmware-style L1
   offset or the kernel's thread context, it's kernel-side. Otherwise
   host-side is cheaper.
2. **Pick the diagnostic format.** Always
   `[ASAN ERROR] <Category>: <details>\n` followed by `abort()`. The
   gtest matches on `<Category>` with a regex.
3. **Gate on the master switch.** First line of the check function is
   `if (!emule_asan_enabled()) return;` (host) or wrap the check in
   `if (__emule_asan_enabled() && ...)` (kernel-side). The host helper
   re-reads the env every call (no caching); the kernel helper caches in a
   per-launch `thread_local` (see the Master switch section) — both pick up an
   env toggle between launches, neither sticks to a process-global first value.
4. **State plumbing if needed.**
   - Host-only: add to `emule_live_ranges.hpp` if it's a registry, or
     keep state local to the helper.
   - Kernel-needed: add `extern thread_local` to
     `jit_kernel_stubs.hpp`, definitions in both
     `emulated_program_runner.cpp` and tt-emule's `kernel_runner.cpp`,
     populator in `launch_cores`, reset to default at kernel exit.
5. **Add a gtest.** Convention: `tests/tt_metal/tt_metal/api/
   test_<name>.cpp`, `EXPECT_DEATH` with a regex on the diagnostic
   category. Each test does `::setenv("TT_METAL_EMULE_ASAN", "1", 1)`
   inside the test body (not the fixture) so other tests in the binary
   aren't polluted.
6. **Update the docs.** Both this file and the reviewer-facing
   `SANITIZERS.md` in tt-metal.

## File index

Host side (tt-metal/tt_metal/impl/emulation/):

- `host_sanitizers.hpp` — master switch + 3 host-only checks.
- `emule_live_ranges.{hpp,cpp}` — 3 registries.
- `emulated_program_runner.cpp` — snapshot/populate/post-launch logic.
  Look for `EmuleOobTensorState` and `launch_cores`.

Kernel side (tt-emule/include/jit_hw/):

- `internal/emule_l1_to_ptr.h` — single definition of the thin
  `__emule_local_l1_to_ptr` chokepoint (master-switch early-out + dispatch to the
  check helpers). Included by both `jit_kernel_stubs.hpp` and `dataflow_api.h`.
- `asan/asan_l1_checks.h` — the lifted L1 check bodies (`__emule_l1_translate`,
  `__emule_asan_check_semaphore`, `__emule_asan_cb_resolve`,
  `__emule_asan_check_oob_tensor`, `__emule_asan_check_padding`) and the sanitizer
  thread-locals they consume.
- `jit_kernel_stubs.hpp` — extern thread_locals, `__emule_dram_ptr`; includes
  `internal/emule_l1_to_ptr.h` for the chokepoint.
- `api/cb_api.h` — the CB sync ops (`cb_reserve_back` / `cb_push_back` /
  `cb_wait_front` / `cb_pop_front`); their sanitizer bookkeeping is delegated to
  `asan/asan_cb.h`.
- `asan/asan_cb.h` — the `__emule_asan_cb_on_{reserve,push,wait,pop}` helpers
  (Dirty-CB dangling flags + call sites, CB-Boundary window counters, the
  always-on Reservation Overflow check, the gated NoC-read-pending race check)
  and the thread-local state they own.
- `api/dataflow/dataflow_api.h` — includes the chokepoint and
  `api/dataflow/asan/asan_dataflow.h`.
- `api/dataflow/asan/asan_dataflow.h` — NOC transfer alignment checks.
- `asan/emule_asan.h` — `__emule_asan_enabled()` (master switch) + the unified
  diagnostic trace. `__emule_asan_panic()` (which every check calls instead of
  `abort()`) prints the kernel/core/processor context + a symbolized backtrace.
  See *Diagnostic trace* below.

Tests (tt-metal/tests/tt_metal/tt_metal/api/test_*.cpp) — one file per
sanitizer category.

## Diagnostic trace

Every `[ASAN ERROR]` calls `__emule_asan_panic()` (in `asan/emule_asan.h`) rather than
a bare `abort()`. That prints:

1. **Kernel identity** — when a kernel is on the stack: the kernel source path
   (`__emule_kernel_name`, a thread-local set per launch in the runner's
   `launch_cores` and cleared at teardown) plus logical/physical core and
   processor/neo/trisc, all read from the existing identity thread-locals.
   Host-API checks (no kernel context) skip this block.
2. **Symbolized backtrace** — `backtrace()` + `dladdr` to find each frame's
   module; frames inside a JIT kernel `.so` are resolved to kernel-source
   `file:line` via `llvm-symbolizer` (→ `addr2line` → raw `backtrace_symbols`).
   The walk stops at `__emule_kernel_entry`.

For frames to resolve, the runner compiles JIT kernels with `-g
-fno-omit-frame-pointer` **when ASAN is on** (still `-O2`; the ASAN flag is part
of the JIT cache key so debug `.so` files don't pollute the normal cache). The
header is self-contained (libc/POSIX only) so it compiles into both kernel `.so`
files and libtt_metal. New checks get the trace for free — just call
`__emule_asan_panic()` after the `fprintf`.

`__emule_asan_panic` is a single real (non-inline, `extern "C"`) symbol. Kernel
`.so` files and the host-API translation unit see only the *declaration* and
resolve it at link/dlopen (like the other `__emule_*` runner symbols), so they
don't pull in `<execinfo.h>`/`<dlfcn.h>` or the tt-emule include path. The one
out-of-line *definition* is emitted by whichever runtime TU defines
`EMULE_ASAN_IMPLEMENTATION` before including the header — `emulated_program_runner.cpp`
for libtt_metal, `kernel_runner.cpp` for the standalone tt-emule runtime. The
report is emitted under one mutex (taken, never released) so that when a kernel
bug trips every core thread at once, exactly one full report prints and aborts.

### Core dumps (`TT_METAL_EMULE_ASAN_ALLOW_CORE`)

Before printing, `__emule_asan_panic` calls `__emule_asan_handle_coredump()` (its
metal mirror is `emule_asan_handle_coredump` in `emule_asan_panic.cpp`) to decide the
fate of the core the `abort()` would otherwise trigger. It is called **under the panic
lock**, so the winning thread acts exactly once — a kernel bug that trips every core at
once never spawns more than one dump.

- **Unset (the default): no core.** The function marks the process non-dumpable via
  `prctl(PR_SET_DUMPABLE, 0)`. This is deliberate and load-bearing: the emulated process
  maps GB-scale L1+DRAM, so each abort would otherwise dump a ~13 GB core, and on hosts
  whose `core_pattern` pipes to a crash handler (e.g. Ubuntu's apport) `ulimit -c 0` /
  `RLIMIT_CORE` is **ignored** — verified — whereas `PR_SET_DUMPABLE=0` the kernel honors
  regardless of `core_pattern`. That is what lets the suite run on any machine with no
  `LD_PRELOAD` shim, generated `nodump.so`, or other per-host setup (an earlier design
  used exactly such a preload; this replaces it). The symbolized trace above already
  captures everything a core would.
- **Set: capture a core.** The function writes a real core of the process to
  `./emule_asan_core.<pid>` in the CWD by `fork`/`exec`'ing `gcore` against its own pid
  (`PR_SET_PTRACER_ANY` first, so the child can attach under a restrictive
  `yama/ptrace_scope`). We dump it ourselves instead of relying on the kernel
  `core_pattern` precisely because that global pipe (apport) silently discards cores from
  locally-built, non-package binaries — so "just enable cores" would yield nothing on
  those hosts. `gcore`/gdb chatter is redirected to `/dev/null` so the `[ASAN ERROR]`
  report stays readable, and the path is printed. Best-effort: if `gcore` is missing or
  ptrace is denied, the process is left dumpable (so a plain-file `core_pattern` still
  produces a core) and a one-line note says so — never a hang.

The output is a standard ELF core; open it with `gdb <binary> emule_asan_core.<pid>`
(`bt`, `thread apply all bt`). It is ~13 GB (the full L1+DRAM snapshot), so use the
opt-in on a single test, not a whole-suite run. Linux-only (`prctl`/`gcore`); the block
is `#if defined(__linux__)`-guarded, matching the rest of the trace facility.
