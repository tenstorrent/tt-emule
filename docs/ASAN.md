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
- Kernel helper: `__emule_asan_enabled()` at the top of
  `tt-emule/include/jit_hw/api/cb_api.h`. (cb_api.h is included by every
  kernel translation unit; placing the helper here makes it visible to
  `dataflow_api.h` and `jit_kernel_stubs.hpp` without a separate include.)

Both helpers re-read the environment on every call (no static caching).
Caching breaks combined test runs where one gtest sets the var via
`setenv` and the next gtest expects it unset — a `static bool` would be
sticky to the first observed value.

When the switch is off:

- The three host helpers in `host_sanitizers.hpp` return at the first line.
- The runner skips the snapshot/registration block in
  `execute_program_emulated` — every thread-local state pointer plumbed
  through `EmuleOobTensorState` stays null/zero.
- Kernel-side checks short-circuit on null/zero state (the inline check
  pattern is always `if (<state> != nullptr) { ... }` or
  `if (<bool> && ...)`).

The one exception is **CB Reservation Overflow** in `cb_reserve_back`,
which is always on. It is structurally load-bearing — see
`cb_api.h:71` and the runner doc for why.

## Live-range registries

Three per-device registries in `tt-metal/tt_metal/impl/emulation/emule_live_ranges.hpp`:

| Class | What it tracks | Updated by |
|---|---|---|
| `LiveL1Ranges` | `(start, end)` of every allocated L1 buffer | `Buffer::allocate_impl` / `deallocate_impl` |
| `LiveDramRanges` | Same for DRAM buffers | Buffer alloc/dealloc |
| `LiveL1PaddingRanges` | `[logical_end, physical_end)` for buffers with declared logical size | `Buffer::set_logical_size` / `deallocate_impl` |

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
| `__emule_sem_l1_range_start` / `_end` | Reserved Semaphore region bounds |
| `__emule_cb_boundary_strict` | Gate for CB Boundary Violation |
| `__emule_cb_reserved_pages[32]` | Per-CB write-window size; updated by `cb_reserve_back`/`cb_push_back` |
| `__emule_cb_waited_pages[32]` | Per-CB read-window size; updated by `cb_wait_front`/`cb_pop_front` |
| `__emule_l1_resolved_ranges` / `_count` / `_capacity` | Object-intent log (kernel writes; runner reads after join) |
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

**Dirty CB (per-kernel exit)** — A CB is "flushed" when every
`cb_reserve_back` was committed by a matching `cb_push_back` and every
`cb_wait_front` was released by a matching `cb_pop_front`. At each
kernel's exit the runner reads the per-kernel thread-local counters
`__emule_cb_reserved_pages[cb]` (bumped by reserve, shrunk by push) and
`__emule_cb_waited_pages[cb]` (set by wait_front, shrunk by pop); either
holding a non-zero net unmatched count means the kernel reserved/waited
without the matching push/pop and left the CB un-flushed (its write/read
pointer desyncs on silicon). This is a per-kernel property, checked
inside the kernel thread before the thread-locals are cleared — NOT a
post-join occupancy scan. Leftover occupancy alone is fine: a producer
that reserves+pushes but is never consumed ends occupied yet flushed.

### Kernel-side checks (in `__emule_local_l1_to_ptr`)

`__emule_local_l1_to_ptr(uint32_t l1_addr)` is the single chokepoint for
every kernel L1 access — it converts a firmware-style L1 offset (or an
absolute host pointer) to a dereferenceable host pointer. The host's
regex-based JIT translation injects calls to it for `l1_arg_ptr`-style
expressions and for the explicit pointer casts that compute kernels
emit.

It is defined in both `jit_kernel_stubs.hpp` and `dataflow_api.h` (the
second include guards with `#ifndef __EMULE_LOCAL_L1_TO_PTR_DEFINED`).
This duplication is intentional — kernels that include only
`jit_kernel_stubs.hpp` (compute kernels via the regex rewriter) still
need the function inlined.

Checks run in this order inside `__emule_local_l1_to_ptr`:

1. **Illegal Semaphore Access** — `l1_addr` inside
   `[sem_l1_range_start, sem_l1_range_end)`. Sem accesses are supposed
   to go through `noc_semaphore_*` APIs; a raw pointer dereference
   bypasses the atomic NoC ops and races.
2. **CB Boundary Violation** — `l1_addr` inside one of the
   `__emule_cbs[i].base + cb_size` ranges. If yes, check whether the
   access page is inside the active write window
   `[write_idx, write_idx + reserved)` modulo num_pages or the read
   window `[read_idx, read_idx + waited)`. If outside both, abort.
   Returning early here is important — CB backing memory isn't
   registered in `LiveL1Ranges`, so a CB address that fell through to
   the OOB check below would always produce "Out-of-Bounds Write" even
   for legitimate CB accesses.
3. **Out-of-Bounds Write (L1)** — for addresses at-or-above
   `l1_unreserved_base` only, scan `__emule_l1_tensor_ranges` for a
   matching extent. If none, abort. When a match is found, also append
   the matched packed `(start, end)` to `__emule_l1_resolved_ranges`
   (deduplicated by linear scan, capped at capacity) — this feeds the
   Object Intent check below.
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

**NOC Transfer Alignment** — Three variants in `dataflow_api.h`,
depending on src/dst type (DRAM/L1 × L1/DRAM). The check is on the low
bits matching between src and dst — required by the NOC hardware's
transaction packing, silently corrupts data on silicon when mismatched.

### Object Intent Violation — the complicated one

This is the check the reviewer explicitly called out. It detects
"unintended writes": a kernel that modifies an L1 buffer it never took
a pointer into via the public API. The classic instance is a kernel
that does pointer arithmetic past the end of buffer A and stomps on
buffer B's bytes — both buffers are legitimately allocated, so the OOB
and Padding checks both pass, but B was never the kernel's intended
target.

Detection has 6 stages, split between the runner (host) and the kernel:

**Stage 1: pre-launch snapshot (runner, in `launch_cores`).**
For each core: if `object_intent_strict && tensor_ranges != null`,
allocate an `oi_snapshots` vector. For every live L1 tensor extent,
`memcpy` the current bytes of L1 in that range into a local snapshot.
Held on the stack of the core thread for the launch's duration.

**Stage 2: refuse multi-kernel launches.**
Exact attribution requires being able to tell, after join, which
kernel resolved which pointer. With >1 kernel per core, the per-kernel
resolved-range arrays would have to be merged somehow, and a write
mismatch couldn't be attributed back to a single kernel. We abort with
a friendly diagnostic when `cs.ki_list->size() != 1`, telling the user
to keep ASAN runs single-kernel-per-core.

**Stage 3: kernel-thread stack-local log.**
Each kernel thread gets a `uint64_t local_resolved[64]` on its own
stack and points `__emule_l1_resolved_ranges` /
`_resolved_ranges_count` / `_resolved_ranges_capacity` at it.
Stack-local because:
- One array per kernel thread, no contention.
- Lifetime ties to the kernel thread exactly.
- Capacity 64 is plenty (even pathological kernels touch <10 distinct
  buffers); overflow drops the excess, which biases the post-launch
  comparison toward false positives, not false negatives.

**Stage 4: kernel records intent.**
Every successful resolution inside `__emule_local_l1_to_ptr` (i.e. the
access hit a live tensor extent in OOB check #3 above) appends the
matched `(start, end)` packed pair to `__emule_l1_resolved_ranges`.
Linear-scan dedup keeps the array small. The append is the only
write — `__emule_local_l1_to_ptr` is the only place that writes the
array.

The semantics are: "this kernel intends to touch buffer
`[start, end)`." Crucially, the intent is recorded at *resolution*
time, not at *write* time. A kernel that resolves a pointer into B but
never actually writes through it is still considered to have intent to
touch B (the runner won't flag B even if B's bytes change — though
that wouldn't normally happen).

**Stage 5: merge after join.**
When each kernel thread exits, it copies its `local_resolved` into the
per-core `oi_resolved_single_kernel` vector (only one kernel thread on
single-kernel-per-core launches, so no merge logic — direct copy). The
runner waits for all kernel threads on the core to join.

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
   `if (__emule_asan_enabled() && ...)` (kernel-side, in a hot path).
   Re-read, no caching.
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

- `jit_kernel_stubs.hpp` — extern thread_locals, `__emule_local_l1_to_ptr`,
  `__emule_dram_ptr`.
- `api/cb_api.h` — `__emule_asan_enabled`, CB ops with the always-on
  Reservation Overflow check and the gated NoC Barrier Missing check.
- `api/dataflow/dataflow_api.h` — second copy of `__emule_local_l1_to_ptr`
  (guarded), NOC transfer alignment checks.
- `emule_asan.h` — unified diagnostic trace. `__emule_asan_panic()` (which every
  check calls instead of `abort()`) prints the kernel/core/processor context +
  a symbolized backtrace. See *Diagnostic trace* below.

Tests (tt-metal/tests/tt_metal/tt_metal/api/test_*.cpp) — one file per
sanitizer category.

## Diagnostic trace

Every `[ASAN ERROR]` calls `__emule_asan_panic()` (in `emule_asan.h`) rather than
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
