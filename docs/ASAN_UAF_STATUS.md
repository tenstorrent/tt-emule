# ASan Use-After-Free Status — 2026-04-30

Snapshot of where the ASan UAF-detection work stands at session pause. Captures all
state needed to resume across the three repos (tt-emule, tt-metal-main, UMD).

## TL;DR

- **L1 UAF detection works.** `MeshDispatchFixture.AsanL1BufferUseAfterFree` fires
  `AddressSanitizer: use-after-poison` on a kernel write through a freed
  `MeshBuffer`'s saved address. Wired into Tier 7 regression as a negative test.
- **DRAM UAF detection broken.** `MeshDispatchFixture.AsanDramBufferUseAfterFree`
  added but does not fire because the per-buffer dealloc hook fails to resolve
  the buffer's bank ranges to host pointers — see "Open bug" below. Test is
  wired into Tier 7 anyway so the failure stays visible.
- **Static-inline header refactor landed** for the `__emule_buffer_alloc/free`
  bridge. No more weak/strong symbol resolution dance. When ASan is off, calls
  are elided; when on, they reach the real `__asan_*_memory_region`.
- **Blanket poisoning is opt-in** via `TT_EMULE_ASAN_BLANKET=1`. Default off.
  Enables `initialize_asan_poison`'s blanket coverage of L1/DRAM unreserved
  regions, intended for op-level tests that consistently route through the
  Buffer allocator.
- **Raw I/O auto-unpoisons.** `SWEmuleChip::write_to_device` /
  `read_from_device` call `__emule_buffer_alloc` on the target range before the
  memcpy. Lets `WriteToDeviceL1` and similar raw-API tests work even when
  blanket poison is on.
- **Regression baseline (blanket OFF, default):** `134 passed, 12 failed, 0 skipped`.
  - 11 pre-existing baseline failures (4 wraparound STRIDED + 7 DFB Config
    Validation) — unrelated to ASan.
  - 12th failure is the DRAM UAF test (open bug below).

## Repo state

### `/localdev/arminale/tt-emule` — branch `armin-asan`

Latest commits (newest first):
```
ce5da0c (HEAD) test(asan): wire L1 + DRAM UAF tests into Tier 7
cfecd3a feat(asan): static-inline __emule_buffer_alloc/free in asan_bridge.h
266b5ac docs(asan): close-out regression at 133/11/0 + Status section update
c0cd591 test(asan): standalone positive control for l1_alloc unpoisoning
4ec63ac test(asan): wire AsanL1BufferInBoundsWrite into Tier 7; 132/11/0
```

Clean working tree. 7 commits ahead of `upstream/armin-asan`.

### `/localdev/arminale/tt-metal-main` — branch `armin-asan-allocator`

Latest commits:
```
412261f36d (HEAD) asan: per-range UNRESOLVED diagnostic in dealloc hook; bump UMD
c8c89bab8e asan: switch to static-inline header for __emule_buffer_alloc/free
d9e9c10d1a feat(asan): per-shard accurate sizing via BufferPageMapping; immediately_poisoned trace field
a7f26597a1 test(asan): AsanDramBufferUseAfterFree + create_dram_mesh_buffer helper
069929bf48 fix(asan): define strong __emule_buffer_alloc/free in libtt_metal.so
e5796e9eac test(asan): un-disable AsanL1BufferUseAfterFree as failing diagnostic
5dbbe1fba8 fix(emule/asan): per-buffer alloc hook now actually unpoisons MeshBuffer ranges
```

Clean working tree. UMD submodule pointer at `5b1cdcb7`.

### `/localdev/arminale/tt-metal-main/tt_metal/third_party/umd` — branch `armin-asan-allocator`

Latest commits:
```
5b1cdcb7 (HEAD) asan: opt-in blanket poison via TT_EMULE_ASAN_BLANKET; auto-unpoison in raw I/O
f50aa85c drop weak no-op fallbacks for __emule_buffer_alloc/free
abda4d14 fix(emule/asan): weak no-op fallbacks for __emule_buffer_alloc/free
```

Clean working tree.

## What was diagnosed and fixed this session

### Root cause #1: weak no-op fallbacks shadowed real impl

The `__emule_buffer_alloc/free` bridge calls were resolving to weak no-op
fallbacks in `libtt-umd.so` because `libtt_metal.so` consumed only
`tt-emule::headers` (interface target, no `.o` files), so the strong defs in
`libtt_emule_lib.a` were never linked into the runtime image. Per-buffer
poison silently did nothing. `AsanL1BufferOverflow` passed for the wrong
reason — WH `worker_l1_size = 1499136 ≈ 0x16e000` placed the OOB-write byte
at the L1Pool slot-tail boundary, which has its own (functional) poison.

**Fix:** moved `__emule_buffer_alloc/free` to `static inline` in
`include/tt_emule/asan_bridge.h`. Each TU that includes the header gets its
own internal-linkage copy. No exported symbol → no weak/strong resolution →
no out-of-tree provider can shadow it. Removed the prior strong defs from
`tt-emule/src/kernel_runner.cpp` and the prior weak fallbacks from
`umd/device/chip/sw_emule_chip.cpp`.

### Root cause #2: blanket poison breaks raw-API tests

Once per-buffer poison started actually firing, ~17 tests began failing with
`use-after-poison` because they use access patterns that bypass
`AllocatorImpl`:
- `WriteToDeviceL1(...)` to hard-coded addresses (Quasar compute kernel test
  writes init_values to `1000 * 1024`).
- `Core::l1_alloc` for DFB / semaphores / kernel binaries (tt-emule's own
  bump allocator, not `AllocatorImpl`).
- Kernels reference raw L1 offsets via runtime args (atomic kernels do
  `reinterpret_cast<std::atomic<T>*>(get_arg_val<u32>(0))`).

These aren't bugs — they're documented patterns. The blanket poison from
`SWEmuleChip::initialize_asan_poison` covers the entire unreserved region,
and any of these accesses trips ASan.

**Fix:** made blanket poison opt-in via `TT_EMULE_ASAN_BLANKET=1`. Default
off. Op-level tests that consistently use the `Buffer`/`MeshBuffer` API can
opt in for stronger OOB-to-never-allocated detection. Lower-level tests run
without the blanket and still get UAF detection through the per-buffer
dealloc hook.

Also added `__emule_buffer_alloc(target, size)` calls inside
`SWEmuleChip::write_to_device` / `read_from_device` so even with blanket on,
host-side raw I/O succeeds. Kernel-side accesses (via
`__emule_local_l1_to_ptr` / `__emule_resolve_noc_addr`) bypass these helpers
and still get full ASan coverage.

## Open bug — DRAM UAF resolution

`AsanDramBufferUseAfterFree` is wired into Tier 7 but currently fails. Trace
output (with `TT_EMULE_ASAN_TRACE=1`) shows:

```
[ASAN_TRACE] dealloc buffer=0x70d4687f0590 is_dram=1 num_ranges=12
[ASAN_TRACE] dealloc range UNRESOLVED core=(1,4) is_dram=1 offset=0x3ffff000 size=4096
[ASAN_TRACE] dealloc range UNRESOLVED core=(4,4) is_dram=1 offset=0x7ffff000 size=4096
... (10 more UNRESOLVED) ...
```

`enumerate_buffer_ranges` in `tt_metal/impl/emulation/asan_hooks.cpp` Case 3
correctly enumerates 12 ranges for a replicated DRAM mesh buffer (one per
DRAM core × mesh device). All 12 fail to resolve to a host pointer.

### Where the resolution fails

`resolve_range_host_base` in `asan_hooks.cpp` builds a `tt::umd::CoreCoord`
from `r.logical_core.x / .y` and calls
`SWEmuleChip::core_for_logical(coord, /*is_dram=*/true)`. That helper
(in `tt_metal/third_party/umd/device/chip/sw_emule_chip.cpp:124-141`) does:

```cpp
if (is_dram) {
    // coord.x is the DRAM channel id — look up the registered DRAM core.
    for (const auto& [xy, channel] : dram_core_to_channel_) {
        if (channel == coord.x) {
            ...
        }
    }
    return nullptr;
}
```

The comment says "coord.x is the DRAM channel id" but the caller's coord
came from `allocator->get_logical_core_from_bank_id(bank_id)`, which returns
the bank's *logical core coordinate* — for WH that's coords like `(1,4)`,
`(4,4)`, `(4,1)`, `(2,2)` (per the trace above). `coord.x` values are 1–7,
not channel ids.

`dram_core_to_channel_` is built from
`soc.get_dram_cores()` (sw_emule_chip.cpp:71-77): keys are SOC-descriptor
DRAM core (x,y) coordinates, values are channel ids 0..(num_channels-1).
For WH N150 there are 6 DRAM channels (1 chip × 6 banks); the SOC's DRAM
cores are at coordinates like `(0,0), (0,1), …, (0,11)` in the noc_translated
indexing — channel ids 0..5 map via the SOC's `get_dram_cores()` API to one
or more (x,y) pairs each.

So the `channel == coord.x` comparison only succeeds by coincidence (when
`coord.x` happens to be in [0, 5]), and even then it might pick the wrong
core.

### What needs to happen to fix it

Two-sided question: who's wrong, the caller or the helper?

Option A — fix `core_for_logical` to look up by logical core coord, not channel:
```cpp
if (is_dram) {
    auto it = dram_core_to_channel_.find(tt_xy_pair(coord.x, coord.y));
    if (it == dram_core_to_channel_.end()) return nullptr;
    std::lock_guard<std::mutex> lock(core_mutex_);
    auto core_it = cores_.find(it->first);
    return core_it == cores_.end() ? nullptr : core_it->second.get();
}
```
This treats `coord` as a real (x,y) coordinate matching what the SOC says,
which is what `allocator->get_logical_core_from_bank_id` actually returns.
Update the comment accordingly.

Option B — fix the caller (`resolve_range_host_base`) to translate the
allocator's logical core coord to a channel id first, then pass channel as
`coord.x`. This requires either calling
`tt::umd::Cluster::get_chip(...)::get_soc_descriptor()::get_dram_channel_from_logical_core(...)`
or similar, AND keeping the documented "coord.x is channel id" contract.

**Option A is preferred** — the allocator is the canonical source of truth
for "which logical core does this bank live on", and `core_for_logical`
should accept that coord directly. Treating coord.x as a channel id is an
internal hack that conflicts with how every other caller of
`core_for_logical` would use it.

### Verification after fix

Per-range traces should print resolved bases instead of UNRESOLVED:
```
[ASAN_TRACE] dealloc buffer=... base=0x... size=4096 offset=0x3ffff000 core=(1,4) is_dram=1 immediately_poisoned=1
```

`MeshDispatchFixture.AsanDramBufferUseAfterFree` should fire
`AddressSanitizer: use-after-poison` on the kernel's `noc_async_read`. Run
regression with `TT_EMULE_ASAN_BLANKET` unset; expected `135 passed, 11
failed, 0 skipped`.

## Diagnostic infrastructure left in place

For future ASan integration debugging:

- `TT_EMULE_ASAN_TRACE=1` — every per-buffer alloc/dealloc hook logs to
  stderr (host base, size, offset, core, is_dram, `immediately_poisoned`
  result of `__asan_address_is_poisoned` after the poison call). Lives in
  `asan_hooks.cpp`.
- `TT_EMULE_ASAN_TRACE=1` also dumps `num_ranges` per dealloc + per-range
  UNRESOLVED lines when `resolve_range_host_base` returns null.
- `TT_EMULE_ASAN_PROBE_RT=1` (in `emulated_program_runner.cpp`) — at kernel
  launch, for each runtime arg that looks like an L1 offset (< 4 MB),
  prints whether `host_l1 + arg` is currently poisoned. Useful for
  diagnosing whether the kernel's address points into a poisoned region
  before the kernel runs.
- `TT_EMULE_ASAN_BLANKET=1` — opt-in to blanket initial poison (described
  above).

All env vars off by default. None affect non-ASan builds.

## How to resume

### Just run UAF tests under the current state
```bash
cd /localdev/arminale/tt-metal-main
LD_LIBRARY_PATH=/usr/lib/llvm-20/lib/clang/20/lib/linux \
  cmake --build build_emule_asan -j$(nproc) --target unit_tests_integration

cd /localdev/arminale/tt-emule
export TT_METAL_MOCK_CLUSTER_DESC_PATH=/localdev/arminale/tt-metal-main/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml
export TT_METAL_EMULE_MODE=1 TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_RUNTIME_ROOT=/localdev/arminale/tt-metal-main TT_METAL_HOME=/localdev/arminale/tt-metal-main
export ASAN_OPTIONS=detect_leaks=0:halt_on_error=1:abort_on_error=0
export LD_LIBRARY_PATH=/usr/lib/llvm-20/lib/clang/20/lib/linux:${LD_LIBRARY_PATH:-}
unset ARCH_NAME
rm -rf /tmp/tt_emule_jit_cache_$(id -u)

/localdev/arminale/tt-metal-main/build_emule_asan/test/tt_metal/unit_tests_integration \
  --gtest_filter="MeshDispatchFixture.Asan*UseAfterFree"
```

### Just run the regression
```bash
cd /localdev/arminale/tt-emule
unset TT_EMULE_ASAN_BLANKET   # or set to 1 for blanket coverage
export ASAN_OPTIONS=detect_leaks=0:halt_on_error=1:abort_on_error=0
export LD_LIBRARY_PATH=/usr/lib/llvm-20/lib/clang/20/lib/linux:${LD_LIBRARY_PATH:-}
BUILD_DIR=/localdev/arminale/tt-metal-main/build_emule_asan ./run_regression.sh \
  2>&1 | tee /tmp/asan_regression.log
grep "Results:" /tmp/asan_regression.log
```

Default expected: `Results: 134 passed, 12 failed, 0 skipped`.

### Resume the DRAM UAF fix
1. Read `tt_metal/third_party/umd/device/chip/sw_emule_chip.cpp:124-141`
   (`SWEmuleChip::core_for_logical`).
2. Apply Option A (look up by `(coord.x, coord.y)` directly in
   `dram_core_to_channel_`). Drop the "coord.x is the DRAM channel id"
   comment.
3. Rebuild, run the DRAM UAF test alone, confirm `[ASAN_TRACE]` shows
   resolved bases (no more UNRESOLVED), and that
   `AddressSanitizer: use-after-poison` fires.
4. Run full regression. Expected: `135 passed, 11 failed, 0 skipped`.
5. Remove the diagnostic `num_ranges` and `UNRESOLVED` traces if desired
   (or keep them — they're env-gated and useful for future debugging).
6. Update `IMPLEMENTATION_REPORT.md` test counts and `docs/ASAN.md`
   "What gets caught" table.
7. Commit on tt-metal-main; bump UMD pointer if the fix is in UMD.

### Open todos beyond DRAM UAF

From the original close-out plan
(`docs/PLAN_asan_allocator_integration.md`):
- `~AllocatorImpl` and `override_state` bypass per-buffer dealloc — leaks
  poison across allocator reconfigure. Practically harmless within a single
  test process; deferred.
- NOC + multi-core UAF tests, DRAM overflow JIT-path test — easy follow-on
  once DRAM UAF resolution is fixed.
- Target-scoped ASan flags (drop the `LD_LIBRARY_PATH` build-time
  workaround) — orthogonal, low priority.

## Pointer to authoritative plan file

`/home/arminale/.claude/plans/proud-tickling-floyd.md` — the planning
document for this debugging session. Has the chronological diagnosis trace,
hypothesis-by-hypothesis, with all the dead ends and pivots. Keep alongside
this status doc; this one is for fast resume, that one is for understanding
why the design is the way it is.
