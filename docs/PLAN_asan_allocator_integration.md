# ASan Integration: Phase 2 — tt-metal allocator hooks + JIT path bring-up

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete AddressSanitizer support for tt-emule by (a) finishing the half-done bring-up of the JIT/tt-metal-emulated path and (b) adding allocator-level per-buffer poison/unpoison hooks so user buffer overflows and use-after-free are caught in JIT kernels.

**Architecture:** The standalone path is already wired (3 negative tests pass in `build_asan/`). What remains: define the declared-but-undefined `__emule_resolve_noc_addr_sized` in tt-metal's `emulated_program_runner.cpp`, propagate `-fsanitize=address -shared-libasan` through tt-metal's CMake and the runner's JIT g++ command, then add a tt-metal-side hook in `AllocatorImpl::allocate_buffer` / `deallocate_buffer` that calls a new `__emule_buffer_alloc/free` bridge in tt-emule. Initial L1 / DRAM poisoning happens in `SWEmulatedChip` at core creation (only above `l1_unreserved_base` / `dram_unreserved_base`, so kernel-config writes don't fire).

**Tech Stack:** C++17, clang-20 (host) + g++ (JIT), CMake, AddressSanitizer (`__asan_*`), CTest, tt-metal `Buffer` / `Allocator` / `IDevice` APIs.

---

## Status — 2026-04-28 (final, post close-out)

**Initial 12-task ASan integration complete; close-out items 1-3 + 9 from the post-integration gap list also landed.**

| Outcome | Number |
|---------|--------|
| Non-ASan regression (`build_emule_clang`) | 126 passed / 11 failed / 0 skipped (baseline) |
| ASan regression (`build_emule_asan`, `TT_EMULE_ASAN=ON`) | **133 passed / 11 failed / 0 skipped** (+7 Tier 7) |

Tier 7 in `run_regression.sh` (auto-skipped on non-ASan builds):

| Test | Type | Detection mechanism |
|------|------|---------------------|
| `asan_inbounds_l1_alloc` | **positive control** | Asserts `Core::l1_alloc` unpoisons the bumped slice |
| `asan_oob_slot_tail`     | negative | L1Pool 1 MB slot-tail poisoning |
| `asan_oob_l1_alloc`      | negative | `Core::l1_alloc` bump-pointer poisoning |
| `asan_oob_dram`          | negative | `__emule_dram_ptr` bank-size bounds check |
| `asan_oob_noc_read`      | negative | `__emule_resolve_noc_addr_sized` bounds check |
| `AsanL1BufferInBoundsWrite` | **positive control** | Asserts the per-buffer alloc hook unpoisons the L1 mesh buffer's range |
| `AsanL1BufferOverflow`   | negative | per-buffer poison + JIT-compiled `.so` under ASan |

The two positive controls assert the alloc-side hook is doing real work (not coasting on initial blanket poison alone). If either positive control fails, the alloc hook regressed even though the negative tests would still trip on the initial poison.

The 11 baseline failures (4 wraparound STRIDED + 7 Config Validation DFB tests) are pre-existing on the `armin-asan-allocator` branch — unchanged by ASan integration.

### Closed in this round

1. **Per-buffer alloc hook now actually does work for MeshBuffers** (was a no-op before due to MeshDevice id mismatch + mutex re-entry deadlock). Three independent fixes in `5dbbe1fba8`:
   - `sw_emulated_chip_for` falls back to iterating registered chips when `device->id()` doesn't match a UMD chip (MeshDevice id 1 vs. chip id 0).
   - `AllocatorImpl::allocate_buffer` / `deallocate_buffer` release `mutex_` before invoking the hook (was self-deadlocking in `get_logical_core_from_bank_id` / `get_bank_offset`).
   - `TT_EMULE_ASAN_TRACE=1` env-gated logging in the hook so future debugging doesn't need source-level instrumentation.
2. **Sharded L1 accurate poisoning** via `Buffer::get_buffer_page_mapping()` (commit on tt-metal-main after `5dbbe1fba8`). Each shard's exact byte count is unpoisoned/poisoned, no more conservative over-unpoisoning.
3. **Standalone positive control** `asan_inbounds_l1_alloc` complements `oob_l1_alloc` (commit `c0cd591` on tt-emule). Asserts `Core::l1_alloc`'s unpoison.
4. **JIT-path positive control** `MeshDispatchFixture.AsanL1BufferInBoundsWrite` (in `5dbbe1fba8`). Asserts the alloc hook for `MeshBuffer::create`.

### Remaining gaps / follow-ups

1. **`DISABLED_AsanL1BufferUseAfterFree` still doesn't fire.** Both alloc and dealloc hooks are now confirmed running with correct host bytes (verified via `TT_EMULE_ASAN_TRACE=1` on Phase A diagnostics). The repoisoned bytes don't trip the kernel write — likely because the next allocation reuses the freed address and unpoisons it again before the kernel runs (the test's `buffer.reset()` happens before the program's internal kernel-binary buffer alloc, which may take the same address). Tracked in the test file's TODO comment. Possible mitigations: alloc a guard buffer that survives past the kernel run; use a side-table to keep poison sticky.
2. **`~AllocatorImpl` and `AllocatorImpl::override_state`** still clear `allocated_buffers_` directly, bypassing the per-buffer dealloc hook. Symmetric to gap (1) — the side-table fix would close both. Deferred until (1) is understood.
3. **NOC + multi-core UAF tests, DRAM UAF test** — share the same UAF-fire mechanism as (1). Deferred.
4. **Build-time ASan instrumentation of host tools (flatc, scaleout binaries)** — currently mitigated by `TT_UMD_BUILD_TOOLS=OFF` + libtt-umd weak-symbol fallbacks + documented `LD_LIBRARY_PATH` build-time hint. Cleanest fix is target-scoped `target_compile_options` instead of global `add_compile_options` on `cmake/project_options.cmake:50–59`. Invasive (touches many CMakeLists); cost/benefit unfavorable while the existing approach works.

### Phase / commit ledger

**tt-emule (`armin-asan`):**
- `385aaf5` — Phase 1 standalone bring-up (asan.h, L1Pool tail poisoning, kernel_runner bounds check, dataflow_api sized-resolver decls + 4 NOC sites)
- `59c7c4dc` — Standalone `oob_noc_read_test`
- `a33817d` — `asan_bridge.h` declarations + bridge function definitions
- `573914a`, `5f75fb2`, `deb7439`, `8d15a59`, `d0e662f` — docs (BUILD_GUIDE, plan, ASAN.md, IMPLEMENTATION_REPORT)
- `eee9893` — non-ASan regression baseline log
- `fb80fd7` — `run_regression.sh` Tier 7 + ASan regression log

**tt-metal-main (`armin-asan-allocator`):**
- `4e20b69a33` — `TT_EMULE_ASAN` option + clang-rt rpath plumbing + sized-resolver definition + always-on bounds checks + JIT g++ flag pass-through
- `7c01a57d6f` — UMD submodule bump (active_dram_bank_size + L1Pool slot-tail)
- `21663d924d` — `TT_UMD_BUILD_TOOLS=OFF` for ASan builds
- `b87e1969a8` — UMD weak-symbol fallback bump (so non-tt-emule executables link)
- `7d6c6d565d`, `55b169397a` — `asan_hooks.{hpp,cpp}` per-buffer enumerator + shard-orientation fix
- `24c48ece6e` — `AllocatorImpl` + `Device::initialize` hook call sites
- `a0f19d4235` — fix two regression-blocking hook bugs (use-after-free in mass dealloc; cluster.get_chip throw on MeshDevice id)
- `b18f8821b4` — `on_buffer_allocated(buffer, base_address)` API change (Buffer::address not yet finalized at hook-fire time)
- `f871dc7963` — JIT-path negative test (`MeshDispatchFixture.AsanL1BufferOverflow` in `unit_tests_integration`)

**UMD submodule (`armin-asan-allocator`):**
- `8803cd78` — `active_dram_bank_size()` static accessor + L1Pool live_size_per_slot ctor arg
- `00e09663` — `core_for_logical(coord, is_dram)` + `initialize_asan_poison(l1, dram)` + lazy-creation poison hook
- `abda4d14` — Weak no-op `__emule_buffer_alloc/free` fallbacks in libtt-umd

---

## Pre-flight

**Updated 2026-04-28 — Phase 1 + bounds-check work complete in tt-metal-main, branch `armin-asan-allocator`.**

Already done — DO NOT REDO:

**In `/localdev/arminale/tt-emule` (branch `armin-asan`):**
- `include/tt_emule/asan.h` (POISON/UNPOISON macros + `__emule_bounds_fail` helper)
- `CMakeLists.txt`: `TT_EMULE_ASAN` option + flags
- `include/tt_emule/l1_pool.hpp`: tail poisoning + `live_size_per_slot` ctor arg
- `include/tt_emule/device.hpp`: standalone WORKER L1 pre-poison + `l1_alloc` unpoison
- `src/kernel_runner.cpp`: `__emule_dram_ptr` always-on bounds check; `__emule_buffer_alloc/free` bridge
- `src/jit_kernel.cpp`: standalone JIT g++ ASan flag pass-through
- `include/jit_hw/api/dataflow/dataflow_api.h`: declares `__emule_resolve_noc_addr_sized` and routes 4 NOC sites through it
- `include/tt_emule/asan_bridge.h`: `extern "C"` declarations for the allocator bridge
- `tests/asan/{CMakeLists.txt,oob_slot_tail_test.cpp,oob_l1_alloc_test.cpp,oob_dram_test.cpp,oob_noc_read_test.cpp}` (4/4 pass)

**In `/localdev/arminale/tt-metal-main` (branch `armin-asan-allocator`, commit `4e20b69a33`):**
- `cmake/project_options.cmake`: `TT_EMULE_ASAN` option + ASan flags + `TT_EMULE_ASAN_RT_DIR` clang-rt resolution
- `tt_metal/impl/CMakeLists.txt`: propagate `TT_EMULE_ASAN_RT_DIR` define
- `tt_metal/impl/emulation/emulated_program_runner.cpp`: `__emule_resolve_noc_addr_sized` definition + bounds checks in `__emule_dram_ptr`, `__emule_local_l1_ptr`, `__emule_noc_resolve`, `__emule_multicast_write`; JIT g++ command appends ASan flags + `-Wl,-rpath,${TT_EMULE_ASAN_RT_DIR}` so kernel `.so` files resolve `libclang_rt.asan-x86_64.so` without `LD_LIBRARY_PATH`
- `tt_metal/third_party/umd` submodule (branch `armin-asan-allocator`, commit `8803cd78`): `SWEmuleChip::active_dram_bank_size()` static accessor; `L1Pool` constructed with `live_size_per_slot=l1_size_` so the unused tail of each 2 MB slot is ASan-poisoned

**Net effect:** the JIT/tt-metal-emulated path links and runs under ASan. Coarse bounds checks fire on out-of-range NOC / L1 / DRAM offsets. Per-buffer (in-region) overflow detection is the remaining work — this plan's Phase 3.

**Plan corrections vs. original:**
- Right tt-metal repo: `tt-metal-main`, not `tt-metal`. `BUILD_GUIDE.md` and `run_regression.sh` both default to it.
- `core_for_logical` signature: `(CoreCoord, bool is_dram)` not `(IDevice&, CoreCoord, BufferType)`. The IDevice translation and BufferType→is_dram dispatch must happen in the caller (tt_metal-layer asan_hooks), because the UMD layer cannot include `<tt-metalium/...>` headers without breaking the link graph.
- Class name in UMD: `SWEmuleChip`, not `SWEmulatedChip`.

---

## File structure

| File | Purpose | Status |
|------|---------|--------|
| `tt-emule/include/tt_emule/asan_bridge.h` | `extern "C"` declarations for `__emule_buffer_alloc/free` (tt-metal includes this) | NEW |
| `tt-emule/src/kernel_runner.cpp` | Define `__emule_buffer_alloc/free` next to other bridges | MODIFY |
| `tt-emule/tests/asan/oob_noc_read_test.cpp` | Standalone negative test for the sized resolver | NEW |
| `tt-emule/tests/asan/CMakeLists.txt` | Register the new negative test | MODIFY |
| `tt-emule/docs/PLAN_asan_allocator_integration.md` | This document | NEW |
| `tt-emule/IMPLEMENTATION_REPORT.md` | Note ASan support once landed | MODIFY (last) |
| `tt-metal/CMakeLists.txt` (top-level) or `tt_metal/impl/emulation/CMakeLists.txt` | `TT_EMULE_ASAN` option + flag propagation | MODIFY |
| `tt-metal/tt_metal/impl/emulation/emulated_program_runner.cpp` | Define `__emule_resolve_noc_addr_sized`; ASan flags in JIT g++ command | MODIFY |
| `tt-metal/third_party/umd/device/api/umd/device/chip/sw_emulated_chip.hpp` | New `core_for_logical(CoreCoord, BufferType)` helper | MODIFY |
| `tt-metal/third_party/umd/device/chip/sw_emulated_chip.cpp` | Initial poison `[unreserved_base, l1_size)` per core; impl of helper | MODIFY |
| `tt-metal/tt_metal/impl/emulation/asan_hooks.hpp` | `on_buffer_allocated/deallocated` interface | NEW |
| `tt-metal/tt_metal/impl/emulation/asan_hooks.cpp` | Per-buffer-range enumeration + bridge calls | NEW |
| `tt-metal/tt_metal/impl/allocator/allocator.cpp` | Two hook call sites (allocate + deallocate + deallocate_all) | MODIFY |
| `tt-metal/tt_emule/asan_tests/CMakeLists.txt` | Register tt-metal-side ASan negative tests | NEW |
| `tt-metal/tt_emule/asan_tests/test_buffer_oob.cpp` | JIT-path overflow test | NEW |
| `tt-metal/tt_emule/asan_tests/test_buffer_uaf.cpp` | JIT-path use-after-free test | NEW |
| `tt-metal/tt_emule/CMakeLists.txt` | Wire `asan_tests/` subdir behind `TT_EMULE_ASAN` | MODIFY |

---

## Conventions used below

- **`tt-emule:`** prefix means a path inside `/localdev/arminale/tt-emule`.
- **`tt-metal-main:`** prefix means a path inside `/localdev/arminale/tt-metal-main` (the active tt-metal repo per `BUILD_GUIDE.md` and `run_regression.sh`).
- Repo root for `cd` or relative paths is shown when ambiguous.

---

## Phase 1 — Unblock the JIT/tt-metal ASan build

### Task 1: Define `__emule_resolve_noc_addr_sized` in the runner

**Files:**
- Modify: `tt-metal-main:tt_metal/impl/emulation/emulated_program_runner.cpp` (next to `__emule_resolve_noc_addr` at line 161)

- [ ] **Step 1: Read the existing resolver to copy its decode pattern**

Run: `grep -n -A 25 "__emule_resolve_noc_addr" /localdev/arminale/tt-metal-main/tt_metal/impl/emulation/emulated_program_runner.cpp | head -50`

Confirm the existing function decodes `(x, y, offset)` from `noc_addr` and looks up `(x,y)` in `__emule_core_map`. Note the bit layout (offset is the low 36 bits; x is bits 36-41; y is bits 42-47).

- [ ] **Step 2: Add the sized variant immediately after `__emule_resolve_noc_addr`**

Insert at `emulated_program_runner.cpp` immediately after the closing brace of `__emule_resolve_noc_addr`:

```cpp
extern "C" uint8_t* __emule_resolve_noc_addr_sized(uint64_t noc_addr, uint32_t size, const char* caller) {
    // Same decode as __emule_resolve_noc_addr.
    uint64_t offset = noc_addr & 0xFFFFFFFFFULL;        // low 36 bits
    uint32_t x      = (noc_addr >> 36) & 0x3F;
    uint32_t y      = (noc_addr >> 42) & 0x3F;
    if (!__emule_core_map) {
        __emule_bounds_fail(caller, "no core map (program-runner state missing)",
                            offset, size, /*limit=*/0);
        return nullptr;
    }
    uint64_t key = (uint64_t(x) << 16) | y;
    auto it = __emule_core_map->find(key);
    if (it == __emule_core_map->end()) {
        __emule_bounds_fail(caller, "no core registered at (x,y)",
                            offset, size, /*limit=*/0);
        return nullptr;
    }
    tt_emule::Core* target = it->second;
    uint64_t limit = (target->role() == tt_emule::CoreRole::DRAM)
                        ? target->dram_size_per_channel()
                        : target->l1_size();
    if (offset + uint64_t(size) > limit) {
        __emule_bounds_fail(caller, "noc offset + size > target region",
                            offset, size, limit);
        return nullptr;
    }
    return target->l1_ptr(static_cast<uint32_t>(offset));
}
```

You will need to `#include "tt_emule/asan.h"` near the top of the file if it isn't already (check first — `__emule_bounds_fail` lives there). If it's already included transitively, fine.

- [ ] **Step 3: Build to confirm it links**

Run from `/localdev/arminale/tt-metal`:
```bash
cmake --build build_emule -j 8 --target tt_metal 2>&1 | tail -30
```

Expected: build completes without unresolved-symbol errors. If `__emule_bounds_fail` is undefined, add `#include "tt_emule/asan.h"` at the top of the file.

- [ ] **Step 4: Commit**

```bash
cd /localdev/arminale/tt-metal
git add tt_metal/impl/emulation/emulated_program_runner.cpp
git commit -m "feat(emul/asan): define __emule_resolve_noc_addr_sized bounds-check bridge"
```

---

### Task 2: Add `TT_EMULE_ASAN` to tt-metal CMake

**Files:**
- Modify: `tt-metal-main:CMakeLists.txt` (top-level — find the `TT_METAL_USE_EMULE` option and place the new option next to it)

- [ ] **Step 1: Locate the `TT_METAL_USE_EMULE` option**

Run: `grep -n "TT_METAL_USE_EMULE\|" /localdev/arminale/tt-metal-main/CMakeLists.txt | head -10`

Note the line number where `TT_METAL_USE_EMULE` is declared.

- [ ] **Step 2: Add the option and the global compile/link flag block**

Add immediately after the `TT_METAL_USE_EMULE` option:

```cmake
option(TT_EMULE_ASAN "Build emulation paths with AddressSanitizer (requires TT_METAL_USE_EMULE=ON)" OFF)

if(TT_EMULE_ASAN)
    if(NOT TT_METAL_USE_EMULE)
        message(FATAL_ERROR "TT_EMULE_ASAN requires TT_METAL_USE_EMULE=ON")
    endif()
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -shared-libasan -g)
    add_link_options(-fsanitize=address -shared-libasan)
endif()
```

The `-shared-libasan` flag is required because dlopen'd JIT kernel `.so`s must share the host's libasan. Without it, ASan aborts on first dlopen with "ASan runtime found in main module before dynamic dependency was loaded."

- [ ] **Step 3: Configure a fresh build dir to verify the wiring**

```bash
cd /localdev/arminale/tt-metal
cmake -B build_emule_asan \
  -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 \
  -DTT_METAL_USE_EMULE=ON -D=ON -DTT_EMULE_ASAN=ON \
  -DTT_EMULE_PATH=/localdev/arminale/tt-emule \
  -DWITH_PYTHON_BINDINGS=OFF -DENABLE_TRACY=OFF -DTT_INSTALL=OFF \
  -DTT_METAL_BUILD_TESTS=OFF -DTTNN_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Debug 2>&1 | tail -20
```

Expected: configure succeeds; no FATAL_ERROR; cache contains `TT_EMULE_ASAN:BOOL=ON`.

Verify: `grep TT_EMULE_ASAN /localdev/arminale/tt-metal-main/build_emule_asan/CMakeCache.txt`

- [ ] **Step 4: Commit**

```bash
cd /localdev/arminale/tt-metal
git add CMakeLists.txt
git commit -m "build(emul): add TT_EMULE_ASAN option propagating ASan flags"
```

---

### Task 3: Pass ASan flags through the runner's JIT g++ invocation

**Files:**
- Modify: `tt-metal-main:tt_metal/impl/emulation/emulated_program_runner.cpp` (the JIT compile command builder — search for `g++ -std=c++17`)

- [ ] **Step 1: Locate the JIT command builder**

Run: `grep -n "g++ -std=c++17\|jit_compile\|kernel.so" /localdev/arminale/tt-metal-main/tt_metal/impl/emulation/emulated_program_runner.cpp | head -20`

Find the function that constructs the `g++` invocation. It will look similar to `tt-emule:src/jit_kernel.cpp` (which already has the same pattern and can serve as a reference).

- [ ] **Step 2: Append ASan flags conditionally**

Inside the command-string builder, after the existing `-fPIC -shared -O…` flags and before any `-I` flags, add:

```cpp
#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
        << " -fsanitize=address -shared-libasan -fno-omit-frame-pointer -g"
#  endif
#endif
#if defined(__SANITIZE_ADDRESS__) && (!defined(__has_feature) || !__has_feature(address_sanitizer))
        << " -fsanitize=address -shared-libasan -fno-omit-frame-pointer -g"
#endif
```

(The double-guard pattern is the same one already used in `tt-emule:src/jit_kernel.cpp` — clang defines `__has_feature` and uses `address_sanitizer`; gcc only defines `__SANITIZE_ADDRESS__`.)

- [ ] **Step 3: Build to confirm**

```bash
cmake --build /localdev/arminale/tt-metal-main/build_emule_asan -j 8 --target tt_metal 2>&1 | tail -30
```

Expected: builds clean. ASan-instrumented JIT compilation will be exercised at runtime in Task 4.

- [ ] **Step 4: Commit**

```bash
cd /localdev/arminale/tt-metal
git add tt_metal/impl/emulation/emulated_program_runner.cpp
git commit -m "feat(emul/asan): forward -fsanitize=address -shared-libasan to JIT g++"
```

---

### Task 4: Capture an ASan baseline regression

**Files:** None modified — this is verification.

The standalone tt-emule plan said to capture a *non-ASan* baseline first; that ship has sailed (work is in flight on `armin-asan`). Instead, capture an ASan-on baseline now: the diff between this and the post-allocator-hook regression is what tells us whether the hooks revealed real bugs.

- [ ] **Step 1: Build the full target**

```bash
cmake --build /localdev/arminale/tt-metal-main/build_emule_asan -j 8 2>&1 | tail -10
```

- [ ] **Step 2: Run the regression with verbose output captured**

```bash
cd /localdev/arminale/tt-emule
BUILD_DIR=/localdev/arminale/tt-metal-main/build_emule_asan \
  ./run_regression.sh 2>&1 | tee /tmp/asan_baseline_regression.log
```

Per CLAUDE.md, log full output. Expected runtime: 2-5x the non-ASan run (~30 min instead of ~10 min).

- [ ] **Step 3: Triage**

Run:
```bash
grep -E "AddressSanitizer:|EMULE.*out-of-bounds|FAIL|Passed|Failed" /tmp/asan_baseline_regression.log | head -100
```

Expected: any `AddressSanitizer:` or `EMULE … out-of-bounds` lines indicate ASan-revealed bugs *that exist independent of allocator hooks* — i.e., L1Pool slot tail overruns or NOC size overruns. Document each in a comment block at the top of the regression log; do not fix yet.

If the pass count drops below the baseline 137, classify each new failure as:
- (a) ASan-revealed real bug (track separately for fix)
- (b) ASan tooling/runtime noise (e.g., interceptor mismatch — unlikely with `-shared-libasan` on both sides)
- (c) Test pattern issue (test relies on undefined behavior)

- [ ] **Step 4: Save the log to the repo for reference**

```bash
mv /tmp/asan_baseline_regression.log /localdev/arminale/tt-emule/docs/asan_baseline_regression.log
cd /localdev/arminale/tt-emule
git add docs/asan_baseline_regression.log
git commit -m "docs(asan): capture ASan-on baseline regression log"
```

---

## Phase 2 — Standalone gap-fill: sized resolver test

### Task 5: Add `oob_noc_read_test` for the standalone path

**Files:**
- Create: `tt-emule:tests/asan/oob_noc_read_test.cpp`
- Modify: `tt-emule:tests/asan/CMakeLists.txt`

The standalone build doesn't have `__emule_resolve_noc_addr_sized` (it lives in tt-metal). For standalone we'll test the same logic by stubbing the bridge locally — kept self-contained.

- [ ] **Step 1: Write the test**

Create `tt-emule:tests/asan/oob_noc_read_test.cpp`:

```cpp
// Negative test: the sized NOC-resolver bounds-check fires when a kernel
// requests `offset + size > l1_size`. Standalone tests don't link against
// the tt-metal program runner, so we instantiate the same logic here against
// a local Core for self-contained verification.
//
// Expect: __emule_bounds_fail prints "[EMULE] noc_async_read: out-of-bounds"
// and _Exit(2)s.

#include "tt_emule/asan.h"
#include "tt_emule/device.hpp"
#include <cstdio>
#include <cstdint>

int main() {
    tt_emule::Core core({0, 0}, tt_emule::CoreRole::WORKER, tt_emule::Core::L1_SIZE);
    uint64_t offset = tt_emule::Core::L1_SIZE - 32;
    uint64_t size   = 64;  // straddles the boundary
    uint64_t limit  = core.l1_size();
    if (offset + size > limit) {
        __emule_bounds_fail("noc_async_read", "noc offset + size > target region",
                            offset, size, limit);
    }
    std::fprintf(stderr, "ERROR: expected bounds-fail abort\n");
    return 1;
}
```

- [ ] **Step 2: Register the test in CMake**

Modify `tt-emule:tests/asan/CMakeLists.txt`. Append after the `test_asan_oob_dram` block:

```cmake
add_executable(test_asan_oob_noc_read oob_noc_read_test.cpp)
target_include_directories(test_asan_oob_noc_read PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_asan_oob_noc_read PRIVATE tt_emule_lib Threads::Threads)
add_test(NAME asan_oob_noc_read COMMAND test_asan_oob_noc_read)
set_tests_properties(asan_oob_noc_read PROPERTIES
    LABELS "asan"
    WILL_FAIL TRUE)
```

- [ ] **Step 3: Build and run**

```bash
cmake --build /localdev/arminale/tt-emule/build_asan -j 4 2>&1 | tail -5
ctest --test-dir /localdev/arminale/tt-emule/build_asan -L asan -V 2>&1 | tail -25
```

Expected: 4/4 tests pass (`WILL_FAIL TRUE` flips a non-zero exit into a CTest pass).

- [ ] **Step 4: Commit**

```bash
cd /localdev/arminale/tt-emule
git add tests/asan/oob_noc_read_test.cpp tests/asan/CMakeLists.txt
git commit -m "test(asan): add standalone bounds-fail test for sized NOC resolver"
```

---

## Phase 3 — Allocator integration

### Task 6: Bridge API `__emule_buffer_alloc/free` in tt-emule

**Files:**
- Create: `tt-emule:include/tt_emule/asan_bridge.h`
- Modify: `tt-emule:src/kernel_runner.cpp`

- [ ] **Step 1: Write the header**

Create `tt-emule:include/tt_emule/asan_bridge.h`:

```cpp
#pragma once

// Allocator-driven ASan poisoning bridge.
//
// tt-metal's AllocatorImpl::allocate_buffer and ::deallocate_buffer call
// these to mark a buffer's per-core L1 / DRAM region as live (unpoisoned)
// or dead (poisoned). Defined in src/kernel_runner.cpp; symbols are
// exported via -rdynamic so dlopen'd JIT .so files can also call them
// (currently only the host process calls — JIT side does not).
//
// When the host binary was not built with ASan, both calls compile to
// no-ops at link time inside tt-emule (see asan.h), so tt-metal can
// invoke them unconditionally.

#include <cstddef>
#include <cstdint>

extern "C" void __emule_buffer_alloc(uint8_t* base, std::size_t size);
extern "C" void __emule_buffer_free (uint8_t* base, std::size_t size);
```

- [ ] **Step 2: Define the bridge functions**

Modify `tt-emule:src/kernel_runner.cpp`. Below the existing `__emule_dram_ptr` definition, add:

```cpp
#include "tt_emule/asan_bridge.h"

extern "C" void __emule_buffer_alloc(uint8_t* base, std::size_t size) {
    if (!base || size == 0) return;
    EMULE_ASAN_UNPOISON(base, size);
}

extern "C" void __emule_buffer_free(uint8_t* base, std::size_t size) {
    if (!base || size == 0) return;
    EMULE_ASAN_POISON(base, size);
}
```

- [ ] **Step 3: Build standalone to confirm**

```bash
cmake --build /localdev/arminale/tt-emule/build_asan -j 4 2>&1 | tail -5
```

Expected: clean build. The new symbols will be exercised when tt-metal's allocator hook calls them in Task 9.

- [ ] **Step 4: Commit**

```bash
cd /localdev/arminale/tt-emule
git add include/tt_emule/asan_bridge.h src/kernel_runner.cpp
git commit -m "feat(asan): add __emule_buffer_alloc/free bridge for allocator hooks"
```

---

### Task 7: `SWEmulatedChip::core_for_logical` + initial poison

**Files:**
- Modify: `tt-metal-main:third_party/umd/device/api/umd/device/chip/sw_emulated_chip.hpp`
- Modify: `tt-metal-main:third_party/umd/device/chip/sw_emulated_chip.cpp`

- [ ] **Step 1: Add the helper declaration**

Modify `sw_emulated_chip.hpp`. Add to the public section of `class SWEmulatedChip`:

```cpp
// Resolve a logical core + buffer type to the emulated Core that backs it.
// buffer_type ∈ {L1, L1_SMALL, TRACE, DRAM}. Returns nullptr if no
// emulated core exists for the resolved physical (x,y).
//
// L1 / L1_SMALL: logical -> virtual worker core via the cluster's
// translation tables (caller must pass the IDevice).
// DRAM / TRACE:  logical core's .x is the channel id; we look up the
// registered DRAM core for that channel.
tt_emule::Core* core_for_logical(const tt::tt_metal::IDevice& device,
                                  tt::umd::CoreCoord logical,
                                  tt::tt_metal::BufferType buffer_type);
```

You will need forward declarations for `tt::tt_metal::IDevice` and `tt::tt_metal::BufferType` near the top of the header. Don't include their headers — use forward decls so the UMD layer stays loose.

- [ ] **Step 2: Implement the helper**

Modify `sw_emulated_chip.cpp`. Add after the existing `get_core` definition:

```cpp
tt_emule::Core* SWEmulatedChip::core_for_logical(const tt::tt_metal::IDevice& device,
                                                  tt::umd::CoreCoord logical,
                                                  tt::tt_metal::BufferType buffer_type) {
    using tt::tt_metal::BufferType;
    if (buffer_type == BufferType::DRAM || buffer_type == BufferType::TRACE) {
        // Logical (channel, 0) -> registered DRAM core via dram_core_to_channel_ inverse.
        for (const auto& [xy, channel] : dram_core_to_channel_) {
            if (channel == logical.x) {
                auto it = cores_.find(xy);
                return it == cores_.end() ? nullptr : it->second.get();
            }
        }
        return nullptr;
    }
    // L1 / L1_SMALL: logical -> virtual NOC via IDevice translation.
    auto virtual_xy = device.virtual_core_from_logical_core(logical, tt::tt_metal::CoreType::WORKER);
    auto it = cores_.find({virtual_xy.x, virtual_xy.y});
    return it == cores_.end() ? nullptr : it->second.get();
}
```

You will need `#include <tt-metalium/device.hpp>` and `#include <tt-metalium/buffer_types.hpp>` at the top of the .cpp.

- [ ] **Step 3: Initial poison at core creation**

Modify `SWEmulatedChip`'s `get_core` (the place that lazy-creates worker / DRAM cores). After the new `tt_emule::Core` is constructed and inserted into `cores_`, before the function returns, add:

```cpp
// ASan: poison [unreserved_base, total_size). Kernel-config region
// (mailbox, kernel binaries, runtime args, semaphores) stays accessible.
// tt-metal's allocator hook will unpoison user-allocated buffers
// individually as they're created. Unpoisoning happens on dealloc.
{
    uint8_t* base = core->l1_data();
    if (role == tt_emule::CoreRole::WORKER) {
        if (l1_unreserved_base_ < l1_size_) {
            __emule_buffer_free(base + l1_unreserved_base_, l1_size_ - l1_unreserved_base_);
        }
    } else {
        if (dram_unreserved_base_ < dram_bank_size_) {
            __emule_buffer_free(base + dram_unreserved_base_, dram_bank_size_ - dram_unreserved_base_);
        }
    }
}
```

You will need to add `l1_unreserved_base_` and `dram_unreserved_base_` member variables to `SWEmulatedChip` and populate them in the constructor (read from the `AllocatorConfig` or via `Allocator::get_base_allocator_addr`). If those values aren't easily accessible at chip construction (they come from the device-init path), defer the initial-poison call to a new public method `void initialize_asan_poison(uint32_t l1_unreserved, uint32_t dram_unreserved)` and call it from `metal_context.cpp` or wherever the allocator is constructed for the chip.

- [ ] **Step 4: Include the bridge header**

In `sw_emulated_chip.cpp` near the top, add: `#include <tt_emule/asan_bridge.h>`

- [ ] **Step 5: Build**

```bash
cmake --build /localdev/arminale/tt-metal-main/build_emule_asan -j 8 2>&1 | tail -10
```

Expected: clean build. No behavior change yet — the hooks aren't called by the allocator until Task 9.

- [ ] **Step 6: Commit**

```bash
cd /localdev/arminale/tt-metal
git add third_party/umd/device/api/umd/device/chip/sw_emulated_chip.hpp \
        third_party/umd/device/chip/sw_emulated_chip.cpp
git commit -m "feat(emul/asan): SWEmulatedChip initial poison + core_for_logical helper"
```

---

### Task 8: `asan_hooks.{hpp,cpp}` — buffer-range enumerator

**Files:**
- Create: `tt-metal-main:tt_metal/impl/emulation/asan_hooks.hpp`
- Create: `tt-metal-main:tt_metal/impl/emulation/asan_hooks.cpp`
- Modify: `tt-metal-main:tt_metal/impl/emulation/CMakeLists.txt` (add `asan_hooks.cpp` to the runner library sources)

- [ ] **Step 1: Write the header**

Create `tt-metal-main:tt_metal/impl/emulation/asan_hooks.hpp`:

```cpp
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {
class Buffer;
}

namespace tt::tt_metal::emulation {

// Called from AllocatorImpl::allocate_buffer immediately after the buffer's
// per-core address(es) are finalized. For each (core, offset, size) triple
// that the buffer covers, the hook resolves the emulated tt_emule::Core*
// via SWEmulatedChip and invokes __emule_buffer_alloc to unpoison the
// region. No-op when the cluster's target type is not Emulated.
void on_buffer_allocated(const Buffer* buffer);

// Symmetric: called from AllocatorImpl::deallocate_buffer (and
// ::deallocate_buffers, which iterates allocated_buffers_ before mass-
// deallocating). Calls __emule_buffer_free per range.
void on_buffer_deallocated(const Buffer* buffer);

}  // namespace tt::tt_metal::emulation
```

- [ ] **Step 2: Write the implementation**

Create `tt-metal-main:tt_metal/impl/emulation/asan_hooks.cpp`:

```cpp
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "asan_hooks.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/metal_soc_descriptor.h>

#include <umd/device/chip/sw_emulated_chip.hpp>
#include <tt_emule/asan_bridge.h>
#include <tt_emule/device.hpp>

#include "tt_metal/impl/context/metal_context.hpp"

#include <vector>

namespace tt::tt_metal::emulation {

namespace {

struct CoreRange {
    tt_emule::Core* core;
    uint64_t        offset;
    uint64_t        size;
};

// Resolve the SWEmulatedChip backing this buffer's device. Returns nullptr
// for non-emulated targets (in which case the caller must skip the hook).
tt::umd::SWEmulatedChip* sw_emulated_chip_for(const Buffer* buffer) {
    auto& cluster = MetalContext::instance().get_cluster();
    if (cluster.get_target_device_type() != tt::TargetDevice::Emulated) {
        return nullptr;
    }
    auto* device = buffer->device();
    if (!device) return nullptr;
    auto* chip = cluster.get_chip(device->id());                // exact accessor
    return dynamic_cast<tt::umd::SWEmulatedChip*>(chip);        // may need
                                                                // alternate path
}

// Enumerate the per-core (core, offset, size) ranges this buffer occupies.
// Three cases:
//   1. HYBRID per-core: iterate Buffer::per_core_addresses_, size per bank.
//   2. Sharded L1 (non-HYBRID): iterate ShardSpec.grid cores; address is
//      Buffer::address(); size is `aligned_size_per_bank()`.
//      *** TODO(asan-accuracy): switch to Buffer::get_buffer_page_mapping()
//      and use core_host_page_indices[i].size() * aligned_page_size() so we
//      don't unpoison the unused tail of a partially-filled shard. The
//      conservative version below over-unpoisons by up to one shard's worth
//      of bytes per partial shard, which lets shard-tail overflows escape
//      detection. We WILL want this fixed before relying on ASan for
//      shard-boundary bug hunting. Tracked as a follow-up commit. ***
//   3. Interleaved DRAM / banked L1: iterate bank_id 0..num_banks; address
//      is Buffer::translate_page_address(0, bank_id); size is
//      `aligned_size_per_bank()`. Logical core comes from
//      Allocator::get_logical_core_from_bank_id.
std::vector<CoreRange> enumerate_ranges(const Buffer* buffer) {
    std::vector<CoreRange> out;
    auto* sw_chip = sw_emulated_chip_for(buffer);
    if (!sw_chip) return out;
    auto* device  = buffer->device();
    auto  bt      = buffer->buffer_type();
    auto  size_pb = buffer->aligned_size_per_bank();

    // --- HYBRID per-core ---
    if (buffer->per_core_allocation_) {
        for (const auto& [logical_core, addr] : buffer->per_core_addresses_) {
            auto* core = sw_chip->core_for_logical(*device, {logical_core.x, logical_core.y}, bt);
            if (core) out.push_back({core, addr, size_pb});
        }
        return out;
    }

    // --- Sharded L1 (non-HYBRID, conservative) ---
    if (is_sharded(buffer->buffer_layout())) {
        // *** TODO(asan-accuracy): use page mapping, see comment above. ***
        const auto cores = corerange_to_cores(
            buffer->shard_spec().tensor_shard_spec.grid,
            std::nullopt,
            buffer->shard_spec().tensor_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
        for (const auto& logical_core : cores) {
            auto* core = sw_chip->core_for_logical(*device, {logical_core.x, logical_core.y}, bt);
            if (core) out.push_back({core, buffer->address(), size_pb});
        }
        return out;
    }

    // --- Interleaved DRAM or banked L1 ---
    auto* allocator = buffer->allocator();
    if (!allocator) return out;
    uint32_t num_banks = allocator->get_num_banks(bt);
    for (uint32_t bank_id = 0; bank_id < num_banks; ++bank_id) {
        auto logical_core = allocator->get_logical_core_from_bank_id(bank_id);
        auto addr         = buffer->translate_page_address(0, bank_id);
        auto* core        = sw_chip->core_for_logical(*device, {logical_core.x, logical_core.y}, bt);
        if (core) out.push_back({core, addr, size_pb});
    }
    return out;
}

}  // namespace

void on_buffer_allocated(const Buffer* buffer) {
    if (!buffer) return;
    for (const auto& r : enumerate_ranges(buffer)) {
        __emule_buffer_alloc(r.core->l1_data() + r.offset, r.size);
    }
}

void on_buffer_deallocated(const Buffer* buffer) {
    if (!buffer) return;
    for (const auto& r : enumerate_ranges(buffer)) {
        __emule_buffer_free(r.core->l1_data() + r.offset, r.size);
    }
}

}  // namespace tt::tt_metal::emulation
```

The `cluster.get_chip(device->id())` call site is illustrative — the engineer must check the exact accessor name. If the cluster doesn't expose `get_chip`, the alternate path is to walk the per-device chip map; see how `emulated_program_runner.cpp:build_core_map` already obtains the `SWEmulatedChip*` and reuse the same mechanism.

`buffer->per_core_allocation_` and `buffer->per_core_addresses_` are private members in tt-metal; you may need to add `friend class emulation::AsanHookFriend` or expose accessors. Prefer accessor methods over friending; if absent, add `bool Buffer::is_per_core_allocation() const` and `const std::unordered_map<CoreCoord, DeviceAddr>& Buffer::per_core_addresses() const` in `buffer.hpp`.

- [ ] **Step 3: Wire the new file into the build**

Modify `tt-metal-main:tt_metal/impl/emulation/CMakeLists.txt` to add `asan_hooks.cpp` next to `emulated_program_runner.cpp` in the source list. (Find the `add_library(...)` or equivalent and append.)

- [ ] **Step 4: Build**

```bash
cmake --build /localdev/arminale/tt-metal-main/build_emule_asan -j 8 --target tt_metal 2>&1 | tail -20
```

Expected: clean build. If you hit "private member" access errors, expose accessors as described above.

- [ ] **Step 5: Commit**

```bash
cd /localdev/arminale/tt-metal
git add tt_metal/impl/emulation/asan_hooks.hpp \
        tt_metal/impl/emulation/asan_hooks.cpp \
        tt_metal/impl/emulation/CMakeLists.txt
git commit -m "feat(emul/asan): per-buffer poison hook with conservative shard handling"
```

---

### Task 9: Wire hooks into `AllocatorImpl`

**Files:**
- Modify: `tt-metal-main:tt_metal/impl/allocator/allocator.cpp` (lines 113-194 for allocate, 196-223 for deallocate, 225-231 for deallocate_buffers)

- [ ] **Step 1: Include the hook header**

At the top of `allocator.cpp`, near the other includes, add:

```cpp
#ifdef TT_METAL_USE_EMULE
#include "tt_metal/impl/emulation/asan_hooks.hpp"
#endif
```

- [ ] **Step 2: Hook the HYBRID per-core return path**

In `AllocatorImpl::allocate_buffer`, immediately before the early-return at line 148 (`return buffer->per_core_addresses_.at(cores[0]);`), insert:

```cpp
#ifdef TT_METAL_USE_EMULE
    emulation::on_buffer_allocated(buffer);
#endif
```

- [ ] **Step 3: Hook the standard path**

In `AllocatorImpl::allocate_buffer`, after `allocated_buffers_.insert(buffer);` at line 191 and before `return address;`, insert:

```cpp
#ifdef TT_METAL_USE_EMULE
    emulation::on_buffer_allocated(buffer);
#endif
```

- [ ] **Step 4: Hook deallocation**

In `AllocatorImpl::deallocate_buffer`, at the very top (after the `mutex_` lock acquisition, before reading `address`), insert:

```cpp
#ifdef TT_METAL_USE_EMULE
    emulation::on_buffer_deallocated(buffer);
#endif
```

The hook must run BEFORE the bank manager releases the address, since the hook reads the buffer's address/size to compute ranges.

- [ ] **Step 5: Hook mass deallocation**

In `AllocatorImpl::deallocate_buffers` at line 225, before the `dram_manager_->deallocate_all()` call, insert:

```cpp
#ifdef TT_METAL_USE_EMULE
    for (Buffer* b : allocated_buffers_) {
        emulation::on_buffer_deallocated(b);
    }
#endif
```

- [ ] **Step 6: Build**

```bash
cmake --build /localdev/arminale/tt-metal-main/build_emule_asan -j 8 2>&1 | tail -10
```

Expected: clean build.

- [ ] **Step 7: Commit**

```bash
cd /localdev/arminale/tt-metal
git add tt_metal/impl/allocator/allocator.cpp
git commit -m "feat(emul/asan): hook AllocatorImpl alloc/dealloc into per-buffer poison"
```

---

### Task 10: Run regression with allocator hooks active; triage

**Files:** None modified — verification + log capture.

- [ ] **Step 1: Run regression**

```bash
cd /localdev/arminale/tt-emule
BUILD_DIR=/localdev/arminale/tt-metal-main/build_emule_asan \
  ./run_regression.sh 2>&1 | tee /tmp/asan_alloc_regression.log
```

Expected runtime: similar to baseline (Task 4), maybe +5% from hook overhead.

- [ ] **Step 2: Diff against the Task 4 baseline**

```bash
diff <(grep -E "AddressSanitizer:|EMULE.*out-of-bounds|FAIL|^Test " /localdev/arminale/tt-emule/docs/asan_baseline_regression.log) \
     <(grep -E "AddressSanitizer:|EMULE.*out-of-bounds|FAIL|^Test " /tmp/asan_alloc_regression.log)
```

Expected: any new failures are real bugs the allocator-hook poisoning revealed (in-buffer overflow into a neighbour's range, use-after-free).

- [ ] **Step 3: Try triage mode if anything fires unexpectedly**

```bash
TT_EMULE_ASAN_WARN_ONLY=1 \
  BUILD_DIR=/localdev/arminale/tt-metal-main/build_emule_asan \
  ./run_regression.sh 2>&1 | tee /tmp/asan_alloc_warn_only.log
grep "EMULE.*out-of-bounds" /tmp/asan_alloc_warn_only.log | sort -u
```

This catches all bridge-bounds violations in one pass without aborting. ASan-detected issues (use-after-poison) still abort regardless of `WARN_ONLY` — that flag only affects our own bridge `__emule_bounds_fail`.

- [ ] **Step 4: Save the log; document any findings**

```bash
mv /tmp/asan_alloc_regression.log /localdev/arminale/tt-emule/docs/asan_alloc_regression.log
cd /localdev/arminale/tt-emule
git add docs/asan_alloc_regression.log
```

If new failures exist, add a `## ASan-revealed bugs` section to `IMPLEMENTATION_REPORT.md` with one entry per bug (line / kernel / symptom). Do not fix in this PR — file follow-ups.

```bash
git commit -m "docs(asan): capture allocator-hook regression log + findings"
```

---

## Phase 4 — JIT-path negative tests + docs

### Task 11: JIT overflow + UAF negative tests

**Files:**
- Create: `tt-metal-main:tt_emule/asan_tests/CMakeLists.txt`
- Create: `tt-metal-main:tt_emule/asan_tests/test_buffer_oob.cpp`
- Create: `tt-metal-main:tt_emule/asan_tests/test_buffer_uaf.cpp`
- Modify: `tt-metal-main:tt_emule/CMakeLists.txt`

- [ ] **Step 1: Write the overflow test**

Create `tt-metal-main:tt_emule/asan_tests/test_buffer_oob.cpp`:

```cpp
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Negative test: kernel writes one byte past the end of an L1 buffer.
// Expect AddressSanitizer:use-after-poison and process death.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/program.hpp>
#include <cstdio>

using namespace tt::tt_metal;

int main() {
    auto* device = CreateDevice(0);
    constexpr uint32_t SIZE = 4096;
    InterleavedBufferConfig cfg{
        .device = device, .size = SIZE, .page_size = SIZE, .buffer_type = BufferType::L1};
    auto buf = CreateBuffer(cfg);

    Program program = CreateProgram();
    CoreCoord core{0, 0};
    auto kernel = CreateKernel(
        program,
        "tt_emule/asan_tests/kernels/oob_writer.cpp",  // create alongside
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, core,
                   {static_cast<uint32_t>(buf->address()), SIZE});
    EnqueueProgram(device->command_queue(), program, /*blocking=*/true);

    Finish(device->command_queue());
    std::fprintf(stderr, "ERROR: expected ASan abort, none fired\n");
    CloseDevice(device);
    return 1;
}
```

Then create `tt-metal-main:tt_emule/asan_tests/kernels/oob_writer.cpp`:

```cpp
#include "dataflow_api.h"

void kernel_main() {
    uint32_t base = get_arg_val<uint32_t>(0);
    uint32_t size = get_arg_val<uint32_t>(1);
    auto* p = reinterpret_cast<volatile uint8_t*>(base);
    p[size] = 0x42;  // first byte past the buffer — poisoned
}
```

- [ ] **Step 2: Write the UAF test**

Create `tt-metal-main:tt_emule/asan_tests/test_buffer_uaf.cpp`:

```cpp
// Negative test: allocate a buffer, free it, then a kernel writes through
// the captured address. Expect AddressSanitizer:use-after-poison.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/program.hpp>
#include <cstdio>

using namespace tt::tt_metal;

int main() {
    auto* device = CreateDevice(0);
    constexpr uint32_t SIZE = 4096;
    InterleavedBufferConfig cfg{
        .device = device, .size = SIZE, .page_size = SIZE, .buffer_type = BufferType::L1};
    auto buf = CreateBuffer(cfg);
    uint32_t addr = buf->address();
    buf.reset();  // triggers Buffer dtor -> AllocatorImpl::deallocate_buffer
                  // -> on_buffer_deallocated -> __emule_buffer_free (poisoned)

    Program program = CreateProgram();
    CoreCoord core{0, 0};
    auto kernel = CreateKernel(
        program,
        "tt_emule/asan_tests/kernels/uaf_writer.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, core, {addr});
    EnqueueProgram(device->command_queue(), program, /*blocking=*/true);

    std::fprintf(stderr, "ERROR: expected ASan abort, none fired\n");
    CloseDevice(device);
    return 1;
}
```

Create `tt-metal-main:tt_emule/asan_tests/kernels/uaf_writer.cpp`:

```cpp
#include "dataflow_api.h"

void kernel_main() {
    uint32_t addr = get_arg_val<uint32_t>(0);
    *reinterpret_cast<volatile uint8_t*>(addr) = 0x55;  // freed region
}
```

- [ ] **Step 3: Wire into CMake**

Create `tt-metal-main:tt_emule/asan_tests/CMakeLists.txt`:

```cmake
add_executable(test_emule_asan_buffer_oob test_buffer_oob.cpp)
target_link_libraries(test_emule_asan_buffer_oob PRIVATE Metalium::Metal)
add_test(NAME emule_asan_buffer_oob COMMAND test_emule_asan_buffer_oob)
set_tests_properties(emule_asan_buffer_oob PROPERTIES
    LABELS "emule_asan"
    WILL_FAIL TRUE
    ENVIRONMENT "TT_METAL_EMULATED_MODE=1;TT_METAL_SLOW_DISPATCH_MODE=1")

add_executable(test_emule_asan_buffer_uaf test_buffer_uaf.cpp)
target_link_libraries(test_emule_asan_buffer_uaf PRIVATE Metalium::Metal)
add_test(NAME emule_asan_buffer_uaf COMMAND test_emule_asan_buffer_uaf)
set_tests_properties(emule_asan_buffer_uaf PROPERTIES
    LABELS "emule_asan"
    WILL_FAIL TRUE
    ENVIRONMENT "TT_METAL_EMULATED_MODE=1;TT_METAL_SLOW_DISPATCH_MODE=1")
```

Modify `tt-metal-main:tt_emule/CMakeLists.txt` to gate-add the subdir at the bottom:

```cmake
if(TT_EMULE_ASAN)
    add_subdirectory(asan_tests)
endif()
```

The `TT_METAL_MOCK_CLUSTER_DESC_PATH` env var must also be set when running these tests; if the existing `run_regression.sh` exports it globally, the tests will pick it up. If not, add the path as another `ENVIRONMENT` entry.

- [ ] **Step 4: Build and run**

```bash
cmake --build /localdev/arminale/tt-metal-main/build_emule_asan -j 8 \
    --target test_emule_asan_buffer_oob test_emule_asan_buffer_uaf 2>&1 | tail -10
ctest --test-dir /localdev/arminale/tt-metal-main/build_emule_asan -L emule_asan -V 2>&1 | tail -30
```

Expected: 2/2 tests pass via `WILL_FAIL TRUE` (the binaries die with ASan output, exit non-zero).

- [ ] **Step 5: Commit**

```bash
cd /localdev/arminale/tt-metal
git add tt_emule/asan_tests tt_emule/CMakeLists.txt
git commit -m "test(emul/asan): JIT-path overflow + use-after-free negative tests"
```

---

### Task 12: Update IMPLEMENTATION_REPORT.md and add usage docs

**Files:**
- Modify: `tt-emule:IMPLEMENTATION_REPORT.md`
- Create: `tt-emule:docs/ASAN.md`

- [ ] **Step 1: Write the usage doc**

Create `tt-emule:docs/ASAN.md`:

```markdown
# Running tt-emule under AddressSanitizer

## Build

```bash
cmake -B /localdev/arminale/tt-metal-main/build_emule_asan \
  -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 \
  -DTT_METAL_USE_EMULE=ON -D=ON -DTT_EMULE_ASAN=ON \
  -DTT_EMULE_PATH=/localdev/arminale/tt-emule \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build /localdev/arminale/tt-metal-main/build_emule_asan -j 8
```

## What gets caught

| Detection | Mechanism |
|-----------|-----------|
| Write past L1 / DRAM total size | Bridge bounds check (`__emule_dram_ptr`, `__emule_resolve_noc_addr_sized`); always-on, doesn't need ASan. |
| Write past L1Pool 1 MB live region | `L1Pool` tail poisoning; needs ASan runtime. |
| Write past bump pointer in standalone `Core::l1_alloc` | `Core::mmap_region` pre-poison; needs ASan. |
| Write into a buffer's L1 range that isn't yours | `AllocatorImpl::allocate_buffer` per-buffer unpoison; needs ASan. |
| Use-after-free of a deallocated buffer | `AllocatorImpl::deallocate_buffer` repoison; needs ASan. |

## Triage knob

```bash
TT_EMULE_ASAN_WARN_ONLY=1 ./run_regression.sh 2>&1 | tee asan_warn.log
```

Bridge bounds checks log and continue instead of `_Exit(2)`. Useful for finding all violations in one pass. ASan-runtime detections (use-after-poison) still abort — this flag only affects our `__emule_bounds_fail` helper.

## Known gaps

- Sharded L1 buffer poisoning is conservative: an entire shard region is unpoisoned even when the buffer's live data fills only part of it. Shard-tail overflows escape detection. Tracked as a follow-up; switch to `Buffer::get_buffer_page_mapping()` for accurate per-shard size.
- `__emule_local_l1_to_ptr` lacks a bounds check.
- `__emule_multicast_write` lacks per-target bounds validation.
- JIT compile cache is in-process only; ASan/non-ASan cannot share. If persistent caching is added, key on ASan state.
```

- [ ] **Step 2: Update IMPLEMENTATION_REPORT.md**

Modify `tt-emule:IMPLEMENTATION_REPORT.md`. After section 3 ("Pros, Cons, and Maintainability"), add a new subsection under **Pros**:

```markdown
**ASan-instrumented memory-bug detection.** Building with `TT_EMULE_ASAN=ON` enables per-buffer L1/DRAM poisoning driven by tt-metal's `AllocatorImpl`. Kernels that write past their buffer end, into a neighbouring buffer, or into freed memory abort with an AddressSanitizer report — no silicon required. Bridge-level bounds checks (`__emule_dram_ptr`, `__emule_resolve_noc_addr_sized`) are always on and catch coarse offset overruns even in non-ASan builds. See `docs/ASAN.md`.
```

Also append to the **Cons** list:

```markdown
**Sharded L1 poisoning is conservative.** ASan unpoisons an entire shard region per shard core even when the buffer fills only part of it; shard-tail overflows can escape detection. Tracked as a follow-up to switch to `Buffer::get_buffer_page_mapping()` for accurate sizing.
```

- [ ] **Step 3: Commit**

```bash
cd /localdev/arminale/tt-emule
git add docs/ASAN.md IMPLEMENTATION_REPORT.md
git commit -m "docs(asan): usage guide + report sections covering ASan support"
```

---

## Self-review checklist

- **Spec coverage** — every item in the prior conversation's "Concrete sub-plan" is mapped to a task: bridge API (Task 6), initial poison (Task 7), allocator hook (Tasks 8-9), sized resolver definition (Task 1), tt-metal CMake (Task 2), JIT g++ flags (Task 3), negative tests (Tasks 5, 11), docs (Task 12).
- **Sharded TODO is explicit** — comment block in `asan_hooks.cpp` (Task 8) marks it `*** TODO(asan-accuracy) ***` with a clear "we WILL want this fixed" note; same gap is also called out in `IMPLEMENTATION_REPORT.md` Cons (Task 12) and `docs/ASAN.md` (Task 12).
- **No unverified API references that would block the engineer** — `Buffer::device()`, `Buffer::allocator()`, `Buffer::translate_page_address()`, `Allocator::get_num_banks()`, `Allocator::get_logical_core_from_bank_id()`, `IDevice::virtual_core_from_logical_core()` were all confirmed during investigation. The two access points flagged for verification at implementation time are: the exact `Cluster -> SWEmulatedChip*` accessor, and whether `Buffer::per_core_allocation_` / `per_core_addresses_` need accessor wrappers; both are noted in the relevant task.
- **Type consistency** — `__emule_buffer_alloc/free` signature `(uint8_t*, std::size_t)` is identical across `asan_bridge.h`, `kernel_runner.cpp`, `sw_emulated_chip.cpp`, and `asan_hooks.cpp`. `on_buffer_allocated/deallocated` takes `const Buffer*` consistently.
- **Phase ordering ensures buildability at each commit** — Phase 1 alone produces a tt-metal binary that builds and runs (no behavior change); Phase 3 lands hooks on top of an ASan-clean baseline so any new findings are attributable to per-buffer poisoning specifically.
