# What kinds of memory bugs the current ASan implementation can and cannot catch

Independent, self-contained summary. Read without prior context.

## Caught — with passing test evidence

| Bug class | What it looks like in a kernel | How ASan catches it | Test that proves it |
|---|---|---|---|
| **L1 write past per-core L1 size** | `*((uint8_t*)__emule_local_l1_to_ptr(0x200000)) = 1` (offset > L1_SIZE) | Bridge bounds check in `__emule_local_l1_ptr` aborts with `[EMULE]` marker, regardless of ASan build flag | `tests/asan/oob_l1_alloc_test.cpp` (standalone, always-on) |
| **L1Pool slot-tail overflow** | A worker core writes past its 1 MB live region into the unused tail of its 2 MB pool slot | L1Pool poisons the tail at construction; ASan `use-after-poison` fires | `tests/asan/oob_slot_tail_test.cpp` (`asan_oob_slot_tail`) |
| **DRAM offset past bank size** | `noc_async_read` from a DRAM offset larger than `active_dram_bank_size()` | Bridge bounds check in `__emule_dram_ptr` (always-on) | `tests/asan/oob_dram_test.cpp` (`asan_oob_dram`) |
| **NOC read with size that runs off the end of the target's L1** | `noc_async_read(noc_addr, ..., size)` where `offset + size > target->l1_size()` | Sized resolver `__emule_resolve_noc_addr_sized` checks both lower bound and `offset + size`; aborts with `[EMULE]` | `tests/asan/oob_noc_read_test.cpp` (`asan_oob_noc_read`) |
| **Multicast write past target L1 (per-target)** | A multicast destination range exceeds one target's L1 size | Per-target bounds check in `__emule_multicast_write` (always-on) | (no dedicated standalone test; covered by the `__emule_bounds_fail` mechanism that the four cases above all share) |
| **Cross-buffer L1 overflow inside a JIT kernel** | Kernel writes one byte past its `MeshBuffer`'s end on a real `unit_tests_integration` run, with `TT_EMULE_ASAN=ON` | Per-buffer poison set by `AllocatorImpl::allocate_buffer` → `asan_hooks` → `__asan_poison_memory_region`; kernel write trips ASan inside the JIT-compiled `.so` | `MeshDispatchFixture.AsanL1BufferOverflow` in `tests/tt_metal/tt_metal/integration/asan/test_asan_negative.cpp` (Tier 7 ASan, fires `AddressSanitizer:`) |
| **Use-after-free on an L1 `MeshBuffer`** | Kernel writes through a saved address after the buffer is destroyed | `AllocatorImpl::deallocate_buffer` → `asan_hooks` repoisons the freed L1 range | `MeshDispatchFixture.AsanL1BufferUseAfterFree` (Tier 7, fires `AddressSanitizer:`) |
| **Allocation-hook regression — silent no-op detector** | An allocator change makes `__emule_buffer_alloc` stop unpoisoning. Without coverage, the OOB tests above could keep passing on the initial blanket poison alone. | Two positive controls write *inside* their just-allocated regions; if poisoning didn't get cleared, they'd fire too. They must pass cleanly. | `asan_inbounds_l1_alloc` (standalone) and `MeshDispatchFixture.AsanL1BufferInBoundsWrite` (Tier 7) |

## Not caught — with the test that would catch it if/when fixed

| Bug class | Why ASan misses it | Test currently failing or absent |
|---|---|---|
| **Use-after-free on a DRAM `MeshBuffer`** | `SWEmuleChip::core_for_logical(coord, is_dram=true)` matches `coord.x` against `dram_core_to_channel_` *values* (channel ids), but `AllocatorImpl::get_logical_core_from_bank_id` passes the bank's *logical core (x,y)*. All ranges return UNRESOLVED in `enumerate_buffer_ranges`, so the dealloc-hook poison never lands. | `MeshDispatchFixture.AsanDramBufferUseAfterFree` — wired into Tier 7, currently FAILS with "exited 1 but did not emit 'AddressSanitizer:'". Fix is documented in `docs/ASAN_UAF_STATUS.md` (Option A: look up by `(coord.x, coord.y)` directly). |
| **Shard-tail overflow on sharded L1 buffers** | `asan_hooks.cpp` unpoisons the full per-shard region from `aligned_size_per_bank()` instead of the actual per-shard byte count from `BufferPageMapping`. Writes past the live data but inside the shard region don't fire. | No test; documented as a known gap in `docs/ASAN.md` and `docs/PLAN_asan_allocator_integration.md`. The Tier 7 close-out plan is to switch to `Buffer::get_buffer_page_mapping()` for accurate per-shard sizing. |
| **Stale poison after `~AllocatorImpl` or `AllocatorImpl::override_state`** | Both clear `allocated_buffers_` directly without invoking `on_buffer_deallocated`, so per-buffer poison can persist past device close or allocator reconfigure into the next run. | No test; documented in `docs/ASAN.md` "Known gaps". Practically harmless inside one test process. |
| **NOC writes on the writer side** (vs. reads) | The sized resolver is only routed through `noc_async_read`; `noc_async_write` and the multicast variants don't call into `__emule_resolve_noc_addr_sized`. Coarse target-L1-size checks still fire via `__emule_multicast_write` and the unsized `__emule_resolve_noc_addr`, but a `noc_async_write` with `offset + size > l1_size()` where `offset < l1_size()` won't trip the **size**-aware path. | No dedicated NOC-write OOB test; in the close-out plan as a follow-on once DRAM UAF is unblocked. |
| **Multi-core / NOC-driven UAF** | The per-buffer poison only fires when the kernel-side host pointer translation matches what was poisoned. Cross-core NOC writes that re-resolve through `__emule_resolve_noc_addr` may bypass the same shadow page if the resolver doesn't apply poison to the target core's address space. | No tests yet; same DRAM-UAF-style follow-on. |
| **Anything compiled out by macros** (`DPRINT`, `ASSERT`) | Stubs are no-ops; they can't catch what they don't see. | N/A — out of scope for ASan. |
| **Concurrency / data race bugs** | ASan is not TSan. The CMake config explicitly makes `TT_EMULE_ASAN` mutually exclusive with the existing TSan path. | Use TSan instead (build without `TT_EMULE_ASAN=ON`, it auto-enables `-fsanitize=thread` in Debug). |
| **Hardware fidelity bugs** (timing, NOC bandwidth contention, DST precision) | ASan is a memory-safety tool, not a hardware simulator. Synchronous `memcpy` NOC and float32-internal DST are emulator approximations and would never fire ASan. | Out of scope. |

## One-line takeaway

ASan in tt-emule reliably catches **OOB and UAF on L1 buffers** (host-side bounds checks always; per-buffer poison when `TT_EMULE_ASAN=ON`), and reliably catches **OOB on DRAM offsets**. It does **not** yet catch DRAM use-after-free or shard-tail overflows on sharded L1 — both have failing/missing tests in Tier 7 and are tracked as the next follow-on work.

---

# How this compares to vanilla CPU ASan

Useful framing: the emulator is *already* a CPU program. Host code gets vanilla ASan for free. The interesting comparison is on the *kernel-facing* memory model — where bugs differ from a normal C++ heap and the tooling has to be hand-plumbed.

## What works the same as vanilla ASan

These are caught with no extra plumbing because tt-emule and tt-metal host code are normal C++ TUs compiled with `-fsanitize=address`:

- Heap overflow / use-after-free / double-free on `malloc`, `new`, `std::vector`, etc., in host code (allocator code, runner, fixtures, gtest harness, UMD).
- Stack overflow in host functions.
- Use-after-return / use-after-scope on host stack frames.
- Global buffer overflow (BSS/data segment) — including `__emule_dst[16][1024]`, the simulated DST register file: a 64 KB global array. ASan inserts redzones around globals, so off-by-one DST writes past `__emule_dst_active_tiles()` would trip ASan even before the explicit `__emule_dst_check` bounds helper.
- ODR / initialization-order issues in host static initializers.
- Shared library linkage: `-shared-libasan` means tt-metal, libtt-umd, libtt_emule, and JIT'd kernel `.so`s share one runtime — no fragmentation.

In short: **CPU-side bugs in the emulator's plumbing are caught for free**.

## What had to be hand-plumbed to work *like* vanilla ASan

Vanilla ASan instruments `malloc` / `free` / `new` / `delete` at the libc/libstdc++ level and inserts redzones automatically. Kernel buffers in tt-emule live in a flat `mmap` region managed by `AllocatorImpl` (a tt-metal-internal allocator with its own bookkeeping) — ASan has zero knowledge of it. Everything below required manual `__asan_poison_memory_region` / `__asan_unpoison_memory_region` calls:

| Vanilla ASan auto-handles | tt-emule equivalent (manual) |
|---|---|
| `malloc(n)` returns a redzone-bracketed region | `AllocatorImpl::allocate_buffer` → `on_buffer_allocated` → `__emule_buffer_alloc(host_base, size)` → `__asan_unpoison_memory_region` |
| `free(p)` repoisons the freed block | `AllocatorImpl::deallocate_buffer` → `on_buffer_deallocated` → `__emule_buffer_free(host_base, size)` → `__asan_poison_memory_region` |
| Inter-allocation redzones | `SWEmuleChip::initialize_asan_poison` blanket-poisons unreserved L1/DRAM up front (opt-in via `TT_EMULE_ASAN_BLANKET=1` for op-level tests) |
| Stack canaries on every function frame | n/a — kernels don't have a separate "device stack"; they run on the host thread's stack which is already vanilla-instrumented |

## Where tt-emule ASan is **weaker** than vanilla

| Gap | Why | Impact |
|---|---|---|
| **No per-buffer redzones** | The allocator hands back contiguous L1 ranges with no "padding poison" between user buffers. Vanilla ASan inserts ~16-128 bytes of redzone around every malloc'd block. | Off-by-one OOB writes that land inside the next *allocated* buffer's range fire only if blanket poisoning is on AND the targeted byte hasn't been unpoisoned yet by a later alloc. Heap-style "stomp on the next object" bugs are less reliably caught. |
| **No `noc_async_write` size-aware bound** | The sized resolver `__emule_resolve_noc_addr_sized` is only routed through `noc_async_read`; writes use the unsized variant. Coarse target-L1 bounds still fire, but `offset + size > l1_size` overruns where `offset < l1_size` slip through on the write path. | NOC-write overruns into target L1 are partially missed. Symmetric to ASan never instrumenting `memcpy` writes — same class of asymmetric coverage. |
| **No leak detection in regression** | `ASAN_OPTIONS=detect_leaks=0` is forced because OpenMPI's `ompi_group_allocate` leaks noise. Vanilla ASan via LSan would catch tt-emule-side host leaks. | Host-side allocator leaks in our own code are silently allowed. Recoverable by running ad-hoc with `detect_leaks=1` on a non-MPI binary. |
| **JIT `.so` symbolization is brittle** | Kernel `.so`s are compiled to `/tmp/tt_emule_jit_*` and dlopen'd. ASan stack traces reference these temp paths; if the test exits and the cache is cleared, post-mortem symbolization may not resolve. Vanilla ASan against a normal binary on disk doesn't have this. | Triage from a stale log requires keeping the temp dir or rebuilding with the same source. |
| **Mutually exclusive with TSan** | CMake enforces `TT_EMULE_ASAN` and the existing TSan flag are not co-linked. Vanilla x86 supports running ASan and TSan in separate builds, same here, but no combined build either. | Two build trees needed if you want both. (Same constraint as upstream.) |
| **8-byte shadow granularity** | This is identical to vanilla ASan — sub-word OOBs within an 8-byte unit are invisible. Listed only because users sometimes assume ASan catches every byte. | Same as upstream ASan. |

## Where tt-emule ASan is **stronger** than vanilla

| Strength | Mechanism |
|---|---|
| **Always-on, build-flag-independent bounds checks** | `__emule_dram_ptr`, `__emule_local_l1_ptr`, `__emule_resolve_noc_addr`, `__emule_resolve_noc_addr_sized`, `__emule_multicast_write` all call `__emule_bounds_fail` even on non-ASan Release builds. Vanilla ASan does *nothing* without `-fsanitize=address`. |
| **Hardware-domain error messages** | Bridge errors print `[EMULE] __emule_dram_ptr: DRAM offset past bank size offset=0x3FFFF000 size=4096 cap=0x10000000`. Vanilla ASan gives raw addresses and access type, not "DRAM bank N, offset X". |
| **Triage knob for sweeps** | `TT_EMULE_ASAN_WARN_ONLY=1` downgrades bounds-fail abort to a warning so you can enumerate every violation in one run. Vanilla ASan has `halt_on_error=0` but the recovery story is messier and only applies to the ASan runtime, not "everywhere we manually check". |
| **Cross-core / cross-language coverage in one process** | A real chip has separate RISC-V cores running unrelated code. In emulation everything is host threads in one process — vanilla ASan can correlate kernel-side and host-side memory bugs across what is, on silicon, an opaque DMA boundary. |

## One-line takeaway (vs. vanilla)

For *host* code in tt-emule, you get vanilla ASan with full strength. For *kernel-facing* memory (L1, DRAM, NOC), you get a hand-built subset of vanilla ASan's coverage — strong on bank/L1-size violations and per-`MeshBuffer` UAF/OOB on L1, weak on inter-buffer redzones, NOC writes, and DRAM UAF (the last is in-flight).

---

*Generated 2026-05-06. Validated against tt-emule `armin-asan-rebased` / tt-metal-main `armin-asan-allocator-rebased`. Regression results: 141/14/0 non-ASan (135/11/0 baseline + 6 Tier 7 pass + 3 Tier 7 design-fails when build_emule_clang is used as JIT host), 143/12/0 ASan (135/11/0 baseline + 8 of 9 Tier 7 pass; 12th fail = DRAM UAF in-flight).*
