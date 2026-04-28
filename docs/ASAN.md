# AddressSanitizer in tt-emule

## Overview

tt-emule integrates AddressSanitizer (ASan) to catch kernel-side memory bugs that on silicon would either silently corrupt neighbouring buffers or crash with no diagnostic. The integration layers two things on top of stock ASan: bridge-level bounds checks at the host/kernel boundary that are always compiled in, and per-buffer L1/DRAM poisoning driven by tt-metal's `AllocatorImpl` that is enabled when building with `TT_EMULE_ASAN=ON`. The result is that out-of-bounds NOC reads/writes, cross-buffer overflows, and use-after-free of deallocated buffers abort with a normal ASan report pointing at the offending kernel line. No silicon required.

## Build

See `BUILD_GUIDE.md` -> "Optional: Build with AddressSanitizer" for the full instructions, including the `LD_LIBRARY_PATH` workaround needed at build time. Minimal invocation:

```bash
cmake -B build_emule_asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTT_METAL_EMULATION=ON \
  -DTT_EMULE_ASAN=ON \
  -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20

LD_LIBRARY_PATH=/usr/lib/llvm-20/lib/clang/20/lib/linux \
  cmake --build build_emule_asan -j$(nproc)
```

Standalone tt-emule negative tests live in a separate `build_asan/` configured with `-DTT_EMULE_ASAN=ON` on tt-emule directly; run with `ctest --test-dir build_asan -L asan`.

## What gets caught

| Detection | Mechanism | Active when |
|-----------|-----------|-------------|
| NOC read/write past target L1 size | `__emule_resolve_noc_addr_sized` bounds check | always (even non-ASan) |
| DRAM offset past bank size | `__emule_dram_ptr` bounds check | always (even non-ASan) |
| Local L1 offset past core size | `__emule_local_l1_ptr` bounds check | always (even non-ASan) |
| Multicast destination past target L1 | per-target bounds in `__emule_multicast_write` | always (even non-ASan) |
| Write past L1Pool 1 MB live region (slot tail) | L1Pool tail poisoning | ASan only |
| Write past `l1_alloc` bump pointer (standalone) | `Core::mmap_region` pre-poison | ASan only, standalone path |
| Write into another buffer's range on the same core | per-buffer poison via Allocator hook | ASan only, JIT path |
| Use-after-free of a deallocated buffer | per-buffer repoison on dealloc | ASan only, JIT path |

The first four rows fire `__emule_bounds_fail` with a printable message naming the offending offset, size, and target core. The last four rows surface as standard ASan `use-after-poison` reports.

## Triage knob: `TT_EMULE_ASAN_WARN_ONLY`

Setting `TT_EMULE_ASAN_WARN_ONLY=1` in the environment downgrades `__emule_bounds_fail` from abort to a single-line warning printed to stderr. Useful for sweeping a test suite to enumerate every out-of-bounds site at once instead of stopping at the first. This knob only affects the bridge-level bounds checks listed in the first four table rows. ASan's own `use-after-poison` detection (the per-buffer rows) ignores this variable and aborts regardless — you cannot warn-only your way past a cross-buffer overflow. Unset or `=0` restores abort-on-first behaviour.

## Known gaps

- **Sharded L1 buffer poisoning is conservative.** For sharded L1 buffers, ASan unpoisons the full `aligned_size_per_bank()` per shard core rather than the accurate per-shard size from `BufferPageMapping`. Partial shards are over-unpoisoned and shard-tail overflows can escape detection. There is a `TODO(asan-accuracy)` comment in `tt_metal/impl/emulation/asan_hooks.cpp`. This is a planned follow-up — we WILL want this fixed before relying on ASan for shard-boundary bug hunting.
- **`~AllocatorImpl` and `AllocatorImpl::override_state` bypass per-buffer dealloc.** Both clear `allocated_buffers_` directly without invoking `on_buffer_deallocated`, so per-buffer poison can persist past device close or allocator reconfigure. Practically harmless inside a single test process, but a wart left as a follow-up.
- **Build-time tools are ASan-instrumented.** `flatc` and other generated build helpers are linked with ASan and `dlopen` libasan; the `cmake --build` step needs `LD_LIBRARY_PATH=/usr/lib/llvm-20/lib/clang/20/lib/linux` set (see BUILD_GUIDE.md). The variable is not needed at test runtime.

## Architecture pointer

The engineering plan is in `docs/PLAN_asan_allocator_integration.md` (background reading; not required to use ASan). The per-buffer poison flow runs allocator -> `asan_hooks` -> `SWEmuleChip` -> `__emule_buffer_alloc` / `__emule_buffer_free` -> ASan `__asan_poison_memory_region` / `__asan_unpoison_memory_region`. Bridge-level bounds checks (`__emule_dram_ptr`, `__emule_resolve_noc_addr_sized`, `__emule_local_l1_ptr`, `__emule_multicast_write`) are independent of the allocator hook and live in `src/kernel_runner.cpp` and `tt_metal/impl/emulation/emulated_program_runner.cpp`.
