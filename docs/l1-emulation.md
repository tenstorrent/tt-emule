# L1 Emulation in tt-emule

How tt-emule emulates per-core **L1 SRAM**. Read this before debugging an
L1-addressing bug, extending the memory model, or auditing the host-pointer
convention against silicon.

On silicon, L1 is per-Tensix scratchpad SRAM that firmware addresses with small
absolute addresses (e.g. a semaphore at `0xFFE30`). Emule backs each core's L1
with a host `mmap` and translates between kernel-visible addresses and host
pointers at well-defined points.

Companion docs: [cb-emulation.md](cb-emulation.md) (CBs live in L1),
[dram-emulation.md](dram-emulation.md) (the banked sibling),
[mem-zeros-handling.md](mem-zeros-handling.md) (the MEM_ZEROS region),
[noc-emulation.md](noc-emulation.md) (NOC address resolution).

---

## 1. Emulation model

Each emulated core owns an L1 region in host memory. The governing convention is
**host-pointer-everywhere**: every L1 address handed to a kernel
(`get_write_ptr`, `get_read_ptr`, `get_semaphore`) is a real host pointer
truncated to `uint32_t`. This works because the L1 mmap is placed below 4 GB
(`MAP_32BIT`) for worker cores, so the truncated pointer round-trips losslessly.

Consequences:
- L1 is **zeroed once** at mmap time (`MAP_ANONYMOUS`), not re-zeroed per program
  — matching silicon's boot-time `wzeromem`. See
  [mem-zeros-handling.md](mem-zeros-handling.md).
- There is **no SRAM size model** beyond the allocated region; no bank
  contention, no access latency.
- A raw firmware-style offset (e.g. `0x1000`) and a truncated host pointer are
  disambiguated at translation time (Section 3).

---

## 2. Backing store: `tt_emule::Core`

`include/tt_emule/device.hpp` — `class Core` owns one core's memory:

```cpp
static constexpr size_t MAX_CBS = 32;
enum class CoreRole { WORKER, DRAM };

uint8_t*  l1_      = nullptr;             // mmap base (host pointer)
uint32_t  l1_base_ = 0;                   // l1_ truncated to 32 bits, for kernels
size_t    l1_size_ = 0;                   // worker L1 / DRAM bank size, from SoC descriptor
CoreRole  role_    = CoreRole::WORKER;
CBSyncState     cb_sync_states_[MAX_CBS] = {};
```

Integrated builds set `l1_size_` from `metal_SocDescriptor` at chip
construction (`SWEmuleChip` sizes worker pool slots and individual DRAM
mmaps from `l1_size` / `dram_bank_size`); the L1 size is arch-dependent
(1.5 MB on wormhole_b0, distinct on blackhole / quasar — see
[BUILD_GUIDE.md](../BUILD_GUIDE.md) for arch defaults).

`mmap_region(size)` maps `MAP_PRIVATE | MAP_ANONYMOUS`, adding **`MAP_32BIT` for
WORKER cores** so `l1_base_` fits in 32 bits. DRAM cores use a plain mmap (they
are reached via the bank-aware NOC path, not a truncated pointer — see
[dram-emulation.md](dram-emulation.md)). `MAP_ANONYMOUS` guarantees the region
starts zeroed.

L1 alignment is `L1_ALIGNMENT` (16 bytes), injected as a JIT define from
`hal::get_l1_alignment()`.

---

## 3. Address resolution

A single contiguous `MAP_32BIT` mmap with 2 MB-aligned slots serves every core
(`include/tt_emule/l1_pool.hpp`: `SLOT_SIZE = 2 MB`). The tt-metal integration
(`SWEmuleChip::worker_pool_`, built with `TT_EMULE_USE_L1_POOL`) owns this pool,
so encoding a host pointer to an L1 offset is a single bitmask
`addr & (SLOT_SIZE - 1)` (i.e., `addr & 0x1FFFFF`).
A thread-local `__emule_bridge_l1` holds the current thread's core L1 base, used
when *decoding* a firmware-style offset back to a host pointer (see
`__emule_local_l1_to_ptr` below).

Translation helpers:

| Helper | Role | Where |
|---|---|---|
| `__emule_addr_to_offset(addr)` | **Encode**: host pointer → L1 offset (`addr & 0x1FFFFF`). | `include/jit_hw/api/dataflow/dataflow_api.h` |
| `__emule_local_l1_to_ptr(l1_addr)` | Kernel-side dual path: if `l1_addr >= l1_base` treat as an absolute host pointer, else add to `__emule_bridge_l1`. | `include/jit_hw/jit_kernel_stubs.hpp` |
| `__emule_resolve_noc_addr(noc_addr)` | **Decode**: NOC xy + lower-`NOC_ADDR_LOCAL_BITS` (36) offset → owning `Core` → `l1_ptr(offset)`. WORKER offsets are masked with `L1_SLOT_MASK 0x1FFFFF`. | `tt-metal/.../emulated_program_runner.cpp` |
| `__emule_local_l1_ptr(offset)` | `extern "C"` legacy fast path (offset → host ptr); most paths now go through the resolver. | `tt-metal/.../emulated_program_runner.cpp` |

> Historical note: an `__emule_fixup_noc_addr` "fixup point" was removed — NOC
> addresses are now constructed correctly by `get_noc_addr` /
> `get_noc_addr_from_bank_id`, so no lower-bit fixup is applied (applying it
> would corrupt DRAM bank offsets > 2 MB). The model is now just encode + decode.

---

## 4. Allocation, semaphores, CBs

- **Bump allocation.** `Core::l1_alloc(bytes)` is a simple bump allocator
  returning an absolute 32-bit address. Live use is the **Quasar DFB
  fallback path** in tt-metal's `emulated_program_runner.cpp` (when no
  upstream `finalize_addr` is supplied for a DFB, the runner bumps a
  fresh region out of the core's L1). Worker WH/BH L1 is otherwise
  addressed directly by truncated CB pointers.
- **Semaphores** live in the kernel-config region of L1 at
  `l1_base + EMULE_SEM_BASE + sem_id * EMULE_SEM_ALIGN` (`EMULE_SEM_ALIGN = 16`),
  where `EMULE_SEM_BASE` is a JIT define computed by the runner from the HAL
  kernel-config base + `ProgramConfig.sem_offset` (the same values real firmware
  uses). See [noc-emulation.md](noc-emulation.md) §5 for semaphore *operations*.
- **Circular buffers** are slices of L1; their sync state lives in
  `Core::cb_sync_states_[]`. See [cb-emulation.md](cb-emulation.md).

---

## 5. Runtime bridge

The program runner (`tt-metal/.../emulated_program_runner.cpp`), per emulated
core/thread, sets `thread_local uint8_t* __emule_bridge_l1 = core->l1_data()`
before launching the kernel. The dlopen'd kernel `.so` reaches the host runtime
only through the `extern "C"` hooks above (exported via `-rdynamic` on
`libtt_metal.so`). In `SWEmuleChip`, `read_from_device` / `write_to_device`
delegate uniformly to `get_core(xy)->l1_ptr(offset)` + `memcpy`.

---

## 6. What's intentionally simplified

- No L1 capacity pressure / eviction / bank conflicts — the mmap is as large
  as the SoC-descriptor-derived `l1_size_` and accesses are direct.
- No per-program re-zeroing (one-time `MAP_ANONYMOUS` zero-init; the Quasar DFB
  fallback bump allocator is the one exception — see
  [mem-zeros-handling.md](mem-zeros-handling.md)).
- **Sub-4 GB address-space pressure.** `MAP_32BIT` forces every worker L1
  region below 4 GB, but that 4 GB host range is shared with the loader's
  text/data, the heap, every thread's stack, and the JIT-compiled kernel
  `.so` mappings. Each worker core mapped individually would compete with
  those for a few megabytes at a time. `L1Pool` mitigates by reserving a
  single contiguous 2-MB-aligned slot pool for all workers up front;
  cores that overflow the pool fall back to individual `MAP_32BIT`
  mappings and re-enter the competition.

---

## 7. Where to read silicon canonical

| Topic | Canonical |
|---|---|
| L1 memory map / addresses | tt-metal HAL (`hal.get_dev_addr(...)`), `dev_mem_map.h` |
| L1 alignment | `hal::get_l1_alignment()` |
| Kernel-facing address APIs | `tt-metal/tt_metal/hw/inc/api/dataflow/*.h` |
