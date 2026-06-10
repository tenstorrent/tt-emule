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
static constexpr size_t L1_SIZE = 1024 * 1024;   // 1 MB (WORKER)
static constexpr size_t MAX_CBS = 32;
enum class CoreRole { WORKER, DRAM };

uint8_t*  l1_      = nullptr;             // mmap base (host pointer)
uint32_t  l1_base_ = 0;                   // l1_ truncated to 32 bits, for kernels
size_t    l1_size_ = L1_SIZE;             // 1 MB (WORKER) or bank size (DRAM)
CoreRole  role_    = CoreRole::WORKER;
CBSyncState     cb_sync_states_[MAX_CBS] = {};
DstRegisterFile dst_;
```

`mmap_region(size)` maps `MAP_PRIVATE | MAP_ANONYMOUS`, adding **`MAP_32BIT` for
WORKER cores** so `l1_base_` fits in 32 bits. DRAM cores use a plain mmap (they
are reached via the bank-aware NOC path, not a truncated pointer — see
[dram-emulation.md](dram-emulation.md)). `MAP_ANONYMOUS` guarantees the region
starts zeroed.

L1 alignment is `L1_ALIGNMENT` (16 bytes), injected as a JIT define from
`hal::get_l1_alignment()`.

---

## 3. Address resolution

Two host-mmap layouts exist; the address *encoding* is identical, only the
offset conversion differs:

- **bridge_l1 mode (default / standalone).** Each `Core` has its own mmap. A
  thread-local `__emule_bridge_l1` holds this thread's core's L1 base; an L1
  offset is `addr - __emule_bridge_l1`.
- **L1Pool mode (`TT_EMULE_USE_L1_POOL`).** A single contiguous `MAP_32BIT` mmap
  with 2 MB-aligned slots serves every core (`include/tt_emule/l1_pool.hpp`:
  `SLOT_SIZE = 2 MB`, `SLOT_MASK = SLOT_SIZE - 1`, `to_offset(addr) = addr &
  SLOT_MASK`). The tt-metal integration (`SWEmuleChip::worker_pool_`) builds with
  this mode; the offset is a single bitmask `addr & 0x1FFFFF`.

Translation helpers:

| Helper | Role | Where |
|---|---|---|
| `__emule_addr_to_offset(addr)` | **Encode**: host pointer → L1 offset. `addr & 0x1FFFFF` under L1Pool, else `addr - bridge_l1` (TLS). | `include/jit_hw/api/dataflow/dataflow_api.h` |
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
  returning an absolute 32-bit address; used by the Quasar DFB path. Worker
  WH/BH L1 is otherwise addressed directly by truncated CB pointers.
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

- No L1 capacity pressure / eviction / bank conflicts — the mmap is as large as
  `L1_SIZE` and accesses are direct.
- No per-program re-zeroing (one-time `MAP_ANONYMOUS` zero-init; the Quasar DFB
  fallback bump allocator is the one exception — see
  [mem-zeros-handling.md](mem-zeros-handling.md)).
- `MAP_32BIT` consumes the shared sub-4 GB address space; L1Pool consolidates
  worker L1 into one allocation to bound this, but pool-exhaustion fallback cores
  still compete for it.

---

## 7. Where to read silicon canonical

| Topic | Canonical |
|---|---|
| L1 memory map / addresses | tt-metal HAL (`hal.get_dev_addr(...)`), `dev_mem_map.h` |
| L1 alignment | `hal::get_l1_alignment()` |
| Kernel-facing address APIs | `tt-metal/tt_metal/hw/inc/api/dataflow/*.h` |
