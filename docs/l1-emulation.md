# L1 Emulation in tt-emule

How tt-emule emulates per-core **L1 SRAM**. Read this before debugging an
L1-addressing bug, extending the memory model, or auditing the address model
against silicon.

On silicon, L1 is per-Tensix scratchpad SRAM that firmware addresses with small
absolute offsets (e.g. a semaphore at `0xFFE30`). Emule backs each core's L1 with
a host `mmap` and follows the same convention: a kernel-visible L1 address is a
small **0-based offset**, translated to a host pointer at the point of use.

Companion docs: [cb-emulation.md](cb-emulation.md) (CBs live in L1),
[dram-emulation.md](dram-emulation.md) (the banked sibling),
[mem-zeros-handling.md](mem-zeros-handling.md) (the MEM_ZEROS region),
[noc-emulation.md](noc-emulation.md) (NOC address resolution),
[multichip/l1-offset-translation.md](multichip/l1-offset-translation.md) (the
offset addressing model + the JIT patch pass, in depth).

---

## 1. Emulation model

Each emulated core owns an L1 region in host memory. A kernel-visible L1 address
is a **0-based firmware offset** into that core's L1 (worker L1 is ~1.5 MB, so an
offset fits in far less than 32 bits) — the same small-absolute-address convention
silicon firmware uses. The offset is converted to a real 64-bit host pointer
**only at the point it is dereferenced**, by adding the running fiber's L1 base
(`__emule_self->bridge_l1`):

```
host_ptr = bridge_l1 + offset          // __emule_l1_translate()
```

A kernel invocation only ever dereferences its **own** core's L1 directly;
cross-core and cross-chip access is expressed as a NOC address and resolved
through `__emule_resolve_noc_addr` (a core-map lookup → the target core's own
backing). So the running fiber's `bridge_l1` is always the correct base at a raw
dereference.

tt-metal kernel source dereferences L1 addresses two ways, both handled without
editing the upstream kernel:
- Through `noc_async_read` / `noc_async_write` (and the CB/semaphore APIs), which
  translate the local address via the chokepoint `__emule_local_l1_to_ptr`.
- By casting the address to a raw pointer and dereferencing it directly
  (`reinterpret_cast<T*>(cb.get_write_ptr())[i] = …`, in mask/scaler generators,
  fill-pad, etc.). These sites are rebased **at the cast** by a JIT-time
  source-transform pass (`apply_x86_rewrites`), which writes a temp
  `patched_kernel.cpp` — upstream kernel files stay pristine. See
  [multichip/l1-offset-translation.md](multichip/l1-offset-translation.md).

Because a kernel-visible address is an offset (not a host pointer), it is
**address-space independent**: the same offset names the same L1 word on any chip
in any process. Worker L1 does not have to be placed in the low 4 GB for the
address to round-trip; where it is mapped is a backing detail (§2, §6).

Consequences:
- L1 is **zeroed once** at mmap time (`MAP_ANONYMOUS`), not re-zeroed per program
  — matching silicon's boot-time `wzeromem`. See
  [mem-zeros-handling.md](mem-zeros-handling.md).
- There is **no SRAM size model** beyond the allocated region; no bank
  contention, no access latency.
- A debug build asserts an offset lands within the core's L1
  (`__emule_self->l1_size`), so a value that is still an absolute/aliased address
  surfaces as a named panic rather than a wild store.

---

## 2. Backing store: `tt_emule::Core`

`include/tt_emule/device.hpp` — `class Core` owns one core's memory:

```cpp
static constexpr size_t MAX_CBS = 32;
enum class CoreRole { WORKER, DRAM };

uint8_t*  l1_      = nullptr;             // mmap base (host pointer)
uint32_t  l1_base_ = 0;                   // l1_ truncated to 32 bits (legacy accessor)
size_t    l1_size_ = 0;                   // worker L1 / DRAM bank size, from SoC descriptor
CoreRole  role_    = CoreRole::WORKER;
CBSyncState     cb_sync_states_[MAX_CBS] = {};
```

Integrated builds set `l1_size_` from `metal_SocDescriptor` at chip construction
(`SWEmuleChip` sizes worker-pool slots and individual DRAM mmaps from `l1_size` /
`dram_bank_size`); the L1 size is arch-dependent (1.5 MB on wormhole_b0, distinct
on blackhole / quasar — see [BUILD_GUIDE.md](../BUILD_GUIDE.md) for arch defaults).

`Core::l1_ptr(offset)` returns `l1_ + offset` with a loud bounds check.
`Core::l1_alloc(bytes)` is a bump allocator returning a **0-based offset** (Quasar
DFB fallback path — see §4).

Worker L1 is currently backed by the low-4 GB pool (`MAP_32BIT`; `l1_pool.hpp`,
`low4g_mmap.hpp`), and DRAM cores use a plain mmap. Under the offset model this
placement is a **backing detail, not an addressing constraint** — the
kernel-visible address is an offset regardless of where L1 is mapped (§6).

L1 alignment is `L1_ALIGNMENT` (16 bytes), injected as a JIT define from
`hal::get_l1_alignment()`.

---

## 3. Address resolution

The runner (`emulated_program_runner.cpp`), per emulated core/fiber, sets the
fiber ctx's `bridge_l1 = core->l1_data()` and `l1_size = core->l1_size()` before
launching the kernel. Every kernel L1 dereference resolves the offset against that
base.

Translation helpers:

| Helper | Role | Where |
|---|---|---|
| `__emule_l1_translate(off)` | **Decode**: L1 offset → host pointer, unconditionally `bridge_l1 + off` (debug-asserts `off < l1_size`). | `include/jit_hw/asan/asan_l1_checks.h` |
| `__emule_local_l1_to_ptr(off)` | The kernel access chokepoint: ASAN checks (when enabled) then `__emule_l1_translate`. Every `noc_async_*` local address and every patched raw-deref cast flows through here. | `include/jit_hw/internal/emule_l1_to_ptr.h` |
| `__emule_addr_to_offset(addr)` | **NOC-construction guard**: `addr & 0x1FFFFF`, idempotent for an in-L1 offset. Applied when building a NOC address in `get_noc_addr` / `get_noc_multicast_addr`; never applied to DRAM bank addresses. | `include/jit_hw/api/dataflow/dataflow_api.h` |
| `__emule_resolve_noc_addr(noc_addr)` | **Decode a NOC address**: xy + lower-`NOC_ADDR_LOCAL_BITS` (36) offset → owning `Core` → `l1_ptr(offset)`. | `emulated_program_runner.cpp` |

Because a local L1 address is already an offset, the NOC encode/decode path and
the `& L1_SLOT_MASK` masking it performs are exact for in-L1 offsets and need no
special handling.

---

## 4. Allocation, semaphores, CBs

- **Bump allocation.** `Core::l1_alloc(bytes)` returns a 0-based offset. Live use
  is the **Quasar DFB fallback path** in `emulated_program_runner.cpp` (when no
  upstream `finalize_addr` is supplied for a DFB, the runner bumps a fresh region
  out of the core's L1). Worker WH/BH L1 is otherwise addressed by CB offsets.
- **Semaphores** are firmware offsets: `get_semaphore(id)` returns
  `EMULE_SEM_BASE + id * EMULE_SEM_ALIGN` (`EMULE_SEM_ALIGN = 16`), where
  `EMULE_SEM_BASE` is a JIT define the runner computes from the HAL kernel-config
  base + `ProgramConfig.sem_offset` (the same values real firmware uses). The
  atomic operation rebases the offset onto `bridge_l1`. See
  [noc-emulation.md](noc-emulation.md) §5 for semaphore *operations*.
- **Circular buffers** are slices of L1; `get_write_ptr` / `get_read_ptr` return
  0-based offsets (the per-RISC ring is maintained in host-pointer space and
  converted to an offset at the accessor). Sync state lives in
  `Core::cb_sync_states_[]`. See [cb-emulation.md](cb-emulation.md).

---

## 5. Runtime bridge

The program runner, per emulated core/thread, sets `thread_local`/fiber-local ctx
fields (`bridge_l1 = core->l1_data()`, `l1_size = core->l1_size()`) before
launching the kernel. The dlopen'd kernel `.so` reaches the host runtime only
through the `extern "C"` hooks and inline chokepoint above (exported via
`-rdynamic` on `libtt_metal.so`). In `SWEmuleChip`, `read_from_device` /
`write_to_device` delegate uniformly to `get_core(xy)->l1_ptr(offset)` + `memcpy`.

---

## 6. What's intentionally simplified

- No L1 capacity pressure / eviction / bank conflicts — the mmap is as large as
  the SoC-descriptor-derived `l1_size_` and accesses are direct.
- No per-program re-zeroing (one-time `MAP_ANONYMOUS` zero-init; the Quasar DFB
  fallback bump allocator is the one exception — see
  [mem-zeros-handling.md](mem-zeros-handling.md)).
- **Worker-L1 placement.** Worker L1 is currently mapped in the low 4 GB
  (`low4g_mmap.hpp`: `MAP_32BIT`'s `[0, 2 GB)` first, then a `/proc/self/maps`-found
  `[2 GB, 4 GB)` gap via `MAP_FIXED`; `L1Pool` reserves one 2-MB-aligned slot per
  Tensix core, `MAP_NORESERVE` so slots cost only virtual address space until
  touched). Under the offset addressing model (§1) this placement is a backing
  detail, not an addressing constraint — the kernel-visible address is an offset
  independent of where L1 is mapped. Lifting the low-4 GB placement (a plain 64-bit
  `mmap` for worker cores) so a mesh whose total L1 exceeds 4 GB — e.g. a 32-chip
  Blackhole galaxy — fits in one process is a separate, behavior-neutral change
  on top of this model; see
  [multichip/l1-offset-translation.md](multichip/l1-offset-translation.md).

---

## 7. Where to read silicon canonical

| Topic | Canonical |
|---|---|
| L1 memory map / addresses | tt-metal HAL (`hal.get_dev_addr(...)`), `dev_mem_map.h` |
| L1 alignment | `hal::get_l1_alignment()` |
| Kernel-facing address APIs | `tt-metal/tt_metal/hw/inc/api/dataflow/*.h` |
