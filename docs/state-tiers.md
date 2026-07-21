<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->
# State tiers in tt-emule

How tt-emule's mutable state is partitioned into **per-chip**, **per-core**, and
**per-thread (per-RISC)** tiers, and how a kernel reaches its per-thread state
through one context pointer. Read this before adding new emulated state, moving a
field between tiers, or working on the fiber execution engine
([fiber-engine.md](fiber-engine.md)) — the fiber step swaps the per-thread tier by
repointing a single pointer, which only works because the state is encapsulated as
described here.

Companion docs (per-subsystem detail): [dest-emulation.md](dest-emulation.md),
[cb-emulation.md](cb-emulation.md), [noc-emulation.md](noc-emulation.md),
[metal-integration.md](metal-integration.md) (where the per-thread context is
allocated in the runner).

---

## Threading model: per-thread ≡ per-RISC

Each `(core, RISC)` kernel runs on a cooperatively-scheduled **fiber**, multiplexed
onto a pool of K worker OS threads (see [fiber-engine.md](fiber-engine.md); the
fibers start together at a barrier — [metal-integration.md](metal-integration.md)
§Layer 3). On WH/BH the three compute TRISCs (UNPACK/MATH/PACK) collapse onto **one**
fiber that runs them in program order; each data-movement RISC (BRISC/NCRISC) is its
own fiber. So an emulated fiber corresponds 1:1 to a RISC, and "per-thread state" is
"per-RISC state" — carried by the fiber.

---

## The three tiers

### Tier 1 — per-chip (already encapsulated)

Owned by the UMD `SWEmuleChip` and the runner; shared read-only by all cores
after init. Not part of this encapsulation work (documented here for completeness):

- `SWEmuleChip`'s `L1Pool` + per-DRAM-channel mmaps ([l1-emulation.md](l1-emulation.md),
  [dram-emulation.md](dram-emulation.md)).
- The runtime-injected bank tables (`dram_bank_to_noc_xy` / `bank_to_dram_offset`
  / `l1_bank_to_noc_xy` / `bank_to_l1_offset`) and the core map used for NOC
  address resolution ([noc-emulation.md](noc-emulation.md)).
- The SoC / cluster descriptor.

### Tier 2 — per-core

Owned by `tt_emule::Core` (lifetime = device open→close), one per logical core:

- The core's L1 mmap; the per-CB `CBSyncState` (geometry + `occupied` semaphore +
  condvars, [cb-emulation.md](cb-emulation.md)); the DFB sync state + tile
  counters ([DFB_EMULATION.md](DFB_EMULATION.md)); L1-resident semaphores.
- **`tt_emule::CoreState`** (`include/jit_hw/internal/emule_core_state.h`) — the
  logical coordinates (`logical_x` / `logical_y`) and the **inter-RISC mailbox**
  (`CoreMailbox mbox[2][2]`, below). Split into its own minimal, dependency-free
  header so `Core` (in `include/tt_emule/device.hpp`) and the umd TU can embed it
  without parsing the kernel-only per-thread context below.

#### Inter-RISC mailbox (`ckernel::mailbox_write` / `mailbox_read`)

On silicon these are a small **blocking** RISC-to-RISC FIFO: `mailbox_write(dst, v)`
hands a scalar from the calling RISC to thread `dst`, and `mailbox_read(src)` blocks
until the peer `src` has written one. `ttnn.sparse_matmul` uses it to keep its in0
reader (BRISC) and compute triad in lockstep — the reader signals, per batch, whether
each expert is a non-zero-sparsity (valid) batch (`is_batch_valid`), and the compute
skips the invalid ones instead of waiting on CB 0 for tiles the reader never pushes.
A no-op stub here (read always returns "valid") makes the compute wait on CB 0 for
skipped batches ⇒ a quiescent fiber deadlock, single-chip and multichip alike; this is
exactly what blocked gpt-oss decode until the model is fully removed as a variable.

emule models it faithfully with a per-core `CoreMailbox` (a bounded ordered ring) and
a **blocking read that parks the fiber** ([fiber-engine.md](fiber-engine.md)) while the
ring is empty — the cooperative analogue of the HW stall. It lives in `CoreState`
because both endpoints (the DM reader fiber and the compute fiber) run on the same core
and share it. Because emule **fuses UNPACK/MATH/PACK into one compute fiber** and has no
per-sub-thread identity, the exact `(src-thread, dst-thread)` pair is not recoverable;
but every mailbox exchange in the tree is either **DM→compute** (sparse matmul) or
**compute→compute** (deepseek / swiglu), so the ring is keyed by the coarse
`(writer_group, reader_group) ∈ {DM, Compute}²`, derived from the caller's
`ThreadCommonCtx::kind` and the `ThreadId` argument (`Brisc`→DM, `Unpack`/`Math`/`Pack`
→Compute). Balanced kernels (equal writes and reads) drain the ring to empty each op.

### Tier 3 — per-thread (per-RISC)

The bulk of the encapsulation: the ~90 mutable JIT globals that used to be loose
`thread_local`s now live in one of three structs in
`include/jit_hw/internal/emule_thread_ctx.h`, specialized by RISC type.

| Struct | Carries | Why here |
|---|---|---|
| `ThreadCommonCtx` (every thread) | Set-once identity / handles (`rt_args`, `common_rt_args`, `core_obj`, `device`, `chip_id`, `bridge_l1`, `bridge_dram`, `cbs`, `dfbs`, `tc_array`, `core_map`, `processor_id`, `neo_id`, `trisc_id`, `num_threads`, `my_thread_id`); a borrowed `CoreState* core`; the **cross-role CB ring pointers** (`local_cb[]` + `cb_self_consume_mask`) | identity is read by all roles; `chip_id` is per-fiber so fabric teleport resolves the source chip when a worker co-runs fibers from several chips (mesh register/run dispatch); CB ring pointers are per-RISC but cross-role — a CB producer is often a DM RISC and the consumer a compute TRISC ([cb-emulation.md](cb-emulation.md)) |
| `ComputeThreadCtx : ThreadCommonCtx` (the fused compute thread) | DST (`dst` / `dst_fresh` / `src_scratch`, [dest-emulation.md](dest-emulation.md)); pack/unpack config (incl. `pack_offset` / `pack_width`) + pack-fused ReLU; the grouped SFPU state (`sfpu`, see `sfpi_types.h::SfpuState` — DST window, predication stack, LReg file, programmable const regs); LLK tilize/untilize/matmul trackers + the matmul operand bridge; op accumulators (welford / ema / dropout RNG / cumsum / cumprod / quant) | DST + SFPU/LReg are compute-TRISC-exclusive Tensix-engine resources |
| `DatamovementThreadCtx : ThreadCommonCtx` (each BRISC/NCRISC thread) | Per-NOC[2] command-buffer set/with-state caches (one-packet / write / TRID / inline-dw / shard) | NOC issuance is a data-movement role (see the convention note below) |

---

## Mechanism: one context pointer, explicit call sites

```cpp
// include/jit_hw/internal/emule_thread_ctx.h
extern thread_local ThreadCommonCtx* __emule_self;        // the only thread_local

ComputeThreadCtx&      __emule_compute_ctx();              // *static_cast<ComputeThreadCtx*>(__emule_self)
DatamovementThreadCtx& __emule_datamovement_ctx();         // (kernel-only; __EMULE_JIT_MODE)
```

- The runner allocates the **derived** context matching the kernel's RISC type
  (`new ComputeThreadCtx` / `new DatamovementThreadCtx`), writes the identity /
  handles through it, and points `core` at the `Core`'s `CoreState`. The context is
  owned by the fiber that runs the kernel; the scheduler points `__emule_self` at it
  on swap-in (see [fiber-engine.md](fiber-engine.md)).
- A kernel reaches **shared** state via `__emule_self->…` and **specialized**
  state via the typed accessor (`__emule_compute_ctx().dst[idst]`,
  `__emule_datamovement_ctx().one_packet_state_size[noc]`). Every call site is
  updated explicitly — **no `#define` aliases**.
- Because a compute-kernel TU never names a `DatamovementThreadCtx` field (and
  vice-versa), the downcast is type-safe per TU. The accessors carry a no-op
  `ASSERT` documenting the kind contract (DST/SFPU only from a compute thread, NOC
  cmd-buf only from a DM thread) rather than a silent guard.

### Ownership / lifetime

| Tier | Owned by | Reached via |
|---|---|---|
| Per-chip | `SWEmuleChip` / `Device` | — |
| Per-core | `tt_emule::Core` | the thread ctx's borrowed `CoreState* core` |
| Per-thread | the fiber running the kernel | the single `thread_local ThreadCommonCtx* __emule_self` |

No state is copied on a context switch — only the pointer moves. That is the
property the fiber engine relies on: the scheduler repoints `__emule_self` on
swap-in.

---

## Two rules that shaped the partition

1. **Only `__emule_`-prefixed (emule-only) symbols migrated into the context.**
   Silicon-named globals read by *unmodified* upstream code must stay
   runner-set globals. The coordinate globals `my_x` / `my_y` have ~hundreds of
   upstream readers (`tensor_accessor.h`, `checkpoint.h`, the real
   `dataflow_api_addrgen.h`, …), so they remain `thread_local` globals set by the
   runner; only the emule-only logical coordinates moved into `CoreState`. Grep
   the upstream tree before migrating any non-`__emule_`-prefixed name.

2. **NOC issuance is a software convention, not a hard wall.** WH resolves a NOC
   proc-type only for DM0/DM1 (compute never issues NOC); Blackhole has no Tensix
   NOC opcode (the NIU is MMIO), so HW does not *forbid* a TRISC driving the NOC.
   NOC cmd-buf state therefore lives in `DatamovementThreadCtx` by convention, and
   the contract is documented (a debug `ASSERT`), not enforced by a clamp.

---

## What's intentionally simplified

- emule fuses the three compute TRISCs into one thread, so the silicon
  MATH/PACK/SFPU intra-compute mutexes collapse to program order — nothing to
  model. A future per-TRISC split (Quasar-style, [QUASAR_EMULATION.md](QUASAR_EMULATION.md))
  must not inherit a "MATH owns DEST/SFPU exclusively" assumption: HW shares the
  SFPU across TRISC1/TRISC2 and supports unpack-to-DEST.
- The per-chip tier is documented as already-encapsulated (`SWEmuleChip` + the
  injected bank tables); formalizing it into an explicit `ChipState` is deferred.

---

## Where to read canonical

| Topic | Location |
|---|---|
| The three context structs + accessors | `include/jit_hw/internal/emule_thread_ctx.h` |
| Per-core `CoreState` | `include/jit_hw/internal/emule_core_state.h`, `include/tt_emule/device.hpp` |
| Grouped SFPU state | `include/jit_hw/sfpi_types.h` (`SfpuState`), `include/jit_hw/sfpi.h` |
| Context allocation / `__emule_self` set | `tt_metal/impl/emulation/emulated_program_runner.cpp` (launch prologue) |
| The fiber engine that builds on this | [fiber-engine.md](fiber-engine.md) |
