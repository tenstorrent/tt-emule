# State tiers in tt-emule

How tt-emule's mutable state is partitioned into **per-chip**, **per-core**, and
**per-thread (per-RISC)** tiers, and how a kernel reaches its per-thread state
through one context pointer. Read this before adding new emulated state, moving a
field between tiers, or working on the fiber execution engine
([multichip/pillar0-fiber-engine.md](multichip/pillar0-fiber-engine.md)) — the
fiber step swaps the per-thread tier by repointing a single pointer, which only
works because the state is encapsulated as described here.

Companion docs (per-subsystem detail): [dest-emulation.md](dest-emulation.md),
[cb-emulation.md](cb-emulation.md), [noc-emulation.md](noc-emulation.md),
[metal-integration.md](metal-integration.md) (where the per-thread context is
allocated in the runner).

---

## Threading model: per-thread ≡ per-RISC

The runner launches **one OS thread per kernel descriptor** and joins them after a
startup barrier (see [metal-integration.md](metal-integration.md) §Layer 3). On
WH/BH the three compute TRISCs (UNPACK/MATH/PACK) collapse onto **one** host
thread that runs them in program order; each data-movement RISC (BRISC/NCRISC) is
its own thread. So an emulated thread corresponds 1:1 to a RISC, and "per-thread
state" is "per-RISC state".

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
  logical coordinates (`logical_x` / `logical_y`). Split into its own minimal,
  dependency-free header so `Core` (in `include/tt_emule/device.hpp`) and the umd
  TU can embed it without parsing the kernel-only per-thread context below.

### Tier 3 — per-thread (per-RISC)

The bulk of the encapsulation: the ~90 mutable JIT globals that used to be loose
`thread_local`s now live in one of three structs in
`include/jit_hw/internal/emule_thread_ctx.h`, specialized by RISC type.

| Struct | Carries | Why here |
|---|---|---|
| `ThreadCommonCtx` (every thread) | Set-once identity / handles (`rt_args`, `common_rt_args`, `core_obj`, `device`, `bridge_l1`, `bridge_dram`, `cbs`, `dfbs`, `tc_array`, `core_map`, `processor_id`, `neo_id`, `trisc_id`, `num_threads`, `my_thread_id`); a borrowed `CoreState* core`; the **cross-role CB ring pointers** (`local_cb[]` + `cb_self_consume_mask`) | identity is read by all roles; CB ring pointers are per-RISC but cross-role — a CB producer is often a DM RISC and the consumer a compute TRISC ([cb-emulation.md](cb-emulation.md)) |
| `ComputeThreadCtx : ThreadCommonCtx` (the fused compute thread) | DST (`dst` / `dst_fresh` / `src_scratch`, [dest-emulation.md](dest-emulation.md)); pack/unpack config + pack-fused ReLU; the grouped SFPU state (`sfpu`, see `sfpi_types.h::SfpuState` — DST window, predication stack, LReg file, programmable const regs); LLK tilize/untilize/matmul trackers + the matmul operand bridge; op accumulators (welford / ema / dropout RNG / cumsum / cumprod / quant) | DST + SFPU/LReg are compute-TRISC-exclusive Tensix-engine resources |
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
  handles through it, points `core` at the `Core`'s `CoreState`, and sets
  `__emule_self` — all before the startup barrier (see
  [metal-integration.md](metal-integration.md) §Layer 3).
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
| Per-thread | the runner today / a fiber object later | the single `thread_local ThreadCommonCtx* __emule_self` |

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
| The fiber engine that builds on this | [multichip/pillar0-fiber-engine.md](multichip/pillar0-fiber-engine.md) |
