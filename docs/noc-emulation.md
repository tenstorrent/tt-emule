# NOC Emulation in tt-emule

Single source of truth for how tt-emule emulates the silicon NOC (Network on
Chip). Read this before extending the mock surface, debugging a NOC-using
kernel, or auditing existing mocks against silicon canonical.

Cross-reference: silicon's canonical user-facing API at
`tt-metal/tt_metal/hw/inc/api/dataflow/*.h`. Mock surface at
`tt-emule/include/jit_hw/api/dataflow/*.h`.

---

## 1. Emulation model

tt-emule runs in **slow-dispatch mode**. Every NOC operation executes inline
before the API returns — there are no in-flight transactions, no outstanding
reads, no per-VC queues, no per-NOC parallel pipelines. The host thread that
issues `noc_async_read` walks the resolver, copies bytes, and returns. State
caches are per-NOC (silicon has two NOC engines and emule models that state
surface); the transfer path is one sync memcpy regardless of which NOC the
API names.

Consequences:
- **Async / posted / non-posted distinctions collapse** — all are sync memcpy.
- **Barriers are no-op** — there is nothing to wait for. `noc_async_write_barrier`,
  `noc_async_writes_flushed`, `noc_async_full_barrier` all return immediately, and
  the per-NOC barrier semantics collapse vacuously (no in-flight transactions on
  either NOC). The one exception is `noc_async_read_barrier`, which models a
  one-time-per-thread first-read latency (timing only, not a correctness wait — the
  data was already memcpy'd by the preceding read); see Section 1.1.
- **TRID (transaction id) is no-op** — `noc_async_*_set_trid`,
  `*_barrier_with_trid`, `ncrisc_noc_*_with_transaction_id_*` return without
  side-effects (or return `true` for completion probes).
- **Dual-NOC state, single-NOC transfer path** — silicon has two NOC engines
  (NOC 0 and NOC 1) each with independent sticky cmd-buf state. Emule models
  the **state surface** per NOC (see Section 8.3) so a kernel that interleaves
  `set_state(noc=0)` and `set_state(noc=1)` gets two independent caches. The
  **transfer path** is single — emule executes one sync memcpy per API call
  regardless of which NOC was named, and there is no NOC0/NOC1 concurrency.

This is correct **by design**, not drift: a kernel that produces the right
value on silicon produces the right value on emule. Timing and ordering
guarantees aren't preserved — emule is a correctness model, not a perf model.

### 1.1 Cross-core start ordering

Multi-core kernels that lean on silicon's launch/NOC timing rather than explicit
handshakes need two minimal timing models, because emule runs each core as a host
thread spawned sequentially with instant reads:

- **Startup barrier** (`emulated_program_runner.cpp::launch_cores`) — all kernel
  threads wait before running any kernel body, modeling simultaneous dispatch and
  removing the spawn stagger that lets an early thread finish before a peer starts.
- **First-read latency** (`dataflow_api.h::noc_async_read_barrier`) — a one-time
  per-thread delay modeling the NOC round-trip a core's first input read incurs
  before it can emit cross-core output. Sleeping threads also cede their cores, so
  a starved peer can run its prologue.

Motivating case: argmax's multi-core reducer resets `done_sem` in its ungated
`k=0` prologue then waits for worker increments; without these models a worker
increment can land before the reset, get clobbered, and hang the wait. See
[`index-based-ops`](../.claude/skills/index-based-ops/SKILL.md).

---

## 2. Address space + resolution

### 2.1 NOC address encoding

A 64-bit NOC address packs the destination's NOC coordinates with a local
L1 offset:

```
[ y_start (N) | x_start (N) | y_end (N) | x_end (N) | L1 offset (L) ]
```

Where `N = NOC_ADDR_NODE_ID_BITS` (e.g. 6 on Wormhole) and `L =
NOC_ADDR_LOCAL_BITS` (e.g. 36 on Wormhole). Both vary per arch — see
`include/jit_hw/noc/noc_parameters.h`.

For unicast: `x_end == x_start`, `y_end == y_start` — one core.
For multicast: the start/end pair defines an inclusive rectangle.

Constructed via `get_noc_addr(noc_x, noc_y, l1_addr, noc)` and
`get_noc_multicast_addr(x_start, y_start, x_end, y_end, l1_addr, noc)`.

### 2.2 Resolution chain

Three runtime hooks, defined in
`tt_metal/impl/emulation/emulated_program_runner.cpp`, bridge the JIT-compiled
kernel back to the host emulation runtime:

| Hook | Purpose |
|---|---|
| `__emule_resolve_noc_addr(uint64_t)` | Central: decode noc_addr → owning core → host pointer. Uses the program runner's core-map. |
| `__emule_local_l1_ptr(uint32_t)` | Fast path for local-core L1 offsets. Skips the noc decode. |
| `__emule_dram_ptr(uint64_t)` | Fast path for DRAM offsets, used by `AllocatorBank<DRAM>` lookups. |

Every dataflow API ultimately routes through one of these.

### 2.3 L1Pool vs bridge_l1

Two host-mmap layouts:
- **L1Pool mode** (`TT_EMULE_USE_L1_POOL`): a single contiguous mmap below 4GB
  serves as the L1 region for every emulated core. Local L1 offsets are direct
  host pointers masked via `addr & 0x1FFFFF` (the 2 MiB L1Pool slot mask —
  fixed, independent of `NOC_ADDR_LOCAL_BITS`).
- **bridge_l1 mode** (default): `__emule_bridge_l1` holds the L1 mmap base for
  this thread's core. Local L1 offsets are computed as `addr - __emule_bridge_l1`.

The address encoding is the same in both modes; only the conversion in
`__emule_addr_to_offset` / `__emule_local_l1_to_ptr` differs.

### 2.4 DRAM channel backing

DRAM is backed by **one host mmap per physical DRAM channel** — 6 on Wormhole
N150, 8 on Blackhole P100 (the outer dimension of the SoC descriptor's `dram:`
array). Every NOC endpoint that fronts a channel aliases onto that single
backing: the channel's subchannel cores (the inner `dram:` array), both
NOC0/NOC1 preferred-worker coords, and the multiple metal `dram_views` that map
to one physical channel (WH exposes 12 views over 6 channels — two views per
channel, distinguished only by `bank_to_dram_offset`). This mirrors silicon:
one physical channel addressed through many coords/views, so a write via any
endpoint is visible via any other (and a NOC 1 read sees a NOC 0 / host write).

The channel is resolved from the core's **UMD LOGICAL coordinate** (`x =
channel`), consistently on both sides: the host path
(`SWEmuleChip::write_to_device` → `get_dram_channel_for_core`) and the kernel
core-map build (`emulated_program_runner.cpp`, translating each preferred-worker
coord TRANSLATED→LOGICAL) — keyed by physical channel (not the metal view
index) so every view of a channel shares one backing. Implementation:
`SWEmuleChip::get_dram_channel_backing(channel)` in `device/chip/sw_emule_chip.cpp`
(umd); the per-NOC bank tables that feed kernel-side DRAM resolution are
described in §8.3.

---

## 3. Async read / write surface

### 3.1 Primary entry points

```cpp
template <uint32_t max_page_size, bool enable_noc_tracing = true>
void noc_async_read(uint64_t src_noc_addr, uint32_t dst_local_l1_addr,
                    uint32_t size, uint8_t noc = noc_index, uint32_t vc = NOC_UNICAST_WRITE_VC);

template <uint32_t max_page_size, bool enable_noc_tracing = true, bool posted = false>
void noc_async_write(uint32_t src_local_l1_addr, uint64_t dst_noc_addr,
                     uint32_t size, uint8_t noc = noc_index, uint32_t vc = NOC_UNICAST_WRITE_VC);
```

Both resolve the noc address via `__emule_resolve_noc_addr`, then `memcpy(size)`.
The template args are accepted-and-ignored for emulation purposes — present so
silicon-side callers that pass them explicitly still compile.

### 3.2 One-packet variants

`noc_async_read_one_packet` / `noc_async_write_one_packet` are silicon fast
paths for ≤16KB transfers. In emule they delegate to the non-one-packet
versions (same per-byte memcpy).

### 3.3 Set-state / with-state pairs

Silicon programs NOC cmd-buf registers in `set_state`, then reuses them in
`with_state` — letting many small reads share the upper-coord setup cost.
Silicon's registers are per `(noc, cmd_buf)`. Emule mirrors that with per-NOC
`[2]` TLS arrays so `set_state(noc=0)` and `set_state(noc=1)` don't clobber
each other:

| TLS (per-NOC `[2]`, indexed by `noc & 1`) | Holds | Used by |
|---|---|---|
| `__emule_one_packet_state_size` | size from one-packet read `set_state` | `noc_async_read_one_packet_with_state` |
| `__emule_noc_trid_state::shard_noc_addr_base` | full src noc addr | `_with_state_with_trid` (upper coords) + multi-packet `_with_state` |
| `__emule_noc_trid_state::shard_size` | size from per-NOC TRID `set_state` | same |
| `__emule_noc_trid_state::shard_vc` | VC from per-NOC TRID `set_state` | same |
| `__emule_write_one_packet_state_dst` | dst noc addr from write `set_state` | write `_with_state` |
| `__emule_write_one_packet_state_size` | size from write `set_state` | write `_with_state` |

`with_state` calls reconstruct the full noc addr from cached upper-32 + caller-
supplied lower-32 (or use the explicit `src_noc_addr` arg directly for the
plain one-packet `_with_state`), then route through the resolver.

The **device-2.0 `class Noc`** exposes the same pattern via its
`set_async_read_state`/`async_read_with_state` and
`set_async_write_state`/`async_write_with_state` methods, backed by
`__emule_noc_cached_size[NUM_NOCS]` (transfer size — shared by the read- and
write-state pairs) and `__emule_noc_cached_write_dst[NUM_NOCS]` (resolved write
destination). These are deliberately per-NOC **globals**, not `Noc` instance
members: the upstream wrapper helpers (`experimental::set_read_state` /
`read_with_state` in `experimental_device_api.hpp`) take `Noc` **by value**, so
state stored on the instance set in `set_state` would be lost before the paired
`with_state` reads it. Modeling it as global per-NOC state mirrors silicon,
where this lives in per-NOC cmd-buf registers that outlive any `Noc` handle
copy.

### 3.4 TRID variants

`*_with_trid` is silicon's per-transaction tag for fine-grained barrier
control. In emule, every write is already complete by the time the call
returns — the `trid` arg is accepted and dropped. Both `noc_async_read_*_trid`
and `noc_async_write_*_trid` collapse to their non-trid siblings.

---

## 4. Multicast

### 4.1 Runtime hook

```cpp
extern "C" void __emule_multicast_write(uint64_t mcast_addr,
                                        const uint8_t* src,
                                        uint32_t size,
                                        bool include_self);
```

The runtime decodes the rectangle (`x_start..x_end` × `y_start..y_end`), looks
up each destination core via the core map, and `memcpy(size)` to each L1.
`include_self` controls whether the sender's coordinates within the rectangle
receive the packet too (silicon's `NOC_CMD_BRCST_SRC_INCLUDE` flag).

### 4.2 Public APIs

```cpp
template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1>
void noc_async_write_multicast(uint32_t src, uint64_t dst, uint32_t size,
                               uint32_t num_dests, bool linked = false,
                               uint8_t noc = noc_index, uint8_t vc = NOC_MULTICAST_WRITE_VC);
// include_self = false

void noc_async_write_multicast_loopback_src(uint32_t src, uint64_t dst,
                                            uint32_t size, uint32_t num_dests,
                                            bool linked = false, uint8_t noc = noc_index);
// include_self = true (silicon: NOC_CMD_BRCST_SRC_INCLUDE)

template <bool enable_noc_tracing = true>
void noc_async_write_multicast_one_packet(/* same args */);
// silicon fast path for size ≤ NOC_MAX_BURST_SIZE; emule delegates to multicast
```

The `Noc::async_write_multicast<NocOptions opts>(...)` class method
dispatches on `has_flag(opts, NocOptions::MCAST_INCL_SRC)` to choose between
the two.

### 4.3 Why one-packet collapses

Silicon's `_one_packet` variant takes a fast path that skips the multi-packet
ncrisc helper. In emule there is no ncrisc helper — every multicast is one
per-core memcpy regardless of size. So `_one_packet` simply delegates.

---

## 5. Semaphore model

Semaphores live in L1 as 4-byte aligned counters. Emule reads/writes them as
`std::atomic<uint32_t>*` for cross-thread visibility (silicon's race-prone
non-atomic `*ptr += value` is strictly weaker — mock is more correct, and any
kernel that depended on the silicon-side race would already be broken).

### 5.1 Local-side operations

```cpp
template <ProgrammableCoreType core_type = ProgrammableCoreType::TENSIX>
class Semaphore {
    void up(uint32_t value);     // local atomic fetch_add
    void down(uint32_t value);   // spin-wait + atomic fetch_sub
    void wait(uint32_t target);  // spin-wait until == target
    void wait_min(uint32_t v);   // spin-wait until ≥ v
    void set(uint32_t value);    // atomic store
    // ...
};
```

Spin-waits include hang detection: 10M iterations without progress triggers
`std::abort()` with a context print. This catches deadlocks in mock-runtime
mismatches without infinite hangs.

The free function `noc_semaphore_wait(ptr, val)` waits for `>= val` when
`val >= 1` (monotonic handshake counter) and exact `== 0` when `val == 0`
(VALID→0 release toggle). Silicon polls `!= val`; that works there because NOC
latency paces increments so the waiter observes every value, but emule's
`noc_semaphore_inc` is a synchronous zero-latency atomic, so a peer can drive a
counter `1→2→3` between two polls and an exact wait would skip the target and
hang. A count-up target is never 0, so the split is unambiguous.

### 5.2 Remote operations

| Call | Behaviour |
|---|---|
| `Semaphore::up(noc, x, y, v, vc)` | Remote atomic fetch_add via `noc_semaphore_inc` |
| `Semaphore::set_multicast<opts>(noc, x_start, ..., num_dests, linked)` | `__emule_multicast_write` 4-byte broadcast; `MCAST_INCL_SRC` opt controls loopback |
| `Semaphore::inc_multicast(noc, x_start, ..., value, num_dests)` | Walks rect, per-core resolve + atomic fetch_add |
| `Semaphore::relay_unicast(noc, dst_sem, x, y)` | 4-byte unicast write of local value to a different sem on remote core |
| `Semaphore::relay_multicast<opts>(noc, dst_sem, ...)` | Mcast write of local value to a different sem on every core in rect |
| `noc_semaphore_set_remote(src_l1, dst_noc, noc)` | 4-byte unicast write (free function, same path as `set` but for a remote target) |
| `noc_semaphore_inc(addr, incr, noc, vc)` | Resolve + atomic fetch_add at addr |
| `noc_semaphore_inc_multicast(addr, incr, num_dests, noc, vc)` | Walks rect, per-core resolve + atomic fetch_add |

The `ProgrammableCoreType` template arg selects which L1 base to use on
silicon (TENSIX vs ACTIVE_ETH vs IDLE_ETH). Emule has only the TENSIX L1
base — the param is accepted-and-ignored.

---

## 6. The trait system

`noc_traits_t<T>` is the type-erasure shim that lets `Noc::async_read<opts>`,
`Noc::async_write<opts>`, etc. take heterogeneous src/dst arguments (CB, DFB,
unicast endpoint, multicast endpoint, allocator bank) and resolve each to a
host pointer + size.

Specializations live in:

| Type | File | Resolves via |
|---|---|---|
| `CircularBuffer` | `circular_buffer.h` | `get_write_ptr()` / `get_read_ptr()` + offset |
| `CircularBufferView<AddrSelector>` | `circular_buffer.h` | as above, parametrized on WRITE_PTR / READ_PTR |
| `DataflowBuffer` | `dataflow_buffer.h` | `get_write_ptr()` / `get_read_ptr()` |
| `UnicastEndpoint` | `endpoints.h` | `__emule_resolve_noc_addr(get_noc_addr(x, y, addr, noc))` |
| `MulticastEndpoint` | `endpoints.h` | raw mcast addr (passed to `__emule_multicast_write`) |
| `AllocatorBank<L1>` | `endpoints.h` | `__emule_resolve_noc_addr(get_noc_addr_from_bank_id<false>(bank_id, addr, noc))` |
| `AllocatorBank<DRAM>` | `endpoints.h` | `__emule_resolve_noc_addr(get_noc_addr_from_bank_id<true>(bank_id, addr, noc))` |
| `TensorAccessor<DSpec>` | `noc_traits.h` | `__emule_resolve_noc_addr(acc.get_noc_addr(page_id, off, noc))` |
| `tensor_accessor::Page` | `noc_traits.h` | `__emule_resolve_noc_addr(page.noc_addr() + off)` |
| `ShardView<Accessor>` | `noc_traits.h` | `__emule_resolve_noc_addr(acc.get_noc_addr(shard_id, off, noc))` |
| `AbstractTensorAccessorWrapper` | `noc_traits.h` | `__emule_resolve_noc_addr(acc.get_noc_addr(page_id, off, noc))` |

The tensor-accessor specializations in `noc_traits.h` are made visible to every
dataflow kernel transitively through `circular_buffer.h` (mirroring upstream's
`circular_buffer.h → noc_zero_dram.inl → noc_traits.h` chain).

The `Noc::async_read<opts>(Src, Dst, size, src_args, dst_args, noc_opts)`
template walks the traits to resolve both pointers, then `memcpy(size)`.

### 6.1 DataflowBuffer implicit-sync overload

```cpp
Noc::async_read<NocOptions::TXN_ID>(Src, DataflowBuffer, ...)
```

Issues `reserve_back` → resolve src → `memcpy` → `push_back`. The synchronous
fence here matches silicon's TXN_ID semantics (the kernel can assume the
buffer is populated by the time the call returns).

---

## 7. Address-generator helpers

`include/jit_hw/internal/dataflow/dataflow_api_addrgen.h` carries the
`interleaved_addr_gen::*` helpers used by `InterleavedAddrGen` and friends.
These are **header-only** in emule — no link-time dependency on tt-metal.

The per-bank lookup tables `bank_to_dram_offset[]`, `bank_to_l1_offset[]`,
`dram_bank_to_noc_xy[]`, `l1_bank_to_noc_xy[]` are runtime-injected by the
emule program runner (`tt_metal/impl/emulation/emulated_program_runner.cpp`).
Their values come from the per-arch device topology.

`get_noc_addr_from_bank_id<DRAM>` (templated bool) computes the per-bank NOC
address by looking up the bank's owning core in the right table, then packing
the result with the bank-local offset.

---

## 8. Runtime bridge

### 8.1 `extern "C"` hooks

These are the only symbols the JIT-compiled kernel links to in the emulation
runtime. Every other API in this doc is header-only and inlined into the
kernel.

| Symbol | Defined in | Used by |
|---|---|---|
| `__emule_resolve_noc_addr` | `emulated_program_runner.cpp` | Every read/write that needs core-map lookup |
| `__emule_local_l1_ptr` | `emulated_program_runner.cpp` | local-core L1 fast path (not on the resolver path) |
| `__emule_dram_ptr` | `emulated_program_runner.cpp` | DRAM-offset fast path (not on the resolver path) |
| `__emule_multicast_write` | `emulated_program_runner.cpp` | Every multicast write / semaphore set_multicast |

### 8.2 Per-thread context state

Most of these are now fields of the per-thread execution context (a `ComputeThreadCtx`
/ `DatamovementThreadCtx`, reached via the single `thread_local ThreadCommonCtx*
__emule_self`), set by the program runner in the launch prologue and read by the
JIT kernel — see [state-tiers.md](state-tiers.md). The exception is `my_x`/`my_y`,
which stay runner-set `thread_local` globals because they are silicon-named symbols
read by unmodified upstream code; the emule-only logical coordinates moved to the
per-core `CoreState` (reached as `__emule_self->core->logical_x/y`).

| State | Type | Home | Read by |
|---|---|---|---|
| `__emule_self->bridge_l1` | `uint8_t*` | `ThreadCommonCtx` | every L1 conversion |
| `__emule_self->core->logical_x` / `_y` | `uint32_t` | per-core `CoreState` | debug prints, `get_absolute_logical_*` |
| `my_x[2]`, `my_y[2]` | `uint8_t` | runner-set global (silicon-named) | `get_noc_addr(addr, noc)` 2-arg overload |
| `__emule_self->cbs` | `CBSyncState*` | `ThreadCommonCtx` | every CB API |
| `__emule_self->dfbs` | `EmuleDFBInterface*` | `ThreadCommonCtx` (Quasar) | every DFB API |

(`noc_index` / `noc_mode` are not context state — they're per-kernel compile-time
constants from host-emitted JIT defines; see §8.3.)

### 8.3 Per-NOC state

Silicon has two NOC engines (NOC 0 and NOC 1) each with independent
sticky cmd-buf state — `set_state(noc=0)` and `set_state(noc=1)` are
independent caches in silicon. Emule mirrors this **at the state level
only**; the transfer path is single (see Section 1). A kernel that
interleaves set_state calls on both NOCs must read back independent
caches, or it gets silently wrong data.

The per-NOC cmd-buf caches are fields of `DatamovementThreadCtx` (the NOC is a
data-movement role; reached via `__emule_datamovement_ctx().<field>[noc]` — see
[state-tiers.md](state-tiers.md)). `my_x`/`my_y` stay runner-set globals
(silicon-named) and the bank tables are per-chip (runtime-injected, shared).

| State | Type | Home / indexed by | Used by |
|---|---|---|---|
| `my_x` / `my_y` | `uint8_t[2]` | runner-set global · `noc & 1` | `get_noc_addr(addr, noc)` |
| `dram_bank_to_noc_xy` | per-arch table | per-chip · `[noc][bank]` | DRAM bank → NOC coords |
| `l1_bank_to_noc_xy` | per-arch table | per-chip · `[noc][bank]` | L1 bank → NOC coords |
| `shard_noc_addr_base` | `uint64_t[2]` | `DatamovementThreadCtx` · `noc & 1` | TRID-tagged stateful reads |
| `shard_size` | `uint32_t[2]` | `DatamovementThreadCtx` · `noc & 1` | same |
| `shard_vc` | `uint32_t[2]` | `DatamovementThreadCtx` · `noc & 1` | same |
| `one_packet_state_size` | `uint32_t[2]` | `DatamovementThreadCtx` · `noc & 1` | `noc_async_read_one_packet_{set,with}_state` |
| `write_one_packet_state_dst` | `uint64_t[2]` | `DatamovementThreadCtx` · `noc & 1` | `noc_async_write_one_packet_{set,with}_state` |
| `write_one_packet_state_size` | `uint32_t[2]` | `DatamovementThreadCtx` · `noc & 1` | same |
| `dw_st` | `__emule_dw_state[2]` | `DatamovementThreadCtx` · `noc & 1` | `noc_inline_dw_write_{set,with}_state` |
| `noc_cached_size` | `uint32_t[NUM_NOCS]` | `DatamovementThreadCtx` · `noc_id_` | `Noc::{set_async_read_state, async_read_with_state, set_async_write_state, async_write_with_state}` transfer size |
| `noc_cached_write_dst` | `uintptr_t[NUM_NOCS]` | `DatamovementThreadCtx` · `noc_id_` | `Noc::{set_async_write_state, async_write_with_state}` resolved dst |

`noc_index` and `noc_mode` are **faithful per-kernel** compile-time constants —
`constexpr uint8_t noc_index = NOC_INDEX; noc_mode = NOC_MODE;` in
`jit_kernel_stubs.hpp`, mirroring the firmware `dataflow_api_common.h`
`KERNEL_BUILD` formula. The host emits `NOC_INDEX` / `NOC_MODE` per kernel
(BRISC→NOC 0, NCRISC→NOC 1; `DM_DEDICATED_NOC` default); emule's `#ifndef`
fallbacks (`NOC_INDEX→0`, `NOC_MODE→DM_DEDICATED_NOC`) cover the compute
wrappers that omit them. The dataflow API signatures default `noc = noc_index`
(mirroring silicon), so a call that omits the arg uses the kernel's own NOC.

The per-NOC bank tables (`dram_bank_to_noc_xy` / `l1_bank_to_noc_xy`) are
declared `extern [2][NUM_*_BANKS]` on the kernel side; the runner lays them out
with the matching **actual-count stride** (`tbl[noc*num_banks + bank]`, where
`num_banks` is the real per-arch count, not a padded `MAX_NUM_BANKS`) so the
`noc=1` row resolves to the right per-NOC coords.

---

## 9. What's intentionally simplified

These are NOT drift — emule deliberately collapses them:

- **Barriers** are no-op. There is nothing in-flight to wait for. Per-NOC
  barrier semantics collapse vacuously because both NOCs have nothing
  pending.
- **TRID counters** are no-op. No transaction tagging needed when every
  op is already complete by the time the API returns.
- **Dual-NOC transfer-path concurrency** — silicon's two NOC engines can
  carry independent transactions in parallel. Emule has no such
  parallelism: every API call is one sync memcpy by the host thread
  regardless of which NOC was named. (Per-NOC **state** is modeled —
  see Section 8.3 — but per-NOC transfer concurrency is not.)
- **Posted vs non-posted writes** collapse — every write is fully complete
  before the call returns.
- **VC (virtual channel)** args are ignored. No NOC bandwidth model.
- **Stream registers (`INLINE_REG` in `Noc::inline_dw_write`)** are not modeled.
  Writes route to memory regardless of the `INLINE_L1` / `INLINE_REG` dispatch.
- **`ProgrammableCoreType::ACTIVE_ETH` / `IDLE_ETH`** is not modeled. Eth-fabric
  is out of scope for single-chip emule. Semaphores in those cores use the
  TENSIX L1 base.
- **CB pages_reservable_at_back / pages_available_at_front** always return true.
  Emule's `cb_reserve_back` / `cb_wait_front` block instead.

---

## 10. Known drift

Divergences from silicon that don't affect correctness for the current
single-chip, non-eth-fabric, TENSIX-only emule scope:

- `ProgrammableCoreType` per-core-type L1 base selection (eth-fabric not
  modeled in emule).
- `Noc::inline_dw_write` `INLINE_REG` stream-register dispatch.
- `noc_traits_t<UnicastEndpoint>::src/dst_addr<LOCAL_L1>` does a resolve+truncate
  through L1Pool's mask (same final pointer either way).

---

## 11. Where to read silicon canonical

When a question arises about what emule should do, the canonical answer is in
the silicon header. Pointers:

| Topic | Silicon canonical |
|---|---|
| All NOC free functions | `tt-metal/tt_metal/hw/inc/api/dataflow/dataflow_api.h` |
| `Noc` class | `tt-metal/tt_metal/hw/inc/api/dataflow/noc.h` |
| `Semaphore` class | `tt-metal/tt_metal/hw/inc/api/dataflow/noc_semaphore.h` |
| `CircularBuffer` | `tt-metal/tt_metal/hw/inc/api/dataflow/circular_buffer.h` |
| `DataflowBuffer` | `tt-metal/tt_metal/hw/inc/api/dataflow/dataflow_buffer.h` |
| `UnicastEndpoint` / `MulticastEndpoint` / `AllocatorBank` | `tt-metal/tt_metal/hw/inc/api/dataflow/endpoints.h` |
| Address-generator helpers | `tt-metal/tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h` |
| NOC HW capability docs | DeepWiki / Confluence — see `.claude/skills/arch-lookup` and the `sage-*` agents |
