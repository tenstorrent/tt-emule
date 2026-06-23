# Circular Buffer Emulation in tt-emule

How tt-emule emulates **circular buffers (CBs)** — the producer/consumer FIFOs
that move tiles between the reader, compute, and writer kernels. Read this before
debugging a CB sync hang, a wrong-tile read, or extending the CB API.

On silicon a CB is a region of L1 plus a pair of hardware read/write pointers
that firmware advances. Emule backs the pointers + occupancy with a host
synchronization struct and stores tiles in L1 (see
[l1-emulation.md](l1-emulation.md)).

Companion docs: [cb-dataformat.md](cb-dataformat.md) (per-CB data-format
dispatch, the bf16-vs-uint16 ambiguity), [tilize-untilize-pack.md](tilize-untilize-pack.md)
(pack/unpack of a CB tile), [l1-emulation.md](l1-emulation.md) (CBs live in L1),
[noc-emulation.md](noc-emulation.md) (readers/writers fill/drain CBs over NOC).

---

## 1. Emulation model

A CB is a single-producer / single-consumer ring over a slice of L1. Under slow
dispatch each core runs reader, compute, and writer as separate host threads, so
the producer/consumer handshake is real cross-thread synchronization — backed by
a mutex + condition variables, with a lock-free fast path on an atomic occupancy
counter. Waits are bounded by a hang-detection timeout. (On Quasar, DFBs replace
CBs with MPMC tile counters — see [DFB_EMULATION.md](DFB_EMULATION.md) — and a
bridge keeps both views coherent; Section 4.)

---

## 2. Backing store: `CBSyncState`

`include/tt_emule/cb_sync_state.hpp`:

```cpp
struct CBSyncState {
    uint8_t* base      = nullptr;     // host pointer to the CB's L1 slice
    uint32_t page_size = 0;           // bytes per tile
    uint32_t num_pages = 0;           // ring capacity in tiles
    uint32_t page_mask = 0;           // num_pages - 1 (pow2 fast modulo)
    std::atomic<uint32_t> occupied{0};  // the shared cross-RISC semaphore
    std::mutex              mu;
    std::condition_variable space_cv;   // producer waits for free space
    std::condition_variable data_cv;    // consumer waits for data
};
```

`CBSyncState` holds **only what is genuinely shared across RISCs** on silicon: the
geometry (`base`/`page_size`/`num_pages`/`page_mask`) and the pages-occupied
**semaphore** (`occupied` + the two condvars) — the analog of the L1
pages_received/acked counter. The read/write *pointers* are **not** here; they are
per-RISC (§2b). Four semaphore-only free functions implement the protocol:
`cb_sync_reserve` (wait for N free), `cb_sync_push` (bump `occupied`, notify
consumer), `cb_sync_wait` (wait for N occupied), `cb_sync_pop` (drop `occupied`,
notify producer). `occupied` is **atomic** so the SPSC fast path can probe it
without taking the lock.

---

## 2b. Per-RISC read/write pointers (`emule_cb_ptr.h`)

On silicon each RISC (DMR / DMW / Tensix) owns its **own** per-CB read and write
pointer registers — a `push_back` on the writer RISC advances only *that* RISC's
write pointer. Emule models this in
[`include/jit_hw/internal/emule_cb_ptr.h`](../include/jit_hw/internal/emule_cb_ptr.h):

```cpp
inline thread_local LocalCBInterface __emule_local_cb[NUM_CIRCULAR_BUFFERS]{};
```

a per-thread (= per-RISC, since each kernel runs on its own host thread) copy of
silicon's `LocalCBInterface` register file. `__emule_cb_{wr,rd}_addr(cb, off)` and
`__emule_cb_advance_{wr,rd}(cb, n)` read/advance *this thread's* `fifo_{wr,rd}_ptr`
(geometry — base / page_size / ring wrap — still comes from the shared
`CBSyncState`). `thread_local` zero-initialises per launch, mirroring silicon's
per-RISC register reset at kernel start.

---

## 3. Kernel-facing API

The JIT compute/dataflow surface (`include/jit_hw/api/cb_api.h`) over the per-core
`Core::cb_sync_states_[]` (exposed to kernels via the `__emule_cbs` TLS):

- `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`
  (uint32_t + int32_t overloads). `cb_push_back`/`cb_pop_front` advance the
  calling RISC's per-RISC pointer (§2b) and then bump/drop the shared `occupied`
  semaphore; `cb_reserve_back` also resets the pack auto-advance counter
  `__emule_pack_offset[cb_id] = 0`.
- `get_write_ptr(cb_id)` / `get_read_ptr(cb_id)` — return the calling RISC's own
  per-RISC pointer (§2b), so concurrent reader/writer RISCs each see their own
  view of the CB.
- `get_tile_size(cb_id)` / `get_tile_hw` / `get_tile_num_faces` — constexpr
  lookups into the `EMULE_TILE_SIZES` array.
- `get_tile_r_dim(cb_id)` / `get_tile_c_dim(cb_id)` — per-CB active tile height /
  width, from the `unpack_tile_r_dim`/`unpack_tile_c_dim` arrays the runner emits
  via `EMULE_TILE_R_DIM`/`EMULE_TILE_C_DIM` (the tiny-tile shape plumbing;
  default 32×32). Drive the tile-shape-aware nfaces / pack / unpack paths.
- `get_dataformat(cb_id)` — from `EMULE_CB_DATA_FORMATS` (enum-only; no
  page-size fallback — see [cb-dataformat.md](cb-dataformat.md)).
- Waits are **`<chrono>`-free by default** (`jit_hw/emule_wait.h`): `cv.wait` +
  `sched_yield`/`usleep`, deliberately avoiding the libstdc++ `<format>`/iostream
  pull that costs ~1 s of JIT parse. `TT_EMULE_WAIT_TIMEOUT=1` restores bounded
  `cv.wait_for` + a per-op deadlock diagnostic; `TT_EMULE_CB_TIMEOUT` sets the
  wait-front timeout.

---

## 4. Format-aware tile access, and CB↔L1 / CB↔DFB

Tiles are read/written through the central format-aware helpers in
`api/compute/common.h`: `__emule_compute::cb_{read,write}_ptr_at` give the raw
slice (resolved via the per-RISC pointer of §2b — `__emule_cb_{wr,rd}_addr`, the
same pointer the dataflow `get_{write,read}_ptr` use, so compute and dataflow
never diverge); `__emule_unpack_cb_tile_to` (CB→DEST) and `pack_dst_to_buf` (DEST→CB)
dispatch on the CB's data format (bf16 / fp32 / int32 / uint16 / Bfp8_b / Bfp4_b)
via the **enum-driven** `cb_is_32bit_format` / `cb_is_bfp8_b_format` etc. The
pack/unpack predicates are **enum-only** (no page-size heuristic fallback); an
`Invalid` format is a known-failure until the runner threads formats through.
by `build_kernel_defines` from each CB's `data_format()` / `page_size()`. The
format dispatch itself is documented in [cb-dataformat.md](cb-dataformat.md); the
pack/unpack mechanics in [tilize-untilize-pack.md](tilize-untilize-pack.md).

CBs are slices of L1 ([l1-emulation.md](l1-emulation.md)). On Quasar, the
**DFB↔CB bridge** keeps the CB view (used by compute) and the DFB tile-counter
view (used by DM) coherent: `cb_push_back`→`inc_posted`,
`cb_pop_front`→`inc_acked`, and `dfb_push_back`/`dfb_pop_front` both advance the
calling RISC's per-RISC pointer (§2b) and bump/drop the shared `occupied`
semaphore via `cb_sync_push`/`cb_sync_pop` — so a consumer reading DFB tiles
through `cb_read_ptr_at` (e.g. matmul-based reduce) tracks the producer
(see [DFB_EMULATION.md](DFB_EMULATION.md)).

---

## 5. What's intentionally simplified

- No backpressure timing / bandwidth model — sync is for correctness only.
- `pages_reservable_at_back` / `pages_available_at_front` are not modeled as
  non-blocking probes; `cb_reserve_back` / `cb_wait_front` block instead.
- Under slow dispatch the reader/compute/writer largely run sequentially; the
  CV machinery is defensive for the multi-threaded harness.

---

## 6. Where to read silicon canonical

| Topic | Canonical |
|---|---|
| CB kernel API | `tt-metal/tt_metal/hw/inc/api/dataflow/circular_buffer.h` |
| CB config / data format | `tt-metal/.../circular_buffer_config` ; [cb-dataformat.md](cb-dataformat.md) |
| Tile / face dims | `tt-metal/tt_metal/hw/inc/.../tensix_types.h` |
