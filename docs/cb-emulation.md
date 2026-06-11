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
    uint32_t write_idx = 0;
    uint32_t read_idx  = 0;
    std::atomic<uint32_t> occupied{0};
    std::mutex              mu;
    std::condition_variable space_cv;   // producer waits for free space
    std::condition_variable data_cv;    // consumer waits for data
};
```

Four free functions implement the protocol: `cb_sync_reserve` (wait for N free),
`cb_sync_push` (advance write_idx + occupied, notify consumer), `cb_sync_wait`
(wait for N occupied), `cb_sync_pop` (advance read_idx, drop occupied, notify
producer). `occupied` is **atomic** so the SPSC fast path can probe it without
taking the lock. Pointer accessors `cb_sync_{read,write}_ptr[_at]` return
`base + (idx & page_mask) * page_size`.

---

## 3. Kernel-facing API

The JIT compute/dataflow surface (`include/jit_hw/api/cb_api.h`) over the per-core
`Core::cb_sync_states_[]` (exposed to kernels via the `__emule_cbs` TLS):

- `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`
  (uint32_t + int32_t overloads). `cb_reserve_back` also resets the pack
  auto-advance counter `__emule_pack_offset[cb_id] = 0`.
- `get_tile_size(cb_id)` / `get_tile_hw` / `get_tile_num_faces` — constexpr
  lookups into the `EMULE_TILE_SIZES` array.
- `get_dataformat(cb_id)` — from `EMULE_CB_DATA_FORMATS` (page-size fallback).
- Waits are **`<chrono>`-free by default** (`jit_hw/emule_wait.h`): `cv.wait` +
  `sched_yield`/`usleep`, deliberately avoiding the libstdc++ `<format>`/iostream
  pull that costs ~1 s of JIT parse. `TT_EMULE_WAIT_TIMEOUT=1` restores bounded
  `cv.wait_for` + a per-op deadlock diagnostic; `TT_EMULE_CB_TIMEOUT` sets the
  wait-front timeout.

A header-only `tt_emule::CircularBuffer` wrapper in
`include/tt_emule/circular_buffer.hpp` still exists from the pre-integration era
when emule shipped a standalone test harness; it owns its own storage around a
`CBSyncState`. It is no longer exercised by the regression suite — the
JIT-compiled path drives `Core::cb_sync_states_[]` directly.

---

## 4. Format-aware tile access, and CB↔L1 / CB↔DFB

Tiles are read/written through the central format-aware helpers in
`api/compute/common.h`: `__emule_compute::cb_{read,write}_ptr_at` give the raw
slice; `__emule_unpack_cb_tile_to` (CB→DEST) and `pack_dst_to_buf` (DEST→CB)
dispatch on the CB's data format (bf16 / fp32 / int32 / uint16 / Bfp8_b / Bfp4_b)
via the **enum-driven** `cb_is_32bit_format` / `cb_is_bfp8_b_format` etc. (page
size is only a fallback). The data-format and tile-size arrays reach the kernel as
the `EMULE_CB_DATA_FORMATS` / `EMULE_TILE_SIZES` JIT defines, emitted per-program
by `build_kernel_defines` from each CB's `data_format()` / `page_size()`. The
format dispatch itself is documented in [cb-dataformat.md](cb-dataformat.md); the
pack/unpack mechanics in [tilize-untilize-pack.md](tilize-untilize-pack.md).

CBs are slices of L1 ([l1-emulation.md](l1-emulation.md)). On Quasar, the
**DFB↔CB bridge** keeps the CB view (used by compute) and the DFB tile-counter
view (used by DM) coherent: `cb_push_back`→`inc_posted`,
`cb_pop_front`→`inc_acked`, `dfb_push_back`→`cb_sync_push`,
`dfb_pop_front`→`cb_sync_pop` (see [DFB_EMULATION.md](DFB_EMULATION.md)).

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
