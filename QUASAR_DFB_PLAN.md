# Dataflow Buffer (DFB) Emulation Plan

This document describes the Quasar Dataflow Buffer API and the plan for emulating it in tt-emule. It serves as a self-contained reference for any agent or developer picking up DFB emulation work.

## References

- **tt-metal repo (upstream)**: `/localdev/arminale/tt-metal-main`
- **tt-emule repo**: `/localdev/arminale/tt-emule`
- **Parent plan**: [QUASAR_EMULATION_PLAN.md](QUASAR_EMULATION_PLAN.md)

---

## 1. What DFBs Replace

On Wormhole/Blackhole, **Circular Buffers (CBs)** are the synchronization mechanism between data movement and compute. CBs are simple SPSC (single-producer, single-consumer) FIFOs: one reader kernel writes tiles, one compute kernel consumes them.

On Quasar, **Dataflow Buffers (DFBs)** replace CBs. The key difference: a single Quasar Tensix has **8 DM cores + 4 Neo compute clusters** sharing **one 4 MB L1**, so DFBs must support **multiple producers and multiple consumers** on the same buffer.

### CB vs DFB at a Glance

| Aspect | CB (Wormhole) | DFB (Quasar) |
|--------|---------------|--------------|
| Producers/consumers | 1:1 (SPSC) | Up to 8 DM + 4 Tensix (MPMC) |
| Sync primitive | Single atomic `occupied` + mutex/CV | Per-TC `posted`/`acked` counters |
| Buffer layout | Ring buffer, single rd/wr index pair | Shared L1 region, per-TC logical rd/wr pointers |
| Access pattern | Sequential | STRIDED (interleaved) or BLOCKED (broadcast) |
| Pointer type | `uint32_t` (mmap below 4 GB) | `uint32_t` L1 addresses |
| API surface | `cb_reserve_back/push_back/wait_front/pop_front` | Same four + `read_in/write_out/finish` |

---

## 2. Quasar Architecture Context

### Confirmed Facts (reviewed with hardware team)

- **24 logical processors per Tensix**: 8 Data Movement (DM0-DM7) + 4 Neo groups x 4 compute slots (E0_MATH0-E3_MATH3) = 16 compute slots. HAL: `max_processors_per_core_ = 24`. Enum: `TensixProcessorTypes` in `core_config.h` (DM0=0..DM7=7, E0_MATH0=8..E3_MATH3=23, COUNT=24).
- **4 MB shared L1**: all processors within one Tensix access the same L1
- **DMs are fully shared**: any of the 8 DMs can interact with any Neo's tile counters (RISC mask determines participation)
- **TRISC roles** (internal names mapping to `MATHn` enum slots): MATH0 = TRISC0 (unpacker/consumer), MATH1 = TRISC1 (math-only, no DFB interaction), MATH2 = TRISC2 (packer/producer), MATH3 = TRISC3 (alternate consumer)
- **Remapper is credits-only**: routes tile counter updates between Neo domains; data stays in shared L1
- **TC slots are logical views**: multiple TC slots point into the same L1 region with independent rd/wr cursors
- **Intra-Tensix DFBs**: simpler variant for same-Neo packer-to-unpacker communication

### Key Constants

```
NUM_DFBS = 32                          # Max DFBs per Tensix
NUM_TENSIX = 4                         # Neo domains
NUM_TILE_COUNTERS_PER_TENSIX = 32      # TCs per Neo
MAX_NUM_TILE_COUNTERS_TO_RR = 4        # Max TC slots per RISC per DFB
NUM_TXN_IDS = 4                        # Transaction IDs for implicit sync
TENSIX_RISC_OFFSET = 8                 # Bits 0-7 = DM, bits 8-15 = Tensix in RISC masks
```

---

## 3. tt-metal Source Files (in `/localdev/arminale/tt-metal-main`)

### Host Side

| File | Role |
|------|------|
| `tt_metal/api/tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp` | Public API: `DataflowBufferConfig`, `CreateDataflowBuffer`, `BindDataflowBufferToProducerConsumerKernels` |
| `tt_metal/impl/dataflow_buffer/dataflow_buffer_impl.hpp` | `DataflowBufferImpl`, `TileCounterAllocator`, `RemapperIndexAllocator`, `TxnIdAllocator`, `ClientTypeAllocator`, `LocalDFBInterfaceHost`, `DFBRiscConfig` (host-side serialization helpers) |
| `tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp` | ~1160 lines: validation, TC allocation, remapper setup, serialization |
| `tt_metal/impl/program/program_impl.hpp` | `ProgramImpl::add_dataflow_buffer`, finalize/allocate hooks |

### Device Side

| File | Role |
|------|------|
| `tt_metal/hw/inc/experimental/dataflow_buffer.h` | `experimental::DataflowBuffer` class -- the kernel-facing API |
| `tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h` | Packed structs: `dfb_initializer_t` (36B), `dfb_initializer_per_risc_t` (44B), `dfb_initializer_intra_tensix_t` (24B) |
| `tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h` | Runtime state: `LocalDFBInterface` (103B), `DFBTCSlot` (17B), `TxnDFBDescriptor` |
| `tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_init.h` | `setup_local_dfb_interfaces()` -- firmware init, remapper config, ISR, TC init barrier |
| `tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_isr.h` | `dfb_implicit_sync_handler` -- IRQ for txn-ID based auto tile counter updates |
| `tt_metal/hw/ckernels/quasar/metal/llk_io/llk_io_unpack.h` | `llk_wait_tiles`, `llk_pop_tiles` (TRISC consumer) |
| `tt_metal/hw/ckernels/quasar/metal/llk_io/llk_io_pack.h` | `llk_wait_for_free_tiles`, `llk_push_tiles` (TRISC producer) |

### Test Kernels

| File | Role |
|------|------|
| `tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp` | DM producer: explicit + implicit sync paths |
| `tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp` | DM consumer: explicit + implicit, strided + blocked |
| `tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6.cpp` | Compute kernel: passthrough (in->out) using DFBs |
| `tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_producer.cpp` | Compute-side producer: `reserve_back`, `push_back`, `finish` |
| `tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_consumer.cpp` | Compute-side consumer: `wait_front`, `pop_front` |
| `tests/tt_metal/tt_metal/api/dataflow_buffer/test_dataflow_buffer.cpp` | Host API integration test |
| `tests/tt_metal/tt_metal/api/dataflow_buffer/test_dataflow_buffer_configs.cpp` | Config/finalization unit tests |

---

## 4. tt-emule Current CB Implementation (for comparison)

### Files to understand before implementing DFBs

| File | Role |
|------|------|
| `include/tt_emule/cb_sync_state.hpp` | `CBSyncState` struct: the SPSC sync primitive with atomic `occupied`, mutex, two CVs |
| `include/tt_emule/circular_buffer.hpp` | `CircularBuffer` class: owns storage (`vector<uint8_t>`), embeds `CBSyncState` |
| `include/tt_emule/device.hpp` | `Core` class: holds 32 `shared_ptr<CircularBuffer>` + 32 `CBSyncState` for JIT path |
| `include/jit_hw/emule_cb_state.h` | `__emule_cb_state` = `CBSyncState`; thread-local `__emule_cbs` pointer |
| `include/jit_hw/api/cb_api.h` | JIT CB ops: `cb_reserve_back/push_back/wait_front/pop_front` with timeout detection |
| `include/kernel_api/dataflow_api.hpp` | Standalone CB ops via `__core->cb(id)->sync_state()` |
| `src/kernel_runner.cpp` | Thread spawning: one `std::thread` per kernel, sets thread-locals, joins |
| `include/tt_emule/program.hpp` | `Program` class: kernel descriptors + CB configs |

### Key CB Sync Pattern (what DFBs must generalize)

```cpp
// CBSyncState -- SPSC, single occupied counter
struct CBSyncState {
    uint8_t*  base;
    uint32_t  page_size, num_pages, page_mask;
    uint32_t  write_idx, read_idx;
    std::atomic<uint32_t> occupied;   // THE sync primitive
    std::mutex mu;
    std::condition_variable space_cv, data_cv;
};

// Producer: wait for free space, then push
cb_sync_reserve(cb, n);  // spin on (num_pages - occupied) >= n, slow: space_cv.wait
cb_sync_push(cb, n);     // advance write_idx, occupied.fetch_add, data_cv.notify

// Consumer: wait for data, then pop
cb_sync_wait(cb, n);     // spin on occupied >= n, slow: data_cv.wait
cb_sync_pop(cb, n);      // advance read_idx, occupied.fetch_sub, space_cv.notify
```

### Two API Paths (both must support DFBs)

| Path | How CB state is reached | Pointer type |
|------|-------------------------|--------------|
| **Standalone** (`kernel_api/`) | `__core->cb(id)->sync_state()` | `uint8_t*` |
| **JIT** (`jit_hw/`) | `__emule_cbs[cb_id]` (thread-local array) | `uint32_t` (truncated) |

---

## 5. Tile Counter (TC) Architecture

The core DFB synchronization primitive. Each TC tracks:
- **`posted`** (atomic): incremented by producer after writing data
- **`acked`** (atomic): incremented by consumer after reading data
- **`capacity`**: set at init time
- **Derived**: `occupancy = posted - acked`, `free_space = capacity - occupancy`

Identified by `PackedTileCounter` (8 bits): 2-bit `tensix_id` (0-3) + 5-bit `counter_id` (0-31).

### Hardware: 4 Neo domains x 32 TCs = 128 total

In emulation: a flat array of 128 `TileCounter` structs, each with atomic posted/acked + a condition variable for blocking waits.

---

## 6. Device-Side API Details

### `experimental::DataflowBuffer` (kernel-facing class)

```cpp
class DataflowBuffer {
    LocalDFBInterface& local_dfb_interface_;  // per-RISC view of this DFB
    uint16_t logical_dfb_id_;

    // Explicit sync
    void reserve_back(uint16_t num_entries);   // producer: wait for free space
    void push_back(uint16_t num_entries);      // producer: signal data written
    void wait_front(uint16_t num_entries);     // consumer: wait for data
    void pop_front(uint16_t num_entries);      // consumer: signal data consumed
    void finish();                             // drain barrier

    // Implicit sync (DM only -- gated by #ifndef COMPILE_FOR_TRISC, deferred for emulation)
    void read_in(noc, src, args);
    void write_out(noc, dst, args);

    // Pointers
    uint32_t get_write_ptr() const;
    uint32_t get_read_ptr() const;
    uint32_t get_entry_size() const;
    uint32_t get_stride_size() const;
    uint16_t get_id() const;              // returns logical_dfb_id_

    // Misc
    Lock scoped_lock();                   // Lock wrapper (body TODO in upstream)
    // noc_traits_t<DataflowBuffer> specialization for NOC addressing (DM-only)
};
```

### Per-RISC View: `LocalDFBInterface`

Each thread has its own `LocalDFBInterface` for each DFB it participates in:

```cpp
struct LocalDFBInterface {
    DFBTCSlot tc_slots[4];           // up to 4 TC slots to round-robin
    uint32_t entry_size;
    uint32_t stride_size;
    uint32_t stride_size_tiles;
    uint32_t rd_entry_idx, wr_entry_idx, wr_entry_ptr;
    uint8_t txn_ids[4];
    uint8_t num_entries_per_txn_id, num_entries_per_txn_id_per_tc;
    uint8_t num_tcs_to_rr;           // how many TC slots this RISC round-robins
    uint8_t num_txn_ids;
    uint8_t tc_idx;                  // current round-robin index
    uint8_t tensix_trisc_mask;       // which TRISCs use this DFB
    uint8_t broadcast_tc;            // BLOCKED mode: post to all TCs
};

struct DFBTCSlot {
    uint32_t rd_ptr, wr_ptr;
    uint32_t base_addr, limit;
    PackedTileCounter packed_tile_counter;  // identifies which TC this slot uses
};
```

### Explicit Sync Behavior (what we emulate)

**`reserve_back(n)` -- producer waits for free space:**
- TRISC packer: hardware stall via `llk_wait_for_free_tiles`
- DM strided: `while (free_space(tensix_id, tc_id) < n);` on current TC slot
- DM blocked: loop over ALL TC slots, wait until every consumer TC has `free_space >= n`

**`push_back(n)` -- producer signals data written:**
- TRISC packer: hardware push via `llk_push_tiles`, advances wr_ptr
- DM strided: `inc_posted(tensix_id, tc_id, n)`, advance wr_ptr, `tc_idx = (tc_idx+1) % num_tcs_to_rr`
- DM blocked: `inc_posted` on ALL TCs, advance wr_ptr on slot 0 only, tc_idx stays

**`wait_front(n)` -- consumer waits for data:**
- TRISC unpacker: hardware stall via `llk_wait_tiles`; gated by `tensix_trisc_mask` (only assigned TRISC acts, others return immediately)
- DM: `while (occupancy(tensix_id, tc_id) < n);`

**`pop_front(n)` -- consumer signals data consumed:**
- TRISC unpacker: `llk_pop_tiles`, advances rd_ptr, round-robins tc_idx
- DM: `inc_acked(tensix_id, tc_id, n)`, advance rd_ptr, `tc_idx = (tc_idx+1) % num_tcs_to_rr`

**`finish()` -- drain barrier:**
- DM: spin until `read_acked == read_posted` for all TCs
- TRISC: spin until `posted == 0` for all TCs

---

## 7. Access Patterns

### STRIDED (default, Phase 1)
- Entries interleaved: producer `i` writes entries `i, i+P, i+2P, ...`
- Each producer/consumer gets its own TC slot(s) and round-robins through them
- `capacity_per_risc = num_entries / max(num_producers, num_consumers)`
- `stride_in_entries = max(num_producers, num_consumers) / num_this_role`

### BLOCKED (consumer only, Phase 2)
- All consumers see ALL data (broadcast semantics)
- Producer posts to every consumer TC simultaneously
- Each consumer has independent read progress tracked by its own TC
- Remapper routes credits for Tensix consumers; DM-DM uses software broadcast loop

---

## 8. Host-Side API

### `DataflowBufferConfig`

> **Note**: The upstream API header explicitly marks this config as a "placeholder" pending a host API redesign. The emulation layer should implement the current interface but avoid over-investing in exact field-level fidelity -- the struct may change upstream.

```cpp
struct DataflowBufferConfig {
    uint32_t entry_size = 0;
    uint32_t num_entries = 0;
    uint16_t producer_risc_mask = 0x0;   // bits 0-7 = DM, 8-15 = Tensix
    uint8_t num_producers = 1;
    AccessPattern pap = STRIDED;
    uint16_t consumer_risc_mask = 0x0;
    uint8_t num_consumers = 1;
    AccessPattern cap = STRIDED;
    bool enable_implicit_sync = false;
    DataFormat data_format = Float16_b;
    std::optional<Tile> tile = std::nullopt;
};
```

### Constraints

- Cannot mix DFBs and CBs in the same program
- BLOCKED producer not supported (only consumer can be BLOCKED)
- One logical core per DFB
- BLOCKED consumer: `num_consumers <= 4`
- STRIDED: `num_entries % max(num_producers, num_consumers) == 0`
- Producer/consumer RISC masks must not overlap

---

## 9. Initialization Flow

```
Host: CreateDataflowBuffer + Bind + Finalize
  -> Serialize dfb_initializer_t + per_risc blobs to L1

Device (all RISCs call setup_local_dfb_interfaces):
  1. Each RISC parses config, fills its own LocalDFBInterface
  2. DM0 exclusively: configure remapper, setup TxnDFBDescriptors, enable ISR
  3. Each producer: reset + set_capacity on its assigned TCs, set tc_init_done
  4. Barrier: all RISCs spin until all producers have set tc_init_done
```

In emulation, steps 2-3 simplify to: allocate tile counter structs, set capacities, signal ready. Step 4 becomes a `std::barrier` or equivalent.

---

## 10. Emulation Strategy

### Phase 1: Explicit Sync, STRIDED Mode

**Scope**: Explicit `reserve_back/push_back/wait_front/pop_front` with configurable cluster count. Design for the general case (up to 8 DMs, 4 Neos) from the start with data-driven thread/cluster counts; initial tests may exercise subsets (e.g. 1 DM + 1 Neo) but the infrastructure must not hard-code these limits.

**Data-driven configuration**: The number of Neos (clusters), DMs, and tile counters per Neo must come from a configuration descriptor, not hard-coded constants. Different Quasar variants will have different cluster counts. The tile counter array dimensions should be `num_neos * NUM_TILE_COUNTERS_PER_TENSIX`.

**New structures needed:**

1. **`TileCounter`** -- emulates one hardware tile counter:
   ```
   struct TileCounter {
       std::atomic<uint32_t> posted{0};
       std::atomic<uint32_t> acked{0};
       uint32_t capacity{0};
       std::mutex mu;
       std::condition_variable space_cv;  // producer waits here
       std::condition_variable data_cv;   // consumer waits here
   };
   // occupancy = posted - acked
   // free_space = capacity - occupancy
   ```

2. **`TileCounterArray`** -- parameterized by cluster count:
   ```
   // num_neos is data-driven (from config), not hard-coded to 4
   std::vector<std::array<TileCounter, 32>> counters;  // counters[tensix_id][counter_id]
   // or: TileCounter counters[num_neos][32] if compile-time
   ```

3. **`DFBSyncState`** -- per-DFB state (replaces `CBSyncState` for Quasar):
   ```
   struct DFBSyncState {
       uint8_t* base;                // shared L1 backing memory
       uint32_t entry_size;
       uint32_t num_entries;
       uint32_t capacity;            // per-RISC capacity
       uint32_t stride_in_entries;
       TileCounterArray* tc_array;   // pointer to shared tile counter matrix
   };
   ```

4. **`EmuleDFBInterface`** -- per-thread view (emulates `LocalDFBInterface`):
   ```
   struct EmuleDFBInterface {
       DFBTCSlot tc_slots[4];
       uint8_t num_tcs_to_rr;
       uint8_t tc_idx;
       uint32_t entry_size, stride_size;
       bool broadcast_tc;
       // ... other fields as needed
   };
   ```

**Key simplification**: Since all emulation threads share the same address space, the tile counter array is a plain shared data structure. The hardware remapper is a no-op -- any thread can directly read/write any tile counter. This eliminates the most complex hardware-specific logic.

### Phase 1 Implementation Status (as of 2026-03-31)

**All Phase 1 infrastructure is implemented and passing tests.** Files created/modified:

| File | What was done |
|------|---------------|
| `include/tt_emule/tile_counter.hpp` | **New.** `TileCounter` (atomic posted/acked, capacity, mutex, 2 CVs) + `TileCounterArray` (flat `unique_ptr<TileCounter[]>`, parameterized by num_neos). Includes `inc_posted`, `inc_acked`, `wait_free_space`, `wait_occupancy`, `reset_all`. |
| `include/tt_emule/dfb_sync_state.hpp` | **New.** `DFBTCSlot` (rd/wr ptr, base/limit, neo/counter id), `EmuleDFBInterface` (4 TC slots, round-robin state, entry/stride/num_entries, broadcast/active flags), `DFBSyncState` (base ptr, entry geometry, capacity). |
| `include/tt_emule/dataflow_buffer.hpp` | **New.** `DataflowBuffer` class: `reserve_back`, `push_back` (strided + broadcast), `wait_front`, `pop_front`, `finish` (drain barrier on space_cv), pointer accessors. Takes `EmuleDFBInterface&` + `TileCounterArray&`. |
| `include/jit_hw/emule_dfb_state.h` | **New.** Thread-local `__emule_dfbs` (EmuleDFBInterface*) + `__emule_tc_array` (TileCounterArray*). |
| `include/jit_hw/api/dfb_api.h` | **New.** JIT DFB ops with timeout detection (`TT_EMULE_DFB_TIMEOUT` env, default 120s). Same semantics as `DataflowBuffer` class but operates directly on thread-locals. |
| `include/kernel_api/dfb_dataflow_api.hpp` | **New.** Standalone DFB ops. Constructs temporary `DataflowBuffer` per call via `__core->tile_counters()` and `__dfb_ifaces`. |
| `include/tt_emule/program.hpp` | **Modified.** Added `QuasarDM`/`QuasarCompute` to `KernelType`, `processor_id` to `KernelDescriptor`, `AccessPattern` enum, `DataflowBufferConfig`, `DFBHandle`, `QuasarDataMovementConfig`, `QuasarComputeConfig`, `Program::add_dfb()`. |
| `include/tt_emule/device.hpp` | **Modified.** Added `TileCounterArray` (unique_ptr) + `DFBSyncState[32]` to `Core`, with `init_tile_counters()`, `init_dfb_sync()`, `reset_dfb_sync()`. |
| `src/kernel_runner.cpp` | **Modified.** Added `build_dfb_interfaces()` (per-thread EmuleDFBInterface construction from DataflowBufferConfig + RISC masks), `std::barrier` init sync, DFB thread-locals (`__dfb_ifaces`, `__emule_dfbs`, `__emule_tc_array`, `__processor_id`), DFB teardown. |
| `src/host_api.cpp` | **Modified.** Added `CreateDataflowBuffer()`, Quasar `CreateKernel()` overloads for `QuasarDataMovementConfig`/`QuasarComputeConfig`. |
| `include/tt_emule/host_api.hpp` | **Modified.** Declared Quasar `CreateKernel` and `CreateDataflowBuffer`. |
| `tests/dfb_passthrough/` | **New.** 1 DM producer + 1 DM consumer, 8x1KB entries, STRIDED 1:1. Validates full reserve/push/wait/pop/finish cycle + pointer accessors. |

**Minor deviations from the planned spec:**
- `TileCounterArray` uses flat `unique_ptr<TileCounter[]>` instead of `vector<array<TileCounter,32>>` (functionally equivalent, avoids non-movability of `std::array<TileCounter,32>`)
- `DFBSyncState` does not contain a `TileCounterArray*` pointer; the TC array is owned by `Core` and accessed via thread-local `__emule_tc_array` or `__core->tile_counters()`
- `DataflowBuffer` in the standalone path is ephemeral (constructed per API call from `EmuleDFBInterface&`); the JIT path operates directly on thread-locals without constructing the class

### Phase 2: BLOCKED Mode + Implicit Sync

- BLOCKED: producer posts to ALL consumer TCs; straightforward once per-TC counters work
- Implicit sync: since NOC ops are synchronous memcpy, `read_in`/`write_out` become memcpy + immediate `inc_posted`/`inc_acked`
- Intra-Tensix DFBs: degenerate case with `num_tcs_to_rr = 1`

### Integration Points in tt-emule

| Component | Change needed | Status |
|-----------|---------------|--------|
| `device.hpp` / `Core` | Add `TileCounterArray`, DFB sync state array, init/reset methods | **Done** -- `tile_counters_`, `dfb_sync_states_[32]`, `init_tile_counters()`, `init_dfb_sync()`, `reset_dfb_sync()` |
| `program.hpp` | Add `DataflowBufferConfig`, `DFBHandle`, `add_dfb()` alongside `add_cb()` | **Done** -- also added `QuasarDM`/`QuasarCompute` to `KernelType`, `processor_id` to `KernelDescriptor`, `QuasarDataMovementConfig`/`QuasarComputeConfig` |
| `kernel_runner.cpp` | Instantiate DFBs, set up per-thread `EmuleDFBInterface`, init barrier | **Done** -- `build_dfb_interfaces()`, `std::barrier`, per-thread `__dfb_ifaces`/`__emule_dfbs`/`__emule_tc_array`/`__processor_id` |
| `host_api.hpp` / `host_api.cpp` | `CreateDataflowBuffer`, Quasar `CreateKernel` overloads | **Done** |
| New: `tile_counter.hpp` | `TileCounter` (atomic posted/acked + mutex/CVs), `TileCounterArray` (parameterized by `num_neos`) | **Done** |
| New: `dfb_sync_state.hpp` | `DFBTCSlot`, `EmuleDFBInterface`, `DFBSyncState` | **Done** |
| New: `dataflow_buffer.hpp` | `DataflowBuffer` class (emulation version) | **Done** |
| New: `jit_hw/emule_dfb_state.h` | Thread-local `__emule_dfbs`, `__emule_tc_array` | **Done** |
| New: `jit_hw/api/dfb_api.h` | JIT-facing DFB operations with timeout detection (parallel to `cb_api.h`) | **Done** |
| New: `kernel_api/dfb_dataflow_api.hpp` | Standalone DFB operations (parallel to `dataflow_api.hpp`) | **Done** |

---

## 11. Target Test Workloads

These are the tt-metal tests that use DFBs on Quasar. They define the emulation surface we must support.

### Workload Summary

| Test | Workload | DFBs | Producers | Consumers | Sync | Compute APIs |
|------|----------|------|-----------|-----------|------|-------------|
| `test_bmm.cpp` | Batched matmul (DRAM->DFB->compute->DFB->DRAM) | 3 (src0, src1, dst) | 1 DM each | 1 Tensix / 1 DM | Explicit | `mm_init`, `matmul_tiles`, `pack_tile`, `acquire/release_dst` |
| `test_matmul_X_tile.cpp` | Block matmul | 3 (src0, src1, dst) | 1 DM each | 1 Tensix / 1 DM | Explicit | `mm_block_init`, `matmul_block`, `pack_tile` |
| `test_direct.cpp` | DRAM round-trip datacopy | 2 (input, output) | 1-4 DM | 1-4 Tensix | Implicit | `unary_op_init_common`, `copy_tile`, `pack_tile` |
| `test_dataflow_buffer.cpp` | Synthetic DFB traffic | 1 or 2 | 1-4 DM/Tensix | 1-4 DM/Tensix | Both | None (DM-DM) or passthrough (dfb_t6) |

### Device-Side Kernels Used

**Dataflow (DM) kernels:**

| Kernel | Role | DFB ops | Other APIs |
|--------|------|---------|-----------|
| `test_kernels/dataflow/reader_bmm_8bank.cpp` | DRAM reader (2 inputs) | `reserve_back`, `push_back`, `get_write_ptr` | `Noc`, `noc_async_read_tile`, `TensorAccessor` |
| `test_kernels/dataflow/writer_bmm_8bank.cpp` | DRAM writer | `wait_front`, `pop_front`, `get_read_ptr` | `noc_async_write_tile`, `Noc` |
| `test_kernels/dataflow/reader_matmul_with_bias_blocked.cpp` | Blocked DRAM reader | `reserve_back`, `push_back` | `Noc`, `AllocatorBank<DRAM>` |
| `kernels/dataflow/writer_unary.cpp` | DRAM writer (production) | `wait_front`, `pop_front` | `Noc`, `AllocatorBank<DRAM>` |
| `test_kernels/dataflow/dfb_producer.cpp` | Synthetic producer | `reserve_back`/`push_back` OR `read_in`; `finish` | `Noc`, `TensorAccessor`, `mhartid` |
| `test_kernels/dataflow/dfb_consumer.cpp` | Synthetic consumer | `wait_front`/`pop_front` OR `write_out` | `Noc`, `TensorAccessor` |
| `test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp` | Implicit sync reader | `read_in` only | `Noc`, `AllocatorBank<DRAM>`, `get_my_thread_id` |
| `test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp` | Implicit sync writer | `write_out` only | `Noc`, raw `g_dfb_interface`, per-txn barrier |

**Compute kernels:**

| Kernel | Role | DFB ops | Compute APIs |
|--------|------|---------|-------------|
| `test_kernels/compute/bmm.cpp` | Matmul A*B=C | `wait_front`, `pop_front` (in); `reserve_back`, `push_back` (out) | `mm_init`, `matmul_tiles`, `pack_tile`, `acquire/release_dst` |
| `test_kernels/compute/eltwise_copy.cpp` | Unary copy | Same pattern | `unary_op_init_common`, `copy_tile`, `pack_tile` |
| `test_kernels/compute/matmul_block.cpp` | Block matmul | Same, with block counts | `mm_block_init`, `matmul_block`, `pack_tile` |
| `test_kernels/compute/dfb_t6.cpp` | Passthrough (in->out) | `wait_front`, `pop_front`, `reserve_back`, `push_back`, `finish` | None (pure sync) |
| `test_kernels/compute/dfb_t6_producer.cpp` | Compute-side producer | `reserve_back`, `push_back`, `finish` | None |
| `test_kernels/compute/dfb_t6_consumer.cpp` | Compute-side consumer | `wait_front`, `pop_front` | None |

### Common DFB Configurations in Tests

Most tests use these patterns:
- **Entry size**: 1024 bytes (synthetic) or 2048 bytes (bf16 tile) or 4096 bytes (fp32 tile)
- **Num entries**: 2-16
- **Access pattern**: STRIDED (most common), BLOCKED consumer (some DFB unit tests)
- **Typical topology**: 1 DM producer -> 1 Tensix consumer (BMM, matmul), or N DM -> N Tensix (datacopy)
- **RISC masks**: `0x1` = DM0, `0x2` = DM1, `0x100` = Tensix Neo0; set explicitly or via `Bind`

### Required Emulation API Surface (Phase 1)

**DFB operations** (minimum for explicit sync tests):
- `reserve_back(n)`, `push_back(n)`, `wait_front(n)`, `pop_front(n)` -- **Done** (standalone + JIT paths)
- `finish()` -- **Done** (DM variant: `posted == acked`; TRISC variant `posted == 0` not yet needed)
- `get_write_ptr()`, `get_read_ptr()`, `get_entry_size()`, `get_stride_size()`, `get_id()` -- **Done**

**Compute operations** (from kernel analysis) -- **Not yet implemented**:
- `mm_init(in0, in1, out)`, `matmul_tiles(in0, in1, tile_a, tile_b, dst_idx)`
- `mm_block_init(in0, in1, out)`, `matmul_block(in0, in1, ...block dims...)`
- `unary_op_init_common(in, out)`, `copy_tile(cb_id, tile_idx, dst_idx)`
- `pack_tile(dst_idx, out_cb_id)`, `acquire_dst()`, `release_dst()`

**NOC operations** (from dataflow kernels) -- **Not yet implemented for DFB path**:
- `noc.async_read(src, dfb, size, src_args, dst_args)` -- with DFB as endpoint
- `noc.async_write(dfb, dst, size, src_args, dst_args)` -- with DFB as endpoint
- `noc.async_read_barrier()`, `noc.async_write_barrier()`
- Legacy: `noc_async_read_tile(tile_id, tensor, addr)`, `noc_async_write_tile(tile_id, tensor, addr)`

**Host operations**:
- `CreateDataflowBuffer(program, config)` -> logical DFB id -- **Done**
- `BindDataflowBufferToProducerConsumerKernels(program, dfb_id, producer, consumer)` -- **Not yet implemented** (RISC masks set manually)
- `experimental::quasar::CreateKernel(program, path, core, config)` with `QuasarDataMovementConfig` / `QuasarComputeConfig` -- **Partially done** (function-pointer overloads work; JIT path with source path not yet done)

**Other kernel APIs** (each needs an emulation stub or implementation):
- `get_compile_time_arg_val(idx)`, `get_arg_val<T>(idx)` -- `get_arg_val` works for Quasar; `get_compile_time_arg_val` needs JIT
- `TensorAccessor`, `TensorAccessorArgs` -- **Not yet implemented**
- `AllocatorBank<DRAM>` -- **Not yet implemented**
- `experimental::Noc` class -- **Not yet implemented** (free-function `noc_async_read/write` exists for WH, but OOP Noc class needed for Quasar)
- `get_my_thread_id()` / `mhartid` -- `__processor_id` thread-local exists; kernel-accessible wrapper not yet exposed
- `DPRINT` -- **Not yet implemented**

**JIT header structure for Quasar build**

The Quasar JIT DFB headers now exist:
- `jit_hw/api/dfb_api.h` -- **Done** (timeout-wrapped DFB ops with `TT_EMULE_DFB_TIMEOUT` env var)
- `jit_hw/emule_dfb_state.h` -- **Done** (thread-local `__emule_dfbs`, `__emule_tc_array`)

### Metal DFB Test JIT Bringup (as of 2026-04-02)

**Two upstream-style DFB tests now pass through the tt-metal emulation path.** These use real `tt-metal` host APIs (`CreateDataflowBuffer`, `BindDataflowBufferToProducerConsumerKernels`, `experimental::quasar::CreateKernel`, `LaunchProgram`) linked against `Metalium::Metal`, with device kernels JIT-compiled by the `emulated_program_runner`.

| Test | Topology | DFBs | Kernels | Status |
|------|----------|------|---------|--------|
| `DFBEmuleDMTest` | 1 DM producer → 1 DM consumer | 1 | `dfb_producer.cpp`, `dfb_consumer.cpp` | **PASS** |
| `DFBEmuleBridgeTest` | 1 DM producer → 1 Compute bridge → 1 DM consumer | 2 | `dfb_producer.cpp`, `dfb_t6.cpp`, `dfb_consumer.cpp` | **PASS** |

**Files created/modified in tt-metal:**

| File | What was done |
|------|---------------|
| `tests/tt_metal/tt_metal/api/dataflow_buffer/test_dfb_emulation.cpp` | **New.** Two GTest cases using real Metal host APIs for DFB configuration, kernel creation, and data verification. |
| `tt_emule/CMakeLists.txt` | **Modified.** Added `test_dfb_emulation` target at Tier 3b. |
| `tt_metal/impl/emulation/emulated_program_runner.cpp` | **Modified.** Added DFB TLS variables (`__emule_dfbs`, `__emule_tc_array`, `__processor_id`), DFB L1 allocation with shared-L1 for connected DFBs, per-thread `EmuleDFBInterface` construction, `mhartid` CSR patching in JIT source, DRAM bank setup. |
| `tt_metal/llrt/tt_cluster.cpp` | **Modified.** Added `ARCH::QUASAR` → `quasar_32_arch.yaml` mapping. |
| `tt_metal/tests/tt_metal/test_utils/env_vars.hpp` | **Modified.** Added `ARCH::QUASAR` string conversion. |
| `tt_metal/third_party/umd/tests/cluster_descriptor_examples/quasar_1chip.yaml` | **New.** Mock cluster descriptor for single-chip Quasar emulation. |
| `tt_metal/soc_descriptors/quasar_32_arch.yaml` | **Modified.** Fixed `dram_views` aliasing (both channels pointed to channel 0). |

**Files created/modified in tt-emule:**

| File | What was done |
|------|---------------|
| `include/jit_hw/experimental/dataflow_buffer.h` | **New.** JIT-compatible `experimental::DataflowBuffer` class wrapping `dfb_api.h` + `noc_traits_t<DataflowBuffer>` specialization. |
| `include/jit_hw/experimental/tensor.h` | **Modified.** Added `noc_traits_t<TensorAccessor>` with `__emule_resolve_noc_addr` for DRAM page resolution. |
| `include/jit_hw/api/tensor/tensor_accessor.h` | **Modified.** Added `InterleavedAddrGen` include for complete type. |
| `include/jit_hw/jit_kernel_stubs.hpp` | **Modified.** Added `emule_dfb_state.h` include and `__processor_id` TLS declaration. |
| `run_regression.sh` | **Modified.** Added Tier 3b DFB Emulation (Quasar) with both test cases. |

**Key debugging findings and fixes:**
- `EmuleDFBInterface` must be per-thread (not shared), matching real HW where each RISC has a separate `LocalDFBInterface`
- DRAM bank aliasing in `quasar_32_arch.yaml` caused data corruption (both `dram_views` mapped to channel 0)
- `__processor_id` must match the kernel's actual RISC ID from `get_kernel_processor_type()`, not the kernel iteration index
- Connected DFBs (same dimensions, same core) share L1 to emulate the HW register file data path through compute bridge kernels

**emulated_program_runner interface construction vs. build_dfb_interfaces():**

The runner uses a simpler `EmuleDFBInterface` setup than the standalone path's `build_dfb_interfaces()`. All threads on a core get `active = true` for all DFBs on that core (`num_tcs_to_rr = 1`, `stride_size = entry_size`, single TC slot). RISC mask enforcement is left to the upstream device kernels (which call `mhartid` to decide whether to participate). `build_dfb_interfaces()` filters by RISC mask at the emulator level and sets `active = false` for non-participants. Both approaches are correct; the runner's simpler approach relies on upstream kernel self-selection.

**Still outstanding**:
- Quasar JIT compilation pipeline (clang-17, correct include paths)
- LLK compute stubs (`llk_io_unpack.h`/`llk_io_pack.h`) wrapping DFB ops for TRISC kernels
