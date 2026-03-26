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

- **24 logical processors per Tensix**: 8 Data Movement (DM) + 4 Neo groups x 4 TRISCs (TRISC0-3) = 16 compute slots. HAL: `max_processors_per_core_ = 24`
- **4 MB shared L1**: all 12 cores access the same L1
- **DMs are fully shared**: any of the 8 DMs can interact with any Neo's tile counters (RISC mask determines participation)
- **TRISC roles**: TRISC0 = unpacker/consumer, TRISC1 = math-only (no DFB), TRISC2 = packer/producer, TRISC3 = alternate consumer
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
| `tt_metal/impl/dataflow_buffer/dataflow_buffer_impl.hpp` | `DataflowBufferImpl`, `TileCounterAllocator`, `RemapperIndexAllocator`, `TxnIdAllocator`, `ClientTypeAllocator` |
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
| `tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6.cpp` | Compute kernel using DFBs |
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

    // Implicit sync (DM only, deferred for emulation)
    void read_in(noc, src, args);
    void write_out(noc, dst, args);

    // Pointers
    uint32_t get_write_ptr() const;
    uint32_t get_read_ptr() const;
    uint32_t get_entry_size() const;
    uint32_t get_stride_size() const;
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

**Scope**: Explicit `reserve_back/push_back/wait_front/pop_front` with configurable cluster count. Validate with 2 DM producers + 4 Tensix consumers initially, then scale to full device (8 DMs, all Neos).

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

### Phase 2: BLOCKED Mode + Implicit Sync

- BLOCKED: producer posts to ALL consumer TCs; straightforward once per-TC counters work
- Implicit sync: since NOC ops are synchronous memcpy, `read_in`/`write_out` become memcpy + immediate `inc_posted`/`inc_acked`
- Intra-Tensix DFBs: degenerate case with `num_tcs_to_rr = 1`

### Integration Points in tt-emule

| Component | Change needed |
|-----------|---------------|
| `device.hpp` / `Core` | Add `TileCounterArray`, DFB storage, init/reset methods |
| `program.hpp` | Add `DataflowBufferConfig`, `add_dfb()` alongside `add_cb()` |
| `kernel_runner.cpp` | Instantiate DFBs, set up per-thread `EmuleDFBInterface`, init barrier |
| New: `dfb_sync_state.hpp` | `TileCounter`, `TileCounterArray`, `DFBSyncState` |
| New: `dataflow_buffer.hpp` | `DataflowBuffer` class (emulation version) |
| New: `jit_hw/api/dfb_api.h` | JIT-facing DFB operations (parallel to `cb_api.h`) |
| New: `kernel_api/dfb_dataflow_api.hpp` | Standalone DFB operations (parallel to `dataflow_api.hpp`) |

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
- `reserve_back(n)`, `push_back(n)`, `wait_front(n)`, `pop_front(n)`
- `finish()`
- `get_write_ptr()`, `get_read_ptr()`, `get_entry_size()`, `get_stride_size()`, `get_id()`

**Compute operations** (from kernel analysis):
- `mm_init(in0, in1, out)`, `matmul_tiles(in0, in1, tile_a, tile_b, dst_idx)`
- `mm_block_init(in0, in1, out)`, `matmul_block(in0, in1, ...block dims...)`
- `unary_op_init_common(in, out)`, `copy_tile(cb_id, tile_idx, dst_idx)`
- `pack_tile(dst_idx, out_cb_id)`, `acquire_dst()`, `release_dst()`

**NOC operations** (from dataflow kernels):
- `noc.async_read(src, dfb, size, src_args, dst_args)` -- with DFB as endpoint
- `noc.async_write(dfb, dst, size, src_args, dst_args)` -- with DFB as endpoint
- `noc.async_read_barrier()`, `noc.async_write_barrier()`
- Legacy: `noc_async_read_tile(tile_id, tensor, addr)`, `noc_async_write_tile(tile_id, tensor, addr)`

**Host operations**:
- `CreateDataflowBuffer(program, core, config)` -> logical DFB id
- `BindDataflowBufferToProducerConsumerKernels(program, dfb_id, producer, consumer)`
- `experimental::quasar::CreateKernel(program, path, core, config)` with `QuasarDataMovementConfig` / `QuasarComputeConfig`

**Other kernel APIs** (each needs an emulation stub or implementation):
- `get_compile_time_arg_val(idx)`, `get_arg_val<T>(idx)` -- already supported in tt-emule for WH; extend for Quasar
- `TensorAccessor`, `TensorAccessorArgs` -- page-oriented DRAM address generator; see emulation plan Section 11
- `AllocatorBank<DRAM>` -- used in `reader_matmul_with_bias_blocked` and `direct_reader/writer_unary`; wraps bank-interleaved DRAM addressing. Needs stub that maps to emulated DRAM
- `experimental::Noc` class -- `async_read`/`async_write`/barriers; emulate as synchronous memcpy
- `get_my_thread_id()` / `mhartid` -- returns per-thread processor ID; used in implicit sync kernels to select DFB config. Emulation: return thread-local processor index matching `TensixProcessorTypes` enum
- `DPRINT` -- debug printing from device kernels; forward to host `printf`/`stdout`

**Outstanding: JIT header structure for Quasar build**

The Quasar build needs a parallel `jit_hw/` header tree (or `#ifdef ARCH_QUASAR` branches) to expose DFB APIs instead of CB APIs. Key new headers:
- `jit_hw/api/dfb_api.h` (parallel to `cb_api.h`)
- `jit_hw/emule_dfb_state.h` (parallel to `emule_cb_state.h`)
- Compute headers referencing DFB IDs instead of CB IDs in `llk_io_unpack.h`/`llk_io_pack.h` stubs

The exact structure depends on whether Quasar kernels use `#include <experimental/dataflow_buffer.h>` (the upstream header) or a tt-emule-specific header. This is an open design question.
