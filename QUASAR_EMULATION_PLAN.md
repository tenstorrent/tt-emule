# Quasar Emulation Plan for tt-emule

This is the top-level plan for adding Quasar (tt-2xx) architecture support to tt-emule. It covers all major subsystems that need changes, links to detailed sub-plans, and tracks exploration/implementation status.

## References

- **tt-metal repo**: `/localdev/arminale/tt-metal-main` (upstream, no tt-emule fork changes)
- **tt-emule repo**: `/localdev/arminale/tt-emule`
- **DFB sub-plan**: [QUASAR_DFB_PLAN.md](QUASAR_DFB_PLAN.md) -- detailed dataflow buffer API analysis and emulation design

---

## Current tt-emule Architecture (Wormhole baseline)

tt-emule is a single-core, multi-threaded, in-process emulator for Tenstorrent Tensix kernels. Key characteristics:

- **Arch**: Wormhole B0 (n150)
- **Grid**: 1x1 (single worker core)
- **L1**: 1 MB per core, `mmap` with `MAP_32BIT`
- **DRAM**: 256 MB, single channel
- **Threads**: one `std::thread` per kernel (DataMovement0, DataMovement1, Compute)
- **Sync**: CBs with SPSC `CBSyncState` (atomic occupied + mutex/CV)
- **Compute**: CPU scalar emulation (32x32 float tiles, 8 DST slots)
- **JIT**: `g++` compile + `dlopen`; `jit_hw/` header tree for metal-compatible API surface
- **NOC**: synchronous `memcpy` (no actual DMA)
- **Dispatch**: always slow dispatch, synchronous `EnqueueProgram`

### Key Files


| File                                     | Role                                                                                   |
| ---------------------------------------- | -------------------------------------------------------------------------------------- |
| `include/tt_emule/device.hpp`            | `Core` (L1 mmap, 32 CBs, DST, bump alloc, TileCounterArray, DFBSyncState[32]) + `Device` (DRAM, NOC resolve, IDevice impl) |
| `include/tt_emule/program.hpp`           | `Program`, `KernelDescriptor` (+processor_id), `KernelType` (DM0/DM1/Compute/QuasarDM/QuasarCompute), `CircularBufferConfig`, `DataflowBufferConfig`, Quasar config structs |
| `src/kernel_runner.cpp`                  | `EnqueueProgram`: instantiate CBs/DFBs, build per-thread DFB interfaces, spawn threads with init barrier, join, cleanup |
| `src/host_api.cpp`                       | Host API wrappers, buffer operations, JIT kernel creation, `CreateDataflowBuffer`, Quasar `CreateKernel` overloads |
| `include/tt_emule/cb_sync_state.hpp`     | `CBSyncState`: the SPSC sync primitive (Wormhole)                                      |
| `include/tt_emule/circular_buffer.hpp`   | `CircularBuffer`: owns storage + `CBSyncState` (Wormhole)                              |
| `include/tt_emule/tile_counter.hpp`      | **New (Quasar):** `TileCounter` (atomic posted/acked + mutex/CVs), `TileCounterArray` (num_neos x 32) |
| `include/tt_emule/dfb_sync_state.hpp`    | **New (Quasar):** `DFBTCSlot`, `EmuleDFBInterface` (per-thread per-DFB), `DFBSyncState` |
| `include/tt_emule/dataflow_buffer.hpp`   | **New (Quasar):** `DataflowBuffer` class: MPMC sync operations + pointer accessors     |
| `include/jit_hw/api/cb_api.h`            | JIT CB API with timeout detection (Wormhole)                                           |
| `include/jit_hw/api/dfb_api.h`           | **New (Quasar):** JIT DFB API with timeout detection                                   |
| `include/jit_hw/emule_dfb_state.h`       | **New (Quasar):** Thread-local `__emule_dfbs`, `__emule_tc_array`                      |
| `include/kernel_api/dataflow_api.hpp`    | Standalone CB + NOC API (Wormhole)                                                     |
| `include/kernel_api/dfb_dataflow_api.hpp`| **New (Quasar):** Standalone DFB operations                                            |
| `include/kernel_api/compute_api.hpp`     | Tile math: add, matmul, copy, pack                                                     |
| `include/tt_emule/dst_register_file.hpp` | 8 DST slots with state machine                                                         |
| `src/jit_kernel.cpp`                     | JIT compilation (g++ for WH; clang-17 for Quasar pending)                              |
| `CMakeLists.txt`                         | Build: tt_emule_lib static + tests (including dfb_passthrough)                         |


---

## Quasar vs Wormhole: What Changes

### 1. Core/Thread Model


| Aspect            | Wormhole              | Quasar                                                     |
| ----------------- | --------------------- | ---------------------------------------------------------- |
| DM cores          | 2 (BRISC, NCRISC)     | 8 (DM0-DM7)                                                |
| Compute cores     | 3 TRISCs (TRISC0/1/2) | 4 Neos x 4 TRISCs = 16 compute slots                       |
| Total per Tensix  | 5                     | 24 (8 DM + 16 compute)                                     |
| Firmware binaries | 2 DM + 3 TRISC        | 1 DM (shared) + 4 TRISC (one per role, reused across Neos) |


**Quasar TRISC roles**: TRISC0 = unpacker, TRISC1 = math, TRISC2 = packer, TRISC3 = alternate consumer. TRISC1 never interacts with DFBs.

**Emulation impact**: `KernelType` enum needs DM0-DM7 + per-Neo compute. Thread pool must support up to 24 threads per Tensix. Thread-local state (`__core`, `__device`, `__emule_cbs` / `__emule_dfbs`) must be per-RISC.

**tt-metal references**:

- `tt_metal/hw/inc/internal/tt-2xx/quasar/core_config.h` -- `TensixProcessorTypes` enum (24 entries)
- `tt_metal/llrt/hal/tt-2xx/quasar/qa_hal_tensix.cpp` -- processor class definitions, Neo naming
- `tt_metal/api/tt-metalium/experimental/host_api.hpp` -- `QUASAR_NUM_DM_CORES_PER_CLUSTER = 8`

### 2. Memory Map


| Aspect           | Wormhole             | Quasar                                                                 |
| ---------------- | -------------------- | ---------------------------------------------------------------------- |
| L1 size          | 1 MB                 | 4 MB (`MEM_L1_SIZE`)                                                   |
| L1 sharing       | Per-core (private)   | Shared across all 24 processors (8 DM + 4 Neo x 4 TRISC) in one Tensix |
| DRAM channels    | 1 (in emule)         | 8 real; emule may keep simplified                                      |
| DRAM per channel | 256 MB (emule)       | 1-4 GB (SoC YAML dependent)                                            |
| Reserved L1      | Small (FW + mailbox) | Large (8x DM FW, 4x TRISC FW, globals, locals, routing tables, fabric) |
| NOC topology     | Torus (2 NOCs)       | Mesh (1 NOC)                                                           |


**Emulation impact**: `Core::L1_SIZE` must become 4 MB (or configurable). `MAP_32BIT` constraint must be verified for larger L1. DRAM model may stay simplified initially.

**tt-metal references**:

- `tt_metal/hw/inc/internal/tt-2xx/quasar/dev_mem_map.h` -- full memory map constants
- `tt_metal/third_party/tt_llk/tt_llk_quasar/llk_lib/llk_memory_checks.h` -- L1 layout narrative

### 3. Synchronization: CBs -> DFBs

This is the largest architectural change. See [QUASAR_DFB_PLAN.md](QUASAR_DFB_PLAN.md) for the full analysis.

**Summary**: CBs use a single atomic `occupied` for SPSC sync. DFBs use per-tile-counter `posted`/`acked` atomics with round-robin distribution across up to 4 TC slots per RISC, supporting MPMC with STRIDED or BLOCKED access patterns.

**Emulation impact**: New `TileCounter` and `TileCounterArray` structs. New `DFBSyncState` and `EmuleDFBInterface` per-thread views. New device-side and JIT API headers. Program model must support DFB configs alongside (or instead of) CB configs.

### 4. Device Grid


| Aspect       | Wormhole (emule) | Quasar (real)         |
| ------------ | ---------------- | --------------------- |
| Tensix grid  | 1x1              | 8x4 = 32 Tensix cores |
| Worker cores | 1                | 32                    |
| DRAM grid    | N/A              | 8x3 = 24 DRAM banks   |


**Emulation impact**: For Phase 1, a single-Tensix emulation (1x1 grid) is sufficient -- the interesting work is inside one Tensix with 8 DM + 4 Neo threads. Multi-Tensix grid can be added later.

**tt-metal references**:

- `tt_metal/third_party/umd/device/api/umd/device/arch/grendel_implementation.hpp` -- grid constants
- `tt_metal/soc_descriptors/quasar_32_arch.yaml` -- SoC descriptor

### 5. Compute / LLK


| Aspect         | Wormhole                                          | Quasar                                       |
| -------------- | ------------------------------------------------- | -------------------------------------------- |
| LLK surface    | Large (all ops)                                   | Subset (matmul, unpack A/AB, pack, datacopy) |
| Math template  | Parameterized by operand IDs, transpose, throttle | Simpler: `MathFidelity` template only        |
| DST alignment  | Standard                                          | 32768-byte alignment                         |
| RISC-V atomics | Not supported                                     | Zaamo + Zalrsc (64-bit atomics, CAS)         |


**Emulation impact**: Current scalar compute emulation (`compute_api.hpp`, `tile.hpp`) may work largely as-is for initial Quasar support. LLK differences are handled by the `jit_hw/` header layer. Atomics support is new.

**tt-metal references**:

- `tt_metal/hw/ckernels/quasar/metal/` -- Quasar-specific ckernels
- `tests/tt_metal/tt_metal/test_riscv_atomics.cpp` -- atomic tests

### 6. NOC


| Aspect              | Wormhole (emule) | Quasar (real)      |
| ------------------- | ---------------- | ------------------ |
| NOC count           | 2                | 1                  |
| Command submission  | MMIO (V1)        | RoCC overlay (V2)  |
| Address translation | None             | Hardware AT tables |
| Topology            | Torus            | Mesh               |


**Emulation impact**: NOC remains synchronous memcpy. The NOC API differences (coordinate encoding, command buffer model) affect the `jit_hw/` header layer and `noc_resolve` address mapping, but not the fundamental emulation approach.

**tt-metal references**:

- `tt_metal/hw/inc/internal/tt-2xx/quasar/noc/` -- NOC register definitions
- `tt_metal/hw/inc/internal/tt-2xx/quasar/noc_nonblocking_api.h` -- V2 (overlay) API

### 7. HAL / Capabilities

Key Quasar HAL flags (from `qa_hal.cpp` / `qa_hal_tensix.cpp`):

- `has_tile_counter_registers_ = true` (WH/BH: false)
- `supports_implicit_dfb_sync_ = true` (WH/BH: false)
- `supports_cbs = true`, `supports_dfbs = true` (WH/BH: dfbs false)
- `noc_topology_ = MESH` (WH/BH: TORUS)
- `has_stream_registers_ = false` (WH/BH: true)
- `max_processors_per_core_ = 24` (WH/BH: 5)
- `firmware_is_kernel_object() = true` (WH/BH: false)

### 8. Kernel Creation and Dispatch

Quasar uses `experimental::quasar::CreateKernel` (in `/localdev/arminale/tt-metal-main/tt_metal/api/tt-metalium/experimental/host_api.hpp`) with dedicated config structs:

**`QuasarDataMovementConfig`** (full struct from `experimental/host_api.hpp`):

```cpp
struct QuasarDataMovementConfig {
    uint32_t num_threads_per_cluster = QUASAR_NUM_DM_CORES_PER_CLUSTER;  // 1-8, default 8
    std::vector<uint32_t> compile_args;
    std::map<std::string, std::string> defines;
    std::unordered_map<std::string, uint32_t> named_compile_args;
    bool is_legacy_kernel = false;        // true = duplicate binary per DM (WH/BH porting aid)
    KernelBuildOptLevel opt_level = KernelBuildOptLevel::O2;
};
```

The runtime reserves the lowest-numbered available DM processors (DM0-DM7) consistent across all clusters. Non-legacy: single shared binary for all threads. Legacy (`is_legacy_kernel = true`): separate binary per DM, with globals treated as thread-local.

**`QuasarComputeConfig`** (full struct):

```cpp
struct QuasarComputeConfig {
    uint32_t num_threads_per_cluster = QUASAR_NUM_TENSIX_ENGINES_PER_CLUSTER;  // 1-4, default 4
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    std::vector<UnpackToDestMode> unpack_to_dest_mode;
    bool bfp8_pack_precise = false;
    bool math_approx_mode = false;
    std::vector<uint32_t> compile_args;
    std::map<std::string, std::string> defines;
    std::unordered_map<std::string, uint32_t> named_compile_args;
    KernelBuildOptLevel opt_level = KernelBuildOptLevel::O3;
};
```

Each Neo has 4 TRISCs (MATH0-MATH3), so N Neos = 4N compute processor slots. Max one compute kernel per cluster. The `compile_args`, `defines`, and `named_compile_args` must be propagated to JIT compilation.

**Dispatch**: `LaunchProgram` is arch-agnostic. Quasar-specific behavior is in kernel group configuration: each DM/Neo slot gets `num_sw_threads` and `kernel_thread_id` metadata. The DM slots are indices 0-7; Neo slots are at offset `QUASAR_NUM_DM_CORES_PER_CLUSTER + neo_id`.

**Emulation impact**: `kernel_runner.cpp` needs to understand `QuasarDataMovementConfig` / `QuasarComputeConfig`, spawn the right number of DM and compute threads, and assign correct thread IDs. The processor reservation logic (lowest available DMs, consistent across clusters) should be replicated.

### 9. Semaphores

Quasar semaphores are **L1-resident atomic counters** with NOC-based remote operations. They are **separate from DFB tile counters** and used for inter-kernel/inter-stage ordering.

- **DM**: `experimental::Semaphore(id)` -- `up`/`down`/`wait`/`set`, plus remote `up(noc, noc_x, noc_y, count)` via `noc_semaphore_inc`
- **Compute**: `ckernel::Semaphore(id)` -- same local ops, no NOC (Quasar-only header)
- **Host**: `CreateSemaphore(program, core_or_range, initial_value)` allocates L1 slots
- **Cross-core**: `INCREMENT_REMOTE_SEM` / `WAIT_FOR_REMOTE_SEM` patterns in pipeline tests

**Emulation impact**: Not needed for Phase 1 DFB workloads (none of them use semaphores). Required for pipeline tests (`test_quasar_semaphores.cpp`) and cross-core communication. Implementation: L1-backed atomic counters with condition variables for wait, remote increment via NOC resolve to target core's L1.

### 10. Multi-Tensix Communication

**All current DFB tests are single-Tensix** (single `CoreCoord(0,0)`). Cross-core patterns exist only in:
- `test_quasar_semaphores.cpp` (`QuasarMultipleClustersMultiSemaphorePipeline`): two cores with remote semaphore signaling
- `test_multi_dm_add_two_ints.cpp`: multi-core placement, no peer L1 traffic
- `data_movement/one_from_one`: cross-core L1 read/write via NOC coordinates

**Emulation impact**: Does not block Phase 1. Multi-Tensix requires: logical-to-physical NOC coordinate map, redirect NOC read/write to target core's L1 when `(noc_x, noc_y)` differs from local core.

### 11. TensorAccessor API

`TensorAccessor` is a page-oriented DRAM/L1 address generator used by DFB dataflow kernels.

- **Device side** (`tt_metal/hw/inc/api/tensor/tensor_accessor.h`): takes `bank_base_address` + `aligned_page_size` + distribution spec; `get_noc_addr(page_id, offset, noc_id)` resolves to 64-bit NOC address
- **Compile-time args** (`TensorAccessorArgs<OFFSET>`): reads config bitmask + page size from kernel compile args; chained via `next_compile_time_args_offset()`
- **NOC integration**: `noc_traits_t<TensorAccessor>` maps `{page_id, offset_bytes}` to NOC addresses for `Noc::async_read/write`
- **Host side**: `TensorAccessorArgs(buffer).append_to(compile_args)` pushes config + page size

**Emulation impact**: Need a stub that maps `(page_id, bank_base, page_size)` to emulated DRAM addresses. For single-bank interleaved buffers (which all current DFB tests use), this simplifies to `bank_base + page_id * page_size`. The `experimental::Noc` class also needs stubs for `async_read`/`async_write`/barriers that perform memcpy between DRAM and DFB L1 backing memory.

---

## Target Test Workloads

These tt-metal tests define the emulation surface. All use DFBs on Quasar with explicit and/or implicit sync. See [QUASAR_DFB_PLAN.md](QUASAR_DFB_PLAN.md) Section 11 for detailed kernel-by-kernel analysis.

### End-to-End Tests

| Test | Workload | DFBs | Threads | Sync | Key APIs |
|------|----------|------|---------|------|----------|
| `test_bmm.cpp` | Batched matmul A*B=C | 3 (src0, src1, dst) | 1 DM reader + 1 DM writer + 1 compute | Explicit | `matmul_tiles`, `pack_tile`, `Noc`, `TensorAccessor` |
| `test_matmul_X_tile.cpp` | Block matmul | 3 | 1+1+1 | Explicit | `matmul_block`, `pack_tile`, `AllocatorBank<DRAM>` |
| `test_direct.cpp` | DRAM datacopy round-trip | 2 (input, output) | 1-4 DM + 1-4 compute | Implicit | `copy_tile`, `pack_tile`, `read_in`/`write_out` |
| `test_dataflow_buffer.cpp` | Synthetic DFB unit tests | 1-2 | 1-4 DM + 0-1 compute | Both | `finish`, STRIDED + BLOCKED, 1/2/4 fan-in/out |

### Recommended Emulation Order

1. **dfb_t6** (synthetic passthrough): simplest -- no compute math, just DFB sync through 2 buffers
2. **eltwise_copy** (datacopy): adds `copy_tile` + `pack_tile` compute
3. **bmm** (matmul): adds `matmul_tiles` + multi-DFB coordination
4. **matmul_block** (block matmul): adds block-level reserve/push/wait/pop counts
5. **Multi-producer/consumer**: 2-4 DM threads with STRIDED access
6. **Implicit sync**: `read_in`/`write_out` path (deferred)

---

## Exploration and Implementation Status


| Area                                      | Status               | Notes                                                            |
| ----------------------------------------- | -------------------- | ---------------------------------------------------------------- |
| DFB API + sync model                      | **Implemented**      | Full Phase 1 (explicit sync, STRIDED) implemented and tested. See [QUASAR_DFB_PLAN.md](QUASAR_DFB_PLAN.md) |
| Core/thread model (8 DM + 4 Neo)          | **Implemented**      | `KernelType::QuasarDM`/`QuasarCompute`, `processor_id` on KernelDescriptor, `QuasarDataMovementConfig`/`QuasarComputeConfig`, `std::barrier` init sync |
| Memory map (4 MB L1, reserved regions)    | **Partially done**   | `Core::l1_size_` is runtime-configurable; default `L1_SIZE` constexpr still 1 MB. Needs Quasar Device constructor or `#ifdef` |
| Device grid (8x4 Tensix, 8 DRAM)          | Explored             | grendel_implementation.hpp, SoC YAML. Not needed for Phase 1 (single-Tensix) |
| NOC differences (mesh, 1 NOC, V2/overlay) | Explored             | noc_nonblocking_api, NOC register headers. Free-function NOC works; OOP `Noc` class needed |
| Compute / LLK differences                 | Explored             | Subset API, simpler templates. LLK DFB stubs not yet implemented |
| Kernel dispatch (24 processors)           | **Implemented**      | kernel_runner spawns per `KernelDescriptor` with `processor_id`, `std::barrier` |
| HAL capabilities                          | Explored             | qa_hal.cpp, qa_hal_tensix.cpp                                    |
| RISC-V atomics (Zalrsc)                   | Noted                | Quasar-only; test_riscv_atomics.cpp                              |
| Quasar kernel APIs                        | **Implemented**      | Config structs done; function-pointer `CreateKernel` overloads done; JIT path pending |
| Quasar program dispatch model             | **Implemented**      | `EnqueueProgram` handles DFBs: allocates L1, builds per-thread interfaces, init barrier, teardown |
| DFB test kernels + workloads              | **Substantially done** | Standalone `dfb_passthrough` + 2 upstream Metal DFB tests pass (DM-only + Compute bridge via `emulated_program_runner` JIT). BMM/matmul/datacopy tests remain (need compute LLK stubs). |
| Quasar semaphore model                    | Explored             | L1-resident counters; not used by DFB workloads; deferred |
| Multi-Tensix NOC communication            | Explored             | All DFB tests are single-Tensix; does not block Phase 1 |
| Tensor/TensorAccessor API                 | **Implemented**      | JIT stub for `TensorAccessor` + `noc_traits_t<TensorAccessor>` with `InterleavedAddrGen` DRAM banking. Used by Metal DFB tests. |
| Global Semaphores                         | Deferred             | Not needed for Phase 1 DFB workloads |


---

## Implementation Phases

### Phase 0: Exploration (complete)

- Understand all Quasar-specific APIs and their emulation requirements (done)
- Review DFB plan (done -- see [QUASAR_DFB_PLAN.md](QUASAR_DFB_PLAN.md))
- All areas explored except Global Semaphores (deferred -- not needed for Phase 1)

### Phase 1 Progress Summary (as of 2026-04-02)

**Infrastructure steps 1-8: DONE.** All core DFB emulation structures, sync operations, API paths (standalone + JIT), kernel runner integration, and program model updates are implemented and working.

**Step 9 (tests): SUBSTANTIALLY DONE.** `dfb_passthrough` standalone test passes (1 DM producer + 1 DM consumer, STRIDED 1:1). Two upstream-style Metal DFB tests now pass through the full tt-metal emulation path (`emulated_program_runner` JIT pipeline):
- `DFBEmuleDMTest` — DM producer + DM consumer, 1 DFB, STRIDED, explicit sync
- `DFBEmuleBridgeTest` — DM producer + Compute bridge (dfb_t6) + DM consumer, 2 DFBs

These tests use real `tt-metal` host APIs (`CreateDataflowBuffer`, `BindDataflowBufferToProducerConsumerKernels`, `experimental::quasar::CreateKernel`, `LaunchProgram`) and JIT-compile upstream device kernels (`dfb_producer.cpp`, `dfb_consumer.cpp`, `dfb_t6.cpp`). See [QUASAR_DFB_PLAN.md](QUASAR_DFB_PLAN.md) for full details.

### Phase 1: Single-Tensix DFB Emulation

Ordered by dependency -- each step builds on the previous.

**Step 1: Core/Thread Model** -- **DONE.** Extended `KernelType` with `QuasarDM`/`QuasarCompute` in `program.hpp`. Added `QuasarDataMovementConfig`/`QuasarComputeConfig` structs. Added `processor_id` (0-23) to `KernelDescriptor`. `Core` L1 is runtime-configurable via existing role-aware constructor (default remains 1 MB; `L1_SIZE` constexpr not yet changed to 4 MB -- needs `#ifdef ARCH_QUASAR` or separate Quasar Device constructor).

**Step 2: Tile Counter Infrastructure** -- **DONE.** New `include/tt_emule/tile_counter.hpp` with `TileCounter` (atomic posted/acked, capacity, mutex, 2 CVs) and `TileCounterArray` (flat `unique_ptr<TileCounter[]>`, parameterized by num_neos, 32 TCs per neo). Added `TileCounterArray` to `Core` (lazy init via `init_tile_counters(num_neos)`).

**Step 3: DFB Sync State** -- **DONE.** New `include/tt_emule/dfb_sync_state.hpp` with `DFBTCSlot`, `EmuleDFBInterface` (per-thread per-DFB view: 4 TC slots, round-robin, entry/stride, broadcast/active flags), and `DFBSyncState` (buffer geometry). `TileCounterArray*` pointer lives on `Core` rather than per-`DFBSyncState`.

**Step 4: DFB Device API** -- **DONE.** New `include/tt_emule/dataflow_buffer.hpp` with `DataflowBuffer` class implementing `reserve_back` (strided + broadcast), `push_back`, `wait_front`, `pop_front`, `finish` (drain barrier), `get_write_ptr`, `get_read_ptr`, `get_entry_size`, `get_stride_size`, `get_id`.

**Step 5: JIT DFB API** -- **DONE.** New `include/jit_hw/emule_dfb_state.h` (thread-local `__emule_dfbs`, `__emule_tc_array`) and `include/jit_hw/api/dfb_api.h` (timeout-wrapped ops with `TT_EMULE_DFB_TIMEOUT` env var, default 120s).

**Step 6: Kernel Runner Update** -- **DONE.** Updated `src/kernel_runner.cpp` with: `build_dfb_interfaces()` (builds per-thread EmuleDFBInterface arrays from DataflowBufferConfig + RISC masks), `std::barrier` init sync, per-thread DFB thread-locals (`__dfb_ifaces`, `__emule_dfbs`, `__emule_tc_array`), per-thread `__processor_id`, DFB teardown via `core.reset_dfb_sync()`.

**Step 7: Program Model** -- **DONE.** Added `DataflowBufferConfig`, `DFBHandle`, `AccessPattern` enum, `Program::add_dfb()` to `program.hpp`. Added `CreateDataflowBuffer()` and Quasar `CreateKernel()` overloads (for function-pointer kernels) in `host_api.cpp`/`host_api.hpp`. Note: `BindDataflowBufferToProducerConsumerKernels` not yet implemented (RISC masks are set manually in config).

**Step 8: Standalone API** -- **DONE.** New `include/kernel_api/dfb_dataflow_api.hpp` for standalone DFB operations via `__core->tile_counters()` and `__dfb_ifaces` thread-local.

**Step 9: Regression Tests** -- **PARTIALLY DONE.** Added `tests/dfb_passthrough/` (1 DM producer + 1 DM consumer, 8x1KB entries, STRIDED 1:1, validates full sync cycle). Extended `run_regression.sh` with Tier 0 standalone section. All 4 standalone tests pass (eltwise_add, matmul, tilize, dfb_passthrough). Remaining from plan: eltwise_copy (needs compute ops), bmm (needs matmul + multi-DFB), multi-producer/consumer tests.

### Phase 1 Remaining Work (to run upstream metal DFB tests)

The core DFB sync infrastructure is complete. The following gaps must be closed to run the actual tt-metal DFB tests listed in Section "Target Test Workloads" above:

**Host API gaps (resolved):**
- ~~`BindDataflowBufferToProducerConsumerKernels`~~ -- **Done.** Used by Metal DFB tests via real tt-metal API.
- ~~Quasar JIT `CreateKernel` with source file path~~ -- **Done.** `emulated_program_runner` handles JIT compilation with g++.

**Device API gaps (resolved):**
- ~~`#include <experimental/dataflow_buffer.h>` shim header~~ -- **Done.** `jit_hw/experimental/dataflow_buffer.h` wraps `dfb_api.h`.
- ~~`experimental::Noc` class~~ -- **Done.** JIT stubs for `Noc::async_read/write/barriers` with memcpy semantics.
- ~~`TensorAccessor` / `TensorAccessorArgs`~~ -- **Done.** JIT stub with `InterleavedAddrGen` DRAM banking.
- ~~`get_my_thread_id()` / `mhartid`~~ -- **Done.** `__processor_id` TLS + JIT `mhartid` CSR patching.
- ~~`DPRINT`~~ -- **Done.** Forward to host `fprintf(stderr)`.

**Device API gaps (remaining):**
- `noc_async_read_tile()` / `noc_async_write_tile()` -- tile-indexed NOC (needed by BMM tests)
- `AllocatorBank<DRAM>` -- bank-interleaved DRAM addressing (needed by matmul tests)
- `read_in()` / `write_out()` -- implicit sync (Phase 2, needed by `test_direct.cpp`)

**Compute API gaps (remaining, for bmm/matmul/datacopy tests):**
- `mm_init`, `matmul_tiles`, `mm_block_init`, `matmul_block`
- `unary_op_init_common`, `copy_tile`
- `pack_tile`, `acquire_dst`, `release_dst`
- LLK stubs: `llk_wait_tiles`, `llk_pop_tiles`, `llk_wait_for_free_tiles`, `llk_push_tiles` (TRISC DFB wrappers)

**JIT / build infrastructure (remaining):**
- Quasar JIT compilation pipeline switch to clang-17 (currently uses g++ which works)
- `L1_SIZE` = 4 MB for Quasar builds (currently still 1 MB default)

**Recommended priority order (updated):**
1. ~~`get_my_thread_id()` / `mhartid`~~ -- **Done**
2. ~~`<experimental/dataflow_buffer.h>` shim header~~ -- **Done**
3. ~~`TensorAccessor` stub~~ -- **Done**
4. ~~`experimental::Noc` class~~ -- **Done**
5. L1_SIZE = 4 MB for Quasar Device
6. LLK DFB stubs for TRISC
7. Compute ops (`mm_init`, `matmul_tiles`, `pack_tile`, etc.)
8. `noc_async_read_tile()` / `noc_async_write_tile()`
9. Implicit sync (`read_in`/`write_out`)
10. Quasar JIT pipeline switch to clang-17

### Phase 2: Extended DFB Features

- BLOCKED consumer access pattern
- Implicit sync (`read_in`/`write_out`) -- simplified to synchronous memcpy + immediate counter update
- Intra-Tensix DFBs

### Phase 3: Device Model Updates

- Quasar build target: separate binary with compile-time `#ifdef ARCH_QUASAR` (resolved: not a runtime flag)
- Grid model (even if still emulating single Tensix)
- Multi-channel DRAM
- NOC address mapping updates
- JIT compiler alignment (clang-17 per workspace rules)
- Global Semaphores: emulate Quasar global semaphore primitive once specs are available (currently host-side sharded L1 buffer; hardware variant TBD)

### Phase 4: Compute / LLK Alignment

- Quasar LLK API differences in `jit_hw/` headers
- DST alignment changes
- RISC-V atomic emulation (if needed for kernel compat)

### Phase 5: Multi-Tensix (future)

- Multi-core grid emulation
- Cross-core NOC communication
- Full SoC descriptor integration

---

## Important Constraints

### Current tests are limited -- emulation should target full device

Current Quasar tt-metal tests only exercise a subset of Neo clusters (typically `num_threads_per_cluster = 1`, occasionally 4) because Quasar is still in hardware bring-up. These tests are useful for initial validation but **do not represent production workloads**.

tt-emule should target **full device utilization** (all Neo clusters, all DMs) as soon as basic emulation accuracy is verified. The emulator's value is precisely in testing configurations that hardware bring-up hasn't reached yet. This means:
- Don't hard-code thread counts or cluster counts to match current test limitations
- Design for the general case (8 DMs, 4 Neos) from the start, even if initial tests only exercise a subset
- Create emulation-specific tests that push beyond current tt-metal test coverage once accuracy is confirmed

### Quasar configurations are data-driven

There will be **many different Quasar configurations**, primarily varying in the **number of clusters** (Neo groups). The emulator must not hard-code cluster count. Instead:
- Cluster count, DM count, and L1 size should come from a **configuration descriptor** (SoC YAML, config struct, or similar data-driven mechanism)
- The `Core` / `Device` model should be parameterized: `num_neos`, `num_dms`, `l1_size`, `num_tile_counters_per_neo`, etc.
- Tile counter array dimensions should derive from the configuration (e.g., `num_neos * 32` total TCs)
- Test fixtures should be parameterizable by configuration so the same test suite runs across all Quasar variants
- Consider loading Quasar SoC YAML descriptors from tt-metal (e.g., `quasar_32_arch.yaml`) to stay in sync with upstream configuration changes

---

## Key Design Decisions (all resolved)

1. **Arch selection**: **Resolved -- Quasar is a separate build.** The emulator binary is compiled for one arch at a time. Arch-specific code paths (DFB vs CB, thread model, L1 size, etc.) use compile-time `#ifdef` rather than runtime dispatch.
2. **CB coexistence**: **Resolved -- DFBs only for Quasar build.** Quasar HAL says `supports_cbs = true` AND `supports_dfbs = true`, but programs cannot mix them (DFB plan Section 8). Since Quasar is a separate build, the Quasar emulator only needs DFB support. CBs remain in the Wormhole build.
3. **Thread count**: **Resolved -- driven by kernel config.** `QuasarDataMovementConfig::num_threads_per_cluster` (1-8) and `QuasarComputeConfig::num_threads_per_cluster` (1-4 Neos, each with 4 TRISCs) determine how many threads to spawn per Tensix. The emulator reads these from the program's kernel descriptors.
4. **L1 addressing**: **Resolved -- no issue.** `MAP_32BIT` gives 2 GB address space; 4 MB L1 fits trivially.
5. **JIT toolchain**: **Resolved -- clang-17 for Quasar build per workspace rules; g++ remains for Wormhole build.** The JIT path in `jit_kernel.cpp` currently hardcodes `g++`; the Quasar build will switch to `clang-17`. This is a Phase 1 change since JIT-compiled Quasar kernels need the correct toolchain from the start.
6. **Configuration format**: **Resolved -- C++ config struct parameterized at build time.** A `QuasarConfig` struct with `num_neos`, `num_dms`, `l1_size`, `num_tile_counters_per_neo` fields, initialized from compile-time defaults with optional runtime override. SoC YAML loading deferred to Phase 3.
7. **JIT header tree structure**: **Resolved -- parallel `jit_hw/` headers with `#ifdef ARCH_QUASAR` gating in shared headers.** Quasar build gets new DFB-specific headers (`jit_hw/api/dfb_api.h`, `jit_hw/emule_dfb_state.h`) paralleling the existing CB headers. Shared headers (e.g. compute) use `#ifdef ARCH_QUASAR` to select DFB vs CB paths. This matches the existing `jit_hw/api/cb_api.h` pattern.

