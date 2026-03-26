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
| `include/tt_emule/device.hpp`            | `Core` (1 MB L1, 32 CBs, DST, bump alloc) + `Device` (DRAM, NOC resolve, IDevice impl) |
| `include/tt_emule/program.hpp`           | `Program`, `KernelDescriptor`, `KernelType` (DM0/DM1/Compute), `CircularBufferConfig`  |
| `src/kernel_runner.cpp`                  | `EnqueueProgram`: instantiate CBs, spawn threads, join, cleanup                        |
| `src/host_api.cpp`                       | Host API wrappers, buffer operations, JIT kernel creation                              |
| `include/tt_emule/cb_sync_state.hpp`     | `CBSyncState`: the SPSC sync primitive                                                 |
| `include/tt_emule/circular_buffer.hpp`   | `CircularBuffer`: owns storage + `CBSyncState`                                         |
| `include/jit_hw/api/cb_api.h`            | JIT CB API with timeout detection                                                      |
| `include/kernel_api/dataflow_api.hpp`    | Standalone CB + NOC API                                                                |
| `include/kernel_api/compute_api.hpp`     | Tile math: add, matmul, copy, pack                                                     |
| `include/tt_emule/dst_register_file.hpp` | 8 DST slots with state machine                                                         |
| `src/jit_kernel.cpp`                     | JIT compilation (g++ wrapper)                                                          |
| `CMakeLists.txt`                         | Build: tt_emule_lib static + tests                                                     |


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

**`QuasarDataMovementConfig`**: `num_threads_per_cluster` (1-8, default 8) determines how many DMs run. The runtime reserves the lowest-numbered available DM processors (DM0-DM7) consistent across all clusters. Non-legacy: single shared binary for all threads. Legacy (`is_legacy_kernel = true`): separate binary per DM.

**`QuasarComputeConfig`**: `num_threads_per_cluster` (1-4, default 4) = number of Neos. Each Neo has 4 TRISCs (TRISC0-3), so N Neos = 4N compute processor slots. Max one compute kernel per cluster. Includes `math_fidelity`, `fp32_dest_acc_en`, and other compute settings.

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

## Exploration Status


| Area                                      | Status               | Notes                                                            |
| ----------------------------------------- | -------------------- | ---------------------------------------------------------------- |
| DFB API + sync model                      | Explored             | Full analysis in [QUASAR_DFB_PLAN.md](QUASAR_DFB_PLAN.md)        |
| Core/thread model (8 DM + 4 Neo)          | Explored             | HAL, core_config, dev_mem_map analyzed                           |
| Memory map (4 MB L1, reserved regions)    | Explored             | dev_mem_map.h, llk_memory_checks.h                               |
| Device grid (8x4 Tensix, 8 DRAM)          | Explored             | grendel_implementation.hpp, SoC YAML                             |
| NOC differences (mesh, 1 NOC, V2/overlay) | Explored             | noc_nonblocking_api, NOC register headers                        |
| Compute / LLK differences                 | Explored             | Subset API, simpler templates                                    |
| Kernel dispatch (24 processors)           | Explored             | qa_hal.cpp linker flags, processor assignment                    |
| HAL capabilities                          | Explored             | qa_hal.cpp, qa_hal_tensix.cpp                                    |
| RISC-V atomics (Zalrsc)                   | Noted                | Quasar-only; test_riscv_atomics.cpp                              |
| Quasar kernel APIs                        | Explored             | `QuasarDataMovementConfig` (num_threads 1-8, DM processor reservation) + `QuasarComputeConfig` (num_threads 1-4 Neos, 4 TRISCs each) |
| Quasar program dispatch model             | Explored             | `LaunchProgram` shared path; Quasar-specific: `num_sw_threads`/`kernel_thread_id` per DM/Neo slot in kernel group config |
| DFB test kernels + workloads              | Explored             | BMM, block matmul, datacopy, synthetic DFB; full API surface in DFB plan Section 11 |
| Quasar semaphore model                    | Explored             | L1-resident counters; `experimental::Semaphore` (DM) + `ckernel::Semaphore` (compute); remote via `noc_semaphore_inc`; not used by DFB workloads |
| Multi-Tensix NOC communication            | Explored             | All DFB tests are single-Tensix; cross-core only in semaphore pipeline test; does not block Phase 1 |
| Tensor/TensorAccessor API                 | Explored             | Page-oriented address generator; `TensorAccessorArgs` from compile-time args; `noc_traits_t` maps page_id -> NOC addr; needs emulation stub |
| Global Semaphores                         | Not yet explored     | Current tt-metal `GlobalSemaphore` is a host-side sharded L1 buffer wrapper for cross-core/cross-chip sync. Quasar may introduce a hardware-level global semaphore primitive (specs TBD). Key files: `tt_metal/api/tt-metalium/global_semaphore.hpp`, `tt_metal/impl/buffers/global_semaphore.cpp`, `tests/tt_metal/tt_metal/api/test_global_semaphores.cpp`. Used by: `point_to_point`, `strided_all_gather_async`, `sdpa`, `deepseek_prefill`, fabric tests |


---

## Implementation Phases

### Phase 0: Exploration (current)

- Understand all Quasar-specific APIs and their emulation requirements
- Review DFB plan (done)
- Explore remaining areas marked "not yet explored" above

### Phase 1: Single-Tensix DFB Emulation

- Implement tile counter array (`TileCounter`, `TileCounterArray`)
- Implement `DFBSyncState` and per-thread `EmuleDFBInterface`
- Implement explicit sync DFB API (reserve_back/push_back/wait_front/pop_front/finish)
- STRIDED access pattern only
- 2 DM producers + 4 Tensix consumers (expandable to 8 DM)
- Update `Core` for 4 MB L1, DFB storage
- Update `kernel_runner.cpp` for multi-DM thread model
- Update `Program` for DFB configs
- Regression tests

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

## Key Design Decisions (to be made)

1. **Arch selection**: **Resolved -- Quasar is a separate build.** The emulator binary is compiled for one arch at a time. Arch-specific code paths (DFB vs CB, thread model, L1 size, etc.) use compile-time `#ifdef` rather than runtime dispatch.
2. **CB coexistence**: **Resolved -- DFBs only for Quasar build.** Quasar HAL says `supports_cbs = true` AND `supports_dfbs = true`, but programs cannot mix them (DFB plan Section 8). Since Quasar is a separate build, the Quasar emulator only needs DFB support. CBs remain in the Wormhole build.
3. **Thread count**: **Resolved -- driven by kernel config.** `QuasarDataMovementConfig::num_threads_per_cluster` (1-8) and `QuasarComputeConfig::num_threads_per_cluster` (1-4 Neos, each with 4 TRISCs) determine how many threads to spawn per Tensix. The emulator reads these from the program's kernel descriptors.
4. **L1 addressing**: **Resolved -- no issue.** `MAP_32BIT` gives 2 GB address space; 4 MB L1 fits trivially.
5. **JIT toolchain**: Current emule uses `g++` for JIT. Workspace rules say clang-17. When do we switch?
6. **Configuration format**: What data-driven format for Quasar variants? Options: SoC YAML from tt-metal, custom JSON/YAML, C++ config struct, or compile-time constants.
7. **JIT header tree structure**: How should the Quasar build expose DFB APIs to JIT-compiled kernels? Options: separate `jit_hw/` tree, `#ifdef ARCH_QUASAR` in existing headers, or redirect to upstream `experimental/dataflow_buffer.h` with emulation shims.

