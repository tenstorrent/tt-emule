# Implementation Report v5: Software-Emulated Device (tt-emule) Integration into tt-metal

## Table of Contents

1. [tt-emule Overview](#1-tt-emule-overview)
2. [Integration into tt-metal](#2-integration-into-tt-metal)
3. [Pros, Cons, and Maintainability](#3-pros-cons-and-maintainability)

---

## 1. tt-emule Overview

### Purpose

tt-emule is a standalone C++ software emulator for Tenstorrent device-level APIs. It emulates the multi-core execution model of Tenstorrent hardware — including per-core L1 SRAM, shared DRAM, circular buffer (CB) synchronization, NOC (Network-on-Chip) communication, semaphore-based cross-core synchronization, and the DST register file — entirely on the host CPU. This enables kernel development, testing, and debugging without access to physical silicon.

### Architecture

Each emulated core runs three concurrent threads mirroring hardware:

| Thread | Entry Point | Role |
|--------|-------------|------|
| NOC Reader | `reader_kernel_main()` | Reads data from DRAM/L1 into circular buffers via NOC |
| Compute | `compute_kernel_main()` | Processes tiles from input CBs, writes results to output CBs |
| NOC Writer | `writer_kernel_main()` | Writes computed results from CBs back to DRAM/L1 via NOC |

Threads synchronize through circular buffers backed by mutex + condition variable pairs (`CBSyncState`). The compute thread operates on a private DST register file (8 slots x 1024 float32 elements) that mirrors the hardware tile register space.

Multiple cores execute concurrently in separate threads, enabling cross-core communication via multicast NOC writes and semaphore signaling — matching the real hardware execution model.

### Memory Model

All memory is owned by `tt_emule::Core` objects. Worker cores draw their memory from a shared `L1Pool` — a single contiguous `MAP_32BIT` mmap with 2 MB aligned slots — enabling bitmask offset extraction (`addr & 0x1FFFFF`) in a single AND instruction. DRAM cores use individual mmaps.

```cpp
// L1Pool — single contiguous MAP_32BIT mmap, 2 MB aligned slots
class L1Pool {
    static constexpr size_t SLOT_SIZE = 2 * 1024 * 1024;  // 2 MB
    static constexpr size_t SLOT_MASK = SLOT_SIZE - 1;
    uint8_t* base_;    // SLOT_SIZE-aligned base
    size_t   num_slots_;
    static uint32_t to_offset(uint32_t addr) { return addr & SLOT_MASK; }
};

enum class CoreRole { WORKER, DRAM };

class Core {
    uint8_t*  l1_      = nullptr;      // from L1Pool slot (WORKER) or individual mmap (DRAM)
    uint32_t  l1_base_ = 0;            // truncated address for kernel use
    size_t    l1_size_ = L1_SIZE;       // 1 MB (WORKER) or bank size (DRAM)
    CoreRole  role_    = CoreRole::WORKER;
    CBSyncState cb_sync_states_[MAX_CBS] = {};
    DstRegisterFile dst_;
};
```

Worker cores use L1-sized regions (1 MB) allocated from L1Pool slots; DRAM cores use bank-sized individual mmaps. The `get_core(tt_xy_pair)` factory lazy-creates with the appropriate role. If the pool is exhausted, worker cores fall back to individual `MAP_32BIT` mmaps.

The pool is sized at 2x the Tensix core count from the SOC descriptor to provide headroom for cores created via virtual coordinates that differ from physical coordinates.

### Address Translation

A key challenge in emulation is that on real hardware, L1 addresses (e.g. `0xFFE30` for a semaphore) are directly dereferenceable by firmware. In emulation, L1 is mmap'd at a host address like `0x41B50000`, so raw L1 offsets are invalid pointers.

tt-emule uses a **host-pointer-everywhere** convention: all addresses returned to kernels (`get_write_ptr`, `get_read_ptr`, `get_semaphore`) are host pointers truncated to `uint32_t`. Three helper functions handle translation at well-defined points:

- **`__emule_addr_to_offset(addr)`** — **Encode point.** Extracts the L1 offset from a host pointer. With L1Pool (`TT_EMULE_USE_L1_POOL`), this is a single bitmask: `addr & 0x1FFFFF`. Without the pool (standalone builds), it falls back to TLS subtraction: `addr - l1_base`. Called by `get_noc_addr` and `get_noc_multicast_addr`.

- **`__emule_fixup_noc_addr(noc_addr)`** — **Fixup point.** Sanitizes the addr bits [35:0] of an already-encoded 64-bit NOC address by applying `__emule_addr_to_offset` to the lower bits. This is necessary because real tt-metal kernels construct NOC addresses by ORing host pointers into pre-computed bases: `uint64_t addr = noc_base | l1_host_ptr` (e.g., matmul sender kernels building data multicast addresses). Called by `noc_async_read`, `noc_async_write`, `noc_async_write_multicast`, `noc_semaphore_inc`, and `noc_semaphore_set_multicast`.

- **`__emule_resolve_noc_addr(noc_addr)`** — **Decode point.** Decodes (x, y, offset) from a 64-bit NOC address, looks up the target core in `__emule_core_map`, and returns a host pointer via `core->l1_ptr(offset)`. Implemented as an `extern "C"` bridge function in the program runner.

The L1Pool bitmask approach eliminates the TLS lookup and conditional branch that the old subtraction-based conversion required, reducing encode-point overhead to a single AND instruction.

### Semaphore Layout

Semaphores are placed in the kernel config region of L1, matching real firmware layout:

```
sem_addr = kernel_config_base + prog_config.sem_offset + sem_id * L1_ALIGNMENT
```

Where `kernel_config_base` is the HAL's `KERNEL_CONFIG` address (`MEM_MAP_END`, ~0x8730 on Wormhole), `sem_offset` is computed by `finalize_sems()` and stored in `ProgramConfig`, and `L1_ALIGNMENT` = 16 bytes. The emulated program runner reads these values from the same `ProgramConfig` that real firmware uses, ensuring address-level fidelity.

For standalone tt-emule builds (where no HAL or ProgramConfig exists), a fallback default of `0xFFE00` is used.

### Circular Buffer Synchronization

`CBSyncState` is the single FIFO primitive shared between all kernel execution paths:

```cpp
struct CBSyncState {
    uint8_t*  base      = nullptr;
    uint32_t  page_size = 0;
    uint32_t  num_pages = 0;
    uint32_t  write_idx = 0;
    uint32_t  read_idx  = 0;
    uint32_t  occupied  = 0;
    std::mutex              mu;
    std::condition_variable space_cv;   // producer waits for free space
    std::condition_variable data_cv;    // consumer waits for data
};
```

Four inline operations (`cb_sync_reserve`, `cb_sync_push`, `cb_sync_wait`, `cb_sync_pop`) implement the producer-consumer protocol. Both standalone and JIT kernel paths use the same `CBSyncState` struct.

### Codebase Structure

```
tt-emule/                           (~6,800 lines across 113 files)
├── include/
│   ├── tt_emule/       (11 files)  Host-side types: Device, Core, L1Pool, Buffer, Program,
│   │                               CircularBuffer, CBSyncState, DstRegisterFile
│   ├── kernel_api/      (4 files)  Standalone device-side API (links tt_emule directly)
│   ├── jit_hw/         (78 files)  JIT kernel stubs (resolved via dlopen + -rdynamic)
│   │   ├── api/compute/            53 compute op headers (30 top-level + 23 eltwise_unary)
│   │   ├── api/dataflow/           NOC ops, multicast, semaphores, addrgen page ops
│   │   ├── api/tensor/             TensorAccessor for page-based addressing
│   │   ├── internal/               Banking infra, RISC attribs, mod_div_lib
│   │   └── experimental/           CoreLocalMem, Noc, AllocatorBank, Lock stubs
│   └── ttkernel/        (4 files)  Forwarding headers for tt-metal kernel include paths
├── src/                 (3 files)  host_api.cpp, kernel_runner.cpp, jit_kernel.cpp
└── tests/              (10 files)  eltwise_add, matmul, tilize, compat tests
```

### JIT Kernel API Coverage

The `jit_hw/` directory provides stub implementations for 75 header files covering:

| Category | Files | Key APIs |
|----------|-------|----------|
| Compute | 53 | `matmul_tiles`, `matmul_block`, `add/sub/mul_tiles`, `pack_tile`, `copy_tile`, `bcast`, `reduce`, `tilize/untilize`, `pack_untilize` (with `experimental::pack_untilize_block`), `transpose_wh`, `quantization`, 23 eltwise_unary SFPU ops (`abs_tile`, `exp_tile`, `negative_tile`, `typecast_tile`, etc.), binary bitwise/shift/comp/fmod/max_min, `gcd/lcm`, `xlogy`, `copy_dest_values` |
| Dataflow | 2 | `noc_async_read/write`, `noc_async_write_multicast`, semaphore ops, `InterleavedAddrGen<DRAM/L1>`, banking arrays |
| CB sync | 1 | `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front` (uint32_t and int32_t overloads) |
| LLK defs | 1 | `llk_unpack_A`, `llk_wait_tiles`, `llk_pop_tiles`, `llk_push_tiles`, `llk_wait_for_free_tiles`, `get_output_id`, `FACE_R_DIM`, `TILE_C_DIM`, coordinate APIs (`get_absolute_logical_x/y`) |
| Tensor | 1 | `TensorAccessor`, `TensorAccessorArgs` |
| Infrastructure | 8 | compile-time args, bfloat16, dprint, assert stubs, DataFormat enum, tile constants |
| Experimental | 5 | `Noc`, `CircularBuffer`, `AllocatorBank`, `Lock`, `CoreLocalMem` |
| Compatibility | 4 | `ckernel.h`, `ckernel_defs.h`, `common_values.hpp`, `risc_attribs.h` |

### Dual API Paths

tt-emule provides two parallel API surfaces for kernel code:

| Aspect | `kernel_api/` (Standalone) | `jit_hw/` (JIT for tt-metal) |
|--------|---------------------------|------------------------------|
| **Linking** | Static link to `tt_emule_lib` | dlopen'd `.so`, symbols via `-rdynamic` |
| **L1 addresses** | `uint8_t*` (host pointers) | `uint32_t` (truncated from mmap below 4 GB) |
| **CB backing** | `tt_emule::CircularBuffer` | `CBSyncState` array on `tt_emule::Core` |
| **Tile format** | Float32 (4096 bytes/tile) | bfloat16 (2048 bytes) or INT32 (4096 bytes) |
| **Context** | Thread-local `__core`, `__device` | Same, plus bridge pointers for DRAM/NOC resolution |

Both paths share a single `CBSyncState` struct and `cb_sync_*` free functions.

### Test Results

**Standalone tests** (3/3 pass): eltwise_add, matmul, tilize

**tt-metal emulated regression** (18/18 pass):

| Tier | Tests | Count | Description |
|------|-------|-------|-------------|
| 1 | Host-only | 13 | bit_utils, host_buffer, tilize_untilize, blockfloat_common, 9 CoreRange/Set tests |
| 2 | Buffer I/O | 2 | SimpleL1Buffer, SimpleDramBuffer (Wormhole N150) |
| 3 | JIT Kernel | 1 | TensixL1Tile — experimental CB/Noc API |
| 4 | TTNN Relational | 1 | 66 sub-cases: eq, ne, gt, lt, ge, le on INT32 (Blackhole P100) |
| 5 | TTNN Matmul Sweep | 1 | 14 sub-cases: multi-core matmul 32x32x32 through 2048x2048x2048 (Wormhole N150) |

**D2M golden test regression** (1589 pass / 147 fail / 142 skip-xfail):

| Test File | Total | Passed | Failed | Skip/XFail | Status |
|-----------|-------|--------|--------|------------|--------|
| test_metal_layout | 94 | 94 | 0 | 0 | **PASS** |
| test_metal_matmul | 127 | 113 | 0 | 14 xfail | **PASS** |
| test_metal_matmul_higher_rank | 10 | 10 | 0 | 0 | **PASS** |
| test_metal_allocate | 6 | 6 | 0 | 0 | **PASS** |
| test_metal_tms | 339 | 332 | 4 | 3 skip | PASS (arange only) |
| test_metal_reductions | 1096 | 932 | 68 | 96 skip | FAIL (unaligned) |
| test_metal_dma | 49 | 39 | 10 | 0 | FAIL (DRAM aborts) |
| test_metal_tilize | 44 | 12 | 32 | 0 | FAIL (untilize/tilize PCC) |
| test_metal_tensor_collapsing | 14 | 12 | 0 | 2 skip | **PASS** |
| test_metal_virtual_grids | 39 | 39 | 0 | 0 | **PASS** |
| test_metal_virtual_grid_rowmajor | 27 | 0 | 0 | 27 skip | SKIP (needs n300) |
| test_metal_masking | 20 | 0 | 20 | 0 | FAIL (runtime crashes) |
| test_metal_bfp8_typecast | 13 | 0 | 13 | 0 | FAIL (PCC mismatch) |

See [D2M_REGRESSION_REPORT.md](D2M_REGRESSION_REPORT.md) for detailed failure analysis and historical progression.

---

## 2. Integration into tt-metal

### Design Philosophy

The integration follows three key principles:

1. **Zero fake headers.** Tests link the real `Metalium::Metal` library and use real tt-metal headers, fixtures, and buffer utilities. Emulation is injected at the UMD (User-Mode Driver) boundary, not via mock headers.

2. **Memory isolation in `tt_emule::Core`.** All device memory — worker L1, DRAM banks — is owned by `tt_emule::Core` objects inside `SWEmulatedChip`. There is no intermediate copy-in/copy-out stage. The program runner uses Core's mmap'd memory directly, and it persists across program runs.

3. **Minimal API surface in `jit_hw/`.** Stub headers only implement what is needed for the target kernel to compile and execute correctly. Where possible, real tt-metal headers are included rather than duplicated.

### Integration Layers

#### Layer 1: UMD Device Injection — SWEmulatedChip

```
tt_metal/third_party/umd/device/
├── api/umd/device/chip/sw_emulated_chip.hpp   (70 lines)
└── chip/sw_emulated_chip.cpp                   (111 lines)
```

`SWEmulatedChip` extends `MockChip` with memory-backed I/O:

```cpp
class SWEmulatedChip : public MockChip {
    std::unique_ptr<tt_emule::L1Pool> worker_pool_;          // 2MB-aligned slots
    size_t next_slot_ = 0;
    std::unordered_map<tt_xy_pair, size_t> core_to_slot_;    // core -> pool slot
    std::unordered_map<tt_xy_pair, std::unique_ptr<tt_emule::Core>> cores_;
    std::unordered_map<tt_xy_pair, uint32_t> dram_core_to_channel_;
    uint32_t l1_size_;
    uint64_t dram_bank_size_;
    std::mutex core_mutex_;
};
```

- **L1Pool for worker cores.** At construction, allocates a single contiguous `MAP_32BIT` mmap with 2x the Tensix core count slots (from `soc.get_cores(CoreType::TENSIX).size()`). Worker cores get their L1 from pool slots; DRAM cores use individual mmaps. Pool exhaustion falls back to individual `MAP_32BIT` mmaps.
- **Lazy core creation.** `get_core(tt_xy_pair)` creates a `tt_emule::Core` on first access with the correct role (`WORKER` or `DRAM`) and memory source (pool slot or individual mmap).
- **Forward declaration only.** The header forward-declares `tt_emule::Core` and `tt_emule::L1Pool`, avoiding UMD include path contamination.
- **Uniform I/O.** All `read_from_device` / `write_to_device` overrides delegate to `get_core(xy)->l1_ptr(offset)` + `memcpy` — uniform for both worker and DRAM cores.

Chip instantiation in `cluster.cpp`:

```cpp
#ifdef TT_METAL_EMULATION
if (chip_type == ChipType::EMULATED) {
    return std::make_unique<SWEmulatedChip>(soc_desc);
}
#endif
```

#### Layer 2: Runtime Activation

Activation is controlled by environment variables:

```
TT_METAL_EMULATED_MODE=1              -> TargetDevice::Emulated + force slow dispatch
TT_METAL_MOCK_CLUSTER_DESC_PATH=...   -> SOC descriptor for core/DRAM topology
TT_METAL_SLOW_DISPATCH_MODE=1         -> Required (no HWCommandQueue in emulation)
```

Both the build flag and the env var are required. The `TT_METAL_EMULATION=ON` CMake flag compiles in the emulated code paths (`SWEmulatedChip`, `execute_program_emulated`, JIT runner) behind `#ifdef TT_METAL_EMULATION`. Without it, these code paths do not exist in the binary. The env var then selects the emulated path at runtime. A single binary built with this flag supports both silicon and emulated execution — the env var toggles which path is taken.

#### Layer 3: JIT Kernel Execution

The emulated program runner is the core integration component:

```
tt_metal/impl/emulation/
├── emulated_program_runner.hpp   (51 lines)
├── emulated_program_runner.cpp  (829 lines)
└── emulated_run_stats.hpp       (20 lines)
```

**JIT Compilation Pipeline:**

1. Kernel source `.cpp` -> temp directory with `wrapper.cpp` (kernel `#define`s + `#include "jit_kernel_stubs.hpp"` + `#include kernel.cpp`)
2. `g++ -std=c++17 -fPIC -shared -O3` with tt-emule and kernel-directory include paths
3. Compile-time args as `-DKERNEL_COMPILE_TIME_ARGS=v0,v1,...`; named args via `-DKERNEL_COMPILE_TIME_ARG_MAP`
4. `dlopen` + `dlsym("__emule_kernel_entry")` -> `std::function<void()>`
5. Results cached by `(source_path : compile_args : defines)` key; cache misses compiled in parallel via `std::async`

**Program Execution — direct Core memory, no copies:**

```
Phase 1a: Collect kernels per logical core from ProgramImpl
    For each HalProgrammableCoreType x each kernel:
        Build JIT cache key, check cache
        Group by logical core with {cache_key, rt_args, common_rt_args}

Phase 1b: Compile cache misses in parallel (std::async)

Phase 1c: Resolve pending kernels to function pointers

Phase 2: Execute all cores concurrently
    For each physical core:
        1. core = sw_emu->get_core(physical_xy)
        2. core->reset_cb_sync() / init_cb_sync(...)
        3. Initialize semaphores at HAL-derived addresses
        4. Set thread-local bridge pointers:
             __emule_bridge_l1   = core->l1_data()
             __emule_bridge_dram = dram_core->l1_data()
             __emule_cbs         = core->cb_sync_array()
             __emule_core_map    = &core_map  (for cross-core NOC resolution)
        5. Launch one thread per kernel (reader + compute + writer)
        6. Join all threads
        7. Done — no copy-back, no munmap
```

**Semaphore Initialization:**

Semaphores are initialized in each core's L1 at addresses matching the real firmware layout. The base address is computed from the HAL kernel config region and ProgramConfig:

```cpp
const auto& hal = MetalContext::instance().hal();
uint32_t kernel_config_base = hal.get_dev_addr(HalProgrammableCoreType::TENSIX,
                                                HalL1MemAddrType::KERNEL_CONFIG);
uint32_t emule_sem_base = kernel_config_base + prog_config.sem_offset;
// Each semaphore: emule_sem_base + sem_id * 16
```

This is passed to JIT kernels as `EMULE_SEM_BASE` so that `get_semaphore(id)` returns the correct L1 address.

**Memory Bridge (JIT <-> host process):**

Thread-local pointers and `extern "C"` bridge functions are exported via `-rdynamic` so that dlopen'd `.so` files resolve them at load time:

```cpp
// Thread-local bridge pointers
thread_local uint8_t*          __emule_bridge_l1   = nullptr;
thread_local uint8_t*          __emule_bridge_dram = nullptr;
thread_local __emule_cb_state* __emule_cbs         = nullptr;
thread_local tt_emule::Core*   __core              = nullptr;

// C-linkage bridge functions
extern "C" uint8_t* __emule_dram_ptr(uint64_t offset);
extern "C" uint8_t* __emule_local_l1_ptr(uint64_t offset);
extern "C" uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr);
extern "C" void     __emule_multicast_write(uint64_t mcast_addr, const uint8_t* src, uint32_t size);
```

This avoids the ABI hazard of JIT kernels inlining any `Device` or `Core` methods — kernels only touch opaque `extern "C"` functions.

#### Layer 4: Dispatch Interception

In `tt_metal.cpp`, `LaunchProgram()` checks the target device type:

```cpp
#ifdef TT_METAL_EMULATION
if (MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Emulated) {
    emule::execute_program_emulated(device, program);
} else
#endif
{
    // Real hardware dispatch path...
}
```

This is the single branch point where emulated execution diverges from silicon dispatch. `CompileProgram` and `WriteRuntimeArgsToDevice` still run to populate `ProgramImpl` data structures; only the hardware dispatch path is bypassed. `finalize_offsets()` also runs before dispatch, which is what populates `ProgramConfig.sem_offset` — making the HAL-based semaphore base available to the emulated runner.

A second interception in `ConfigureDeviceWithProgram()` skips binary writing for emulated mode.

### Guard Pattern: `is_mock_or_emulated()`

32 call sites across 15 files guard hardware-specific operations:

| Subsystem | Files | Guards | Operations Skipped |
|-----------|-------|--------|--------------------|
| Dispatch queues | `fd_mesh_command_queue.cpp`, `sd_mesh_command_queue.cpp` | 8 | HW queue operations |
| Firmware | `risc_firmware_initializer.cpp` | 5 | RISC firmware load |
| Context init | `metal_context.cpp` | 4 | Fabric, FW, watcher, dprint |
| Program dispatch | `program.cpp`, `dispatch.cpp` | 2 | Dispatch data population |
| Device init | `device.cpp`, `device_manager.cpp` | 2 | HAL/HW initialization |
| Fabric | `control_plane.cpp`, `channel_trimming_export.cpp` | 2 | Fabric control plane |
| Buffer/event dispatch | `buffers/dispatch.cpp`, `event/dispatch.cpp` | 2 | HW buffer/event ops |
| Other | `system_memory_manager.cpp`, `prefetch.cpp`, `core_descriptor.cpp`, etc. | 7 | Misc HW operations |

Definition (in `tt_cluster.hpp`):

```cpp
bool is_mock_or_emulated() const {
    return this->target_type_ == tt::TargetDevice::Mock ||
           this->target_type_ == tt::TargetDevice::Emulated;
}
```

### Architecture Stack

```
User Code (ttnn::matmul, ttnn::add, etc.)
    +-- tt-metal API (CreateBuffer, CreateProgram, LaunchProgram)
            +-- [TT_METAL_EMULATION guard] execute_program_emulated()
                    |-- JIT compile: kernel .cpp -> .so via g++ + dlopen
                    |-- Per-core: SWEmulatedChip::get_core() -> tt_emule::Core*
                    |-- Init CB sync states from ProgramImpl
                    |-- Compute sem base from HAL kernel config + ProgramConfig
                    |-- Populate banking arrays from metal_SocDescriptor
                    +-- Launch N cores x M threads, set TLS context
                        |-- reader_kernel_main()  --- NOC read, CB push
                        |-- compute_kernel_main() --- DST math, CB pop/push
                        +-- writer_kernel_main()  --- CB pop, NOC write

tt::umd::SWEmulatedChip (extends MockChip)
    +-- L1Pool: single MAP_32BIT mmap, 2MB-aligned slots for all worker cores
    +-- Per-{x,y} lazy-created tt_emule::Core (pool slot or individual mmap)
    +-- write_to_device / read_from_device -> memcpy to/from Core::l1_ptr()

tt_emule::L1Pool
    |-- Single contiguous MAP_32BIT mmap (num_slots x 2MB + alignment overhead)
    |-- SLOT_SIZE=2MB, SLOT_MASK=0x1FFFFF
    +-- to_offset(addr) -> addr & SLOT_MASK (one AND instruction)

tt_emule::Core
    |-- L1 from L1Pool slot (WORKER) or individual mmap (DRAM)
    |-- CBSyncState[32] for multi-thread CB sync
    |-- DstRegisterFile for compute thread
    +-- l1_base_addr() -> 32-bit truncated address for JIT kernels
```

### CMake Configuration

Two independent flags control the build:

| Flag | Scope | Effect |
|------|-------|--------|
| `TT_METAL_EMULATION` | tt_metal library + UMD | Compiles `emulated_program_runner.cpp`, enables `SWEmulatedChip`, defines JIT include paths, adds `-rdynamic` |
| `TT_METAL_USE_TT_EMULE` | Top-level build | Adds `tt_emule/` test subdirectory |

```bash
cmake -B build_emule \
  -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 \
  -DTT_METAL_USE_TT_EMULE=ON \
  -DTT_METAL_EMULATION=ON \
  -DTT_EMULE_PATH=/path/to/tt-emule \
  -DWITH_PYTHON_BINDINGS=OFF -DENABLE_TRACY=OFF -DTT_INSTALL=OFF \
  -DTT_METAL_BUILD_TESTS=OFF -DTTNN_BUILD_TESTS=OFF
```

### Test Infrastructure

Tests in `tt_emule/` are organized in tiers and use standard tt-metal fixtures — no custom test infrastructure:

| Tier | Binary | Cases | Description | Cluster |
|------|--------|-------|-------------|---------|
| 1a | test_bit_utils | 4 | Pure bit manipulation | None |
| 1b | 9 CoreRange/Set tests | 9 | API-level CoreRange/CoreRangeSet | None |
| 1c | test_host_buffer | 1 | HostBuffer RAII | None |
| 1d | test_tilize_untilize, test_blockfloat_common | 2 | Data format conversion | None |
| 2 | test_simple_l1_buffer | 2 | L1 buffer read/write with MeshDeviceFixture | Wormhole N150 |
| 2 | test_simple_dram_buffer | 1 | DRAM buffer round-trip | Wormhole N150 |
| 3 | test_simple_l1_buffer (Tensix filter) | 1 | JIT kernel execution with experimental API | Wormhole N150 |
| 4 | test_ttnn_relational_int | 66 | eq, ne, gt, lt, ge, le on INT32 | Blackhole P100 |
| 5 | test_ttnn_matmul_sweep | 14 | Multi-core matmul with semaphore sync (32^2--2048^2) | Wormhole N150 |

Additional ttnn tests built but not in regression: `test_ttnn_add`, `test_ttnn_add_int`, `test_ttnn_sub_int`, `test_ttnn_rsub_int`, `test_ttnn_matmul`.

D2M golden test regression: `run_d2m_regression.sh` — runs 13 tt-mlir test files (1878 tests) against the emulated backend. 1589 pass, 147 fail, 142 skip. See [D2M_REGRESSION_REPORT.md](D2M_REGRESSION_REPORT.md).

Regression scripts: `run_regression.sh` (18 tt-metal tests) + `run_d2m_regression.sh` (13 D2M test files, 1863 tests).

### tt-metal Files Modified

The complete set of tt-metal modifications for emulation support:

| Category | Files | Lines |
|----------|-------|-------|
| New: program runner | `emulated_program_runner.{hpp,cpp}`, `emulated_run_stats.hpp` | ~900 |
| New: UMD chip | `sw_emulated_chip.{hpp,cpp}` | ~181 |
| New: test infrastructure | `tt_emule/CMakeLists.txt`, `tt_emule/ttnn_tests/CMakeLists.txt` | ~207 |
| Modified: dispatch | `tt_metal.cpp` | ~20 |
| Modified: guards | 15 files (`is_mock_or_emulated()`) | ~32 one-liners |
| Modified: enum additions | `rtoptions.cpp`, `tt_cluster.hpp`, UMD types | ~15 |
| Modified: CMake | `CMakeLists.txt` (top-level + UMD) | ~40 |

---

## 3. Pros, Cons, and Maintainability

### Pros

**Development velocity without hardware.** Developers can write, compile, and test kernels on any x86 Linux machine. The full loop — kernel compilation, buffer I/O, multi-threaded multi-core execution with CB synchronization and NOC communication — runs in seconds without requiring a Tenstorrent card. The matmul sweep (14 cases up to 2048x2048) completes in under 30 seconds.

**Zero fake headers reduces drift.** By linking the real `Metalium::Metal` library and using real tt-metal headers, tests exercise the actual host API code paths. Header changes in tt-metal are immediately visible as compile errors in emulated tests, rather than silently diverging behind a fake header layer.

**Memory isolation eliminates copy overhead.** `tt_emule::Core` is the single memory owner for both worker L1 and DRAM banks. The program runner sets bridge pointers directly to Core's mmap'd region — no copy-in / copy-out, no temporary allocations, no munmap lifecycle. Core memory persists across program runs, matching hardware semantics.

**Multi-core execution is realistic.** All cores launch concurrently in separate threads, with cross-core communication via NOC multicast writes and semaphore atomic operations. This catches real concurrency bugs: the semaphore address translation fix was discovered only because the multi-core matmul sweep exercises genuine cross-thread synchronization.

**Semaphore layout matches hardware.** The HAL-based semaphore base (`kernel_config_base + prog_config.sem_offset`) uses the same values computed by `finalize_sems()` for real firmware. This eliminated the CB/semaphore overlap bug that caused matmul_block-2048x2048 to abort, and ensures that any future changes to semaphore placement in tt-metal are automatically reflected in emulation.

**Runtime toggle from a single binary.** A binary built with `TT_METAL_EMULATION=ON` supports both silicon and emulated execution — the `TT_METAL_EMULATED_MODE` environment variable selects which path is taken at runtime. The build flag is a prerequisite: without it, the emulated code paths (`SWEmulatedChip`, `execute_program_emulated`) are not compiled in (`#ifdef TT_METAL_EMULATION`).

**Interleaved DRAM banking is production-accurate.** Bank mapping arrays (`dram_bank_to_noc_xy`, `bank_to_dram_offset`, etc.) are populated from the real `metal_SocDescriptor` at program execution time. `InterleavedAddrGen<DRAM>` computes proper banked NOC addresses matching the real firmware's banking logic.

**Incrementally extensible.** New compute ops are single-file headers in `jit_hw/api/compute/`. The pattern (DST-to-DST operations with format-aware load/store) has been applied consistently across 53 compute headers.

**D2M golden test coverage.** 1589 of 1878 D2M golden tests pass (85% pass rate), covering layout transforms, buffer allocation, matmul (single-core, multi-core, double-buffered, 3D/4D batched up to 2048x2048x2048), reductions (sum, max, mean), DMA (L1-to-L1 and partial DRAM), TMS (reshape, permute, concatenate_heads), virtual grids, and tensor collapsing. This provides broad regression coverage for D2M-generated kernels.

### Cons

**Behavioral fidelity gap.** The emulator approximates hardware behavior but does not replicate it:
- NOC operations are synchronous `memcpy` (no latency modeling, no bandwidth contention)
- DST register file uses float32 internally; bfloat16 precision loss only occurs at pack/unpack boundaries, not during intermediate computation
- No tile pipeline stalls, no resource conflicts, no firmware overhead
- No watcher/dprint server — `DPRINT` is a host `printf` stub

Tests that pass in emulation may fail on silicon due to timing, precision, or resource constraints.

**Dual API surface.** `kernel_api/` and `jit_hw/` remain separate API surfaces with different type conventions (`uint8_t*` vs `uint32_t`, `Tile` vs raw bfloat16). Adding a new kernel API function requires implementation in both paths.

**JIT compilation overhead.** Each unique kernel requires a `g++` invocation (typically 1--3 seconds). Results are cached within a process but not persisted across runs. Cache misses are compiled in parallel via `std::async`, but test suites with many distinct kernels still pay full compilation cost on every invocation.

**`MAP_32BIT` address space pressure.** The JIT path relies on `mmap` with `MAP_32BIT` to ensure L1 pointers fit in `uint32_t`. The 32-bit address space region (~2 GB) is shared across all cores. L1Pool mitigates this by consolidating all worker L1 into a single contiguous allocation (Wormhole 80 cores x 2 MB = 160 MB, Blackhole 140 cores x 2 MB = 280 MB — both fit easily). DRAM cores use regular mmap outside the 32-bit region. The remaining risk is that non-pool fallback cores (from pool exhaustion) still compete for the same 32-bit space.

**Format detection heuristic.** The `page_size > 2048` heuristic used by `copy_tile` and `pack_tile` to distinguish bfloat16 from 32-bit formats is fragile. It works for current tile sizes but would break for new formats with different tile sizes.

**No persistent JIT cache.** The in-memory JIT cache is lost when the process exits. Suites with many distinct kernels pay compilation cost on every run.

**Address translation complexity.** The host-pointer convention requires careful translation at three well-defined points: encode (`__emule_addr_to_offset` in `get_noc_addr`/`get_noc_multicast_addr`), fixup (`__emule_fixup_noc_addr` in NOC operation functions), and decode (`__emule_resolve_noc_addr` in the program runner). The fixup point exists because real tt-metal kernels construct NOC addresses by ORing host pointers into pre-computed NOC bases. With L1Pool, the fixup is a fast bitmask (`addr & 0x1FFFFF`) rather than a TLS lookup, but each new NOC-level function must still remember to apply it.

**D2M coverage gaps.** 2 test files fail entirely: masking (20 runtime crashes) and bfp8_typecast (13 PCC mismatches). 3 files have partial failures: reductions (68/1096 fail on unaligned shapes), tilize (32/44 fail on standalone untilize/tilize PCC), and DMA (10/49 fail on DRAM paths). The remaining 8 files pass fully. The primary gaps are unaligned tensor masking, standalone tilize/untilize data layout correctness, and DRAM DMA address resolution.

### Maintainability

**Low ongoing cost for tt-emule standalone.** The standalone emulator (~6,000 lines) is self-contained with no external dependencies beyond the C++ standard library and pthreads. Changes to tt-metal do not affect standalone tests.

**Medium ongoing cost for tt-metal integration.** The primary maintenance burden is keeping JIT stubs (`jit_hw/`) aligned with tt-metal kernel APIs. When tt-metal adds or changes a kernel API function, the corresponding stub must be created or updated. The current 78 `jit_hw/` files cover the most common operations but represent a fraction of the full kernel API surface.

**Rebase risk is bounded.** The tt-metal integration is concentrated in:
- `tt_metal/impl/emulation/` — the program runner (900 lines)
- `SWEmulatedChip` in UMD (181 lines, with L1Pool)
- `is_mock_or_emulated()` guards (32 one-liners across 15 files)
- One dispatch branch in `tt_metal.cpp`

Rebasing onto new tt-metal versions primarily requires:
1. Checking that `is_mock_or_emulated()` guards cover any new hardware-dependent code paths
2. Updating JIT stubs if kernel APIs change
3. Verifying that `ProgramImpl` and `CircularBufferConfig` interfaces haven't changed

**Guard pattern is predictable.** All guards follow the same `is_mock_or_emulated()` pattern. New hardware-dependent code paths in tt-metal that need guarding are identifiable by crashes when running in emulated mode — they fail loudly rather than producing silent corruption.

**Test maintenance scales with op coverage.** Each new ttnn operation requires: (1) a JIT stub header if the compute kernel uses uncovered APIs, (2) a test binary entry in `tt_emule/ttnn_tests/CMakeLists.txt`, and (3) a `run_test` line in `run_regression.sh`. The pattern is mechanical.

**Single point of truth for CB sync is durable.** Any bug fix to circular buffer synchronization is made in one place (`CBSyncState` + `cb_sync_*` functions) and propagated automatically.

**Banking infrastructure is self-maintaining.** Bank mapping arrays are populated dynamically from the SOC descriptor at runtime, so adding support for new chip architectures requires no code changes — only a new cluster descriptor YAML file.

**HAL-based semaphore placement is self-maintaining.** By reading `kernel_config_base` and `sem_offset` from the HAL and ProgramConfig respectively, the emulator automatically tracks any changes to the L1 memory map or semaphore layout in tt-metal. No hardcoded constants to update.

### Changes from v3 to v4

| Aspect | v3 | v4 |
|--------|----|----|
| Codebase size | ~5,990 lines, 87 files | ~6,000 lines, 110 files |
| JIT stubs | 55 files | 78 files (+23 compute headers) |
| Compute headers | 26 | 53 (30 top-level + 23 eltwise_unary) |
| Program runner | 674 lines | 829 lines (parallel JIT compile, HAL-based semaphore base) |
| Semaphore base | Dynamic `max(0xFFE00, max_cb_end)` from CB pre-scan | HAL-based: `kernel_config_base + prog_config.sem_offset` (matches real firmware) |
| JIT compilation | Sequential | Parallel via `std::async` for cache misses |
| `is_mock_or_emulated()` guards | 35 across 16 files | 32 across 15 files |
| D2M golden tests | Not tracked | 249 pass / ~244 fail / 56 skip-xfail (13 test files, ~1146 tests) |
| Matmul D2M coverage | 15 pass, abort at 2048x matmul_block | 56 pass (incl. 2048x2048x2048 bf16+f32), timeout at 48% |
| Regression scripts | `run_regression.sh` (18 tests) | + `run_d2m_regression.sh` (13 D2M test files, ~1146 tests) |

### Changes from v4 to v5

| Aspect | v4 | v5 |
|--------|----|----|
| Codebase size | ~6,000 lines, 110 files | ~6,800 lines, 113 files |
| JIT stubs | 78 files | 75 files (consolidated) |
| tt-mlir base | Pre-uplift | Rebased on `milant/uplift_mar_25` (new LLK APIs, expanded test suites) |
| D2M golden tests | 832 pass / 162 fail / 81 skip (1075 tests) | **1589 pass** / 147 fail / 142 skip (1878 tests) |
| D2M pass rate | ~77% | **85%** |
| D2M fully-passing files | 5 of 13 | **8 of 13** |
| Matmul D2M | 108 pass, 7 double-buffered fail | **113 pass**, 14 xfail (all functional tests pass) |
| Pack untilize | Not supported | `experimental::pack_untilize_block` with `copy_tile` + `__llk_pack_untilize` |
| LLK CB stubs | Not present | `llk_wait_tiles`, `llk_pop_tiles`, `llk_push_tiles`, `llk_wait_for_free_tiles` delegating to `cb_api.h` |
| SFPU ops | Stubs only (no logic) | Functional: `abs_tile`, `exp_tile`, `negative_tile`, `typecast_tile` with DST element-wise math |
| CB API | uint32_t only | + int32_t overloads for D2M-generated code |
| Coordinate API | Not present | `get_absolute_logical_x/y` in compute context |
| New D2M wins | — | virtual_grids (0→39), tensor_collapsing (2→12), TMS (169→332), reductions (420→932), DMA (16→39) |

---

*Report updated 2026-03-30. Covers tt-emule on branch `armin` / tt-mlir rebased on `milant/uplift_mar_25`.*
