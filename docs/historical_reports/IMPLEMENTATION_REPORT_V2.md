# Implementation Report v2: Software-Emulated Device (tt-emule) Integration into tt-metal

## Table of Contents

1. [tt-emule Overview](#1-tt-emule-overview)
2. [Integration into tt-metal](#2-integration-into-tt-metal)
3. [Pros, Cons, and Maintainability](#3-pros-cons-and-maintainability)

---

## 1. tt-emule Overview

### Purpose

tt-emule is a standalone C++ software emulator for Tenstorrent device-level APIs. It emulates the multi-core execution model of Tenstorrent hardware — including per-core L1 SRAM, shared DRAM, circular buffer (CB) synchronization, and the DST register file — entirely on the host CPU. This enables kernel development, testing, and debugging without access to physical silicon.

### Architecture

Each emulated core runs three concurrent threads mirroring hardware:

| Thread | Entry Point | Role |
|--------|-------------|------|
| NOC Reader | `reader_kernel_main()` | Reads data from DRAM/L1 into circular buffers |
| Compute | `compute_kernel_main()` | Processes tiles from input CBs, writes results to output CBs |
| NOC Writer | `writer_kernel_main()` | Writes computed results from CBs back to DRAM/L1 |

Threads synchronize through circular buffers backed by mutex + condition variable pairs (`CBSyncState`). The compute thread operates on a private DST register file (8 slots × 1024 float32 elements) that mirrors the hardware tile register space.

### Memory Model

All memory is owned by `tt_emule::Core` objects. Each Core mmaps a contiguous region below 4 GB using `MAP_32BIT`, enabling JIT-compiled kernels to use truncated 32-bit addresses as hardware-style pointers.

```cpp
enum class CoreRole { WORKER, DRAM };

class Core {
    // MAP_32BIT mmap'd region — addressable with uint32_t
    uint8_t*  l1_      = nullptr;
    uint32_t  l1_base_ = 0;        // truncated address for kernel use
    size_t    l1_size_ = L1_SIZE;   // 1 MB (WORKER) or bank size (DRAM)
    CoreRole  role_    = CoreRole::WORKER;

    // Per-core CB synchronization array (32 slots)
    CBSyncState cb_sync_states_[MAX_CBS] = {};

    // Compute engine state
    DstRegisterFile dst_;
};
```

Worker cores use L1-sized regions (1 MB); DRAM cores use bank-sized regions (e.g. 256 MB). The `get_core(tt_xy_pair)` factory lazy-creates with the appropriate role.

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
    std::condition_variable space_cv;  // producer waits for free space
    std::condition_variable data_cv;   // consumer waits for data
};
```

Four inline operations (`cb_sync_reserve`, `cb_sync_push`, `cb_sync_wait`, `cb_sync_pop`) implement the producer-consumer protocol. Both standalone and JIT kernel paths use the same `CBSyncState` struct, eliminating the earlier duplication of ~80 lines of synchronization code.

### Codebase Structure

```
tt-emule/                           (~4,660 lines across 65 files)
├── include/
│   ├── tt_emule/       (10 files)  Host-side types: Device, Core, Buffer, Program,
│   │                               CircularBuffer, CBSyncState, DstRegisterFile
│   ├── kernel_api/      (4 files)  Standalone device-side API (links tt_emule directly)
│   ├── jit_hw/         (33 files)  JIT kernel stubs (resolved via dlopen + -rdynamic)
│   │   ├── api/compute/            15 SFPU op headers (INT32, float, bitwise, etc.)
│   │   ├── api/dataflow/           NOC read/write with uint32_t addresses
│   │   ├── api/tensor/             TensorAccessor for page-based addressing
│   │   └── experimental/           CoreLocalMem, Noc endpoint stubs
│   └── ttkernel/        (4 files)  Forwarding headers for tt-metal kernel include paths
├── src/                 (3 files)  host_api.cpp, kernel_runner.cpp, jit_kernel.cpp
└── tests/              (15 files)  eltwise_add, matmul, tilize, compat tests
```

### Dual API Paths

tt-emule provides two parallel API surfaces for kernel code:

| Aspect | `kernel_api/` (Standalone) | `jit_hw/` (JIT for tt-metal) |
|--------|---------------------------|------------------------------|
| **Linking** | Static link to `tt_emule_lib` | dlopen'd `.so`, symbols via `-rdynamic` |
| **L1 addresses** | `uint8_t*` (host pointers) | `uint32_t` (truncated from mmap below 4 GB) |
| **CB backing** | `tt_emule::CircularBuffer` | `CBSyncState` array on `tt_emule::Core` |
| **Tile format** | Float32 (4096 bytes/tile) | bfloat16 (2048 bytes) or INT32 (4096 bytes) |
| **Context** | Thread-local `__core`, `__device` | Same, plus bridge pointers for DRAM resolution |

Both paths share a single `CBSyncState` struct and `cb_sync_*` free functions.

### Test Results

**Standalone tests** (3/3 pass): eltwise_add, matmul, tilize

**tt-metal emulated tests** (25/25 pass):
- `test_ttnn_add` — bfloat16 element-wise addition (1 test case)
- `test_ttnn_add_int` — INT32 addition (8 test cases)
- `test_ttnn_sub_int` — INT32 subtraction (8 test cases)
- `test_ttnn_rsub_int` — INT32 reverse subtraction (8 test cases)

---

## 2. Integration into tt-metal

### Design Philosophy: Zero Fake Headers, Memory Isolation

The integration follows two key principles:

1. **Zero fake headers.** Tests link the real `Metalium::Metal` library and use real tt-metal headers, fixtures, and buffer utilities. Emulation is injected at the UMD (User-Mode Driver) boundary, not via mock headers.

2. **Memory isolation in `tt_emule::Core`.** All device memory — worker L1, DRAM banks — is owned by `tt_emule::Core` objects inside `SWEmulatedChip`. There is no intermediate copy-in/copy-out stage. The program runner uses Core's mmap'd memory directly, and it persists across program runs.

### Integration Layers

The integration touches tt-metal at four layers:

#### Layer 1: UMD Device Injection — SWEmulatedChip

```
tt_metal/third_party/umd/device/
├── api/umd/device/chip/sw_emulated_chip.hpp   (61 lines)
└── chip/sw_emulated_chip.cpp                   (88 lines)
```

`SWEmulatedChip` extends `MockChip` with memory-backed I/O. All non-memory operations (barriers, resets, power management) remain as no-ops from `MockChip`.

```cpp
class SWEmulatedChip : public MockChip {
    std::unordered_map<tt_xy_pair, std::unique_ptr<tt_emule::Core>> cores_;
    std::unordered_map<tt_xy_pair, uint32_t> dram_core_to_channel_;
    uint32_t l1_size_;
    uint64_t dram_bank_size_;
    std::mutex core_mutex_;
};
```

Key design decisions:
- **Lazy core creation.** `get_core(tt_xy_pair)` creates a `tt_emule::Core` on first access with the correct role (`WORKER` or `DRAM`) and memory size.
- **Forward declaration.** The header forward-declares `tt_emule::Core` rather than including `device.hpp`, avoiding UMD include path contamination. The out-of-line destructor `~SWEmulatedChip() = default` is defined in the `.cpp` where the full type is visible (required for `unique_ptr<Core>` deletion).
- **DRAM lookup.** The constructor pre-builds `dram_core_to_channel_` from the SOC descriptor for O(1) role determination.
- **Memory I/O.** All `read_from_device` / `write_to_device` overrides delegate to `get_core(xy)->l1_ptr(offset)` + `memcpy` — uniform for both worker and DRAM cores.

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
TT_METAL_EMULATED_MODE=1              → TargetDevice::Emulated + force slow dispatch
TT_METAL_MOCK_CLUSTER_DESC_PATH=...   → SOC descriptor for core/DRAM topology
TT_METAL_SLOW_DISPATCH_MODE=1         → Required (no HWCommandQueue in emulation)
```

In `rtoptions.cpp`:

```cpp
case EnvVarID::TT_METAL_EMULATED_MODE:
    this->runtime_target_device_ = tt::TargetDevice::Emulated;
    this->using_slow_dispatch = true;
    this->fast_dispatch = false;
    break;
```

In `tt_cluster.cpp`, the emulated target creates a UMD `Cluster` with `ChipType::EMULATED`, which routes to `SWEmulatedChip`.

#### Layer 3: JIT Kernel Execution

The emulated program runner is the core integration component:

```
tt_metal/impl/emulation/
├── emulated_program_runner.hpp   (49 lines)
└── emulated_program_runner.cpp  (393 lines)
```

**JIT Compilation Pipeline:**

1. Kernel source → temp directory with `wrapper.cpp` (includes `jit_kernel_stubs.hpp` + kernel)
2. `g++ -std=c++17 -fPIC -shared -O1` with tt-emule and kernel directory include paths
3. Compile-time args passed as `-DKERNEL_COMPILE_TIME_ARGS=v0,v1,...`
4. `dlopen` + `dlsym("__emule_kernel_entry")` → `std::function<void()>`
5. Results cached by `(source_path : compile_args : defines)` key

**Program Execution — direct Core memory, no copies:**

```
Phase 1: Collect kernels per logical core from ProgramImpl
    For each HalProgrammableCoreType × each kernel:
        JIT compile (with cache)
        Group by logical core with {fn, rt_args, common_rt_args}

Phase 2: Execute per core
    1. core = sw_emu->get_core(physical_xy)    // Real Core, L1 already mmap'd below 4GB
    2. core->reset_cb_sync()                    // Clear from previous run
    3. For each CB: core->init_cb_sync(...)     // From program's circular buffer config
    4. Set thread-local bridge pointers:
         __emule_bridge_l1   = core->l1_data()
         __emule_bridge_dram = dram_core->l1_data()
         __emule_cbs         = core->cb_sync_array()
         __core              = core
    5. Launch one thread per kernel
    6. Join all threads
    7. Done — no copy-back, no munmap
```

**Memory Bridge (JIT ↔ host process):**

Thread-local pointers and `extern "C"` bridge functions are exported via `-rdynamic` so that dlopen'd `.so` files resolve them at load time:

```cpp
thread_local uint8_t*          __emule_bridge_l1   = nullptr;
thread_local uint8_t*          __emule_bridge_dram = nullptr;
thread_local __emule_cb_state* __emule_cbs         = nullptr;
thread_local tt_emule::Core*   __core              = nullptr;

extern "C" uint8_t* __emule_dram_ptr(uint64_t offset);
extern "C" uint8_t* __emule_noc_resolve(uint32_t x, uint32_t y, uint64_t addr);
```

This avoids the ABI hazard of JIT kernels inlining any `Device` or `Core` methods — kernels only touch opaque `extern "C"` functions that resolve at dlopen time.

#### Layer 4: Dispatch Interception

In `tt_metal.cpp`, `LaunchProgram()` checks the target device type:

```cpp
#ifdef TT_METAL_EMULATION
if (MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Emulated) {
    detail::CompileProgram(device, program);
    detail::WriteRuntimeArgsToDevice(device, program, /*force_slow_dispatch=*/true);
    emule::execute_program_emulated(device, program);
    return;
}
#endif
```

This is the single branch point where emulated execution diverges from silicon dispatch. `CompileProgram` and `WriteRuntimeArgsToDevice` still run to populate `ProgramImpl` data structures; only the hardware dispatch path is bypassed.

### Guard Pattern: `is_mock_or_emulated()`

Throughout tt-metal, hardware-specific operations are guarded with `is_mock_or_emulated()` checks:

```cpp
bool is_mock_or_emulated() const {
    return this->target_type_ == tt::TargetDevice::Mock ||
           this->target_type_ == tt::TargetDevice::Emulated;
}
```

38 call sites across the codebase, organized by subsystem:

| Subsystem | Files | Guards | Operations Skipped |
|-----------|-------|--------|--------------------|
| Dispatch queues | `fd_mesh_command_queue.cpp`, `sd_mesh_command_queue.cpp` | 12 | HW queue operations |
| Firmware | `risc_firmware_initializer.cpp` | 5 | RISC firmware load |
| Context init | `metal_context.cpp` | 4 | Fabric, FW, watcher, dprint |
| Program dispatch | `program.cpp`, `dispatch.cpp` | 4 | Dispatch data population |
| Device init | `device.cpp`, `device_manager.cpp` | 3 | HAL/HW initialization |
| Fabric | `control_plane.cpp`, `channel_trimming_export.cpp` | 3 | Fabric control plane |
| Buffer/event dispatch | `buffers/dispatch.cpp`, `event/dispatch.cpp` | 2 | HW buffer/event ops |
| Other | `system_memory_manager.cpp`, `prefetch.cpp`, `core_descriptor.cpp` | 3 | Misc HW operations |

### Architecture Stack

```
tt-metal API (LaunchProgram)
    └── [TT_METAL_EMULATION guard] execute_program_emulated()
            ├── JIT compile: kernel .cpp → .so via g++ + dlopen
            ├── Per-core: SWEmulatedChip::get_core() → tt_emule::Core*
            ├── Init CB sync states from ProgramImpl
            └── Launch N threads (reader + compute + writer), set TLS context

tt::umd::SWEmulatedChip (extends MockChip)
    └── Per-{x,y} lazy-created tt_emule::Core with MAP_32BIT mmap
    └── write_to_device / read_from_device → memcpy to/from Core::l1_ptr()

tt_emule::Core
    ├── MAP_32BIT mmap region (WORKER=1MB, DRAM=bank_size)
    ├── CBSyncState[32] for multi-thread CB sync
    ├── DstRegisterFile for compute thread
    └── l1_base_addr() → 32-bit truncated address for JIT kernels
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

Key CMake details:
- `-rdynamic` on the `tt_metal` target exports thread-local bridge symbols to dlopen'd JIT kernels
- `TT_EMULE_JIT_INCLUDE_DIR` and `TT_EMULE_INCLUDE_DIR` are compile-time defines for JIT include paths
- `TT_EMULE_PROJECT_SOURCE_DIR` provides the project root for ttnn kernel includes
- UMD's `device` library gets `${TT_EMULE_PATH}/include` as a private include directory (only when `TT_METAL_EMULATION` is set)

### Test Infrastructure

Tests in `tt_emule/` are organized in tiers:

| Tier | Tests | Description |
|------|-------|-------------|
| 1a | test_bit_utils (4 cases) | Pure host computation, no device |
| 1b | 9 CoreRange tests | API-level testing |
| 1c | test_host_buffer | RAII buffer management |
| 1d | test_tilize_untilize, test_blockfloat_common | Data format conversion |
| 2 | test_simple_{l1,dram}_buffer | MeshDeviceFixture with SWEmulatedChip |
| 3 | test_ttnn_{add,add_int,sub_int,rsub_int} | Full ttnn eltwise ops with JIT kernels |

Tests use standard `MeshDeviceFixture` and `buffer_test_utils.hpp` from tt-metal — no custom fixtures.

---

## 3. Pros, Cons, and Maintainability

### Pros

**Development velocity without hardware.** Developers can write, compile, and test kernels on any x86 Linux machine. The full loop — kernel compilation, buffer I/O, multi-threaded execution with CB synchronization — runs in seconds without requiring a Tenstorrent card.

**Zero fake headers reduces drift.** By linking the real `Metalium::Metal` library and using real tt-metal headers, tests exercise the actual host API code paths. Header changes in tt-metal are immediately visible as compile errors in emulated tests, rather than silently diverging behind a fake header layer.

**Memory isolation eliminates copy overhead.** With `tt_emule::Core` owning all memory (worker L1 and DRAM banks), the program runner no longer needs to copy data between backing stores and mmap'd regions. This removes the entire copy-in / copy-out lifecycle (`mmap_l1_low()` → memcpy in → kernel execution → memcpy out → munmap) that existed in the previous design. Core memory is persistent across program runs, matching hardware semantics more closely.

**Single memory owner simplifies reasoning.** In the v1 design, memory state was split across three owners: `EmulatedChip` (L1/DRAM vectors), `emulated_program_runner` (temporary mmap'd regions), and `tt_emule::Core` (standalone tests). In v2, `tt_emule::Core` is the single owner. The `SWEmulatedChip` at UMD level is a thin wrapper that delegates all I/O to Core objects.

**Minimal tt-metal footprint.** The integration touches ~12 files in tt-metal with narrowly scoped changes:
- 2 new files (`emulated_program_runner.{hpp,cpp}`)
- 2 new files (`sw_emulated_chip.{hpp,cpp}`)
- 38 guard additions (`is_mock_or_emulated()`)
- 1 dispatch branch in `tt_metal.cpp`
- Enum additions to existing types (`TargetDevice::Emulated`, `ChipType::EMULATED`)

**Runtime activation, not compile-time.** The same binary supports both silicon and emulated execution. Activation is purely via environment variables. Only the `TT_METAL_EMULATION=ON` flag compiles in the JIT runner; the emulated code path is dead code in non-emulation builds.

**Shared synchronization logic.** The `CBSyncState` + `cb_sync_*` consolidation ensures that circular buffer FIFO behavior is defined in one place and shared across standalone tests, JIT kernels, and the `kernel_api/` path.

**CB state embedded in Core.** The `cb_sync_states_[MAX_CBS]` array is directly inside `Core`, so its address is stable across kernel runs and requires no extra heap allocation. The program runner calls `core->init_cb_sync()` / `core->reset_cb_sync()` to configure CBs per program, keeping CB lifecycle tied to the memory owner.

### Cons

**Behavioral fidelity gap.** The emulator approximates hardware behavior but does not replicate it:
- NOC operations are synchronous `memcpy` (no latency modeling, no bandwidth contention)
- DST register file uses float32 internally; bfloat16 precision loss only occurs at pack/unpack boundaries
- No tile pipeline stalls, no resource conflicts, no firmware overhead
- Single-bank interleaved addressing only (multi-bank not yet implemented)

Tests that pass in emulation may fail on silicon due to timing, precision, or resource constraints.

**Dual API surface.** Despite consolidation, `kernel_api/` and `jit_hw/` remain separate API surfaces with different type conventions (`uint8_t*` vs `uint32_t`, `Tile` vs raw bfloat16). Adding a new kernel API function requires implementation in both paths.

**JIT compilation overhead.** Each unique kernel requires a `g++` invocation (typically 1–3 seconds). Results are cached within a process but not persisted across runs.

**`MAP_32BIT` dependency.** The JIT path relies on `mmap` with `MAP_32BIT` to ensure L1 pointers fit in `uint32_t`. This is Linux-specific and limits maximum addressable memory. It also assumes the 32-bit address space region is available, which may conflict with other large allocations (DRAM bank size, multiple devices).

**Format detection heuristic.** The `page_size > 2048` heuristic used by `copy_tile` and `pack_tile` to distinguish bfloat16 from 32-bit formats is fragile. It works for current tile sizes (bfloat16 = 2048 bytes, INT32/Float32 = 4096 bytes) but would break for new formats.

**No persistent JIT cache.** The in-memory JIT cache (`g_jit_cache`) is lost when the process exits. Test suites with many distinct kernels pay the full compilation cost on every invocation.

### Maintainability

**Low ongoing cost for tt-emule standalone.** The standalone emulator (~4,660 lines) is self-contained with no external dependencies beyond the C++ standard library and pthreads. Changes to tt-metal do not affect standalone tests.

**Medium ongoing cost for tt-metal integration.** The primary maintenance burden is keeping JIT stubs (`jit_hw/`) aligned with tt-metal kernel APIs. When tt-metal adds or changes a kernel API function, the corresponding stub must be created or updated. The current 33 `jit_hw/` files cover the most common operations but are far from complete.

**Rebase risk is bounded.** The tt-metal integration is concentrated in the `emulation/` directory, `SWEmulatedChip`, and `is_mock_or_emulated()` guards. Rebasing onto new tt-metal versions primarily requires:
1. Checking that `is_mock_or_emulated()` guards cover any new hardware-dependent code paths
2. Updating JIT stubs if kernel APIs change
3. Verifying that `ProgramImpl` and `CircularBuffer` interfaces haven't changed

**Memory owner consolidation simplifies debugging.** With `tt_emule::Core` as the single memory owner, there is one place to set breakpoints for memory access issues. The old three-way ownership (EmulatedChip vectors ↔ mmap'd regions ↔ standalone Core) required tracing data through copy stages to find corruption sources.

**Test maintenance scales with op coverage.** Each new ttnn operation requires: (1) a JIT stub header in `jit_hw/api/compute/`, (2) a test entry in `tt_emule/ttnn_tests/CMakeLists.txt`, and (3) verification that the kernel's include paths are covered by JIT compiler flags.

**Single point of truth for CB sync is a durability win.** Any bug fix or enhancement to circular buffer synchronization is made in one place (`CBSyncState` + `cb_sync_*` functions) and propagated to all consumers automatically.

### Changes from v1 to v2

| Aspect | v1 (EmulatedChip) | v2 (SWEmulatedChip + Core) |
|--------|-------------------|---------------------------|
| Memory owner | Split: EmulatedChip vectors + mmap'd temp regions + standalone Core | Single: `tt_emule::Core` owns all memory |
| I/O path | memcpy to/from `vector<uint8_t>` | memcpy to/from Core's mmap'd L1 |
| Program execution | mmap → copy-in → run → copy-out → munmap | Set bridge pointers → run → done |
| CB state | Locally allocated `CoreCBs` struct per run | Embedded `cb_sync_states_[32]` in Core |
| DRAM cores | Separate `dram_memory_` map in EmulatedChip | `tt_emule::Core` with `CoreRole::DRAM` |
| Name | `EmulatedChip` | `SWEmulatedChip` |
| mmap management | `mmap_l1_low()` per core per run + `munmap()` | Core constructor (once, persistent) |
| Lines removed | — | ~120 lines (mmap_l1_low, CoreCBs, copy-in/out, munmap) |

---

*Report generated 2026-03-10. Covers tt-emule integration at commit `6f37e12` (master) / tt-metal commit `79495f4c1b`.*
