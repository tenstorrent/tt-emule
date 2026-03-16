# Implementation Report: Software-Emulated Device (tt-emule) Integration into tt-metal

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

### Codebase Structure

```
tt-emule/                           (~3,750 lines across 64 files)
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
| **CB backing** | `tt_emule::CircularBuffer` | `__emule_cb_state` (mmap'd raw memory) |
| **Tile format** | Float32 (4096 bytes/tile) | bfloat16 (2048 bytes) or INT32 (4096 bytes) |
| **Context** | Thread-local `__core`, `__device` | Same, plus bridge pointers for DRAM resolution |

Both paths share a single `CBSyncState` struct and `cb_sync_*` free functions for circular buffer FIFO logic, eliminating what was previously ~80 lines of duplicated synchronization code.

### Test Results

**Standalone tests** (4/4 pass): eltwise_add, matmul, tilize, tt_metal_compat

**tt-metal emulated tests** (25/25 pass):
- `test_ttnn_add` — bfloat16 element-wise addition (1 test case)
- `test_ttnn_add_int` — INT32 addition (8 test cases)
- `test_ttnn_sub_int` — INT32 subtraction (8 test cases)
- `test_ttnn_rsub_int` — INT32 reverse subtraction (8 test cases)

---

## 2. Integration into tt-metal

### Design Philosophy: Zero Fake Headers

The integration follows a "zero fake headers" approach. Rather than creating mock versions of tt-metal headers, tt-emule injects an emulated device at the UMD (User-Mode Driver) boundary. Tests link the real `Metalium::Metal` library and use real tt-metal headers, fixtures, and buffer utilities. Emulation is activated at runtime via environment variables, not at compile time.

### Integration Points

The integration touches tt-metal at four layers:

#### Layer 1: UMD Device Injection

`EmulatedChip` extends `MockChip` with memory-backed I/O:

```
tt_metal/third_party/umd/device/chip/
├── emulated_chip.hpp    (73 lines)   Memory-backed chip interface
└── emulated_chip.cpp   (103 lines)   Lazy-allocated L1 + DRAM backing stores
```

- Reads L1 and DRAM bank sizes from SOC descriptor
- Lazily allocates `vector<uint8_t>` backing stores per core/channel on first access
- All `read_from_device` / `write_to_device` operations are `memcpy` on backing stores

New enum values were added: `TargetDevice::Emulated` and `ChipType::EMULATED`.

#### Layer 2: Runtime Activation

Activation is controlled by environment variables and propagated through runtime options:

```
TT_METAL_EMULATED_MODE=1              → TargetDevice::Emulated + force slow dispatch
TT_METAL_MOCK_CLUSTER_DESC_PATH=...   → SOC descriptor for core/DRAM topology
TT_METAL_SLOW_DISPATCH_MODE=1         → Required (no HWCommandQueue in emulation)
```

`rtoptions.cpp` detects `TT_METAL_EMULATED_MODE` and sets the target device type. `tt_cluster.cpp` then creates UMD `Cluster` instances with `ChipType::EMULATED`, which instantiates `EmulatedChip` objects instead of `SiliconChip` or `MockChip`.

#### Layer 3: JIT Kernel Execution

The emulated program runner (`emulated_program_runner.cpp`, 461 lines) is the core integration component:

```
tt_metal/impl/emulation/
├── emulated_program_runner.hpp   (49 lines)   Public API declarations
└── emulated_program_runner.cpp  (461 lines)   JIT compiler + multi-threaded executor
```

**JIT Compilation Pipeline:**
1. Kernel source → temp directory with `wrapper.cpp` (includes `jit_kernel_stubs.hpp` + kernel)
2. `g++ -std=c++17 -fPIC -shared -O1` with tt-emule include paths
3. Compile-time args passed as `-DKERNEL_COMPILE_TIME_ARGS=v0,v1,...`
4. `dlopen` + `dlsym("__emule_kernel_entry")` → `std::function<void()>`
5. Results cached by `(source_path : compile_args : defines)` key

**Program Execution:**
1. Group kernels by logical core from `ProgramImpl`
2. Per core: mmap L1-sized region with `MAP_32BIT` (ensures 32-bit pointer compatibility)
3. Copy EmulatedChip L1 → mmap'd region
4. Initialize `__emule_cb_state` array from program's circular buffer config
5. Launch one thread per kernel, setting thread-local context (`__rt_args`, `__core`, bridge pointers)
6. Join all threads
7. Copy mmap'd L1 → EmulatedChip
8. `munmap`

**Memory Bridge:** Thread-local pointers (`__emule_bridge_l1`, `__emule_bridge_dram`) and `extern "C"` bridge functions (`__emule_dram_ptr`, `__emule_noc_resolve`) are resolved at `dlopen` time via `-rdynamic`, allowing JIT-compiled kernels to access emulated memory without linking tt-emule directly.

#### Layer 4: Dispatch Interception

In `tt_metal.cpp`, `LaunchProgram()` checks the target device type:

```cpp
if (device->get_target_device_type() == TargetDevice::Emulated) {
    // Compile program, write runtime args
    emule::execute_program_emulated(device, program);
    return;  // Skip normal hardware dispatch
}
```

This is the single branch point where emulated execution diverges from silicon dispatch. All upstream API calls (buffer allocation, program construction, runtime args) follow the normal tt-metal code paths.

### Guard Pattern: `is_mock_or_emulated()`

Throughout tt-metal, hardware-specific operations (fabric setup, firmware initialization, watcher/dprint server management) are guarded with `is_mock_or_emulated()` checks. This pattern skips operations that require physical hardware while allowing all software-layer operations to proceed normally.

### CMake Configuration

Two independent flags control the build:

| Flag | Scope | Effect |
|------|-------|--------|
| `TT_METAL_EMULATION` | tt_metal library | Compiles `emulated_program_runner.cpp`, enables EmulatedChip, defines JIT include paths |
| `TT_METAL_USE_TT_EMULE` | Top-level build | Adds `tt_emule/` test subdirectory to build |

```bash
cmake -B build_emule \
  -DTT_METAL_USE_TT_EMULE=ON \
  -DTT_METAL_EMULATION=ON \
  -DTT_EMULE_PATH=/path/to/tt-emule \
  -DWITH_PYTHON_BINDINGS=OFF -DENABLE_TRACY=OFF -DTT_INSTALL=OFF
```

### Test Infrastructure

Tests in `tt_emule/` are organized in tiers:

| Tier | Tests | Description |
|------|-------|-------------|
| 1a | test_bit_utils | Pure host computation, no device |
| 1b | 9 CoreRange tests | API-level testing |
| 1c | test_host_buffer | RAII buffer management |
| 1d | test_tilize_untilize, test_blockfloat_common | Data format conversion |
| 2 | test_simple_{l1,dram}_buffer | MeshDeviceFixture with EmulatedChip |
| 3 | test_ttnn_{add,add_int,sub_int,rsub_int} | Full ttnn eltwise ops with JIT kernels |

Tests use standard `MeshDeviceFixture` and `buffer_test_utils.hpp` from tt-metal — no custom fixtures.

---

## 3. Pros, Cons, and Maintainability

### Pros

**Development velocity without hardware.** Developers can write, compile, and test kernels on any x86 Linux machine. The full loop — kernel compilation, buffer I/O, multi-threaded execution with CB synchronization — runs in seconds without requiring a Tenstorrent card.

**Zero fake headers reduces drift.** By linking the real `Metalium::Metal` library and using real tt-metal headers, tests exercise the actual host API code paths. Header changes in tt-metal are immediately visible as compile errors in emulated tests, rather than silently diverging behind a fake header layer.

**Minimal tt-metal footprint.** The integration touches ~10 files in tt-metal with narrowly scoped changes:
- 2 new files (emulated_program_runner.{hpp,cpp})
- 2 new files (emulated_chip.{hpp,cpp})
- ~5 guard additions (`is_mock_or_emulated()`)
- 1 dispatch branch in `tt_metal.cpp`
- Enum additions to existing types

This makes rebasing onto new tt-metal versions manageable.

**Runtime activation, not compile-time.** The same binary supports both silicon and emulated execution. No separate "emulation build" of tt-metal is needed — only the additional `TT_METAL_EMULATION=ON` flag compiles in the JIT runner, and activation is purely via environment variables.

**Shared synchronization logic.** The `CBSyncState` + `cb_sync_*` consolidation ensures that circular buffer FIFO behavior is defined in one place and shared across standalone tests, JIT kernels, and the kernel_api path. This eliminates a class of bugs where the two implementations could diverge.

**Incrementally extensible.** New compute ops are added as single-file headers in `jit_hw/api/compute/` with no changes to the framework. The SFPU stub pattern (DST-to-DST operations with format-aware load/store) is well-established and has been applied consistently across 15 operations.

### Cons

**Behavioral fidelity gap.** The emulator approximates hardware behavior but does not replicate it:
- NOC operations are synchronous `memcpy` (no latency modeling, no bandwidth contention)
- DST register file uses float32 internally; bfloat16 precision loss only occurs at pack/unpack boundaries, not during intermediate computation
- No tile pipeline stalls, no resource conflicts, no firmware overhead
- Single-bank interleaved addressing only (multi-bank not yet implemented)

Tests that pass in emulation may fail on silicon due to timing, precision, or resource constraints that the emulator does not model.

**Dual API surface.** Despite consolidation efforts, `kernel_api/` and `jit_hw/` remain separate API surfaces with different type conventions (`uint8_t*` vs `uint32_t`, `Tile` vs raw bfloat16). Adding a new kernel API function requires implementation in both paths, and the two can still diverge in subtle ways (e.g., pointer arithmetic, format handling).

**JIT compilation overhead.** Each unique kernel requires a `g++` invocation (typically 1–3 seconds). While results are cached within a process, the cache is not persisted across runs. Test suites with many distinct kernels pay this cost on every invocation.

**`MAP_32BIT` dependency.** The JIT path relies on `mmap` with `MAP_32BIT` to ensure L1 pointers fit in `uint32_t`. This is a Linux-specific flag and limits maximum L1 size to ~2 GB. It also assumes the 32-bit address space region is available, which may conflict with other large allocations.

**Format detection heuristic.** The `page_size > 2048` heuristic used by `copy_tile` and `pack_tile` to distinguish bfloat16 from 32-bit formats is fragile. It works for the current tile sizes (bfloat16 = 2048 bytes, INT32/Float32 = 4096 bytes) but would break if new formats with different tile sizes are introduced.

### Maintainability

**Low ongoing cost for tt-emule standalone.** The standalone emulator (3,750 lines) is self-contained with no external dependencies beyond the C++ standard library and pthreads. Changes to tt-metal do not affect standalone tests.

**Medium ongoing cost for tt-metal integration.** The primary maintenance burden is keeping JIT stubs (`jit_hw/`) aligned with tt-metal kernel APIs. When tt-metal adds or changes a kernel API function, the corresponding stub must be created or updated. This is manual work that requires understanding both the hardware semantics and the emulation approximation. The current inventory of 33 jit_hw files covers the most common operations but is far from complete.

**Rebase risk is bounded.** The tt-metal integration is concentrated in the `emulation/` directory and `is_mock_or_emulated()` guards. The dispatch interception is a single branch. Rebasing onto new tt-metal versions primarily requires:
1. Checking that `is_mock_or_emulated()` guards cover any new hardware-dependent code paths
2. Updating JIT stubs if kernel APIs change
3. Verifying that `ProgramImpl` and `CircularBufferConfig` interfaces haven't changed

**Test maintenance scales with op coverage.** Each new ttnn operation requires: (1) a JIT stub header in `jit_hw/api/compute/`, (2) a test entry in `tt_emule/ttnn_tests/CMakeLists.txt`, and (3) verification that the kernel's include paths are covered by JIT compiler flags. The pattern is mechanical but must be done per-op.

**Single point of truth for CB sync is a durability win.** The `CBSyncState` consolidation means that any bug fix or enhancement to circular buffer synchronization (e.g., adding deadlock detection, changing the blocking strategy) is made in one place and propagated to all consumers automatically.

---

*Report generated 2026-03-10. Covers tt-emule at commit `6f37e12` (master branch).*
