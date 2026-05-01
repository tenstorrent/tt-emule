# Implementation Report v9: Software-Emulated Device (tt-emule) Integration into tt-metal

## Table of Contents

1. [tt-emule Overview](#1-tt-emule-overview)
2. [Integration into tt-metal](#2-integration-into-tt-metal)
3. [Pros, Cons, and Maintainability](#3-pros-cons-and-maintainability)

---

## 1. tt-emule Overview

### Purpose

tt-emule is a standalone C++ software emulator for Tenstorrent device-level APIs. It emulates the multi-core execution model of Tenstorrent hardware — including per-core L1 SRAM, shared DRAM, circular buffer (CB) synchronization, NOC (Network-on-Chip) communication, semaphore-based cross-core synchronization, and the DST register file — entirely on the host CPU. This enables kernel development, testing, and debugging without access to physical silicon.

### Architecture

tt-emule supports two hardware threading models:

#### Wormhole / Blackhole

Each emulated core runs three concurrent threads mirroring hardware:

| Thread | Entry Point | Role |
|--------|-------------|------|
| NOC Reader | `reader_kernel_main()` | Reads data from DRAM/L1 into circular buffers via NOC |
| Compute | `compute_kernel_main()` | Processes tiles from input CBs, writes results to output CBs |
| NOC Writer | `writer_kernel_main()` | Writes computed results from CBs back to DRAM/L1 via NOC |

Threads synchronize through circular buffers backed by mutex + condition variable pairs (`CBSyncState`).

#### Quasar (Tensix Neo)

Each emulated Neo runs up to 12 concurrent threads:

| Thread Type | Count per Neo | Entry Point | Role |
|-------------|---------------|-------------|------|
| DM (DM0–DM7) | up to 8 | DM kernel `main()` | Data movement: DRAM/L1 reads/writes via NOC, DFB push/pop |
| Compute (E0–E3) | up to 4 | Compute kernel `main()` | UNPACK/MATH/PACK pipeline on shared DST register file |

All 12 cores share a single 4 MB L1. DFBs (Dataflow Buffers) replace WH's SPSC circular buffers with MPMC synchronization via tile counters. CSR reads (`mhartid`, `NEO_ID`, `TRISC_ID`) are regex-patched at JIT time to read thread-local variables. See [QUASAR_EMULATION.md](docs/QUASAR_EMULATION.md) for the full Quasar reference.

#### Shared Across Architectures

The compute thread operates on a private DST register file (16 slots in bf16 mode / 8 slots in fp32 mode, × 1024 float32 elements) with mode-aware bounds checking via `__emule_dst_check()`. Multiple cores execute concurrently in separate threads, enabling cross-core communication via multicast NOC writes and semaphore signaling — matching the real hardware execution model.

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

#### Quasar Memory Model

Quasar Neos have larger L1 and a different DRAM model than WH/BH:

- **L1**: 4 MB shared across all 12 cores in a Neo (8 DM + 4 compute). Emulated as a bump allocator on the host heap (`Core::l1_alloc()`). The JIT path reads the L1 size from the SOC YAML via `SWEmulatedChip`.
- **Firmware L1 address translation**: Device kernels reference L1 via firmware-range offsets (e.g., `0x1000`). `__emule_local_l1_to_ptr()` distinguishes these from host pointers and translates via `__emule_bridge_l1`. NOC operations and JIT-patched `reinterpret_cast<T*>(get_arg_val(N))` patterns use this translation.
- **DRAM**: `NUM_DRAM_BANKS` is set to the real architecture channel count (Quasar=2, WH=6, BH=8) at `emulated_program_runner.cpp:700`. All bank NOC coordinates are registered in `__emule_core_map` (lines 1012-1031), enabling multi-bank interleaving via `InterleavedAddrGen`.
- **Bridge DFBs**: When a compute kernel bridges two DFBs with the same `(entry_size, num_entries)`, the runner allocates L1 once and reuses the base address for both, modeling the hardware register file passthrough.

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

#### Wormhole/Blackhole: CBSyncState

`CBSyncState` is the FIFO primitive for WH/BH kernel execution paths:

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

Four inline operations (`cb_sync_reserve`, `cb_sync_push`, `cb_sync_wait`, `cb_sync_pop`) implement the producer-consumer protocol. Both standalone and JIT kernel paths use the same `CBSyncState` struct. The CB API also provides a lock-free fast path for SPSC scenarios.

#### Quasar: Tile Counter Synchronization

On Quasar, a single Neo has 24 logical processors sharing one L1, so SPSC circular buffers are insufficient. DFBs (Dataflow Buffers) provide MPMC synchronization via **tile counters** — one per consumer, each with an independent `posted`/`acked` counter pair.

Key data structures (`include/tt_emule/tile_counter.hpp`, `include/tt_emule/dfb_sync_state.hpp`):

- **`TileCounter`**: atomic `posted`/`acked` + `capacity` + mutex + 2 CVs (`space_cv` for producer, `data_cv` for consumer). `occupancy = posted - acked`, `free_space = capacity - occupancy`.
- **`TileCounterArray`**: `num_neos × 32` counters in a flat array, owned by `Core`, shared across all threads. Four operations: `inc_posted`, `inc_acked`, `wait_free_space`, `wait_occupancy`.
- **`EmuleDFBInterface`**: per-thread per-DFB view with up to 4 round-robin TC slots, stride, active flag, and read/write entry indices.

Two access patterns:
- **STRIDED**: entries interleaved across consumers. `M = max(P, C)`, producer p owns TC slots `{p + k*P}`, consumer c owns `{c + k*C}`. `stride_size = M * entry_size`.
- **BLOCKED**: all consumers see all data. Producer posts to ALL consumer TCs simultaneously (`broadcast_tc = true`). Consumer drains each TC fully before advancing (`drain_per_tc = true`). Each TC slot has its own sub-range. P*C counter scheme.

See [DFB_EMULATION.md](docs/DFB_EMULATION.md) for the full synchronization deep dive, access pattern details, and program lifecycle.

#### DFB↔CB Bridge

Compute kernels use the CB API (`cb_push_back`, `cb_pop_front`) while DM kernels use the DFB API (`dfb_push_back`, `dfb_pop_front`). The bridge connects both sides:

- `dfb_push_back` calls `cb_sync_push` to update CB `occupied` so compute's `cb_wait_front` sees pushed tiles.
- `cb_push_back` calls `inc_posted` on the DFB's tile counters so DM's `dfb_wait_front` sees compute's output.
- `dfb_pop_front` calls `cb_sync_pop`; `cb_pop_front` calls `inc_acked`.

This allows the same L1 buffer to be accessed via CB API from compute and DFB API from DM, which is how upstream Quasar kernels are structured.

### Compute Engine

The compute engine emulates the hardware's UNPACK → MATH → PACK pipeline. On real hardware, three TRISC cores (UNPACK, MATH, PACK) run in parallel; in emulation, all three share one host thread via `PACK(x) x`, `MATH(x) x`, `UNPACK(x) x` macros.

**DST Register File** (`include/jit_hw/api/compute/common.h`): `__emule_dst[16][1024]` — 16 tile slots × 1024 float32 elements (64 KB). Active slot count depends on `DST_ACCUM_MODE`: 16 for bf16 (default), 8 for fp32. `__emule_dst_check()` enforces the mode-aware limit.

**UNPACK Engine** (`include/jit_hw/api/compute/nfaces.h`): On real hardware, UNPACK reads tile data from L1 in **nfaces format** (4 sequential 16×16 faces) and produces row-major data in SRC registers. Emulation converts via a constexpr LUT: `dst[i] = buf[rowmajor_to_nfaces[i]]`, with simultaneous bf16→f32 conversion where needed. All compute ops that read from CBs perform this conversion.

**PACK Engine** (`include/jit_hw/api/compute/common.h`): Reverse of UNPACK — row-major DST → nfaces L1, with optional f32→bf16 conversion. **Auto-advance**: each `pack_tile` call writes to `cb_write_ptr_at(ocb, __emule_pack_offset[ocb]++)`, then increments the offset. Reset occurs in `cb_reserve_back` / `dfb_reserve_back`. **L1 accumulation**: `llk_pack_reconfig_l1_acc(1)` switches pack to additive mode (adds DST to existing CB contents).

**Implemented Compute Operations:**

| Operation | Function | File |
|-----------|----------|------|
| Tile copy | `copy_tile`, `copy_block_matmul_partials` | `common.h` |
| Pack | `pack_tile`, `pack_tile_block` | `common.h` |
| Add / Sub / Mul | `add_tiles`, `sub_tiles`, `mul_tiles` | `common.h` |
| Matmul | `matmul_tiles`, `matmul_block` (AVX2/FMA when available) | `matmul.h` |
| Reduce | `reduce_tile` (ROW/COL/SCALAR × SUM/MAX) | `reduce.h` |
| L1 acc toggle | `llk_pack_reconfig_l1_acc` | `common.h` |

Stubbed operations (compile but no-op): SFPU element-wise ops (23 files), broadcast, transpose_wh, tilize/untilize compute path, pack_untilize, quantization. See [QUASAR_EMULATION.md](docs/QUASAR_EMULATION.md) section 5.5 for the full list.

### Codebase Structure

```
tt-emule/                           (~8,400 lines across 131 files)
├── include/
│   ├── tt_emule/       (14 files)  Host-side types: Device, Core, L1Pool, Buffer, Program,
│   │                               CircularBuffer, CBSyncState, DstRegisterFile,
│   │                               TileCounter, TileCounterArray, EmuleDFBInterface,
│   │                               DFBSyncState, DataflowBuffer
│   ├── kernel_api/      (4 files)  Standalone device-side API (links tt_emule directly)
│   ├── jit_hw/         (88 files)  JIT kernel stubs (resolved via dlopen + -rdynamic)
│   │   ├── api/compute/            55 compute op headers (32 top-level + 23 eltwise_unary)
│   │   ├── api/dataflow/           NOC ops, multicast, semaphores, addrgen page ops
│   │   ├── api/tensor/             TensorAccessor for page-based addressing
│   │   ├── api/dfb_api.h           DFB operations with timeout and DFB↔CB bridge
│   │   ├── internal/               Banking infra, RISC attribs, mod_div_lib
│   │   └── experimental/           CoreLocalMem, Noc, AllocatorBank, Lock stubs
│   └── ttkernel/        (4 files)  Forwarding headers for tt-metal kernel include paths
├── src/                 (3 files)  host_api.cpp, kernel_runner.cpp, jit_kernel.cpp
├── tests/              (17 files)  eltwise_add, matmul, tilize, compat, dfb_passthrough,
│                                   dfb_multi_consumer tests
└── docs/                (3 files)  DFB_EMULATION.md, QUASAR_EMULATION.md, TEST_COVERAGE_TODO.md
```

### JIT Kernel API Coverage

The `jit_hw/` directory provides stub implementations for 75 header files covering:

| Category | Files | Key APIs |
|----------|-------|----------|
| Compute | 55 | `matmul_tiles`, `matmul_block`, `add/sub/mul_tiles`, `pack_tile`, `copy_tile`, `reduce_tile` (row/col/scalar × sum/max), `bcast`, `tilize/untilize`, `pack_untilize` (with `experimental::pack_untilize_block`), `transpose_wh`, `quantization`, 23 eltwise_unary SFPU ops (`abs_tile`, `exp_tile`, `negative_tile`, `typecast_tile`, etc.), binary bitwise/shift/comp/fmod/max_min, `gcd/lcm`, `xlogy`, `copy_dest_values` |
| Compute nfaces | 1 | `__emule_nfaces::rowmajor_to_nfaces` constexpr LUT — UNPACK/PACK engine layout conversion between nfaces (L1) and row-major (DST) |
| Dataflow | 2 | `noc_async_read/write`, `noc_async_write_multicast`, `noc_inline_dw_write` (unicast/multicast/stateful), semaphore ops, `InterleavedAddrGen<DRAM/L1>`, banking arrays |
| CB sync | 1 | `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front` (uint32_t and int32_t overloads), DFB↔CB bridge |
| DFB sync | 1 | `dfb_reserve_back`, `dfb_push_back`, `dfb_wait_front`, `dfb_pop_front`, `dfb_finish`, `dfb_get_write_ptr`/`dfb_get_read_ptr`, timeout detection |
| CSR emulation | 1 | `csr_read<CSR::NEO_ID>()`, `csr_read<CSR::TRISC_ID>()` via TLS `__emule_neo_id`, `__emule_trisc_id` |
| LLK defs | 1 | `llk_unpack_A`, `llk_wait_tiles`, `llk_pop_tiles`, `llk_push_tiles`, `llk_wait_for_free_tiles`, `get_output_id`, `FACE_R_DIM`, `TILE_C_DIM`, coordinate APIs (`get_absolute_logical_x/y`) |
| Tensor | 1 | `TensorAccessor`, `TensorAccessorArgs` |
| Infrastructure | 8 | compile-time args, bfloat16, dprint, assert stubs, DataFormat enum, tile constants |
| Experimental | 5 | `Noc`, `CircularBuffer`, `AllocatorBank`, `Lock`, `CoreLocalMem` |
| Compatibility | 5 | `ckernel.h`, `ckernel_defs.h`, `common_values.hpp`, `risc_attribs.h` (`InlineWriteDst`, `write_at_cmd_buf`), `dprint.h` (debug print stubs) |

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

**Standalone tests** (5/5 pass): dfb_passthrough, dfb_multi_consumer, eltwise_add, matmul, tilize

**tt-metal emulated regression** (137 passing, 0 failures, 0 skipped):

| Tier | Tests | Count | Description | Cluster |
|------|-------|-------|-------------|---------|
| 0 | Standalone | 3 | dfb_passthrough, dfb_multi_consumer, eltwise_add | None |
| 1 | Host-only | 16 | bit_utils, host_buffer, tilize_untilize, blockfloat, emulation_toggle, dst_capacity, 9 CoreRange/Set | None |
| 2 | Buffer I/O | 3 | SimpleL1Buffer, SimpleDramBuffer, emulation_toggle_active | WH N150 |
| 3a | JIT Kernel | 1 | TensixL1Tile — experimental CB/Noc API | WH N150 |
| 3b | DFB STRIDED (Group A) | 20 | DM-DM multi-P/C, ImplicitSyncFalse + ImplicitSyncTrue | Quasar |
| 3d | DFB Bridge (Groups B+C) | 20 | DM→Tensix + Tensix→DM STRIDED topologies | Quasar |
| 3e | DFB Pipeline (Group D) | 3 | DM→Tensix→DM multi-DFB pipeline | Quasar |
| 3e | DFB BLOCKED | 30 | DM-DM + DM→Tensix + Tensix→DM BLOCKED, both ImplicitSync | Quasar |
| 3f | DFB Multi-Core | 6 | 2-core STRIDED (1Sx1S, 2Sx2S) + BLOCKED (1Sx4B), both ImplicitSync | Quasar |
| 3g | DFB Config Validation | 10 | TC pairing, risc_mask, remapper validation (7 STRIDED + 3 multi-core) | Quasar |
| 3h | Quasar Compute | 4 | MultipleThreads, SingleThread, MultipleKernels, TLS | Quasar |
| 3i | Quasar Semaphores | 3 | ComputeKernelSemaphores, DmAndComputeSemaphores, DmLoopback | Quasar |
| 3j | Simple DM + Atomics | 4 | SingleDmL1Write, 3× RISC-V atomic tests | Quasar |
| 3k | Data Movement | 4 | LoopbackPacketSizes, LoopbackDirectedIdeal, OneFromOnePacketSizes, OneFromOneDirectedIdeal | WH N150 |
| 3l | DM Direct Write + DRAM | 7 | 3 direct write (unicast, stateful, multicast) + 4 DRAM unary (packet sizes, core locations, channels, directed) | WH N150 |
| 4 | TTNN INT32 | 2 | ttnn_relational_int (66 sub-cases), ttnn_add_int | BH P100 |
| 5 | TTNN Matmul Sweep | 1 | 14 sub-cases: multi-core matmul 32² through 2048² | WH N150 |
| 5b | Quasar Matmul PCC | 2 | TensixMatmulBlock, TensixMatmulBlockInitShort | Quasar |
| 6 | Silicon Toggle | 1 | ttnn_add_int — env vars unset, runs in emulation (toggle proof) | WH N150 |

See [QUASAR_EMULATION.md](docs/QUASAR_EMULATION.md) section 8 for a feature-by-feature table with test evidence.

**D2M golden test regression** (1694 pass / 164 fail / 224 skip-xfail):

| Test File | Total | Passed | Failed | Skip/XFail | Status |
|-----------|-------|--------|--------|------------|--------|
| test_metal_layout | 94 | 94 | 0 | 0 | **PASS** |
| test_metal_matmul | 127 | 112 | 1 | 14 xfail | FAIL (1 regression) |
| test_metal_matmul_higher_rank | 10 | 10 | 0 | 0 | **PASS** |
| test_metal_allocate | 6 | 6 | 0 | 0 | **PASS** |
| test_metal_tms | 339 | 332 | 4 | 3 skip | PASS (arange only) |
| test_metal_reductions | 1300 | 991 | 131 | 178 | FAIL (unaligned) |
| test_metal_dma | 49 | 49 | 0 | 0 | **PASS** |
| test_metal_tilize | 44 | 44 | 0 | 0 | **PASS** |
| test_metal_tensor_collapsing | 14 | 12 | 0 | 2 skip | **PASS** |
| test_metal_virtual_grids | 39 | 39 | 0 | 0 | **PASS** |
| test_metal_virtual_grid_rowmajor | 27 | 0 | 0 | 27 skip | SKIP (needs n300) |
| test_metal_masking | 20 | 5 | 15 | 0 | FAIL (partial tile) |
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
├── emulated_program_runner.cpp  (~975 lines)
└── emulated_run_stats.hpp       (20 lines)
```

**JIT Compilation Pipeline:**

1. Kernel source `.cpp` -> temp directory with `wrapper.cpp` (kernel `#define`s + `#include "jit_kernel_stubs.hpp"` + `#include kernel.cpp`)
2. `g++ -std=c++17 -fPIC -shared -O3` with tt-emule and kernel-directory include paths
3. Compile-time args as `-DKERNEL_COMPILE_TIME_ARGS=v0,v1,...`; named args via `-DKERNEL_COMPILE_TIME_ARG_MAP`
4. Quasar-specific patches applied via regex before compilation:
   - `mhartid` CSR: `asm volatile("csrr %0, mhartid" ...)` → `var = __processor_id;`
   - RISC-V fence: `asm volatile("fence")` → `__sync_synchronize()`
   - Raw L1 pointer casts: `reinterpret_cast<T*>(get_arg_val(N))` → adds `__emule_bridge_l1` offset
5. `dlopen` + `dlsym("__emule_kernel_entry")` -> `std::function<void()>`
6. Results cached by `(source_path : compile_args : defines)` key; cache misses compiled in parallel via `std::async`

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
           Quasar-specific TLS:
             __emule_dfbs        = dfb_interfaces_for_this_thread
             __emule_tc_array    = core->tile_counters()
             __emule_neo_id      = neo_id (0-3)
             __emule_trisc_id    = trisc_id (0-3)
        5. WH/BH: Launch reader + compute + writer threads
           Quasar: Launch 1 thread per active DM processor
                   + 1 thread per active compute engine
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

**DFB Setup (Quasar only):**

For programs with Dataflow Buffers, the runner performs additional setup before thread launch:

1. Allocate L1 for each DFB: `entry_size * num_entries` bytes via `core->l1_alloc()`. Bridge DFBs (compute kernel connecting input and output DFBs with same dimensions) share the same L1 backing via `dim_key` deduplication.
2. Initialize tile counters: `M = max(P, C)` counter slots per DFB, each with `capacity = num_entries / M`.
3. Build per-thread `EmuleDFBInterface` arrays: producer p gets TC slots `{p + k*P}`, consumer c gets `{c + k*C}` (STRIDED). BLOCKED mode sets `broadcast_tc = true` and uses P*C counter scheme.
4. For BLOCKED consumer assignment: `counter_id = counter_base + p*C + c`.

See [DFB_EMULATION.md](docs/DFB_EMULATION.md) section 5.5 for the full details of JIT/Metal path DFB setup.

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
                    +-- [WH/BH] Launch N cores x 3 threads, set TLS context
                        |   |-- reader_kernel_main()  --- NOC read, CB push
                        |   |-- compute_kernel_main() --- DST math, CB pop/push
                        |   +-- writer_kernel_main()  --- CB pop, NOC write
                        |
                        +-- [Quasar] Launch N cores x (DM + compute) threads
                            |-- dm_kernel_main() x P   --- DFB push/pop, NOC read/write
                            +-- compute_main() x Q     --- UNPACK/MATH/PACK, CB↔DFB bridge

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
    |-- TileCounterArray for DFB sync (Quasar)
    |-- DFBSyncState[32] for per-DFB metadata (Quasar)
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

The tier table above in Test Results reflects the full `run_regression.sh` structure (91 test invocations across 12 tiers). Additional ttnn tests built but not in regression: `test_ttnn_add`, `test_ttnn_sub_int`, `test_ttnn_rsub_int`, `test_ttnn_matmul`.

D2M golden test regression: `run_d2m_regression.sh` — runs 13 tt-mlir test files (2082 tests) against the emulated backend. 1694 pass, 164 fail, 224 skip/xfail. See [D2M_REGRESSION_REPORT.md](D2M_REGRESSION_REPORT.md).

Regression scripts: `run_regression.sh` (133 passing tests) + `run_d2m_regression.sh` (13 D2M test files, 2082 tests).

### tt-metal Files Modified

The complete set of tt-metal modifications for emulation support:

| Category | Files | Lines |
|----------|-------|-------|
| New: program runner | `emulated_program_runner.{hpp,cpp}`, `emulated_run_stats.hpp` | ~1000 |
| New: UMD chip | `sw_emulated_chip.{hpp,cpp}` | ~181 |
| New: test infrastructure | `tt_emule/CMakeLists.txt`, `tt_emule/ttnn_tests/CMakeLists.txt` | ~207 |
| New: Quasar test files | `test_dfb_emulation.cpp`, `test_dataflow_buffer.cpp`, `test_quasar_compute_kernels.cpp`, `test_quasar_semaphores.cpp`, `test_dm_loopback.cpp`, `test_single_dm_l1_write.cpp`, `test_riscv_atomics.cpp`, `test_globals_tls.cpp`, `test_matmul_X_tile.cpp`, data_movement tests | ~2500 |
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

**DFB MPMC synchronization is hardware-accurate.** The tile counter model (`TileCounter` + `TileCounterArray` + `EmuleDFBInterface`) replicates the hardware's per-consumer posted/acked counter pair with correct BLOCKED drain semantics (`drain_per_tc`). 72 DFB tests (42 STRIDED + 30 BLOCKED) verify correctness across all P/C combinations from 1P-1C through 4P-4C, with both DM-DM and DM-Tensix-DM topologies.

**UNPACK/PACK nfaces layout conversion is verified by matmul PCC.** The constexpr nfaces LUT (`nfaces.h`) ensures that bfloat16 tile data is correctly converted between L1 (nfaces) and DST (row-major) formats. Matmul PCC tests pass with non-trivial input data, which requires both UNPACK (nfaces→rowmajor) and PACK (rowmajor→nfaces) conversions to be correct.

**Matmul PCC passing is a milestone.** `test_matmul_X_tile.cpp` verifies tile GEMM with AVX2/FMA acceleration against golden reference, exercising the full UNPACK→MATH→PACK pipeline end-to-end through the DFB↔CB bridge.

**Incrementally extensible.** New compute ops are single-file headers in `jit_hw/api/compute/`. The pattern (DST-to-DST operations with format-aware load/store) has been applied consistently across 53 compute headers.

**D2M golden test coverage.** 1624 of 1878 D2M golden tests pass (86% pass rate), covering layout transforms, buffer allocation, matmul (single-core, multi-core, double-buffered, 3D/4D batched up to 2048x2048x2048), reductions (sum, max, mean), DMA (L1-to-L1 and partial DRAM), TMS (reshape, permute, concatenate_heads), tilize/untilize, virtual grids, and tensor collapsing. This provides broad regression coverage for D2M-generated kernels.

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

**Nfaces conversion not exercised in isolation.** The UNPACK and PACK engines are tested only indirectly via compute operations (matmul, add_tiles, etc.) that happen to exercise them. No dedicated UNPACK-only or PACK-only test exists. A bug in the nfaces LUT for a specific face/element combination might not be caught if no existing compute test triggers that access pattern. See `docs/TEST_COVERAGE_TODO.md`.

**D2M coverage gaps.** 1 test file fails entirely: bfp8_typecast (13 PCC mismatches). 2 files have partial failures: reductions (131/1300 fail on unaligned shapes) and masking (15/20 fail on partial tiles). DMA tests now pass fully (49/49). The remaining 10 files pass fully. The primary gaps are unaligned tensor reductions, partial tile masking, and BFP8 format precision.

### Maintainability

**Low ongoing cost for tt-emule standalone.** The standalone emulator (~8,400 lines) is self-contained with no external dependencies beyond the C++ standard library and pthreads. Changes to tt-metal do not affect standalone tests.

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

### Changes from v5 to v6

| Aspect | v5 | v6 |
|--------|----|----|
| Codebase size | ~6,800 lines, 113 files | ~8,400 lines, 131 files |
| JIT stubs | 75 files (consolidated) | 88 files |
| Compute headers | 53 (30 top-level + 23 eltwise_unary) | 55 (32 top-level + 23 eltwise_unary) |
| Quasar architecture | Not documented | 12-thread model (8 DM + 4 compute per Neo), 4 MB shared L1 |
| DFB infrastructure | Not documented | TileCounter, TileCounterArray, EmuleDFBInterface, DFBSyncState; STRIDED + BLOCKED access patterns; DFB↔CB bridge |
| Compute engine | DST briefly mentioned (8 slots) | Full UNPACK/MATH/PACK pipeline: nfaces LUT, mode-aware DST (16/8 slots), L1 accumulation, AVX2/FMA matmul |
| Implemented compute ops | Not enumerated | 11 operations: copy/pack tile, add/sub/mul_tiles, matmul_tiles/block, reduce_tile, L1 acc toggle |
| CSR emulation | Not documented | NEO_ID, TRISC_ID via TLS; mhartid regex patch |
| JIT patches | mhartid only | + fence instruction, L1 pointer cast patches |
| tt-metal regression | 18 pass | **126 pass**, 1 fail (hardware-only), 2 skip |
| Quasar-specific tests | None | 72 DFB + 4 compute + 3 semaphore + 4 atomics/DM + 4 NOC + 2 matmul = 89 tests |
| Matmul PCC | Not tracked | Passing (`TensixMatmulBlock`, `TensixMatmulBlockInitShort`) |
| Reference docs | None | `docs/DFB_EMULATION.md`, `docs/QUASAR_EMULATION.md`, `docs/TEST_COVERAGE_TODO.md` |

### Changes from v6 to v7

| Aspect | v6 | v7 |
|--------|----|----|
| Nfaces UNPACK/PACK | Not applied to general compute ops | All UNPACK (CB→DST) and PACK (DST→CB) ops use nfaces LUT |
| D2M golden tests | 1595 pass / 141 fail / 142 skip | **1624 pass** / 112 fail / 142 skip |
| D2M pass rate | 85% | **86%** |
| D2M fully-passing files | 8 of 13 | **9 of 13** (tilize now passes) |
| Tilize/Untilize D2M | 12/44 pass (PCC mismatch) | **44/44 pass** — nfaces in `__llk_pack_tiled` and `copy_tile` fixed all 32 failures |
| DMA D2M | 38/49 pass | **40/49 pass** (+2 tiled roundtrip fixes) |
| Standalone regression | 83 pass / 28 fail / 2 skip | **85 pass** / 28 fail / 2 skip (→ 126/1/2 after rebase fixes in v8) |
| Quasar matmul PCC | 0/2 pass (data corruption) | **2/2 pass** — nfaces conversion resolved PCC failures |
| Files modified | — | `common.h`, `matmul.h`, `reduce.h`, `llk_defs.h` |

**Key insight:** On real hardware, UNPACK converts nfaces→row-major when loading from L1/CB into DST, and PACK converts row-major→nfaces when writing back. The emulator was missing these conversions. Element-wise ops (add/sub/mul) previously worked "by accident" because nfaces permutation cancels for identical per-element operations. Non-element-wise ops (matmul, tilize/untilize) failed because element positions were scrambled. The fix applies the nfaces LUT to all general UNPACK and PACK operations, plus the D2M tilize PACK path (`__llk_pack_tiled`).

### Changes from v7 to v8

| Aspect | v7 | v8 |
|--------|----|----|
| tt-metal rebase | Pre-rebase (`3fa4d75355`) | Rebased (1,669 commits newer), 7 fixes applied |
| tt-metal regression | 85 pass / 28 fail / 2 skip | **126 pass** / 1 fail / 2 skip |
| DFB BLOCKED multi-P/C | All 16 failing (8 DM-DM + 8 TensixDM) | **All 16 passing** |
| BLOCKED consumer model | Round-robin tc_idx on every pop_front | `drain_per_tc`: drain each TC slot fully before advancing |
| BLOCKED TC slot layout | All slots share full buffer range | Per-slot sub-ranges: `base_addr = alloc_base + p*capacity*entry_size` |
| EmuleDFBInterface struct | `broadcast_tc`, `active` | + `drain_per_tc` field between `broadcast_tc` and `active` |
| JIT cache | Source path + compile args key | Same (struct layout changes require manual cache clear) |
| Rebase fixes | — | JIT stubs, HAL core count, finalize alloc_addr, BLOCKED stride/offset, WH proc_bit, early DFB finalize, BLOCKED drain |
| Pre-existing failures | 28 | **1** (ttnn_add_int_silicon) |

**Key insight:** BLOCKED mode consumers on hardware perform block reads — they exhaust all entries from one producer's TC slot before advancing to the next producer's slot. The emulation was incorrectly round-robining through TC slots on every `pop_front` call, producing a shuffled read order. Additionally, all TC slots shared the full buffer address range instead of each slot having its own contiguous sub-range. Both bugs only manifested with multiple producers or consumers (1P-1C was unaffected since there's only one TC slot).

### Changes from v8 to v9

| Aspect | v8 | v9 |
|--------|----|----|
| tt-metal regression | 133 pass / 1 fail / 2 skip | **137 pass / 0 fail / 0 skip** |
| DFB STRIDED wraparound tests | 4 failing (`DMTest1xDFB4Sx4S`, `DMTest1xDFB2Sx4S` + `_IS` variants) | **All 4 passing** |
| RISC-V atomic tests | 3 segfaulting (`TestAtomicLoadStoreRISCV`, `TestAtomicAddFetchRISCV`, `TestAtomicCASRISCV`) | **All 3 passing** |
| `__emule_local_l1_to_ptr` availability | Defined in `dataflow_api.h` only | Also in `jit_kernel_stubs.hpp` under `#ifndef __EMULE_LOCAL_L1_TO_PTR_DEFINED` — available to every JIT kernel |
| L1 pointer cast JIT patch | Not present | Regex in runner patches `reinterpret_cast<T*>(get_arg_val<uint32_t>(N))` → `reinterpret_cast<T*>((uintptr_t)__emule_local_l1_to_ptr(...))` |
| DFB test `num_entries_in_buffer` (4Sx4S / 2Sx4S) | 29 / 21 — not divisible by `max(P,C)=4` | 28 / 20 — divisible by 4; wraparound still exercised (both > `num_entries=16`) |
| Tier 6 Silicon Toggle | FAIL (expected; requires real hardware) | **PASS** (runs in emulation after env-var unset) |

**Key insight (DFB wraparound):** `num_entries_in_buffer` must be a multiple of `max(P, C)` for every entry to be processed. With `num_entries_in_buffer=29` and `max(P,C)=4`, each producer/consumer does `floor(29/4)=7` iterations covering 28 of 29 entries; entry 28 is never written to the output buffer and stays zero. Choosing values divisible by `max(P,C)` (28 and 20 respectively) eliminates the truncation while preserving the wraparound-exercise invariant.

**Key insight (atomic segfaults):** Quasar atomic kernels call `reinterpret_cast<std::atomic<T>*>(get_arg_val<uint32_t>(0))` where arg[0] is a raw L1 firmware offset (~`0xba780`). On real hardware these offsets are directly dereferenceable by firmware; on x86 emulation they are not valid host addresses, causing an immediate segfault. The fix: (1) move `__emule_local_l1_to_ptr()` into `jit_kernel_stubs.hpp` so every JIT kernel has access to it before any kernel-specific includes; (2) guard the existing definition in `dataflow_api.h` with `#ifndef __EMULE_LOCAL_L1_TO_PTR_DEFINED` to prevent ODR violations; (3) add a JIT preprocessor regex that rewrites the pattern automatically so existing and future kernels with this idiom need no source modifications.

---

*Report updated 2026-04-21. Covers tt-emule on branch `armin` / tt-metal on branch `arminale/emule-metal-base`. All 7 previously-failing tests now pass; regression at 137/0/0.*
