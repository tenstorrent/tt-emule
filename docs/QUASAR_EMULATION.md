# Quasar Emulation Overview

This is a running document updated as emulation phases progress.

---

## 1. Overview

tt-emule provides software emulation of Tenstorrent hardware for host-side testing without silicon. It supports two execution paths:

- **Standalone path**: Kernels are native C++ functions linked directly against tt-emule. Tests under `tests/`. State accessed via `__core` and `__dfb_ifaces` thread-locals.
- **JIT path**: Kernels are compiled from source at runtime (g++), loaded via `dlopen`. Integrates with real tt-metal host APIs (`CreateDataflowBuffer`, `LaunchProgram`). State accessed via `__emule_dfbs` and `__emule_tc_array` thread-locals set by `emulated_program_runner.cpp`.

---

## 2. Quasar Architecture (Tensix Neo)

"Tensix Neo" (or "Neo") is one core on a Quasar chip. Key facts:

- One Neo contains: **8 Data Movement (DM) processors** + **4 compute engines** + **4 MB shared L1**
- All 12 cores (8 DM + 4 compute) share the same 4 MB L1 — any core can read/write any L1 address
- A Quasar chip has many Neo cores (e.g., 32 in `quasar_32_arch.yaml`)
- 2 DRAM channels per chip
- DFBs (Dataflow Buffers) replace Wormhole's CBs (Circular Buffers) — MPMC instead of SPSC
- Hardware thread IDs (hartids): DM0-DM7 = 0-7, compute engines labeled NEO0-NEO3 internally in `hw_thread.h` = hartids 8-11 (Note: "NEO0-NEO3" are the 4 compute engines within a single Neo core — don't confuse with the architecture name)

---

## 3. Threading Model

How processors map to host threads:

| Processor Type | Hardware Count | Emulated Threads | Mapping |
|----------------|---------------|-----------------|---------|
| DM (DM0-DM7) | 8 per Neo | 1 thread per DM processor used | `get_dm_processors()` |
| Compute (engines 0-3) | 4 per Neo | 1 thread per compute engine | `get_compute_processors()` grouped by engine |

- Total: a Neo is emulated as up to **12 host threads** (8 DM + 4 compute)
- `mhartid` CSR: regex-patched in JIT-compiled kernel source to read `__processor_id` TLS variable
- `num_threads_per_cluster` in `QuasarComputeConfig`: controls how many of the 4 compute engines are active (default: 4)

**NEO_ID/TRISC_ID CSR: emulated (Phase 4).** `ckernel::csr_read<ckernel::CSR::NEO_ID>()` returns `__emule_neo_id` TLS, `csr_read<CSR::TRISC_ID>()` returns `__emule_trisc_id` TLS. Both are set by the program runner before each compute thread starts. Tests `risc_math.cpp` and `simple_tls_check.cpp` pass.

---

## 4. DRAM Emulation

- Emulated as a flat host buffer (single contiguous allocation per device)
- `NUM_DRAM_BANKS` is forced to **1** in the JIT defines, regardless of the real architecture count (Quasar=2, WH=6, BH=8)
- **Why:** `InterleavedAddrGen` with N>1 banks maps page IDs to bank-specific NOC addresses. Banks 1+ generate NOC (x,y) coordinates not registered in `__emule_core_map`. `__emule_resolve_noc_addr()` returns null for unmapped cores, causing `noc_async_read` to skip the memcpy and produce zeros.
- `noc_async_read` / `noc_async_write` are synchronous `memcpy` in emulation — no actual NOC transfer
- `__emule_dram_ptr(offset)` returns a host pointer into the flat DRAM buffer

---

## 5. DFB Emulation (Summary)

See `DFB_EMULATION.md` for the full deep dive. Key points:

- **STRIDED mode** (Phase 2, complete): `M = max(P, C)` interleaving factor. Producer p owns TC slots `{p + k*P}`, consumer c owns `{c + k*C}`. `stride_size = M * entry_size`. `capacity = num_entries / M`.
- **BLOCKED mode** (Phase 3, complete): `broadcast_tc = true` for producers. P*C tile counters: `counter_id = counter_base + p*C + c`. Consumer round-robins through `num_producers` TCs.
- **Timeout detection**: Blocking DFB operations (`reserve_back`, `wait_front`) use `wait_for` with configurable timeout (default 120s, `TT_EMULE_DFB_TIMEOUT` env var). Aborts with diagnostic on hang.
- **Max DFBs**: 8 per program with `neo_id=0` (limited by `TILE_COUNTERS_PER_NEO / MAX_TC_SLOTS_PER_DFB`)

---

## 6. L1 Memory

- 4 MB shared between all 12 cores in a Neo (8 DM + 4 compute engines)
- Emulated as a bump allocator on the host heap (`Core::l1_alloc()`)
- Standalone path: 1 MB default (sufficient for current test sizes)
- JIT path: size from SOC YAML (4 MB for Quasar via `SWEmulatedChip`)
- Bridge DFBs (compute kernel connecting input and output DFBs) share L1 backing store via `dim_key` deduplication — models the hardware register file passthrough

---

## 7. DST Register File

The DST (destination) register file is the same physical size on all architectures (WH, BH, Quasar). The number of usable tile slots depends on the **accumulation mode**, not the architecture:

| Mode | `DST_ACCUM_MODE` | Active Slots | Element Size | Use Case |
|------|-------------------|-------------|-------------|----------|
| bf16 (half-dest) | 0 (default) | 16 | 2 bytes | Most compute kernels |
| fp32 (half-dest) | != 0 | 8 | 4 bytes | High-precision accumulation |

In emulation (`include/jit_hw/api/compute/common.h`):
- Physical array: `__emule_dst[16][1024]` — always 16 slots × 1024 float32 elements
- Active slot count: `__emule_dst_active_tiles()` returns 16 or 8 based on `DST_ACCUM_MODE`
- Bounds checking: `__emule_dst_check()` enforces the mode-aware limit and aborts with a diagnostic if exceeded
- `tile_regs_acquire()` zeroes only the active slots

`DST_ACCUM_MODE` is a compile-time define passed by the JIT compiler. Kernels that need fp32 accumulation set it via `QuasarComputeConfig`.

---

## 8. JIT Compilation

- Compiler: **g++** (user preference; overrides CLAUDE.md's clang-20 rule for JIT specifically)
- Process: `emulated_program_runner` writes patched kernel source to temp file → `g++ -std=c++17 -fPIC -shared -O1` → `dlopen` the resulting `.so` → call `kernel_main` symbol
- `mhartid` CSR patch: regex replaces `asm volatile("csrr %0, mhartid" ...)` with `var = __processor_id;`
- TLS variables set per thread before kernel launch: `__emule_dfbs`, `__emule_tc_array`, `__processor_id`
- JIT cache: compiled `.so` files cached in `/tmp/tt_emule_jit_*`

---

## 9. Known Gaps

| Gap | Impact | Status |
|-----|--------|--------|
| `NEO_ID` CSR emulated | `csr_read<CSR::NEO_ID>()` returns `__emule_neo_id` TLS | Done (Phase 4) |
| Multi-bank DRAM interleaving | Always flat (1 bank); higher banks' NOC addresses unmapped | By design |
| DFB↔CB sync bridge | `dfb_push_back`/`pop_front` now bridge to CB `occupied` counter; `cb_push_back`/`pop_front` bridge to DFB tile counters. Matmul data flows end-to-end. | Done (Phase 7) |
| CB base pointer for DFB-backed CBs | `core->l1_ptr(base_addr)` double-added L1 base; fixed to cast `base_addr` directly | Done (Phase 7) |
| L1 bump allocator reset | `reset_l1_bump()` called between program runs to prevent L1 OOM | Done (Phase 7) |
| Multi-core NOC communication | Loopback and one-from-one tests pass (single-core self-write + two-core read) | Done (Phase 8) |
| Firmware L1 address translation | `__emule_local_l1_to_ptr()` distinguishes firmware offsets from host pointers; NOC ops use it | Done (Phase 8) |
| NOC virtual channels | `noc_async_read/write` accept optional virtual channel parameter (ignored in emulation) | Done (Phase 8) |
| Profiler macros | `DeviceZoneScopedN`, `DeviceTimestampedData` stubbed as no-ops | Done (Phase 8) |
| `GenericMeshDeviceFixture` for emulation | `MeshDeviceFixtureBase::SetUp()` allows slow dispatch when `TT_METAL_EMULATED_MODE` is set | Done (Phase 8) |
| NOC 2.0 API (NocSendDescriptor) | New stateful NOC API used by `requestor_2_0.cpp`, `direct_write` tests | Not implemented |
| LLK numerical correctness | Matmul data flow works but numerical results wrong (tile format mismatch) | Deferred |
| LLK math/SFPU operations | Only datacopy/tilize/untilize stubbed in `llk_defs.h`; binary, reduce, matmul, SFPU not implemented | Not implemented |
| TRISC `finish()` variant | Compute `finish()` behavior untested | Deferred |
| `EMULE_SEM_BASE` default conflict | Was: stubs.hpp defaulted to 0 vs api.h to 0xFFE00. Fixed: stubs.hpp no longer defines a default; defers to dataflow_api.h or JIT define | Fixed |
| `MathFidelity` enum values | Was: common.h auto-assigned (1,2,3). Fixed: explicit values matching tt-metal (HiFi2=2,HiFi3=3,HiFi4=4) | Fixed |
| `get_absolute_logical_x/y` return type | Was: uint32_t vs uint8_t. Fixed: both return uint32_t with `#ifndef` guard | Fixed |
| Standalone DstRegisterFile | Was: 8 slots. Fixed: 16 physical slots with mode-aware active count (16 bf16, 8 fp32) | Fixed |
| TileCounter lockless fast path | Was: non-atomic pair read could underflow in MPMC. Fixed: removed lockless fast path | Fixed |
| DFB `dfb_id` bounds check | Was: no bounds check. Fixed: all DFB API functions validate `dfb_id < MAX_DFBS` | Fixed |
| `get_local_cb_interface` duplication | Was: incompatible definitions in llk_defs.h and dataflow_api.h. Fixed: `#ifndef` guard | Fixed |
| Standalone L1 bump reset | Was: not called between EnqueueProgram runs. Fixed: reset_l1_bump() in teardown | Fixed |
| Compile-time L1 addresses | Handled by `__emule_local_l1_to_ptr()` — firmware offsets translated via `__emule_bridge_l1` | Done (Phase 8) |
| RISC-V fence instruction | `asm volatile("fence")` patched to `__sync_synchronize()` at JIT time | Done (Phase 6) |
| Raw L1 pointer casts | `reinterpret_cast<T*>(get_arg_val(N))` patched at JIT time to add `__emule_bridge_l1` | Done (Phase 6) |

---

## 10. Quasar-Specific Test Status

All tests below use `experimental::quasar::CreateKernel` with `QuasarDataMovementConfig` and/or `QuasarComputeConfig`. Architecture-independent tests (CoreRange, bit_utils, tilize, host_buffer, TTNN, D2M) are omitted.

**Regression total:** 107 passing, 3 pre-existing failures, 2 skipped.

### 10.1 Supported — Passing

| Test File | Test Names | Count | Notes |
|-----------|-----------|-------|-------|
| `api/dataflow_buffer/test_dfb_emulation.cpp` | DFBEmuleDMTest, DFBEmuleBridgeTest | 2 | 1P-1C DM, DM→compute→DM bridge |
| `api/dataflow_buffer/test_dataflow_buffer.cpp` | DMTest1xDFB\*, DMTensixTest\*, TensixDMTest\*, DMTensixDMTest\* | 72 | STRIDED (24 ISF + 18 IST) + BLOCKED (30) |
| `test_quasar_compute_kernels.cpp` | QuasarComputeKernelMultipleThreads, QuasarComputeKernelSingleThread, QuasarCreateMultipleComputeKernelsSingleCluster | 3 | Multi-thread/single-thread compute, multi-kernel |
| `test_globals_tls.cpp` | QuasarComputeKernelTLS | 1 | NEO_ID, TRISC_ID, thread ID TLS isolation |
| `test_quasar_semaphores.cpp` | QuasarComputeKernelSemaphores, QuasarDmAndComputeKernelSemaphores | 2 | Compute + DM semaphore spin-wait |
| `test_dm_loopback.cpp` | DmLoopback | 1 | DM semaphore + DRAM NOC |
| `test_single_dm_l1_write.cpp` | SingleDmL1Write | 1 | Named CT args, common RT args |
| `test_riscv_atomics.cpp` | TestAtomicLoadStoreRISCV, TestAtomicAddFetchRISCV, TestAtomicCASRISCV | 3 | GCC atomic builtins on L1 |
| `data_movement/loopback/test_loopback.cpp` | LoopbackPacketSizes, LoopbackDirectedIdeal | 2 | Single-core L1→L1 write via NOC |
| `data_movement/one_from_one/test_one_from_one.cpp` | OneFromOnePacketSizes, OneFromOneDirectedIdeal | 2 | Two-core L1 read |

**Total passing Quasar-specific: 89**

### 10.2 Supported — Partial / Blocked

| Test File | Test Names | Pass/Total | Blocking Gap |
|-----------|-----------|------------|-------------|
| `test_quasar_semaphores.cpp` | QuasarMultiSemaphorePipeline, QuasarMultipleClustersMultiSemaphorePipeline | 0/2 | Multi-cluster semaphore pipeline |
| `data_movement/one_from_one/test_one_from_one.cpp` | OneFromOneVirtualChannels, OneFromOneCustom | 0/2 | Skipped (2 cores needed in specific config) |
| `data_movement/one_from_one/test_one_from_one.cpp` | OneFromOnePacketSizes2\_0 | 0/1 | NOC 2.0 API (`requestor_2_0.cpp`) |
| `data_movement/direct_write/test_direct_write.cpp` | DirectWritePerformanceComparison, DirectWriteAddressPatterns, DirectWriteMulticast | 0/3 | NOC 2.0 API (`noc_inline_dw_write`, `InlineWriteDst`) |
| `data_movement/dram_unary/test_unary_dram.cpp` | DRAMPacketSizes, DRAMCoreLocations, DRAMChannels, DRAMDirectedIdeal, DRAMPacketSizes2\_0 | 0/5 | Firmware internal headers (`noc/noc_parameters.h`) |
| `integration/matmul/test_matmul_X_tile.cpp` | TensixMatmulSingleTile, TensixMatmulMultiTile, TensixMatmulBlock, TensixMatmulBlockInitShort, TensixMatmulBlockInitShortWithDt | 0/5 | Data path works end-to-end; numerical correctness blocked on tile format |

### 10.3 Unsupported — Not in tt-emule

| Test File | Test Names | Count | Blocking Gap |
|-----------|-----------|-------|-------------|
| `test_bmm.cpp` | Bmm | 1 | DFB pointer translation + full matmul compute pipeline |
| `api/test_runtime_args.cpp` | QuasarCRTASharedL1Address, QuasarCRTAUniqueL1Addresses | 2 | CRTA L1 address computation |
| `api/test_direct.cpp` | TensixSingleCoreDirectDramReaderDatacopyWriter (Quasar path) + 3 others | 4 | MeshCoordinate workload dispatch |
| `data_movement/dram_sharded/test_dram_sharded.cpp` | DRAMShardedReadDirectedIdeal, DRAMShardedReadTileNumbers, DRAMShardedReadBankNumbers, DRAMShardedReadTridDirectedIdeal, DRAMShardedReadTileNumbers2\_0, DRAMShardedReadTridDirectedIdeal\_2\_0 | 6 | Sharded DRAM read, transaction ID support |
| `data_movement/noc_api_latency/test_noc_api_latency.cpp` | NocApiLatencyUnicastWrite, NocApiLatencyUnicastRead, NocApiLatencyStatefulWrite, NocApiLatencyStatefulRead, NocApiLatencyMulticastWrite2x2, NocApiLatencyMulticastWrite5x5, NocApiLatencyMulticastWriteAll | 7 | Stateful NOC 2.0 API, latency measurement |
| `data_movement/one_from_all/test_one_from_all.cpp` | OneFromAllPacketSizes, OneFromAllDirectedIdeal, OneFromAllVirtualChannels, OneFromAllCustom | 4 | Multi-core gather pattern with subordinate cores |
| `data_movement/one_packet/test_one_packet.cpp` | OnePacketReadSizes, OnePacketWriteSizes, OnePacketReadDirectedIdeal, OnePacketWriteDirectedIdeal, OnePacketReadSizes\_2\_0, OnePacketWriteSizes\_2\_0 | 6 | Single-packet NOC read/write, NOC 2.0 variant |

**Total unsupported Quasar-specific: 30**

---

## 11. File Reference

| File | Role |
|------|------|
| `src/kernel_runner.cpp` | Standalone path: `build_dfb_interfaces()`, `EnqueueProgram` |
| `src/jit_kernel.cpp` | JIT compilation (g++) |
| `include/tt_emule/device.hpp` | `Core` — owns L1, TileCounterArray, DFBSyncState |
| `include/tt_emule/tile_counter.hpp` | `TileCounter`, `TileCounterArray` |
| `include/tt_emule/dfb_sync_state.hpp` | `EmuleDFBInterface`, `DFBSyncState` |
| `include/tt_emule/dst_register_file.hpp` | `DstRegisterFile` — 16 physical slots, mode-aware active count |
| `include/jit_hw/api/dfb_api.h` | JIT DFB operations with timeout and bounds checks |
| `include/jit_hw/experimental/dataflow_buffer.h` | `experimental::DataflowBuffer` JIT wrapper |
| `include/jit_hw/api/compute/common.h` | DST register file, compute ops (pack/copy/matmul tiles) |
| `include/jit_hw/llk_defs.h` | LLK stubs: datacopy, tilize/untilize, CB delegation |
| `run_regression.sh` | C++ regression (6 tiers) |
| `run_d2m_regression.sh` | D2M Python golden test regression |
| *(tt-metal)* `emulated_program_runner.cpp` | JIT path: DFB setup, thread spawning, mhartid patch |
