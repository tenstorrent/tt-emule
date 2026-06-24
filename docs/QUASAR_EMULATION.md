# Quasar Emulation in tt-emule

tt-emule provides software emulation of Tenstorrent Quasar hardware for host-side testing without silicon. This document is the reference for anyone reading, testing, or extending the Quasar emulation.

---

## 1. Overview

tt-emule runs kernels through a single JIT path: kernels are compiled from source at runtime (g++), loaded via `dlopen`, and integrate with the real tt-metal host APIs. State is accessed via the `__emule_dfbs` and `__emule_tc_array` thread-locals set by `emulated_program_runner.cpp`. This lets upstream device kernels run unmodified (modulo CSR patching) against real tt-metal host APIs, so host-side code exercises the same paths as on silicon.

---

## 2. Quasar Architecture (Tensix Neo)

"Tensix Neo" (or "Neo") is one core on a Quasar chip. Key facts:

- One Neo contains: **8 Data Movement (DM) processors** + **4 compute engines** + **4 MB shared L1**
- All 12 cores (8 DM + 4 compute) share the same 4 MB L1 — any core can read/write any L1 address
- A Quasar chip has many Neo cores (e.g., 32 in `quasar_32_arch.yaml`)
- 2 DRAM channels per chip
- DFBs (Dataflow Buffers) replace Wormhole's CBs (Circular Buffers) — MPMC instead of SPSC
- Hardware thread IDs (hartids): DM0-DM7 = 0-7, compute engines NEO0-NEO3 = hartids 8-11 (the 4 compute engines within a single Neo core — distinct from the chip-level "Neo" architecture name)

---

## 3. Threading Model

How processors map to host threads:

| Processor Type | Hardware Count | Emulated Threads | Mapping |
|----------------|---------------|-----------------|---------|
| DM (DM0-DM7) | 8 per Neo | 1 thread per DM processor used | `get_dm_processors()` |
| Compute (engines 0-3) | 4 per Neo | 1 thread per compute engine | `get_compute_processors()` grouped by engine |

- Total: a Neo is emulated as up to **12 host threads** (8 DM + 4 compute)
- `num_threads_per_cluster` in `QuasarComputeConfig`: controls how many of the 4 compute engines are active (default: 4)

**CSR emulation:** Hardware CSR reads are regex-patched at JIT compile time:
- `mhartid`: `asm volatile("csrr %0, mhartid" ...)` → `var = __processor_id;`
- `NEO_ID`: `ckernel::csr_read<ckernel::CSR::NEO_ID>()` → returns `__emule_neo_id` TLS
- `TRISC_ID`: `csr_read<CSR::TRISC_ID>()` → returns `__emule_trisc_id` TLS

Both `__emule_neo_id` and `__emule_trisc_id` are set by the program runner before each compute thread starts.

**Test evidence:** `test_globals_tls.cpp:QuasarComputeKernelTLS` verifies NEO_ID, TRISC_ID, and thread ID TLS isolation across compute engines.

---

## 4. Memory Model

### 4.1 L1 Memory

- 4 MB shared between all 12 cores in a Neo (8 DM + 4 compute engines)
- Emulated as a bump allocator on the host heap (`Core::l1_alloc()`)
- L1 size comes from the SOC YAML (4 MB for Quasar via `SWEmulatedChip`)
- Bridge DFBs (compute kernel connecting input and output DFBs) share L1 backing store via `dim_key` deduplication — models the hardware register file passthrough

**Firmware L1 address translation:** Device kernels reference L1 via firmware-range offsets (e.g., `0x1000`). `__emule_local_l1_to_ptr()` distinguishes these from host pointers and translates via `__emule_bridge_l1`. NOC operations use this translation.

**Compile-time L1 addresses:** `reinterpret_cast<T*>(get_arg_val(N))` patterns are regex-patched at JIT time to add `__emule_bridge_l1` offset.

### 4.2 DRAM

- Emulated as a flat host buffer (single contiguous allocation per device)
- `NUM_DRAM_BANKS` is set to the **real architecture channel count** (Quasar=2, WH=6, BH=8) at `emulated_program_runner.cpp:700`
- All bank-to-NOC-coordinate arrays (`dram_bank_to_noc_xy`, `bank_to_dram_offset`) are populated for every channel at lines 531-541
- DRAM cores are registered in `__emule_core_map` at lines 1012-1031, so `__emule_resolve_noc_addr()` resolves all bank addresses correctly
- `noc_async_read` / `noc_async_write` are synchronous `memcpy` — no actual NOC transfer
- `__emule_dram_ptr(offset)` returns a host pointer into the flat DRAM buffer

**Test evidence:** `test_dm_loopback.cpp:DmLoopback` exercises DM semaphore + DRAM NOC read/write. `test_single_dm_l1_write.cpp:SingleDmL1Write` tests named CT args and common RT args via DRAM. `test_dm_unary_dram.cpp:DRAMChannels` exercises multi-bank DRAM read/write across all architecture channels.

### 4.3 NOC

NOC operations (`noc_async_read`, `noc_async_write`) are synchronous memcpy in emulation. Supported features:

- **Loopback**: Single-core L1→L1 write via NOC — tested by `test_dm_loopback_noc.cpp:LoopbackPacketSizes, LoopbackDirectedIdeal`
- **One-from-one**: Two-core L1 read — tested by `test_dm_one_from_one.cpp:OneFromOnePacketSizes, OneFromOneDirectedIdeal`
- **Inline direct write**: `noc_inline_dw_write` (unicast 32-bit write), `noc_inline_mcast_dw_write` (multicast), `noc_inline_dw_write_set_state`/`noc_inline_dw_write_with_state` (stateful cached address/value pattern). Byte-enable masking supported. Tested by `test_dm_direct_write.cpp:PerformanceComparison, AddressPatterns, Multicast`
- **DRAM read/write**: Multi-bank `noc_async_read`/`noc_async_write` to DRAM via `get_noc_addr_from_bank_id<dram>()`. All architecture bank counts supported. Tested by `test_dm_unary_dram.cpp:PacketSizes, CoreLocations, DRAMChannels, DirectedIdeal`
- **Virtual channels**: Optional parameter accepted and ignored
- **RISC-V fence**: `asm volatile("fence")` patched to `__sync_synchronize()` at JIT time

---

## 5. Compute Engine

The compute engine emulates the hardware's UNPACK → MATH → PACK pipeline. On real hardware, three TRISC cores (TRISC0=UNPACK, TRISC1=MATH, TRISC2=PACK) run in parallel. In emulation, all three share one host thread — operations execute sequentially via `PACK(x) x`, `MATH(x) x`, `UNPACK(x) x` macros.

### 5.1 DST Register File

The DST (destination) register file is where compute results accumulate before being packed to L1.

| Mode | `DST_ACCUM_MODE` | Active Slots | Element Size | Use Case |
|------|-------------------|-------------|-------------|----------|
| bf16 (half-dest) | 0 (default) | 16 | 4 bytes (f32 internal) | Most compute kernels |
| fp32 (half-dest) | != 0 | 8 | 4 bytes (f32 internal) | High-precision accumulation |

Implementation (`include/jit_hw/api/compute/common.h`):
- Physical array: `__emule_dst[16][1024]` — 16 slots × 1024 float32 elements (64 KB)
- Active slot count: `__emule_dst_active_tiles()` returns 16 or 8 based on `DST_ACCUM_MODE`
- Bounds checking: `__emule_dst_check()` aborts with diagnostic if slot exceeds mode-aware limit
- `tile_regs_acquire()` zeroes only the active slots

`DST_ACCUM_MODE` is a compile-time define passed by the JIT compiler. Kernels that need fp32 accumulation set it via `QuasarComputeConfig`.

### 5.2 UNPACK Engine (L1 → SRC/DST)

On real hardware, the UNPACK engine reads tile data from L1 in **NFACES format** and produces **row-major** data in the SRC registers for the MATH engine. In emulation, UNPACK is integrated into each compute op that reads from a CB.

**NFACES format:** A 32×32 tile is stored as 4 sequential 16×16 faces, each row-major internally:

```
face 0 (offsets   0-255): rows  0-15, cols  0-15  (top-left)
face 1 (offsets 256-511): rows  0-15, cols 16-31  (top-right)
face 2 (offsets 512-767): rows 16-31, cols  0-15  (bottom-left)
face 3 (offsets 768-1023): rows 16-31, cols 16-31 (bottom-right)
```

The conversion is done via a constexpr LUT (`include/jit_hw/api/compute/nfaces.h`):
```cpp
// rowmajor_to_nfaces[i] = nfaces offset for row-major element i
// Usage (UNPACK): dst[i] = buf[rowmajor_to_nfaces[i]]
namespace __emule_nfaces {
    inline constexpr auto rowmajor_to_nfaces = make_rowmajor_to_nfaces();
}
```

Format conversion (bf16→f32 or passthrough for f32 data) happens in the same loop as the layout conversion. For example, in `copy_tile`:
```cpp
// bfloat16: UNPACK nfaces→rowmajor + bf16→f32
for (uint32_t i = 0; i < n; i++)
    __emule_dst[idst][i] = __emule_bf16::to_f32(bf[rowmajor_to_nfaces[i]]);
```

All compute ops that read from CBs (`copy_tile`, `add_tiles`, `sub_tiles`, `mul_tiles`, `matmul_tiles`) perform this UNPACK conversion.

**Test evidence:** `test_matmul_X_tile.cpp:TensixMatmulBlock, TensixMatmulBlockInitShort` — nfaces layout conversion is required for matmul PCC to pass with non-trivial input data.

**TODO: No dedicated UNPACK-only test exists.**

### 5.3 PACK Engine (DST → L1)

On real hardware, the PACK engine reads row-major DST data and writes it to L1 in NFACES format, with optional format conversion (f32→bf16). In emulation, PACK is implemented in `pack_dst_to_buf()` (`common.h`):

```cpp
// bf16 non-L1-acc path:
bf[rowmajor_to_nfaces[i]] = __emule_bf16::from_f32(__emule_dst[dst_slot][i]);
```

**Auto-advance:** On real hardware, each `pack_tile` call auto-advances the PACK engine's L1 write pointer to the next tile slot. In emulation, this is tracked by `__emule_pack_offset[cb_id]` (per-CB, thread-local). `pack_tile` writes to `cb_write_ptr_at(ocb, __emule_pack_offset[ocb]++)`. The offset is reset to 0 in `cb_reserve_back` and `dfb_reserve_back`.

**L1 accumulation:** When `llk_pack_reconfig_l1_acc(1)` is called, `pack_dst_to_buf` adds DST values to existing CB contents instead of overwriting. This is used by matmul kernels that accumulate partial products across multiple `tile_regs_acquire/commit` cycles.

**Test evidence:** `test_matmul_X_tile.cpp:TensixMatmulBlock, TensixMatmulBlockInitShort` — pack auto-advance is required for multi-tile matmul output to write to distinct L1 slots.

### 5.4 Implemented Compute Operations

| Operation | Function | File | Description |
|-----------|----------|------|-------------|
| Tile copy | `copy_tile(icb, itile, idst)` | `common.h` | UNPACK CB tile → DST |
| Block copy | `copy_block_matmul_partials(...)` | `common.h` | Reload block of tiles from CB to DST |
| Pack tile | `pack_tile(idst, ocb)` | `common.h` | PACK DST → CB with auto-advance |
| Pack block | `pack_tile_block(ifrom, ocb, n)` | `common.h` | Pack N consecutive DST tiles to CB |
| Add | `add_tiles(icb0, icb1, it0, it1, idst)` | `common.h` | Element-wise add with UNPACK |
| Subtract | `sub_tiles(icb0, icb1, it0, it1, idst)` | `common.h` | Element-wise sub with UNPACK |
| Multiply | `mul_tiles(icb0, icb1, it0, it1, idst)` | `common.h` | Element-wise mul with UNPACK |
| Matmul | `matmul_tiles(in0, in1, it0, it1, idst)` | `matmul.h` | 32×32 tile GEMM, accumulates into DST |
| Block matmul | `matmul_block(...)` | `matmul.h` | Block of rt_dim × ct_dim output tiles |
| Reduce | `reduce_tile(icb, icb_scaler, ...)` | `reduce.h` | Row/col/scalar reduce with SUM/MAX. MAX uses a per-DST-slot fresh flag (set by `tile_regs_acquire`, cleared by writers) to overwrite on first call rather than max-accumulate against the zero-init, so `min(x) = -max(-x)` works correctly when all values are negative. |
| L1 acc toggle | `llk_pack_reconfig_l1_acc(enable)` | `common.h` | Enable/disable PACK L1 accumulation |

The matmul implementation uses AVX2/FMA intrinsics when available (`-mavx2 -mfma`) for ~4-8x speedup over the scalar fallback.

### 5.5 Stubbed Compute Operations (Compile but No-Op)

These operations exist as empty function bodies so device kernels compile. They produce no output:

| Category | Examples | File |
|----------|----------|------|
| Init/reconfig | `mm_init`, `mm_init_short`, `binary_op_init_common`, `copy_tile_init`, all `reconfig_data_format*` | `common.h`, `matmul.h` |
| SFPU element-wise | `add_int_sfpu`, `sub_int_sfpu`, `mul_int_sfpu`, `div_int32_sfpu`, `binary_bitwise_sfpu`, etc. | `add_int_sfpu.h`, `sub_int_sfpu.h`, etc. |
| Broadcast | `bcast` variants (COL, ROW, SCALAR) | `bcast.h` |
| Transpose | `transpose_wh` | `transpose_wh.h` |
| Tilize/Untilize | `tilize_init`, `untilize_init` | `tilize.h`, `untilize.h` |
| Pack untilize | `pack_untilize_init`, `pack_untilize` | `pack_untilize.h` |
| Eltwise binary SFPU | SFPU-accelerated binary ops | `eltwise_binary_sfpu.h` |
| Quantization | Quantize/dequantize | `quantization.h` |
| LLK reduce/matmul init | `state_configure`, `llk_math_matmul_init`, `llk_unpack_AB_matmul_init`, `llk_unpack_reconfig_data_format_srca`, `llk_math_reconfig_data_format_srca`, `llk_unpack_AB_reduce_init`, `llk_math_reduce_init` | `llk/llk_reduce_primitives.h` |

---

## 6. Dataflow Buffers (Summary)

See `docs/DFB_EMULATION.md` for the full deep dive. Key points:

- **STRIDED mode**: `M = max(P, C)` interleaving factor. Producer p owns TC slots `{p + k*P}`, consumer c owns `{c + k*C}`. `stride_size = M * entry_size`. `capacity = num_entries / M`. Tested by 42 tests in `test_dataflow_buffer.cpp`.
- **BLOCKED mode**: `broadcast_tc = true` for producers. P*C tile counters: `counter_id = counter_base + p*C + c`. Consumer round-robins through `num_producers` TCs. Tested by 30 tests in `test_dataflow_buffer.cpp`.
- **DFB↔CB bridge**: `dfb_push_back`/`pop_front` bridge to CB `occupied` counter; `cb_push_back`/`pop_front` bridge to DFB tile counters. This allows compute kernels to use CB API while DM kernels use DFB API on the same buffer.
- **Timeout detection**: Blocking DFB operations use `wait_for` with configurable timeout (default 120s, `TT_EMULE_DFB_TIMEOUT` env var). Aborts with diagnostic on hang.
- **Max DFBs**: 8 per program with `neo_id=0` (limited by `TILE_COUNTERS_PER_NEO / MAX_TC_SLOTS_PER_DFB`)

---

## 7. JIT Compilation

- Compiler: **g++** (user preference; overrides CLAUDE.md's clang-20 rule for JIT specifically)
- Process: `emulated_program_runner` writes patched kernel source to temp file → `g++ -std=c++17 -fPIC -shared -O1` → `dlopen` the resulting `.so` → call `kernel_main` symbol
- `mhartid` CSR patch: regex replaces `asm volatile("csrr %0, mhartid" ...)` with `var = __processor_id;`
- `fence` instruction patch: `asm volatile("fence")` → `__sync_synchronize()`
- Raw L1 pointer cast patch: `reinterpret_cast<T*>(get_arg_val<uint32_t>(N))` → `reinterpret_cast<T*>((uintptr_t)__emule_local_l1_to_ptr(get_arg_val<uint32_t>(N)))` — translates raw L1 firmware offsets to valid host pointers via the per-thread `__emule_bridge_l1` base
- TLS variables set per thread before kernel launch: `__emule_dfbs`, `__emule_tc_array`, `__processor_id`, `__emule_neo_id`, `__emule_trisc_id`
- JIT cache: compiled `.so` files cached in `/tmp/tt_emule_jit_*`
- Profiler macros (`DeviceZoneScopedN`, `DeviceTimestampedData`) stubbed as no-ops

---

## 8. Implemented Features

Every feature listed here is verified by at least one passing test.

| Feature | Implementation | Test Evidence |
|---------|---------------|---------------|
| DM threading (1 thread per DM) | `emulated_program_runner.cpp` | `test_quasar_compute_kernels.cpp:QuasarComputeKernelMultipleThreads` |
| Compute threading (1 thread per engine) | `emulated_program_runner.cpp` | `test_quasar_compute_kernels.cpp:QuasarComputeKernelSingleThread` |
| NEO_ID / TRISC_ID CSR | `__emule_neo_id`, `__emule_trisc_id` TLS | `test_globals_tls.cpp:QuasarComputeKernelTLS` |
| mhartid CSR patch | Regex in `emulated_program_runner.cpp` | `test_quasar_compute_kernels.cpp` (all 3 tests) |
| DRAM read/write (multi-bank) | `noc_async_read/write` as memcpy, all architecture bank counts | `test_dm_loopback.cpp:DmLoopback`, `test_dm_unary_dram.cpp:DRAMChannels` |
| L1 shared memory (bump alloc) | `Core::l1_alloc()` | All DFB tests (72+ tests allocate L1) |
| L1 address translation | `__emule_local_l1_to_ptr()` in `jit_kernel_stubs.hpp`; JIT regex auto-patches `reinterpret_cast<T*>(get_arg_val<uint32_t>(N))` | `test_single_dm_l1_write.cpp:SingleDmL1Write`, `test_riscv_atomics.cpp` |
| NOC loopback (single-core) | Synchronous memcpy | `test_dm_loopback_noc.cpp:LoopbackPacketSizes, LoopbackDirectedIdeal` |
| NOC one-from-one (two-core) | Synchronous memcpy | `test_dm_one_from_one.cpp:OneFromOnePacketSizes, OneFromOneDirectedIdeal` |
| NOC inline direct write (unicast) | `noc_inline_dw_write` with byte-enable | `test_dm_direct_write.cpp:PerformanceComparison, AddressPatterns` |
| NOC inline direct write (stateful) | `set_state`/`with_state` TLS-cached | `test_dm_direct_write.cpp:PerformanceComparison, AddressPatterns` |
| NOC inline direct write (multicast) | `noc_inline_mcast_dw_write` rectangle decode | `test_dm_direct_write.cpp:Multicast` |
| DRAM unary read/write (multi-bank) | `get_noc_addr_from_bank_id<dram>()` + `noc_async_read/write` | `test_dm_unary_dram.cpp:PacketSizes, CoreLocations, DRAMChannels, DirectedIdeal` |
| Semaphores (compute + DM) | Atomic spin-wait | `test_quasar_semaphores.cpp:QuasarComputeKernelSemaphores, QuasarDmAndComputeKernelSemaphores` |
| RISC-V atomics on L1 | GCC builtins | `test_riscv_atomics.cpp:TestAtomicLoadStoreRISCV, TestAtomicAddFetchRISCV, TestAtomicCASRISCV` |
| DFB STRIDED (1P-1C through 4P-4C) | `EmuleDFBInterface` + `TileCounterArray` | 42 tests in `test_dataflow_buffer.cpp` (38 pass / 4 fail on `arminale/emule-metal-base`: 4Sx4S and 2Sx4S wraparound — fix not yet upstream) |
| DFB BLOCKED | `broadcast_tc` path | 30 tests in `test_dataflow_buffer.cpp` |
| DFB↔CB bridge | `cb_sync_push/pop` in dfb_api.h / cb_api.h | DFB compute bridge tests in `test_dataflow_buffer.cpp` |
| DFB timeout detection | `wait_for` with `TT_EMULE_DFB_TIMEOUT` | Triggered by hang scenarios in regression |
| DST register file (16/8 slots) | `__emule_dst[16][1024]` + mode-aware bounds | `test_matmul_X_tile.cpp` (uses 16 slots) |
| UNPACK nfaces→rowmajor | `__emule_nfaces::rowmajor_to_nfaces` LUT | `test_matmul_X_tile.cpp:TensixMatmulBlock, TensixMatmulBlockInitShort` |
| PACK rowmajor→nfaces + auto-advance | `pack_dst_to_buf()` + `__emule_pack_offset` | `test_matmul_X_tile.cpp:TensixMatmulBlock, TensixMatmulBlockInitShort` |
| Matmul (tile GEMM, AVX2) | `matmul_tiles()` / `matmul_block()` | `test_matmul_X_tile.cpp:TensixMatmulBlock, TensixMatmulBlockInitShort` |
| Element-wise add/sub/mul | `add_tiles()`, `sub_tiles()`, `mul_tiles()` | D2M `test_add_int` (add path) |
| Reduce (row/col/scalar, sum/max/min via -max(-x)) | `reduce_tile()` with per-DST fresh flag for first-call MAX overwrite | Tier 5b: all 16 `Sum*` + `MinMax*` cases in `test_reduction.cpp`; D2M regression (reduce ops) |
| Metal 2.0 named-args | `experimental/kernel_args.h` (RtaArg, CrtaArg, CtaVal); emulator runner emits `kernel_bindings_generated.h` / `kernel_args_generated.h` per kernel | Tier 5b W-reduce on bbradel-tip via `SumTensorLastDimFixture.SumTensorCorrectly/1` |
| L1 accumulation | `llk_pack_reconfig_l1_acc()` | `test_matmul_X_tile.cpp` (matmul accumulates partials) |
| RISC-V fence patch | `__sync_synchronize()` | All JIT tests (compilation succeeds) |
| `GenericMeshDeviceFixture` | Allows slow dispatch in emulated mode | All JIT-path tests |

**Regression total against `arminale/emule-metal-20`:** 135 passing, 11 failing, 0 skipped. Tier 5b covers all 16 cases in `tests/ttnn/unit_tests/gtests/test_reduction.cpp` (6 `Sum*` and 10 `MinMax*`, both aligned and unaligned shapes).

The 11 failures are all in DFB tests that depend on follow-up fixes not yet merged to upstream `main`:
- **Tier 3b STRIDED wraparound (4):** `DMTest1xDFB4Sx4S`, `DMTest1xDFB4Sx4S_IS`, `DMTest1xDFB2Sx4S`, `DMTest1xDFB2Sx4S_IS`
- **Tier 3g DFB Config Validation (7):** `DMTest1xDFB1Sx4SConfig`, `DMTensixTest1xDFB4Sx1SConfig`, `DMTest1xDFB4Sx1SConfig`, `DMTest1xDFB4Sx4SConfig`, `DMTest1xDFB2Sx4SConfig`, `DMTest1xDFB4Sx2SConfig`, `DMTest1xDFB1Sx1BConfig`

These failures will resolve once the wraparound + DFB-config fixes upstream and the base pointer is bumped.

---

## 9. Not Implemented

| Feature | Blocking Gap | Tests Affected |
|---------|-------------|----------------|
| NOC 2.0 API (`NocSendDescriptor`) | Stateful NOC API (`requestor_2_0.cpp`) not implemented | `OneFromOnePacketSizes2_0`, `NocApiLatency*` (7 tests), `OnePacket*_2_0` (2 tests), `DRAMPacketSizes2_0` |
| Multi-cluster semaphore pipeline | Cross-cluster semaphore coordination not implemented | `QuasarMultiSemaphorePipeline`, `QuasarMultipleClustersMultiSemaphorePipeline` |
| SFPU operations | All SFPU ops are no-op stubs | Any test requiring SFPU math |
| Broadcast (COL/ROW/SCALAR) | `bcast` variants are no-op stubs | Tests requiring broadcast compute |
| Transpose WH | `transpose_wh` is a no-op stub | Tests requiring tile transpose |
| Tilize/Untilize compute | `tilize.h`, `untilize.h` are empty stubs | Tests requiring compute-path tilize |
| Pack untilize | `pack_untilize` is a no-op stub | Tests requiring untilized pack output |
| Sharded DRAM | No sharded read/write support | `test_dram_sharded.cpp` (6 tests) |
| Multi-core gather (one-from-all) | Multi-core gather pattern not implemented | `test_one_from_all.cpp` (4 tests) |
| CRTA L1 address computation | Common runtime args L1 address resolution | `test_runtime_args.cpp:QuasarCRTA*` (2 tests) |
| MeshCoordinate workload dispatch | Multi-core workload dispatch not implemented | `test_direct.cpp` Quasar paths (4 tests) |
| BMM full pipeline | DFB pointer translation + full compute pipeline | `test_bmm.cpp:Bmm` |
| Quasar matmul (3 skipped variants) | `TensixMatmulSingleTile`, `TensixMatmulMultiTile`, `TensixMatmulBlockInitShortWithDt` not yet enabled | 3 tests skipped |
| TRISC `finish()` variant | Compute `finish()` should wait for `posted == 0`, not `posted == acked` | No test exercises this distinction |

---

## 10. File Reference

| File | Role |
|------|------|
| `tt-metal/.../emulated_program_runner.cpp` | JIT compilation (g++), per-core thread launch, DFB interface setup |
| `include/tt_emule/device.hpp` | `Core` — owns L1, TileCounterArray, DFBSyncState |
| `include/tt_emule/tile_counter.hpp` | `TileCounter`, `TileCounterArray` |
| `include/tt_emule/dfb_sync_state.hpp` | `EmuleDFBInterface`, `DFBSyncState` |
| `include/jit_hw/api/dfb_api.h` | JIT DFB operations with timeout and bounds checks |
| `include/jit_hw/api/cb_api.h` | JIT CB operations with DFB bridge |
| `include/jit_hw/api/compute/common.h` | DST register file, compute ops, PACK engine |
| `include/jit_hw/api/compute/common_globals.h` | Thread-local DST, pack offset tracking |
| `include/jit_hw/api/compute/nfaces.h` | NFACES↔rowmajor constexpr LUT |
| `include/jit_hw/api/compute/matmul.h` | Tile GEMM (AVX2/scalar) |
| `include/jit_hw/api/compute/reduce.h` | Row/col/scalar reduce with SUM/MAX |
| `include/jit_hw/api/bfloat16.h` | bf16↔f32 conversion helpers |
| `include/jit_hw/experimental/dataflow_buffer.h` | `experimental::DataflowBuffer` JIT wrapper |
| `include/jit_hw/llk_defs.h` | LLK stubs: datacopy, CB delegation |
| `include/jit_hw/jit_kernel_stubs.hpp` | TLS declarations, stubbed firmware symbols |
| `include/jit_hw/ckernel.h` | CSR read emulation (NEO_ID, TRISC_ID) |
| `run_regression.sh` | C++ regression (6 tiers) |
| `run_d2m_regression.sh` | D2M Python golden test regression |
| *(tt-metal)* `emulated_program_runner.cpp` | JIT path: DFB setup, thread spawning, CSR/fence/L1 patches |
| *(tt-metal)* `test_dataflow_buffer.cpp` | 72 STRIDED + BLOCKED DFB tests |
| *(tt-metal)* `test_matmul_X_tile.cpp` | Quasar matmul PCC tests |
