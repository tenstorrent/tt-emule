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

**Known gap: `NEO_ID` CSR not emulated.** Compute kernels that read `ckernel::csr_read<ckernel::CSR::NEO_ID>()` to determine their engine index (0-3) need a JIT stub similar to the `mhartid` patch. Current DFB compute kernels (`dfb_t6.cpp`) don't use this. Tests like `risc_math.cpp` and `simple_tls_check.cpp` would fail.

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
- **BLOCKED mode** (Phase 3, in progress): `broadcast_tc = true` for producers. Each consumer sees all entries. Runtime code in `dfb_api.h` handles broadcast; TC assignment needs update.
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

## 7. JIT Compilation

- Compiler: **g++** (user preference; overrides CLAUDE.md's clang-20 rule for JIT specifically)
- Process: `emulated_program_runner` writes patched kernel source to temp file → `g++ -std=c++17 -fPIC -shared -O1` → `dlopen` the resulting `.so` → call `kernel_main` symbol
- `mhartid` CSR patch: regex replaces `asm volatile("csrr %0, mhartid" ...)` with `var = __processor_id;`
- TLS variables set per thread before kernel launch: `__emule_dfbs`, `__emule_tc_array`, `__processor_id`
- JIT cache: compiled `.so` files cached in `/tmp/tt_emule_jit_*`

---

## 8. Known Gaps

| Gap | Impact | Status |
|-----|--------|--------|
| `NEO_ID` CSR not emulated | Compute kernels using `csr_read<CSR::NEO_ID>()` fail | Needs JIT stub |
| Multi-bank DRAM interleaving | Always flat (1 bank); higher banks' NOC addresses unmapped | By design |
| BLOCKED mode TC assignment | `*B` test variants fail | Phase 3 in progress |
| Multi-core NOC communication | All current tests are single-core | Not needed yet |
| LLK numerical correctness | Matmul stubs are flow-control only | Phase 4 |
| TRISC `finish()` variant | Compute `finish()` behavior untested | Deferred |

---

## 9. Test Coverage (Phase 2)

43 passing C++ tests across 6 tiers:

- **Tier 0**: Standalone DFB tests (`dfb_passthrough`, `dfb_multi_consumer`, `eltwise_add`)
- **Tier 1**: Host-only unit tests (bit_utils, tilize, CoreRange, etc.)
- **Tier 2**: Buffer I/O (L1, DRAM, emulation toggle)
- **Tier 3**: JIT kernel execution + DFB emulation (Groups A-D: DM-DM, DM-Tensix, Tensix-DM, multi-DFB pipeline)
- **Tier 4-5**: TTNN relational/add/matmul sweep
- **D2M**: 1589/1878 Python golden tests passing (WH/BH only, no Quasar)

---

## 10. File Reference

| File | Role |
|------|------|
| `src/kernel_runner.cpp` | Standalone path: `build_dfb_interfaces()`, `EnqueueProgram` |
| `src/jit_kernel.cpp` | JIT compilation (g++) |
| `include/tt_emule/device.hpp` | `Core` — owns L1, TileCounterArray, DFBSyncState |
| `include/tt_emule/tile_counter.hpp` | `TileCounter`, `TileCounterArray` |
| `include/tt_emule/dfb_sync_state.hpp` | `EmuleDFBInterface`, `DFBSyncState` |
| `include/jit_hw/api/dfb_api.h` | JIT DFB operations with timeout |
| `include/jit_hw/experimental/dataflow_buffer.h` | `experimental::DataflowBuffer` JIT wrapper |
| `run_regression.sh` | C++ regression (6 tiers) |
| `run_d2m_regression.sh` | D2M Python golden test regression |
| *(tt-metal)* `emulated_program_runner.cpp` | JIT path: DFB setup, thread spawning, mhartid patch |
