# IMPLEMENTATION_REPORT.md Changelog

Historical per-version diffs of the implementation report. Latest entry first.

---

## v12 → v13 (2026-05-26)

| Aspect | v12 | v13 |
|--------|-----|-----|
| tt-metal base | `arminale/emule-metal-20` @ `5c6ffaca751` | `arminale/emule-umd-fix-pin` @ `c380834a` (pin bumped to upstream `8385f58b`; umd CI guard cherry-pick; Metal 2.0 JIT support) |
| Metal 2.0 JIT integration | called `write_kernel_{bindings,args}_generated_header` from `tt_metal/jit_build/genfiles.cpp` — required promoting two functions out of an anonymous namespace into `genfiles.hpp`'s public API surface (`mkdtemp` per kernel + `-I<dir>` + cleanup) | self-contained in `emulated_program_runner.cpp`: `Metal2BindingsSnapshot` + `build_metal2_snapshot()` + `emit_metal2_namespaces()` inline-emit the same `args::`/`dfb::`/`sem::`/`ta::` blocks into the JIT wrapper. **No public-API leak into `genfiles.hpp`.** Adds `sem::` coverage (missing in earlier inline version) and a binding-aware cache-key suffix (DFB/sem/RTA/CRTA/TA names + IDs) that prevents two Metal 2.0 kernels with identical sources/CTAs/defines but different binding identity from colliding on the cached `.so` |
| CI matrix | `wormhole` + `blackhole` (Quasar tests ran inside `wormhole`, polluting its known-failures) | `wormhole` + `blackhole` + `quasar` as three parallel matrix jobs; per-arch self-contained `scripts/run_regression_{arch}.sh`; `.github/known-failures-{wormhole,blackhole,quasar}.txt` (Quasar's 9 failures split out into their own file) |

---

## v11 → v12

| Aspect | v11 | v12 |
|--------|-----|-----|
| tt-metal base | `arminale/emule-metal-base` @ `8711ac3d0ba` | `arminale/emule-metal-20` (`emule-metal-base` + cherry-picked Metal 2.0 emule-JIT genfiles commit `5c6ffaca751`) |
| `test_reduction.cpp` coverage | 1 / 16 (W-aligned sum only) | **16 / 16** — all 6 `Sum*` (Last/First/Both × aligned/unaligned) and all 10 `MinMax*` (Last/First/Both × 4/4/2 params) |
| Metal 2.0 named-args | not supported in emule JIT | `experimental/kernel_args.h` ported; emulator runner emits `kernel_bindings_generated.h` / `kernel_args_generated.h` per kernel via tt-metal's `genfiles.cpp` and adds `-I<dir>` to JIT compile; cache key incorporates DFB/sem/RTA/CRTA names |
| New JIT stubs | — | `llk_math_eltwise_binary.h` (empty shim — dead include in `_neg` reduce kernels), `tt-metalium/buffer_types.hpp` (4 enums consumed via `sharded_tensor_addr_gen.hpp`), `DYNAMIC_NOC_X/Y` + `NOC_XY_ADDR` macros, single-arg `reconfig_data_format_*` overloads, DPRINT stream-manipulator stubs (`HEX`/`DEC`/`OCT`/`BIN`/`FIXED`/`DEFAULTFLOAT`/`SETW`/`SETPRECISION`) |
| `reduce_tile<MAX>` | accumulated as `max(DST[i], result)`, which clamped to 0 when DST was zero-initialized and inputs were all negative — broke `ttnn::min` (lowers to `min(x) = -max(-x)`) | per-DST-slot fresh flag: `tile_regs_acquire` marks slots fresh; writers (`copy_tile`, `add/sub/mul_tiles`, `matmul_tiles`) clear it; `reduce_tile<MAX>` overwrites on fresh slots, max-accumulates on dirty ones |
| Code review | — | second-pass review on Phase A + C: relocated `MEM_ZEROS_BASE` to top of L1 (was colliding with bump allocator), aligned `set_async_read_state`/`async_read_with_state` with upstream contract (size from cached state for fits-in-one-packet, src always recomputed), switched header-level macros to `constexpr`, refactored `__emule_matmul_state` from 6 TLS vars into one struct |

**Key insight:** emule's `tile_regs_acquire()` zero-init of DST diverges from real HW (which leaves DST undefined). The previous `reduce_tile<MAX>` accumulation logic worked when the kernel pre-loaded a running max via `copy_tile(cb_acc)` but produced wrong results on the first call after acquire when all inputs were negative. The fresh-flag distinguishes "first call" from "accumulating into a kernel-staged running max" without changing behavior for `tile_regs_acquire`'s zero-init contract that other ops (notably `reduce_tile<SUM>`) rely on.

---

## v10 → v11

| Aspect | v10 | v11 |
|--------|-----|-----|
| tt-metal base | `arminale/emule-metal-base` @ `c812fbb1cc` (first frozen pointer) | `arminale/emule-metal-base` @ `8711ac3d0b` ("Implement warmup flow in demo (#43109)", merge-base of `origin/main` and `origin/bbradel-41067_generic_reduce_host20`, +40 commits) |
| `tt_llk` source | submodule at `tt_metal/third_party/tt_llk` | promoted into the main tree at `tt_metal/tt-llk/` (submodule no longer initialized) |
| New test tiers | — | Tier 5b `ttnn_sum_last_dim_wh` (W-axis reduction, BF16, 3200×64 → 3200×1). A Quasar variant was attempted but the upstream W-reduce factory rejects the Quasar arch (`DataMovementKernel is not supported on Quasar`) — out of emule's scope. |
| Reduce kernel JIT stubs | implicit | added: `ckernel::PoolType` / `ckernel::ReduceDim` namespace alias, `MEM_ZEROS_SIZE`, `experimental::Noc::VcSelection`, `set_async_read_state`, `async_read_with_state` (consumed by `ttnn/cpp/ttnn/kernel_lib/{l1_helpers.hpp, reduce_helpers_*.hpp}`) |

**Key insight:** the merge-base reduce kernels at `8711ac3d0b` reference a layered helper library (`ttnn/cpp/ttnn/kernel_lib/`) that wraps NOC state-cache primitives. The static stub-coverage check missed these because they live behind transitive `#include`s. With the stubs added, the same kernels compile under the emulator JIT and the (already-functional) `reduce_tile<SUM, REDUCE_ROW>` math produces correct PCC.

---

## v9 → v10

| Aspect | v9 | v10 |
|--------|----|-----|
| tt-metal base | `arminale/quasar-rebased` (in-flight working branch) | `arminale/emule-metal-base` @ `c812fbb1cc` (frozen pointer at upstream merge commit `tt-emule support for Quasar (#43091)`) |
| Branch policy | Working branch could be force-pushed | `arminale/emule-metal-base` is a stable pointer at an upstream main commit; downstream consumers can pin against it |

**Key insight:** v9 measured the emulator against the in-flight branch that contained wraparound and BLOCKED-config fixes. The new `arminale/emule-metal-base` is intentionally pinned to the upstream merge commit (which predates those fixes); the resulting failures are the known set of follow-up work not yet merged back to `main`.

---

## v8 → v9

| Aspect | v8 | v9 |
|--------|----|----|
| DFB STRIDED wraparound tests | 4 failing (`DMTest1xDFB4Sx4S`, `DMTest1xDFB2Sx4S` + `_IS` variants) | **All 4 passing** |
| RISC-V atomic tests | 3 segfaulting (`TestAtomicLoadStoreRISCV`, `TestAtomicAddFetchRISCV`, `TestAtomicCASRISCV`) | **All 3 passing** |
| `__emule_local_l1_to_ptr` availability | Defined in `dataflow_api.h` only | Also in `jit_kernel_stubs.hpp` under `#ifndef __EMULE_LOCAL_L1_TO_PTR_DEFINED` — available to every JIT kernel |
| L1 pointer cast JIT patch | Not present | Regex in runner patches `reinterpret_cast<T*>(get_arg_val<uint32_t>(N))` → `reinterpret_cast<T*>((uintptr_t)__emule_local_l1_to_ptr(...))` |
| DFB test `num_entries_in_buffer` (4Sx4S / 2Sx4S) | 29 / 21 — not divisible by `max(P,C)=4` | 28 / 20 — divisible by 4; wraparound still exercised (both > `num_entries=16`) |
| Tier 6 Silicon Toggle | FAIL (expected; requires real hardware) | **PASS** (runs in emulation after env-var unset) |

**Key insight (DFB wraparound):** `num_entries_in_buffer` must be a multiple of `max(P, C)` for every entry to be processed. With `num_entries_in_buffer=29` and `max(P,C)=4`, each producer/consumer does `floor(29/4)=7` iterations covering 28 of 29 entries; entry 28 is never written to the output buffer and stays zero. Choosing values divisible by `max(P,C)` (28 and 20 respectively) eliminates the truncation while preserving the wraparound-exercise invariant.

**Key insight (atomic segfaults):** Quasar atomic kernels call `reinterpret_cast<std::atomic<T>*>(get_arg_val<uint32_t>(0))` where arg[0] is a raw L1 firmware offset (~`0xba780`). On real hardware these offsets are directly dereferenceable by firmware; on x86 emulation they are not valid host addresses, causing an immediate segfault. The fix: (1) move `__emule_local_l1_to_ptr()` into `jit_kernel_stubs.hpp` so every JIT kernel has access to it before any kernel-specific includes; (2) guard the existing definition in `dataflow_api.h` with `#ifndef __EMULE_LOCAL_L1_TO_PTR_DEFINED` to prevent ODR violations; (3) add a JIT preprocessor regex that rewrites the pattern automatically so existing and future kernels with this idiom need no source modifications.

---

## v7 → v8

| Aspect | v7 | v8 |
|--------|----|----|
| tt-metal rebase | Pre-rebase (`3fa4d75355`) | Rebased (1,669 commits newer), 7 fixes applied |
| DFB BLOCKED multi-P/C | All 16 failing (8 DM-DM + 8 TensixDM) | **All 16 passing** |
| BLOCKED consumer model | Round-robin tc_idx on every pop_front | `drain_per_tc`: drain each TC slot fully before advancing |
| BLOCKED TC slot layout | All slots share full buffer range | Per-slot sub-ranges: `base_addr = alloc_base + p*capacity*entry_size` |
| EmuleDFBInterface struct | `broadcast_tc`, `active` | + `drain_per_tc` field between `broadcast_tc` and `active` |
| Rebase fixes | — | JIT stubs, HAL core count, finalize alloc_addr, BLOCKED stride/offset, WH proc_bit, early DFB finalize, BLOCKED drain |

**Key insight:** BLOCKED mode consumers on hardware perform block reads — they exhaust all entries from one producer's TC slot before advancing to the next producer's slot. The emulation was incorrectly round-robining through TC slots on every `pop_front` call, producing a shuffled read order. Additionally, all TC slots shared the full buffer address range instead of each slot having its own contiguous sub-range. Both bugs only manifested with multiple producers or consumers (1P-1C was unaffected since there's only one TC slot).

---

## v6 → v7

| Aspect | v6 | v7 |
|--------|----|----|
| Nfaces UNPACK/PACK | Not applied to general compute ops | All UNPACK (CB→DST) and PACK (DST→CB) ops use nfaces LUT |
| Tilize/Untilize D2M | partial (PCC mismatch) | **all passing** — nfaces in `__llk_pack_tiled` and `copy_tile` fixed all 32 failures |
| Quasar matmul PCC | failing (data corruption) | **passing** — nfaces conversion resolved PCC failures |
| Files modified | — | `common.h`, `matmul.h`, `reduce.h`, `llk_defs.h` |

**Key insight:** On real hardware, UNPACK converts nfaces→row-major when loading from L1/CB into DST, and PACK converts row-major→nfaces when writing back. The emulator was missing these conversions. Element-wise ops (add/sub/mul) previously worked "by accident" because nfaces permutation cancels for identical per-element operations. Non-element-wise ops (matmul, tilize/untilize) failed because element positions were scrambled. The fix applies the nfaces LUT to all general UNPACK and PACK operations, plus the D2M tilize PACK path (`__llk_pack_tiled`).

---

## v5 → v6

| Aspect | v5 | v6 |
|--------|----|----|
| Quasar architecture | Not documented | 12-thread model (8 DM + 4 compute per Neo), 4 MB shared L1 |
| DFB infrastructure | Not documented | TileCounter, TileCounterArray, EmuleDFBInterface, DFBSyncState; STRIDED + BLOCKED access patterns; DFB↔CB bridge |
| Compute engine | DST briefly mentioned (8 slots) | Full UNPACK/MATH/PACK pipeline: nfaces LUT, mode-aware DST (16/8 slots), L1 accumulation, AVX2/FMA matmul |
| Implemented compute ops | Not enumerated | 11 operations: copy/pack tile, add/sub/mul_tiles, matmul_tiles/block, reduce_tile, L1 acc toggle |
| CSR emulation | Not documented | NEO_ID, TRISC_ID via TLS; mhartid regex patch |
| JIT patches | mhartid only | + fence instruction, L1 pointer cast patches |
| Reference docs | None | `docs/DFB_EMULATION.md`, `docs/QUASAR_EMULATION.md`, `docs/TEST_COVERAGE_TODO.md` |

---

## v4 → v5

| Aspect | v4 | v5 |
|--------|----|----|
| tt-mlir base | Pre-uplift | Rebased on `milant/uplift_mar_25` (new LLK APIs, expanded test suites) |
| Pack untilize | Not supported | `experimental::pack_untilize_block` with `copy_tile` + `__llk_pack_untilize` |
| LLK CB stubs | Not present | `llk_wait_tiles`, `llk_pop_tiles`, `llk_push_tiles`, `llk_wait_for_free_tiles` delegating to `cb_api.h` |
| SFPU ops | Stubs only (no logic) | Functional: `abs_tile`, `exp_tile`, `negative_tile`, `typecast_tile` with DST element-wise math |
| CB API | uint32_t only | + int32_t overloads for D2M-generated code |
| Coordinate API | Not present | `get_absolute_logical_x/y` in compute context |

---

## v3 → v4

| Aspect | v3 | v4 |
|--------|----|----|
| Program runner | 674 lines | 829 lines (parallel JIT compile, HAL-based semaphore base) |
| Semaphore base | Dynamic `max(0xFFE00, max_cb_end)` from CB pre-scan | HAL-based: `kernel_config_base + prog_config.sem_offset` (matches real firmware) |
| JIT compilation | Sequential | Parallel via `std::async` for cache misses |
| Regression scripts | `run_regression.sh` | + `run_d2m_regression.sh` (D2M test files) |
