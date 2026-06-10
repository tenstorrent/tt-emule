# emule-mapping: how silicon HW concepts map to emule simulation

Catalog of the abstractions tt-emule uses to simulate Tenstorrent
silicon. Each row: a silicon HW concept, what the real HW does, how
emule simulates it, and which file owns the implementation. Use this
table when picking a mock strategy — match the new mock's "shape" to
an existing strategy where possible.

This is the strategy-picking catalog cited by `sage-*` agents and the
`/implement-mock` skill. For the authoritative per-subsystem deep-dives, see the
docs the project's `IMPLEMENTATION_REPORT.md` indexes — especially
`docs/{l1,dram,dest,cb,noc}-emulation.md`, `docs/metal-integration.md`,
`docs/tilize-untilize-pack.md`, and `docs/cb-dataformat.md`. This table stays
concise; those docs carry the detail (and are the source of truth if they
disagree with a row here).

---

## 1. Memory model

### 1.1 L1 SRAM

| Silicon | Emule | Owner |
|---|---|---|
| 1.5 MB per Tensix core, 32-bit firmware addresses, RISC dereferences directly | Per-core `mmap(MAP_PRIVATE\|MAP_ANONYMOUS\|MAP_32BIT)` of `Core::l1_size()` bytes. 2-MB-aligned slot. Worker cores get MAP_32BIT (low 4GB) so the truncated host pointer is itself a valid host pointer. | `tt_emule::Core` (`tt-emule/include/tt_emule/device.hpp`); `L1Pool` (`tt-emule/include/tt_emule/l1_pool.hpp`) |

**Key conversion**: `__emule_local_l1_to_ptr(uint32_t l1_addr)` —
returns `__emule_bridge_l1 + l1_addr` for firmware offsets, or casts
back to host pointer for already-absolute values.
Lives in `tt-emule/include/jit_hw/jit_kernel_stubs.hpp`.

**Pitfall**: `Core::reset_l1_bump()` must NOT memset the top of L1 — the
tt-metal allocator places user buffers there, so wiping it corrupts
host-written tensor data.

### 1.2 DRAM

| Silicon | Emule | Owner |
|---|---|---|
| Banked across DRAM cores at the chip edges; `InterleavedAddrGen` computes which bank a page lives in | Per-DRAM-core `mmap` (regular, no MAP_32BIT — saves the low 2GB) sized per bank. `bank_to_dram_offset[]` + `dram_bank_to_noc_xy[2][]` (sized `MAX_NUM_BANKS`) populated from `metal_SocDescriptor` at program init. `NUM_DRAM_BANKS` is dynamic per arch (`get_num_dram_views()`; 12 on WH-N150, non-pow2). | `SWEmuleChip` (UMD level), `tt-emule/include/jit_hw/internal/dataflow/dataflow_api_addrgen.h`. See `docs/dram-emulation.md`. |

**Bridge**: `__emule_dram_ptr(uint64_t offset)` (`extern "C"` resolved at dlopen).

### 1.3 DST register file

| Silicon | Emule | Owner |
|---|---|---|
| Compute engine's per-tile float register file; format depends on PACK config | `thread_local float __emule_dst[__EMULE_DST_TILES][__EMULE_TILE_ELEMS]` — always row-major float32, regardless of CB format (16 bf16 / 8 fp32 active slots via `DST_ACCUM_MODE`) | `tt-emule/include/jit_hw/api/compute/common.h`. See `docs/dest-emulation.md`. |

`tile_regs_acquire()` zeros active tiles; `tile_regs_commit/wait/release` are no-ops.

---

## 2. NOC

### 2.1 NOC address encoding

| Silicon | Emule |
|---|---|
| Coordinates packed above a local-L1 offset; bit widths vary per arch | Unicast `(y << NODE_ID_BITS+LOCAL_BITS) \| (x << LOCAL_BITS) \| offset`; multicast adds an end-coord pair. `NOC_ADDR_LOCAL_BITS=36`, `NOC_ADDR_NODE_ID_BITS=6`. **See `docs/noc-emulation.md` §2.1 for the authoritative encoding** (don't rely on this row for exact bit positions). |

### 2.2 NOC transactions

| Silicon | Emule | Owner |
|---|---|---|
| Async NOC transactions; `noc_async_*_barrier()` waits for ack from receiver | Synchronous memcpy via `__emule_resolve_noc_addr` bridge; barriers are no-ops | `__emule_resolve_noc_addr` and `__emule_multicast_write` in `${TT_METAL_DIR}/tt_metal/impl/emulation/emulated_program_runner.cpp` |

**Critical fix**: `__emule_resolve_noc_addr` masks `l1_offset` with
`L1_SLOT_MASK = 0x1FFFFF` (2MB-1). This handles the "NOC-OR truncated
host pointer" pattern where a consumer kernel does
`dst_noc_coord | (uint64_t)get_write_ptr(cb)` instead of the canonical
`get_noc_addr(x, y, get_write_ptr(cb))`. Without the mask the resolver
gets a host pointer where it expected a firmware offset.

### 2.3 Multicast

| Silicon | Emule |
|---|---|
| Single NOC packet broadcast to rectangle of cores via cmd-buf registers | Loop over rectangle and memcpy to each receiver. `noc_async_write_multicast` resolves to `__emule_multicast_write` |
| `NOC_CMD_BRCST_SRC_INCLUDE` bit controls whether the sender NIU receives its own packet | 4-argument bridge `__emule_multicast_write(mcast_addr, src, size, include_self)`. `noc_async_write_multicast` passes `include_self=false`; the `_loopback_src` variant passes `true`. Loopback-vs-not is faithfully modeled (the runner skips/includes the sender's coords in the rectangle). |

The Mcast semantic rewrite pattern used in downstream consumers
replaces 25+ raw NOC_CMD_BUF register writes with a single
`noc_async_write_multicast` + `noc_semaphore_set_multicast`.

---

## 3. Compute pipeline

### 3.1 UNPACK / MATH / PACK pipelines

| Silicon | Emule |
|---|---|
| Three async pipelines on the compute core; LLK macros gate code into specific pipelines | All three macros (`UNPACK(x)`, `MATH(x)`, `PACK(x)`) just **evaluate the expression inline**. No real pipeline state. |

Implication: many silicon kernels do `PACK((sigmoid_tile(0)))` to
gate the sigmoid into the PACK pipeline; on emule the call simply
runs in line on the compute thread.

### 3.2 LLK templates

| Silicon | Emule |
|---|---|
| `llk_math_*`, `llk_pack_*`, `llk_unpack_*` configure UNPACK/MATH/PACK pipeline state and emit specific instruction sequences | Mostly no-op stubs. There's no HW pipeline state. The actual math happens in the compute API (`api/compute/common.h::add_tiles`, `mul_tiles`, etc.) |

When an op's TRISC body uses a deep LLK chain (>5 LLK template errors
on first compile), use the **semantic rewrite** pattern instead of
stubbing every LLK header (see Strategy C in `/implement-mock`).

### 3.3 nfaces tile layout

| Silicon | Emule |
|---|---|
| Tiles are face-packed: 32×32 tile = 4 faces × 256 elements (16×16 each), packed sequentially | `__emule_nfaces::rowmajor_to_nfaces[1024]` permutation table maps row-major index i → face-packed index ni. UNPACK reads CB nfaces → row-major DST; PACK writes row-major DST → CB nfaces. Symmetric. |

Owner: `tt-emule/include/jit_hw/api/compute/nfaces.h`.

### 3.4 Compute primitives

| Silicon | Emule |
|---|---|
| `add_tiles/sub_tiles/mul_tiles` use FPU dataflow with format-specific paths | Direct scalar math on row-major float32 DST; format-aware path picks bf16/fp32 read+conversion on UNPACK side and write on PACK side |
| `matmul_tiles` uses MM accumulator with optional transpose | `__emule_dst[idst][r*32+c] += sum_k A[r,k]*B[k,c]` (or transposed if `__emule_mm_transpose_in1`). AVX2-enabled FMA inner loop. |
| `recip_tile`, `exp_tile`, `sigmoid_tile`, `silu_tile`, `relu_tile` | `expf`, `1/x`, `1/(1+exp(-x))`, `x*sigmoid(x)`, `max(0,x)` per element on row-major DST |
| `reduce_tile<MAX/SUM, REDUCE_ROW/COL/SCALAR>` | Real per-axis reduction on DST |
| `sub_tiles_bcast<COL>` etc. | `any_tiles_bcast<op, bcast, bool acc_to_dest=false>` with `bcast_b(r,c) = load_b(r,0)` for COL, `load_b(0,c)` for ROW, `load_b(0,0)` for SCALAR. `acc_to_dest=true` accumulates DST in place; `false` overwrites. Silicon's `llk_math_eltwise_binary<...>` takes this as a runtime flag; emule does it as a compile-time template. (Note: callers like `deepseek_mul_tiles_bcast_scalar` pass `acc_to_dest=true`.) |

Owner files:
- `tt-emule/include/jit_hw/api/compute/common.h` (add/sub/mul_tiles, pack_tile)
- `tt-emule/include/jit_hw/api/compute/matmul.h`
- `tt-emule/include/jit_hw/api/compute/reduce.h`
- `tt-emule/include/jit_hw/api/compute/bcast.h`
- `tt-emule/include/jit_hw/api/compute/eltwise_unary/{recip,exp,sigmoid,silu,relu}.h`

### 3.5 sfpi:: vector intrinsics

| Silicon | Emule |
|---|---|
| `sfpi::vFloat`, `v_if(cond) {...}`, `sfpi::dst_reg[k]`, `sfpi::reinterpret` — SIMD on SFPU | Modeled as a 32-lane SIMD type **executed scalar-per-lane** (`sfpi.h`), not bit-exact to the silicon SFPU. Usable for simple sfpi bodies; deep/complex sfpi chains are still often cleaner to semantic-rewrite (Strategy C). |

Owner: `tt-emule/include/jit_hw/sfpi.h`.

### 3.6 Cross-RISC synchronization

| Silicon | Emule |
|---|---|
| `unified_kernels::sync_riscs_enter/exit` + `invalidate_l1_cache` — multi-RISC barrier with explicit L1 cache invalidate | **No-op under `__EMULE_JIT_MODE`** — emule's CBSyncState (mutex+condvar) already serializes consumer-producer; no L1 cache to invalidate |

Gated in `mcast/op.hpp` post-mcast sync block.

---

## 4. Circular buffers (CBs)

### 4.1 CB sync

| Silicon | Emule |
|---|---|
| Producer-consumer ring; sync via in-L1 semaphores | `CBSyncState` struct: `base` + `page_size` + `num_pages` + `page_mask` + `write_idx` + `read_idx` + atomic `occupied` + mutex + 2 condvars. `cb_wait_front(cb, n)` blocks until `occupied >= n`; `cb_push_back(cb, n)` advances `write_idx`/`occupied` + notifies. See `docs/cb-emulation.md`. |

Owner: `tt-emule/include/tt_emule/cb_sync_state.hpp`; per-core array
in `tt_emule::Core::cb_sync_states_[MAX_CBS]`.

The CB sync is the **single most important shared-state primitive**:
it's what lets emule run BRISC + NCRISC + TRISC threads concurrently
within a single core.

### 4.2 CB-tile reads/writes from compute API

`__emule_compute::cb_read_ptr_at(cb_id, tile_offset)` and
`cb_write_ptr_at(cb_id, tile_offset)` — return raw `uint8_t*` into
Core's L1.

### 4.3 Tile format dispatch

`__emule_compute::cb_is_32bit_format(cb_id)` is **enum-driven**: it reads the
CB's real `DataFormat` (from the `EMULE_CB_DATA_FORMATS` JIT define) and only
falls back to the `page_size > 2048` heuristic when the format is `Invalid`
(standalone builds). Sibling predicates: `cb_is_bfp8_b_format` /
`cb_is_bfp4_b_format` / `cb_is_uint16_format`. The page-size heuristic alone is
**wrong** for small-page int32/uint32, thin tiles, and the bf16-vs-uint16
ambiguity (both 2048 B) — always use the enum predicate when adding a
format-aware path. Drives `add_tiles`/`mul_tiles`/`pack_tile`. See
`docs/cb-dataformat.md` and `docs/cb-emulation.md`.

---

## 5. Semaphores

| Silicon | Emule |
|---|---|
| In-L1 32-bit words at fixed offsets, atomic ops from RISCs and NOC remote writes | Same — at `EMULE_SEM_BASE + id*16` per core. Accessed via `std::atomic_*` on raw `uint32_t`. NOC remote sets go through `__emule_resolve_noc_addr`. `noc_semaphore_wait` is a spin-wait with `std::this_thread::yield()`. |

Owner: `tt-emule/include/jit_hw/api/dataflow/dataflow_api.h::noc_semaphore_*`.

---

## 6. Kernel JIT compile + dispatch

| Silicon | Emule |
|---|---|
| RISC-V kernels compiled at build time, loaded onto cores by tt-metal | JIT: emit wrapper.cpp + `#include patched_kernel.cpp` (with regex-rewritten L1 casts) + clang++ → `.so` → dlopen. Each (core, processor) gets its own `std::thread` running `__emule_kernel_entry()`. |

Owner: `emulated_program_runner.cpp::jit_compile_kernel`,
`execute_program_emulated`.

### 6.1 Source rewriter

Three regex patterns rewrite L1 pointer casts in the top-level
kernel.cpp (NOT in included headers — that's why per-op patches
exist):

1. `reinterpret_cast<T*>(get_arg_val<uint32_t>(N))`
2. `reinterpret_cast<T*>(static_cast<uintptr_t>(get_arg(args::NAME)))`
3. `reinterpret_cast<volatile tt_l1_ptr T*>(<expr>)`

All become wrapped with `__emule_local_l1_to_ptr(...)`.

### 6.2 JIT cache key

`<src_path>:<compile_args>:<named_args>:<defines>:<metal2_suffix>`

When a downstream consumer aggregates per-op headers into a hub, the
hub's content hash should be added as an extra cache-key component so
edits to consumer op headers invalidate cached `.so`s.

---

## 7. The three strategies

When implementing a mock, pick from:

### A. Stub in `jit_hw/`

Default for new compute primitives, dataflow helpers, generic API
surface. New file at `tt-emule/include/jit_hw/api/{compute, dataflow,
tensor}/<name>.h`. No `__EMULE_JIT_MODE` guard needed (jit_hw is
already inside `-I` ahead of tt-metal's hw/inc).

### B. Per-op `#ifdef __EMULE_JIT_MODE` patch in a consumer's op header

For per-op silicon-specific patterns that can't be generically stubbed
in `jit_hw/`. Most common: constexpr L1 firmware address cast routed
through `__emule_local_l1_to_ptr`.

### C. Semantic rewrite under `__EMULE_JIT_MODE`

For deep LLK/sfpi chains. Gate off the entire silicon include block
+ operator() body, reimplement directly against CB + DST math.
Validated 4× (RMSNorm, clamped_silu, Mcast, eltwise_mul).

---

## 8. Out-of-scope (today)

| Silicon | Status |
|---|---|
| Fabric / multichip | All `tt_metal/fabric/hw/inc/**/*.{h,hpp}` shadowed as opaque types — ops PARSE but don't execute. |
| Bit-exact sfpi:: SIMD math | `sfpi.h` runs scalar-per-lane (see §3.5) — functional but not silicon-bit-exact. |
| Ethernet / socket / D2D / H2D | Same status as fabric. |
| Quasar architecture | Dedicated regression coverage exists for supported Quasar DFB/compute/semaphore/atomics paths (`TT_EMULE_ARCH=quasar`, `quasar_Q1.yaml`); other Quasar-specific APIs may still need targeted mocks. |
| topk multi-core dataflow | `topk` blocks on a `VALID`/`INVALID` NoC-semaphore enum gap in `reader_final_topk`/`writer_local_topk` — tracked in issue #137. (Compute primitives `matmul_tiles`/`matmul_block` and the `{deepseek,glm,kimi}_moe_gate` shims ARE implemented — no longer out of scope.) |
| Mock cluster DRAM bank shape | "No DRAM bank exists for core 7-0" issues when bank topology doesn't match silicon. |

---

