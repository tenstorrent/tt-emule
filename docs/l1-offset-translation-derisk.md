# L1 offset addressing — patch taxonomy & per-site audit

Companion to [l1-offset-translation.md](l1-offset-translation.md). This is the
reference behind the design: the taxonomy of kernel L1-dereference idioms and the
rule that covers each, and the site-by-site audit of everything the offset switch
touches (what changes vs. what is already correct).

## 1. Patch taxonomy (real compiled corpus)

Scoped to the kernels emule actually JIT-compiles (the C++ regression + ttnn pytest
kernels under `tt_metal/**/kernels`, `tests/**/kernels`, `ttnn/cpp/ttnn/**/kernels`,
`ttnn/cpp/ttnn/kernel`) — **not** the `models/demos/**` trees, which emule does not
run. Kernel bodies turn an L1 address into a pointer through a small, closed set of
idioms; each is covered by one rewrite rule (or is already handled).

| Family | Idiom | Covered by |
|---|---|---|
| get_arg_val cast | `reinterpret_cast<T*>(get_arg_val<uint32_t>(N))` | existing `l1_arg_ptr_re` |
| Metal-2.0 arg cast | `reinterpret_cast<T*>(static_cast<uintptr_t>(get_arg(...)))` | existing `l1_named_arg_ptr_re` |
| tt_l1_ptr cast, one-step | `reinterpret_cast<volatile tt_l1_ptr T*>(get_write_ptr(cb))` | **P1** |
| tt_l1_ptr cast, two-step | `uint32_t a = get_write_ptr(cb); … reinterpret_cast<volatile tt_l1_ptr T*>(a)` | **P1** (deref cast re-adds the attribute) |
| tt_l1_ptr C-style | `(volatile tt_l1_ptr T*)a` | **P1** is `reinterpret_cast`-only today → C-style falls to the SIGSEGV net |
| attr-less get_*_ptr cast | `reinterpret_cast<uint16_t*>(get_write_ptr(cb))`, `cb.get_write_ptr()` | **P2** |
| get_semaphore cast | `reinterpret_cast<volatile uint32_t*>(get_semaphore(id))` | **P1** (if tt_l1_ptr) / **P2** |
| Semaphore class | `Semaphore s(id); s.up()/.wait()` | class edit (`atom()` rebases) — no kernel rewrite |
| Quasar DFB | `reinterpret_cast<T*>(dfb.get_read_ptr() << N)` | falls to the SIGSEGV net (Quasar) |
| reverse narrowing | `reinterpret_cast<uint32_t>(ptr)` | **reverse rule** (subtract `bridge_l1`) |
| reverse C-style | `(uint32_t)ptr` (fabric/CCL) | **gap R2** — caught loudly by the debug assert |

**Rule ordering** (`apply_x86_rewrites`): P1 (tt_l1_ptr) runs before the get_arg_val
rules so their output is not re-wrapped; P2 excludes `tt_l1_ptr` by negative lookahead
so it never overlaps P1; the reverse rule matches `reinterpret_cast<uint32_t>` (no
`*`) and cannot re-match the pointer-typed forward outputs. Each site is wrapped
exactly once.

**Loud-failure net.** A missed forward site dereferences a small 0-based offset as a
pointer → unmapped low page → immediate SIGSEGV at the real kernel `file:line`
(`#line` preserved). Unlike the old aliasing model (where a missed translation
happened to "work" because the value already was a pointer), coverage gaps here
announce themselves. The rule set is grown against real kernels rather than trying to
enumerate every idiom up front.

## 2. Per-site audit (BREAKS = changed; SAFE = already correct)

### Changed (the offset switch)

| Site | File | Change |
|---|---|---|
| `__emule_l1_translate` | `jit_hw/asan/asan_l1_checks.h` | drop the `>= l1_base` disambiguation → unconditional `bridge_l1 + offset`; debug assert `offset < l1_size` |
| `ThreadCommonCtx::l1_size` | `jit_hw/internal/emule_thread_ctx.h` + runner launch | new POD field, set beside `bridge_l1` (keeps the assert out of the host `device.hpp` include) |
| `get_semaphore` ×2 | `dataflow_api.h`, `jit_kernel_stubs.hpp` | drop `l1_base +` → pure offset (kept byte-identical under `__EMULE_GET_SEMAPHORE_DEFINED`) |
| `Semaphore` | `jit_hw/api/dataflow/noc_semaphore.h` | store the offset; `atom()` rebases `bridge_l1 + offset` directly |
| `__emule_sem_atomic` | `dataflow_api.h` | receives an already-translated pointer; trivial cast; no per-chip remap |
| `get_write_ptr`/`get_read_ptr` | `jit_hw/api/cb_api.h` | subtract `bridge_l1` (ring stays in host-pointer space) |
| ASAN `cb_resolve` `cb_start` | `jit_hw/asan/asan_l1_checks.h` | rebase `cb.base` into offset space for the offset-vs-offset comparison |
| `Core::l1_alloc` | `include/tt_emule/device.hpp` | return a 0-based offset |
| DFB base setup | runner `allocate_dfbs_on_core` | `finalize_addr` = offset (found-flag sentinel, not `!= 0`); `base = l1_data() + base_addr` |
| forward/reverse rules | runner `apply_x86_rewrites` | P1, P2, reverse-narrowing |

### Already correct (no change)

| Site | Why |
|---|---|
| `noc_async_read/write` + multicast | already translate the local address via `__emule_local_l1_to_ptr` |
| `__emule_resolve_noc_addr`, `__emule_multicast_write` | decode the NOC-local offset then `l1_ptr(offset)` — offset-based |
| `__emule_addr_to_offset` | `& 0x1FFFFF` is idempotent for an in-L1 offset (kept as a NOC-construction guard) |
| addr-gen (`TensorAccessor`, `InterleavedAddrGen*`) | build NOC addresses only; never a local raw deref |
| `init_core_cb_sync` / `init_core_semaphores` | already call `Core::l1_ptr(offset)` |
| rt-args (`ctx->rt_args`, `get_arg_addr`, `get_arg_val`) | `rt_args` is a host-pointer array; arg *values* are device offsets translated at the cast |
| compute-side CB accessors (`api/compute/common.h`) | return host pointers consumed internally, not kernel-visible uint32s |
| ASAN sem/tensor/padding range tables | already expressed as offsets |
| DRAM | reached via bank offsets on non-`MAP_32BIT` backing; never used the host-pointer identity |

## 3. Known gaps / follow-ons

- **R2 — C-style `(uint32_t)ptr` reverse narrowing** (fabric/CCL). Not yet rewritten
  (`-fms-extensions` still downgrades the truncation to a warning, but it is no longer
  value-preserving under offsets). Caught loudly by the `__emule_l1_translate` debug
  assert (out-of-range offset). Add a name-constrained rewrite when a real failing
  kernel surfaces; a blanket `(uint32_t)EXPR` rule would mistranslate non-pointer int
  casts.
- **Quasar DFB shift cast** (`reinterpret_cast<T*>(dfb.get_read_ptr() << N)`) — falls
  to the SIGSEGV net; add a targeted rule if the Quasar suite trips it.
- **Worker-L1 placement** — worker L1 is still `MAP_32BIT` low-4 GB. Removing that
  (plain 64-bit `mmap`) to fit a >4 GB mesh in one process is a separate,
  behavior-neutral change enabled by this model (see l1-offset-translation.md).
- **`__emule_chip_relative_l1` (DM-1)** — removed; one shared offset space makes the
  per-chip global-semaphore pointer remap unnecessary.
