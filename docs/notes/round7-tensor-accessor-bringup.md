# Round 7 — Sharded TensorAccessor Bring-up: Complete Walkthrough

Date: 2026-06-01
Branch: `arminale/tensor-accessor` (off `arminale/mass-llk-bringup`)

## The goal

Unblock ~500+ sharded ttnn tests by giving emule's JIT a working
`TensorAccessor` for sharded tensors. Strategy: **delete emule's
simplification shim and route through upstream's full implementation**,
instead of reimplementing ~600 LOC of iterator/dspec/page machinery.

## What was committed (5 tt-emule commits + 1 tt-metal commit)

### tt-emule branch `arminale/tensor-accessor`

| Commit | Subject | Purpose |
|---|---|---|
| `26f1a18` | Phase 1: delete simplification shim | Remove `include/jit_hw/api/tensor/tensor_accessor.h`; let JIT include resolution fall through to upstream's `tt_metal/hw/inc/api/tensor/tensor_accessor.h`. |
| `877ff86` | Phase 2: noc_traits specializations | Add `noc_traits_t<tensor_accessor::Page>` and `noc_traits_t<ShardView<Accessor>>` so upstream's iterator-yielded types resolve NOC addresses through emule's `__emule_resolve_noc_addr`. |
| `25f37a5` | InterleavedPow2AddrGenFast shim | Add the pow2 fast-path variant in `include/jit_hw/internal/dataflow/dataflow_api_addrgen.h`. |
| `3a54185` | Promote `test_pad_subcoregrids` | New entry in `scripts/run_ttnn_pytests.sh` — 57 tests now passing. |
| `41f0043` | Skill update | New "Pass-through-to-upstream" section in `.claude/skills/llk-bringup/SKILL.md` documenting the workflow + traps. |

### tt-metal branch `arminale/emule-dispatch-fix` (local only)

| Commit | Subject |
|---|---|
| `593e36cfa63` | `emule: emit IS_NOT_POW2_NUM_DRAM_BANKS + LOG_BASE_2_* + KERNEL_BUILD` |

This is the most load-bearing fix and is **not yet pushed** — needs explicit
authorization per the standing rule. The override the user granted for this
round applied only to the tt-emule branch.

## The findings — what we learned about emule's JIT vs upstream

### Finding 1: The simplification shim was unnecessary

Emule had a 76-line `tensor_accessor.h` shim explaining "Implementation is
simplified: we only store `bank_base_address` + `page_size`..." Inspecting
upstream's full version: pure C++17 templates, no silicon-specific
intrinsics. The transitive include chain (`page.h`, `dspec.h`,
`tensor_accessor_args.h`, iterator headers) all reduces to pure C++ that
depends only on primitives emule already shims (`InterleavedAddrGen<>`,
`dram_bank_to_noc_xy[][]`, `bank_to_dram_offset[]`).

**Lesson:** simplification shims are tech debt when the upstream code is
generic. Delete them and let JIT include resolution flow through.

### Finding 2: `KERNEL_BUILD` is a load-bearing gate

Upstream uses `#if defined(KERNEL_BUILD) || defined(FW_BUILD)` in several
headers to switch between host-side and kernel-side codepaths. Emule's JIT
compiles kernel code as x86 host code, so neither was defined. Result:

- `tensor_accessor.h:352-486` — the entire interleaved DRAM specialization
  (inheriting `InterleavedAddrGen<true>`) was disabled. The deduction guide
  at line 488 still routed `TensorAccessor(args, addr)` to that
  specialization, leading to "incomplete type" errors.
- `internal/tensor/dspec.h:16` — `get_common_arg_addr` forward decl was also
  gated, so dspec.h was emitting a stub that collided with emule's existing
  definition.
- `sharding_addrgen.hpp` — its KERNEL_BUILD-gated overload references
  `InterleavedPow2AddrGenFast`, which then became required.

**Fix:** define `KERNEL_BUILD=1` in the JIT defines map. Side effect: had
to make `get_compile_time_arg_val<N>` bounds-safe (return 0 for
`N >= size`) so upstream's `TensorAccessorArgs<CTA_OFFSET>::args_config =
get_ct_arg(0)` constexpr parsing doesn't fail when the host emits fewer
CTA slots than the template scans.

### Finding 3: The non-pow2 bank-count bug — the root of S1's 0.99 ATOL

This is the **most subtle bug** found this round. WH-N150 has 12 DRAM banks
— not a power of two.

Upstream's `interleaved_addr_gen::get_bank_offset_index<DRAM>`:

```cpp
template <bool DRAM>
FORCE_INLINE uint32_t get_bank_offset_index(uint32_t id) {
    if constexpr (DRAM) {
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
        return udivsi3_const_divisor<NUM_DRAM_BANKS>(id);
#else
        return id >> LOG_BASE_2_OF_NUM_DRAM_BANKS;
#endif
    }
    ...
```

Real tt-metal builds set one of these defines in
`jit_build/build_env_manager.cpp:118-129` based on whether the bank count
is pow2. **Emule's `build_kernel_defines` was emitting `NUM_DRAM_BANKS=12`
but neither `IS_NOT_POW2_NUM_DRAM_BANKS` nor
`LOG_BASE_2_OF_NUM_DRAM_BANKS`.** With `LOG_BASE_2_OF_NUM_DRAM_BANKS`
undefined, the preprocessor treats it as 0, so the bit-shift path silently
became `id >> 0 = id`. Then `bank_index = id - id*12 = -11*id` → unsigned
underflow → out-of-bounds array access → essentially random bank routing
for `id != 0` (id=0 accidentally maps to bank 0).

**Fix in `emulated_program_runner.cpp:989-1009`:** mirror upstream's logic
— emit `LOG_BASE_2_OF_NUM_DRAM_BANKS` if pow2, else
`IS_NOT_POW2_NUM_DRAM_BANKS`. Same for L1 banks (currently 1, pow2 with
shift=0).

This single fix routes interleaved DRAM addresses correctly for all
non-pow2 bank counts.

### Finding 4: `noc_traits_t` partial-template chain

Upstream's `TensorAccessor::shard_pages()` returns a
`ShardPages<TensorAccessor>` proxy whose iterator yields
`tensor_accessor::Page` (an opaque struct holding a pre-baked NOC address).
Upstream's `noc.async_write(p, dst, ...)` then dispatches through
`noc_traits_t<tensor_accessor::Page>::src_addr(...)` — a specialization
emule didn't have.

Emule's existing `noc_traits_t<TensorAccessor<DSpecT>>` extracts addresses
via `acc.get_noc_addr(page_id, ...)` → `__emule_resolve_noc_addr`. The new
specializations follow the same pattern:

```cpp
template <>
struct noc_traits_t<tensor_accessor::Page> {
    static uintptr_t src_addr(const tensor_accessor::Page& src, ...) {
        uint64_t noc_addr = src.noc_addr() + args.offset_bytes;
        return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
    }
    // dst_addr similar
};

template <typename Accessor>
struct noc_traits_t<ShardView<Accessor>> {
    static uintptr_t src_addr(const ShardView<Accessor>& src, ...) {
        uint64_t noc_addr = src.get_noc_addr(args.shard_id, ...);
        return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
    }
};
```

### Finding 5: `InterleavedPow2AddrGenFast` was a transitive blocker

Once `KERNEL_BUILD` was defined, `sharding_addrgen.hpp:320-350` activated
overloads referencing `InterleavedPow2AddrGenFast<DRAM>`. Emule had
`InterleavedAddrGenFast` but not the pow2 variant. Every kernel that
includes `sharding_addrgen.hpp` (e.g. `reader_unary_start_id.cpp` used by
untilize / repeat_interleave / many DM ops) silently broke. The shim
addition in `dataflow_api_addrgen.h` follows the existing
`InterleavedAddrGenFast` pattern with shift-based page-size arithmetic.

### Finding 6: NOC encoding shift mismatch (sanity check, no fix needed)

Wormhole's real `NOC_ADDR_COORD_SHIFT = 32`, Blackhole/Quasar use 36.
Emule's shim uses 36 ("Blackhole-style") throughout. Traced a few NOC
addresses and confirmed the kernel-side `get_noc_addr_helper` packs at 36
and the host-side `__emule_resolve_noc_addr` decodes at 36 — internally
consistent.

## The remaining bug — Phase 3 task #59 (CLOSED in commit `6d05222`)

> Original diagnosis (kept for trail; **the diagnosis was wrong**):
>
> *`test_s2i_dram_height_sharded` lands 240 of 320 sticks; last 8 sticks
> per shard arrive as zeros. Hypothesis: emule's CB-to-buffer binding
> exposes only 24 of 32 sticks in the readable region.*

The CB-binding hypothesis didn't survive contact with the code — emule's
CB binding doesn't truncate at sub-page granularity. The actual cause:

`reset_l1_bump` was zeroing the top 512 B of every core's L1 each
program, where the allocator places user-visible buffers. The MEM_ZEROS
region is supposed to live at `MEM_ZEROS_BASE = 0xFFE00` and be
512 B wide for the `zero_buffer()` helper — but emule's reset was using
`l1_size - 512` as the start, which on WH-N150 happens to land inside
where tt-metal's allocator places stick buffers for these uneven shard
specs. Each program's reset wiped the last 8 sticks of L1-resident
shards before the kernel even ran. Subsequent NOC writes were faithful;
the writers were just reading already-zeroed memory.

Fix: rezero at MEM_ZEROS_BASE specifically (commit `6d05222`). Task #59
closed. The tt-metal companion commit `593e36cfa63` (emit
`IS_NOT_POW2_NUM_DRAM_BANKS` / `LOG_BASE_2_OF_NUM_DRAM_BANKS`) is now
pushed to `arminale/tensor-accessor-fix`.

## Tooling infrastructure added during debug (then reverted)

I temporarily added these to `emulated_program_runner.cpp` to debug S1,
then reverted:
- `TT_EMULE_KEEP_JIT_SRC` extension to also keep the JIT build dir + write
  `cmd.sh`
- `TT_EMULE_TRACE_NOC=<path>` to log every `__emule_resolve_noc_addr` call

Reverted because they were single-use diagnostics. If they need to come
back as durable env-var-gated tools, they're easy to recreate from the
conversation history.

## What's now passing that wasn't before

- `test_pad_subcoregrids` — 57/58 (1 expected-failure test)
- `test_untilize_nd_shard_to_same_shard_spec_uneven_input_shard_spec` —
  **36/36** (the 4 uneven-shape variants were the MEM_ZEROS bug, now closed)
- The InterleavedPow2AddrGenFast shim plus IS_NOT_POW2 fix also benefit
  any test that uses `sharding_addrgen.hpp` overloads — likely a broader
  unblock than what I sampled.

The "Round 8 sharded harvest" pass (post-Round-7 work; see
`round8-sharded-harvest.md`) added two more emule fixes on top of these
and pushed the sharded-pass count further:
- `TensorAccessor::get_aligned_page_size()` branch in
  `noc_async_{read,write}_page` → **+648 test_full_like sharded** (0/648 → 648/648).
- TRID no-op shims (`noc_async_{read,write}_set_trid`,
  `ncrisc_noc_*_with_transaction_id_{flushed,sent}`) → unblocks JIT
  compile for the stick-layout reader kernel; +4 passes on
  test_interleaved_to_sharded. The 23 test_pad sharded variants compile
  but surface a separate non-aligned multi-slot data-mismatch
  (documented in the Round 8 register).

## Branch state at end of session

- `arminale/tensor-accessor` (tt-emule): 5 commits ahead of
  `arminale/mass-llk-bringup`, all pushed.
- `arminale/tensor-accessor-fix` (tt-metal): the companion tt-metal
  commit (`593e36cfa63` — emit `IS_NOT_POW2_NUM_DRAM_BANKS` /
  `LOG_BASE_2_OF_NUM_DRAM_BANKS`) is pushed to this branch. The Round 8
  harvest commits live on top of `arminale/tensor-accessor`.
