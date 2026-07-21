# JIT source patch pass (`apply_x86_rewrites`)

Reference for every regex the emule JIT applies to kernel source before
compiling it for the x86 host. The pass is a self-contained module in tt-emule —
public API `tt::emule::patch_kernel_source` in `include/tt_emule/kernel_patcher.hpp`,
rules in `include/tt_emule/detail/kernel_patcher.hpp` — which the tt-metal emulation
runner (`tt_metal/impl/emulation/emulated_program_runner.cpp`) calls before compiling.
This document enumerates the regexes, the code each one targets, the rewrite it performs, and why.

## Where this runs in the pipeline

`preprocess_tu_recursive(src_path, …)`:

1. Reads the kernel `.cpp` (or an included header) as raw text.
2. Calls **`apply_x86_rewrites(src)`** — the rewrites below — on that text.
3. Scans the text for quote-includes (`#include "…"`) and **recurses** into each
   one it can resolve against the kernel include roots, so a shared helper header
   (e.g. `kernel_helper_functions/pad_tile.hpp`, `data_movement/common/kernels/common.hpp`)
   is patched too. Includes that have an emule shadow (jit_hw `api/…`) are skipped
   — the shadow is already offset-correct. Angle-includes and system headers are
   left to `-I` resolution.
4. Emits `#line 1 "<real kernel path>"` at the top of the patched copy.

Two consequences shape every rule:

- **The pass runs on raw text — before the C preprocessor.** `#define`d types are
  *not* expanded, so a macro that hides a pointer type or the `tt_l1_ptr` token
  (e.g. `#define u32_l1_ptr volatile tt_l1_ptr uint32_t*`) is opaque to these
  regexes. See "Known gaps".
- **Rewrites are line-count-preserving** (`emule_line_preserving_replace` pads each
  replacement with the newlines its match consumed). Combined with the `#line`
  directive, DWARF — and therefore gdb/ASAN backtraces — resolve to the real
  `kernel.cpp:<line>`. Two rules (`ptr_to_l1_addr_re`, `zero_len_noc_coords_re`)
  use plain `std::regex_replace`; their replacements are single-line so the count
  is preserved anyway.

## The L1 offset model, in one paragraph

Under the offset model a kernel L1 address is a **0-based firmware offset** (not a
host pointer). `get_write_ptr` / `get_read_ptr` / `get_semaphore` /
`get_tile_address` / `get_arg_val` all return offsets. Any point where the kernel
turns that offset into a pointer and dereferences it must rebase onto this fiber's
`bridge_l1`, via the chokepoint `__emule_local_l1_to_ptr(offset)` (=
`bridge_l1 + offset`). The rewrites below insert that call at each deref idiom.
The reverse rule handles a pointer being narrowed back to a device address. A
**missed** L1 site derefs a small offset and SIGSEGVs immediately at the real
kernel line — that "SIGSEGV net" is intentional: gaps surface loudly instead of
silently corrupting memory. Diagnose with `EMULE_JIT_DEBUG=1` + gdb (ASAN off).

## Rule summary (in application order)

| # | Regex | Purpose |
|---|-------|---------|
| 1 | `mhartid_re` | RISC-V `csrr … mhartid` → host processor-id read |
| 2 | `fence_re` | RISC-V `fence` → host memory barrier |
| 3 | `l1_ptr_cast_re` (P1) | translate a `tt_l1_ptr`-attributed pointer cast |
| 4 | `l1_arg_ptr_re` | translate an attr-less cast of `get_arg_val<uint32_t>(N)` |
| 5 | `l1_named_arg_ptr_re` | translate the Metal 2.0 `get_arg(args::NAME)` form |
| 6 | `l1_getptr_cast_re` (P2) | translate an attr-less cast of an inline `get_*_ptr()` |
| 7 | `l1_mem_move_copy_re` (P3) | translate both L1 operands of `memmove`/`memcpy` |
| 8 | `l1_addr_var_re` + per-var casts (P4) | set-gated closure: casts (attr-less / C-style / `v+arith` / `_l1_ptr` macro) of vars derived from `get_*_ptr` or `get_arg_val` |
| 8b | `l1_arg_addr_widen_re` (P4b) | widen (not rebase) a `uint32_t` local fed by `get_arg_addr`/`get_common_arg_addr` |
| 9 | `l1_pad_tile_re` (P5.1) | translate `fill_pad_tile`'s `l1_tile_ptr` param cast |
| 10 | `l1_curr_addr_re` (P5.2) | translate `fill_with_val_async`'s `curr_addr` cast |
| 10b | `l1_shard_map_param_re` (P5.3) | widen (not rebase) `get_shard_map`'s `L1_address` param |
| 11 | `ptr_to_l1_addr_re` (reverse) | pointer → 0-based device offset (subtract `bridge_l1`) |
| 12 | `zero_len_noc_coords_re` | make a dead zero-length array compile on x86 clang |
| P1 | `include_re` | (pipeline) find quote-includes to recurse into |
| P2 | `inc_flag_re` | (pipeline) parse `-I"…"` kernel include roots |

Ordering matters: **P1 runs before the `get_arg` rules** so their output isn't
re-wrapped; **P2/P4 carry a `(?![^>]*\btt_l1_ptr\b)` negative lookahead** so they
never touch a cast P1 already owns (no double translation); **P4 is gated on a
collected variable set** so it can't match arbitrary scalar casts.

---

## 1. `mhartid_re` — RISC-V hart-id read

```
asm\s+volatile\s*\(\s*"csrr\s+%0\s*,\s*mhartid"\s*:\s*"=r"\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\)\s*;
```

**Triggers on** (`tests/tt_metal/.../dataflow/riscv_atomics.cpp:43`):

```cpp
asm volatile("csrr %0, mhartid" : "=r"(thread_idx));
```

**Rewrites to** `thread_idx = __emule_self->processor_id;`

x86 assembler rejects RISC-V CSR instructions; the runner sets the per-fiber
`processor_id` before each kernel launch.

## 2. `fence_re` — RISC-V memory fence

```
asm\s+volatile\s*\(\s*"fence"\s*(:::\s*"memory"\s*)?\)\s*;
```

**Triggers on** (`tt_metal/hw/inc/internal/tt-2xx/quasar/overlay/llk_intf_api.hpp:69`):

```cpp
asm volatile("fence" : : : "memory");
```

**Rewrites to** `__sync_synchronize();` — the closest host equivalent. The clobber
list is optional (some kernels, e.g. the embedding_backward `ARCH_BLACKHOLE`
cache-flush fence, omit it).

## 3. `l1_ptr_cast_re` (P1) — `tt_l1_ptr`-attributed cast

```
reinterpret_cast<\s*([^>]*\btt_l1_ptr\b[^>]*\*)\s*>\s*\(\s*(?!\s*&)(?!\s*get_(?:common_)?arg_addr\b)((?:[^()]|\([^()]*\))*?)\s*\)
```

**Triggers on** (any kernel using the canonical L1 pointer attribute, e.g.):

```cpp
volatile tt_l1_ptr uint32_t* status_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
```

**Rewrites to** `reinterpret_cast<volatile tt_l1_ptr uint32_t*>((uintptr_t)__emule_local_l1_to_ptr((uint32_t)(get_arg_val<uint32_t>(0))))`.

The operand allows one nested-paren level. Two operand exclusions:
- **`(?!\s*&)`** skips an address-of operand: `reinterpret_cast<tt_l1_ptr T*>(&scalar)`
  reinterprets the bytes of a *stack* local (e.g. a pad value) whose address is a host
  pointer, **not** a 0-based offset — translating it would corrupt it. A bitwise
  `base & mask` operand does not start with `&` and still matches.
- **`(?!\s*get_(?:common_)?arg_addr\b)`** skips the rt-args accessors:
  `get_arg_addr(i)` / `get_common_arg_addr(i)` return a **host pointer**
  (`&__emule_self->rt_args[i]`), not an L1 offset, so their casts must never be
  rebased. (Most such casts are C-style — which P1 doesn't touch — but the
  `reinterpret_cast` form exists, e.g. the moe selective-reduce-combine writer.)

P1 runs before rules 4–5 so a `tt_l1_ptr`+`get_arg_val` cast is wrapped here and not
re-matched there.

## 4. `l1_arg_ptr_re` — attr-less `get_arg_val<uint32_t>(N)` cast

```
reinterpret_cast<([^>]+\*)>\s*\(\s*get_arg_val<uint32_t>\s*(\([^)]*\))\s*\)
```

**Triggers on** (attr-less form, no `tt_l1_ptr`; e.g. a kernel passing an address
as a runtime arg):

```cpp
sem_addrs[i] = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
```

**Rewrites to** `reinterpret_cast<volatile uint32_t*>((uintptr_t)__emule_local_l1_to_ptr(get_arg_val<uint32_t>(arg_idx++)))`.

Kernels pass raw L1 firmware offsets as runtime args; the host needs the rebase.
(The `tt_l1_ptr` variant of this same idiom is already handled by P1.)

## 5. `l1_named_arg_ptr_re` — Metal 2.0 named-arg cast

```
reinterpret_cast<([^>]+\*)>\s*\(\s*static_cast<uintptr_t>\s*\(\s*get_arg\s*\(\s*([^)]+)\s*\)\s*\)\s*\)
```

**Triggers on**:

```cpp
reinterpret_cast<atomic_type*>(static_cast<uintptr_t>(get_arg(args::l1_counter_addr)));
```

**Rewrites to** `reinterpret_cast<atomic_type*>((uintptr_t)__emule_local_l1_to_ptr(static_cast<uint32_t>(get_arg(args::l1_counter_addr))))`.

The Metal 2.0 host runtime exposes runtime args through the typed
`get_arg(args::NAME)` accessor rather than `get_arg_val<uint32_t>(index)`.

## 6. `l1_getptr_cast_re` (P2) — attr-less cast of an inline `get_*_ptr()`

```
reinterpret_cast<\s*((?![^>]*\btt_l1_ptr\b)[^>]*\*)\s*>\s*\(\s*((?:[A-Za-z_]\w*\s*\.\s*)?(?:get_write_ptr|get_read_ptr|get_semaphore|get_tile_address)\s*(?:<[^>]*>)?\s*\([^()]*\))\s*\)
```

**Triggers on** (both the free-function and the `CircularBuffer` method form):

```cpp
auto ptr = reinterpret_cast<uint16_t*>(cb_scaler.get_write_ptr());
uint32_t* brisc_counts_ptr = reinterpret_cast<uint32_t*>(cb_brisc_expert_counts.get_write_ptr());
```

**Rewrites to** `reinterpret_cast<uint16_t*>((uintptr_t)__emule_local_l1_to_ptr((uint32_t)(cb_scaler.get_write_ptr())))`.

The negative lookahead partitions this from P1 (P1 owns `tt_l1_ptr` casts). The
operand is the `get_*_ptr` call itself (one arg-paren level); a cast of the call
plus **trailing arithmetic** (`get_write_ptr() + off`) is *not* matched here — it
falls to P4 (if via a variable) or the SIGSEGV net.

## 7. `l1_mem_move_copy_re` (P3) — L1↔L1 `memmove` / `memcpy`

```
\b(memmove|memcpy)\s*\(\s*\(\s*void\s*\*\s*\)\s*\(\s*((?:[^()]|\([^()]*\))*?)\s*\)\s*,\s*\(\s*void\s*\*\s*\)\s*\(\s*((?:[^()]|\([^()]*\))*?)\s*\)\s*,
```

**Triggers on** (`data_movement/common/kernels/common.hpp:115`, `tt_memmove`):

```cpp
memmove((void*)(dst_l1_addr), (void*)(src_l1_addr), (size_t)(bytes));
```

**Rewrites to** `memmove((void*)__emule_local_l1_to_ptr((uint32_t)(dst_l1_addr)), (void*)__emule_local_l1_to_ptr((uint32_t)(src_l1_addr)), (size_t)(bytes));`

Both operands are 0-based L1 offsets cast to `void*`; each is rebased. Narrowly
anchored on a `memmove`/`memcpy` with **two** `(void*)`-cast args, so it never
touches an unrelated `(void*)` cast. One nested-paren level per operand.

## 8. `l1_addr_var_re` + per-var casts (P4) — the set-gated closure

The workhorse for casts of L1-address *variables* the inline rules miss (attribute-less,
C-style, arithmetic operand, or `#define`d pointer type). Two passes.

**(A) Collect the L1-address variable set.** Direct producers:

```
\b(?:uint32_t|auto)\s+([A-Za-z_]\w*)\s*=\s*(?:[A-Za-z_]\w*\s*\.\s*)?(?:get_write_ptr|get_read_ptr|get_semaphore|get_tile_address|get_arg_val)\s*(?:<[^>]*>)?\s*\(
```

`get_arg_val` is a producer for the `uint32_t addr = get_arg_val<uint32_t>(i)` store-then-cast
L1-offset idiom (rule 4 covers the inline-cast form; this adds the store-then-cast case). It is the
broadest producer — runtime args aren't always L1 offsets (a count, a DRAM base) — but only a
collected var actually cast to an L1 pointer is rewritten, and a wrong rebase fails loudly (the
SIGSEGV net / the `[EMULE_L1] … out of range` assert).

then the **transitive closure** (fixpoint) over derivations — `w = <collected> [+/- …]`:

```
\b(?:uint32_t|auto)\s+([A-Za-z_]\w*)\s*=\s*<collected-var>\b
```

**(B) Per collected `v`, translate its casts** (operand = `v`, optionally `v + arith`
with one nested-paren level), in three forms — reinterpret_cast (any non-`tt_l1_ptr`
pointer target **or** a `\w*_l1_ptr` macro type P1 can't see pre-macro), C-style with a
parenthesised operand, and C-style with a bare operand:

```
reinterpret_cast<\s*((?![^>]*\btt_l1_ptr\b)(?:[^>]*\*|\w*_l1_ptr))\s*>\s*\(\s*(v\b(?:[^()]|\([^()]*\))*?)\s*\)
\(\s*([\w:][\w:\s]*\*)\s*\)\s*\(\s*(v\b(?:[^()]|\([^()]*\))*?)\s*\)
\(\s*([\w:][\w:\s]*\*)\s*\)\s*(v)\b
```

**Triggers on** — six real shapes, all with a **collected** operand:

```cpp
// (i) attr-less bare var — full/writer_full.cpp
uint32_t write_addr = get_write_ptr(cb_value);
auto ptr = reinterpret_cast<uint16_t*>(write_addr);
// (ii) C-style cast — tilize reader_unary_pad_dims_split_rows.cpp
uint32_t l1_write_addr = cb_in0.get_write_ptr();
auto* dst = (volatile tt_l1_ptr uint32_t*)(l1_write_addr + block_size);   // C-style + arith
// (iii) arithmetic operand, attr-less C-style — fold writer_cb2dram_for_rm_input.cpp
uint32_t l1_addr = cb_in0.get_read_ptr();
x = *(volatile uint16_t*)(l1_addr + j * 2);
// (iv) derived var — reduction writer_welford_hw.cpp
auto means_addr = cb_partial_obj.get_read_ptr();          // collected
auto vars_addr  = means_addr + partial_tile_size_bytes;   // collected transitively
auto* vp = reinterpret_cast<volatile float*>(vars_addr);
// (v) macro pointer type — pad writer_pad_dims_rm_sharded_stickwise.cpp
uint32_t cb_write_addr = cb.get_write_ptr();
auto p = reinterpret_cast<u32_l1_ptr>(cb_write_addr);     // u32_l1_ptr = #define volatile tt_l1_ptr uint32_t*
// (vi) C-style cast, bare (unparenthesised) operand — fold writer_cb2dram_for_rm_input.cpp
uint32_t intermed_l1_scratch = cb_in1.get_write_ptr();
auto* patch_data = (volatile uint16_t*)intermed_l1_scratch;
```

**Gating on the collected set** is the entire safety mechanism — it is what makes
handling C-style casts safe without a blanket rule: a var not assigned (directly or by
derivation) from a `get_*_ptr` producer is never collected, so no arbitrary cast is
touched. `tt_l1_ptr` reinterpret_casts stay with P1 (negative lookahead); C-style casts
are P1-untouched so they carry no `tt_l1_ptr` partition. The reinterpret form runs
before the C-style form, whose `(uint32_t)(…)` output (no `*`) it cannot re-match.
Residual risk: a collected var later reassigned to a non-L1 value — it fails loudly
(SIGSEGV / the `__emule_l1_translate` OOB assert). (Interprocedural *parameter* casts —
where the L1 address arrives through a function parameter, not a local — are **not**
collected here; those are P5's per-site rules.)

## 8b. `l1_arg_addr_widen_re` (P4b) — widen a `get_arg_addr`-fed local, don't rebase it

```
\buint32_t(\s+[A-Za-z_]\w*\s*=\s*get_(?:common_)?arg_addr\s*\()
```

**Triggers on** (`ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/padded_slice_reader_rm_interleaved_start_id.cpp:21,23`):

```cpp
const uint32_t num_unpadded_sticks_addr = get_arg_addr(9);
tt_l1_ptr uint32_t* num_unpadded_sticks = (tt_l1_ptr uint32_t*)(num_unpadded_sticks_addr);
```

**Rewrites to** `const uintptr_t num_unpadded_sticks_addr = get_arg_addr(9);` (the cast on the
next line is left untouched — it now casts a full-width pointer, needing no rebase).

`get_arg_addr`/`get_common_arg_addr` return a genuine **host pointer**, not an L1
offset (see rule 3's exclusion) — safe to narrow into a `uint32_t` only while the L1
pool is guaranteed to live below 4 GB. Dropping that guarantee (`worker_l1_mmap.hpp`,
the >4 GB galaxy work) means a `uint32_t` local storing this value can silently lose
its high bits; the later cast back to a pointer then dereferences garbage and
SIGSEGVs. This is the **opposite** of every other rule here: those rebase a genuine
0-based offset into a pointer; this one *prevents a truncation* of a value that was
already a correct pointer — rebasing it (adding/subtracting `bridge_l1`) would produce
a different, equally-wrong address. Scoped to the exact `uint32_t <name> = get_arg_addr(...)`
shape; `auto`-declared locals already deduce `uintptr_t` and need no rewrite.

## 9. `l1_pad_tile_re` (P5.1) — `fill_pad_tile` param cast

```
reinterpret_cast<\s*T\s*\*\s*>\s*\(\s*l1_tile_ptr\s*\)
```

**Triggers on** (`ttnn/cpp/ttnn/operations/kernel_helper_functions/pad_tile.hpp`,
`fill_pad_tile(uint32_t l1_tile_ptr)` — used by the matmul in0/in1 padding readers
via `pad_last_ktile` / `pad_last_transposed_ktile`):

```cpp
auto tile_ptr = reinterpret_cast<T*>(l1_tile_ptr);
```

**Rewrites to** `auto tile_ptr = reinterpret_cast<T*>((uintptr_t)__emule_local_l1_to_ptr((uint32_t)(l1_tile_ptr)));`

Interprocedural: the L1 address flows from `cb.get_write_ptr()` at the call site
through the `l1_tile_ptr` parameter, so it is neither a `tt_l1_ptr` cast (P1) nor a
locally `get_*_ptr`-assigned var (P4). Anchored on the exact cast (target `T*` +
the helper's documented L1-address operand), so it matches only this one site.

## 10. `l1_curr_addr_re` (P5.2) — `fill_with_val_async` cast

```
reinterpret_cast<\s*uint16_t\s*\*\s*>\s*\(\s*curr_addr\s*\)
```

**Triggers on** (`operations/data_movement/pad/device/kernels/dataflow/reader_pad_dims_rm_interleaved.cpp:23`):

```cpp
uint32_t curr_addr = begin_addr;      // begin_addr param == cb.get_write_ptr()
while (curr_addr < begin_addr_aligned && size_nbytes > 0) {
    reinterpret_cast<uint16_t*>(curr_addr)[0] = pad_value;   // translated
    curr_addr += 2;
    ...
}
```

**Rewrites to** `reinterpret_cast<uint16_t*>((uintptr_t)__emule_local_l1_to_ptr((uint32_t)(curr_addr)))[0] = pad_value;`

Translation happens **at the cast**, leaving `curr_addr` a 0-based offset for its
other uses (the sibling `CoreLocalMem<uint32_t> dst(curr_addr)` NoC-read path,
whose ctor rebases separately). `curr_addr` is a copy of the `begin_addr`
parameter, so the value is again interprocedural and P4 cannot reach it.

## 10b. `l1_shard_map_param_re` (P5.3) — `get_shard_map` param widening

```
get_shard_map\s*\(\s*uint32_t\s+L1_address\s*\)
```

**Triggers on** (`ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp:136`):

```cpp
template <typename ShardingInfoType>
std::pair<const mapping_table_t* const, uint32_t> get_shard_map(uint32_t L1_address) {
    const mapping_table_t* const map = reinterpret_cast<const mapping_table_t* const>(L1_address);
    ...
}
```

**Rewrites to** `get_shard_map(uintptr_t L1_address)` (the return type's own `uint32_t`
is untouched — the pattern anchors on the parameter only).

Same widen-not-rebase reasoning as P4b (rule 8b): `get_shard_map` is called as
`get_shard_map<T>(get_arg_addr(rt_index))` from `reader_unary_stick_start_id.cpp`,
`writer_unary_stick_start_id.cpp`, and other sharded/reshard/CCL kernels that share
this header — the `uint32_t` parameter silently truncates `get_arg_addr`'s host
pointer at the call boundary once the L1 pool can live above 4 GB, and the
`reinterpret_cast` inside then dereferences the truncated value. Interprocedural
(the parameter, not a local variable), so neither P4 nor P4b's collector can reach
it — anchored per-site like P5.1/P5.2. Widening `uint32_t`→`uintptr_t` is an exact
zero-extension for every other value that already flows through this parameter, so
this is safe for all callers, not just the ones observed crashing.

## 11. `ptr_to_l1_addr_re` (reverse) — pointer → device offset

```
reinterpret_cast<\s*(?:std::)?uint32_t\s*>\s*\(\s*((?:[^()]|\([^()]*\))*?)\s*\)
```

**Triggers on** (fabric/CCL, narrowing an L1 pointer to a device address):

```cpp
… reinterpret_cast<uint32_t>(header) …
```

**Rewrites to** `static_cast<uint32_t>(reinterpret_cast<uintptr_t>(header) - reinterpret_cast<uintptr_t>(__emule_self->bridge_l1))`.

The reverse direction: the device address is the 0-based offset, so subtract
`bridge_l1` (it was a bare truncation under the old host-pointer-aliasing model).
Requiring `uint32_t>` (no `*`) skips the pointer-typed `reinterpret_cast<T*>`
forms handled by rules 3–10.

## 12. `zero_len_noc_coords_re` — dead zero-length array

```
(using\s+RemoteNocCoords\s*=\s*RemoteNocCoord\s*\[)\s*N\s*(\])
```

**Triggers on** (`operations/normalization/layernorm/device/kernels/dataflow/layernorm_dataflow_utils.h:32`):

```cpp
using RemoteNocCoords = RemoteNocCoord[N];
```

**Rewrites to** `using RemoteNocCoords = RemoteNocCoord[(N) == 0 ? 1 : (N)];`

The two-stage reduce path instantiates this with `N==0` for a core with no
second-stage workers. Silicon's compiler accepts a zero-length array as a GNU
extension; stock x86 clang rejects it. The array is dead (the `N==0` loop never
runs, it is never indexed), so a 1-element dummy is behavior-identical. The
instantiation is unavoidable because `kernel_main()` is not a template, so the
`if constexpr (use_two_stage_reduce)` dead branch is still type-checked.

## Pipeline regexes (not source rewrites)

### `include_re` — recursion into quote-includes

```
#[ \t]*include[ \t]*"([^"]+)"
```

In `preprocess_tu_recursive`: finds `#include "…"` directives so the pass can
patch shared kernel helpers in other directories (a `*_common.hpp` /
`kernel_lib/*.inl` can hold the same raw-L1 idioms). Includes with an emule shadow
(jit_hw `api/…`) are skipped; angle-includes and system headers are left to `-I`.

**Escaping `../…` includes are normalized, not skipped.** A patched header is mirrored
into `out_dir` under its include name so the compiler finds it (via `-I out_dir`) before
the original. A deep relative include like the softmax writer's
`../../../../../../kernel_helper_functions/pad_tile.hpp` mirrors to a path that escapes
`out_dir`. Rather than leave it unpatched (→ raw-offset SIGSEGV in the pristine
`pad_tile.hpp`), the pass remaps it to an include-root-relative name (e.g.
`ttnn/operations/kernel_helper_functions/pad_tile.hpp` — the same form other kernels use),
writes the patched copy at `out_dir/<that>`, and rewrites the `#include` directive in the
source to match. Only a genuinely unresolvable escape (canonical path under no include
root) is left to `-fms-extensions`.

### `inc_flag_re` — kernel include-root discovery

```
-I"([^"]+)"
```

Parses the `-I"…"` flags out of the kernel compile command to build the include
roots the recursion resolves cross-dir headers against; `-I out_dir` is placed
first so a patched copy wins over the original on quote-relative resolution.

## Abandoned / narrowed patterns (do not reintroduce)

These broader forms were tried and caused regressions; the current rules are the
narrowed survivors. They are recorded so the broad form is not reintroduced.

- **Blanket ungated attr-less cast — abandoned.** The precise ancestor of P4
  without its collected-var gate:

  ```
  reinterpret_cast<\s*((?![^>]*\btt_l1_ptr\b)[^>]*\*)\s*>\s*\(\s*([A-Za-z_]\w*)\s*\)
  ```

  i.e. translate *every* attr-less pointer cast of *any* bare variable. It
  over-matched: it wrapped variables that were not L1 offsets (already host
  pointers, NOC/DRAM addresses) and **double-translated** operands other rules
  already owned, producing wild pointers — it broke a large batch of
  previously-passing matmul/sdpa/conv entries and was reverted. The lesson: gate on
  a collected L1-address set (P4) or anchor per-site (P5); never translate a cast
  by shape alone.

- **P1 without the `(?!\s*&)` operand exclusion — narrowed.** The original P1
  matched an address-of operand, so `reinterpret_cast<tt_l1_ptr T*>(&pad_value)`
  ran the address of a **stack** local through `__emule_local_l1_to_ptr`,
  corrupting a pad value (the address is a host pointer, not a 0-based offset). P1
  now carries `(?!\s*&)` to skip that operand — see rule #3.

- **Blanket C-style `tt_l1_ptr` cast rule — rejected (never landed).** Tempting to
  mirror P1 for C-style casts (translate every `(…tt_l1_ptr…*)(expr)`), since the
  crashing tilize casts are C-style. But `get_arg_addr`/`get_common_arg_addr` return
  **host pointers** (`&__emule_self->rt_args[i]`), and of the ~34 C-style `tt_l1_ptr`
  casts in kernels, **13 are over `get_arg_addr`** — a blanket rule would rebase those
  host pointers into garbage. It is unnecessary anyway: all 21 non-`arg_addr` C-style
  `tt_l1_ptr` casts are over **collected** vars (`= get_write_ptr()/get_read_ptr()`), so
  P4's set gate covers them. Lesson: the `tt_l1_ptr` attribute is *not* a sufficient
  L1 signal for C-style casts (host pointers wear it too); the **collected-var gate**
  is. Handle C-style casts inside P4 (set-gated), never as a standalone shape rule.

- **Redirect-shim shadows that bypass patching — deleted, do not recreate.** An emule
  shadow under `jit_hw/cpp/ttnn/…` that merely `#include`s the real upstream header
  (a `cpp/ttnn/…`-prefix resolution workaround) is a **trap**: `preprocess_tu_recursive`
  treats the shadow as offset-correct and skips patching, but the shadow pulls in the
  **unpatched** original, so P1–P5 never run on it (e.g. `common.hpp`'s `tt_memmove`
  SIGSEGV'd on raw offsets). The `cpp/ttnn/…` prefix already resolves via
  `-I <src>/ttnn`, so such shims are obsolete — delete them and let the recursion patch
  the real header. Never add a shadow that only forwards to a patchable header.

## Deliberate non-rules and known gaps

These are not "tried and reverted"; they are cases deliberately left to fail loudly
or handled elsewhere.

- **Fabric header-narrow rule ("R2") — name-pattern-constrained, NOT blanket.** Fabric/CCL
  kernels narrow a translated packet-header L1 *pointer* to a device address via C-style
  `(uint32_t)<header>`; under the offset model that must yield the 0-based offset
  (`ptr - bridge_l1`), not the truncated host pointer. The rule matches only a
  packet-header-pointer identifier (`\w*(hdr|packet_header|header_ptr|header_addr)\w*`) —
  a blanket `(uint32_t)EXPR` rule would mistranslate ordinary int casts, and even the
  header-ish pattern deliberately excludes `current_cmd_header` (a `CclCommandHeader`
  *value*). It pairs with the fabric shim's offset↔pointer translation
  (`__emule_fabric_stubs.h`) and the chip-qualified route key so the fabric carries offsets
  end-to-end and survives worker L1 above 4 GB. An over-match of a non-pointer is loud:
  `reinterpret_cast<uintptr_t>(non_pointer)` is ill-formed and fails the JIT compile.
- **Macro-hidden pointer casts — covered only for collected vars.** The pass runs
  before the C preprocessor, so a `#define`d pointer type hides both the `*` and the
  `tt_l1_ptr` token from every regex (e.g. the pad *stickwise* kernels'
  `#define u32_l1_ptr volatile tt_l1_ptr uint32_t*` +
  `reinterpret_cast<u32_l1_ptr>(cb_write_addr)`). P4(B) now matches a `\w*_l1_ptr`
  macro target **when the operand is a collected var** (as here, `cb_write_addr =
  cb.get_write_ptr()`). The residual gap is a macro-typed cast over a **non-collected**
  operand, or a pointer macro that doesn't end in `_l1_ptr` — neither observed today.
