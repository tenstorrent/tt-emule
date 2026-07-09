# L1 offset addressing

How emule represents a kernel's L1 address, and how that address becomes a host
pointer at the point of use. Read this with [l1-emulation.md](l1-emulation.md)
(the L1 backing-store model) and [l1-offset-translation-derisk.md](l1-offset-translation-derisk.md)
(the patch taxonomy + per-site audit).

## The model

A kernel's 32-bit L1 address is a **0-based firmware offset** into its own core's
L1 (worker L1 is 1.5 MB on Blackhole, 1.5 MB on Wormhole — an offset always fits in
far less than 32 bits). This is what silicon uses: firmware addresses L1 with small
absolute offsets. The offset is converted to a real 64-bit host pointer **only at the
point it is dereferenced**, by adding the running fiber's L1 base
(`ThreadCommonCtx::bridge_l1`):

```
host_ptr = bridge_l1 + offset          // __emule_l1_translate()
```

Each kernel invocation only ever dereferences its **own** core's L1 directly;
cross-core and cross-chip access is expressed as a NOC address and resolved through
`__emule_resolve_noc_addr` (a `core_map` lookup → the target core's own backing), so
the per-fiber `bridge_l1` is always the correct base at a raw dereference.

Because the address is an offset (not a host pointer), it is **address-space
independent**: the same offset names the same L1 word on any chip in any process. A
mesh of chips no longer has to fit all its worker L1 into one process's low 4 GB —
the constraint that previously capped emule at ~16 Blackhole chips.

## Why the change is small

Two properties of the existing runtime make offsets nearly free:

- **The data path already funnels the local address through the translation
  chokepoint.** `noc_async_read` translates its local `dst` via
  `__emule_local_l1_to_ptr`; `noc_async_write` / the multicast variants translate
  their local `src` the same way (`jit_hw/api/dataflow/dataflow_api.h`). So once
  `get_write_ptr`/`get_read_ptr`/`get_semaphore` return offsets and
  `__emule_l1_translate` rebases unconditionally, every `noc_async_*` consumer is
  correct with no change.
- **NOC and DRAM are already offset-based.** NOC addresses carry the offset in their
  low bits (decoded by `__emule_resolve_noc_addr`); DRAM is reached via bank offsets
  through a non-`MAP_32BIT` backing. Neither depended on the host-pointer identity.

What remains is: (1) make the address-producing APIs return offsets, and (2) rebase
at the **raw dereference** sites in kernel bodies that don't go through a `noc_async_*`
call. (1) is a handful of one-line edits; (2) is a JIT source-patch.

## The producer-side edits

The APIs that hand an L1 address to a kernel return an offset; the translation
chokepoint rebases unconditionally. See the per-site audit in the derisk doc.

- `__emule_l1_translate` (`jit_hw/asan/asan_l1_checks.h`) — unconditional
  `bridge_l1 + offset`; a debug build additionally asserts `offset < l1_size`
  (`ThreadCommonCtx::l1_size`, set per core at launch) so a value that is still an
  absolute address surfaces as a named panic.
- `get_semaphore` (`jit_hw/api/dataflow/dataflow_api.h` + `jit_hw/jit_kernel_stubs.hpp`)
  — returns `EMULE_SEM_BASE + id*ALIGN` (a pure offset).
- `Semaphore` (`jit_hw/api/dataflow/noc_semaphore.h`) — stores the offset; `atom()`
  rebases `bridge_l1 + offset` directly (bypassing the sanitizer chokepoint, which
  would otherwise trap the legitimate semaphore-region access).
- `get_write_ptr`/`get_read_ptr` (`jit_hw/api/cb_api.h`) — the CB ring is maintained
  in host-pointer space; these subtract `bridge_l1` so the value a kernel sees is an
  offset.
- `Core::l1_alloc` (`include/tt_emule/device.hpp`) and the Quasar DFB base setup
  (runner `allocate_dfbs_on_core`) — the DFB bump/base is a 0-based offset, rebased to
  a host pointer where the DFB/CB sync state stores its `base`.
- `__emule_sem_atomic` receives an already-translated host pointer (the semaphore
  cast is rewritten at JIT time) and uses it directly — no per-chip remap. This
  retires the `__emule_chip_relative_l1` (DM-1) global-semaphore workaround: one
  shared offset space means a peer chip's semaphore is reached by translating the
  offset with the reader's own `bridge_l1`, not by remapping a host pointer.

## The JIT patch pass

Kernel bodies that dereference an L1 address without going through a `noc_async_*`
call (`reinterpret_cast<T*>(get_write_ptr(cb))[i] = …`, and the mask/scaler/fill-pad
idioms) must rebase the offset at the cast. This is done by extending the existing
line-preserving regex pass `apply_x86_rewrites` in the runner (the same pass that
already rewrites `get_arg_val` L1-pointer casts) — upstream tt-metal kernel source is
untouched; the transform writes a temp `patched_kernel.cpp`, and
`preprocess_tu_recursive` also patches the shared kernel helpers it includes.

The rules insert an offset→host-pointer translation at every deref idiom (`tt_l1_ptr`
casts, `get_*_ptr` casts, `memmove`/`memcpy`, and a set-gated closure over variables
derived from an L1-address producer that covers reinterpret_cast and C-style casts,
bare or arithmetic operands, and macro-typed pointers), plus the reverse
pointer→offset narrowing. rt-args accessors (`get_arg_addr`, which return host
pointers) are excluded. The enumerated rules, their triggering idioms, and the
deliberate non-rules are documented in [jit-l1-patch-pass.md](jit-l1-patch-pass.md).

A missed forward site dereferences a small offset as a pointer → an immediate SIGSEGV
at the real kernel `file:line` (the `#line` is preserved). This is the loud-failure
net: coverage gaps announce themselves rather than silently corrupting, so the rule
set is grown against real kernels. The fabric client API's C-style
`(uint32_t)<packet-header>` narrowings ("R2") are rewritten to the 0-based offset by a
name-pattern-constrained rule, paired with the fabric shim's offset↔pointer translation
(`__emule_fabric_stubs.h`) and a chip-qualified route key — so the fabric carries offsets,
not truncated pointers, and survives worker L1 mapped above 4 GB (see
[fabric-ccl-emulation.md](fabric-ccl-emulation.md) and [jit-l1-patch-pass.md](jit-l1-patch-pass.md)).

## Relationship to worker-L1 placement

Offset addressing **decouples** the kernel-visible address from where worker L1 is
mapped. Worker L1 is still backed by the low-4 GB `MAP_32BIT` pool
([l1-emulation.md](l1-emulation.md) §2) — that placement is now a backing
detail, not an addressing constraint. Lifting it (a plain 64-bit `mmap` for worker
cores, so a >4 GB mesh such as a 32-chip Blackhole galaxy fits one process) is a
separate, behavior-neutral change on top of this model.
