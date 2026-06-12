# Kernel API Layers in tt-emule

The abstraction layers a Tenstorrent compute kernel can be written against, and
where each one is emulated under tt-emule. Read this before deciding **how** to
bring up a kernel that doesn't yet run under emule — the layer a kernel bottoms
out at determines whether emule needs a high-level shadow, an LLK shim, or a
deeper hardware model.

This is a *vocabulary* doc: the layer-1 / 1.5 / 2 / 3 terms appear throughout
the `include/jit_hw/` header comments (e.g. `api/compute/rmsnorm.h`,
`api/compute/common.h`, the MoE-gate shadows) and in PR titles. Collecting the
definitions here keeps everyone using them the same way.

> **Not to be confused with** the *build/integration* "Layer 1–4" in
> [metal-integration.md](metal-integration.md) (UMD device injection → runtime
> activation → JIT kernel execution → dispatch interception). That is a
> different axis — *how emule plugs into tt-metal*, not *what API a kernel
> targets*. The two numbering schemes are unrelated.

---

## The layers

A silicon compute kernel is a stack: the kernel author writes against a
high-level tile-op API (the **PoR** — Program-of-Record — API), which expands
to **LLK** (low-level kernel) calls, which expand to **hardware instructions**
operating on the SFPU/PACK/UNPACK/MATH engines and the DEST register file.
Emule can intercept at any rung. From highest (easiest to emulate faithfully)
to lowest:

### layer-1 — kernel-author tile-op API (PoR)

The high-level, documented compute API a kernel author calls: `add_tiles`,
`reduce_tile`, `matmul_tiles`, `exp_tile`, `pack_tile`, the `cb_*` circular-
buffer ops, `tile_regs_acquire`/`_commit`. One call = one semantic tile
operation. Emule models these directly as scalar/`float` math over its
row-major DEST array — this is where emule is most faithful and most of the
`api/compute/` surface lives.

When a whole algorithm is packaged as a single high-level helper, that is a
**layer-1 lift**. Examples in-tree:
- `api/compute/rmsnorm.h` — `rmsnorm_compute_impl` packages `x / sqrt(mean(x²) +
  eps) * gamma` as one helper, so the consumer `op.hpp`'s `#ifdef
  __EMULE_JIT_MODE` block becomes a single call.
- `api/compute/{deepseek,glm,kimi}_moe_gate.h`, `api/compute/topk_xl.h` —
  layer-1 (PoR) shadows of MoE-gate / distributed-top-K kernels.
- `api/compute/eltwise_unary/clamped_silu.h` — layer-1 shadow of the GPT-OSS
  SwiGLU activation.

### layer-1.5 — pack-engine config capture

A transitional rung between layer-1 and the hardware model. Some kernels reach
*below* the tile-op API to reconfigure the PACK engine directly — e.g. an
`TTI_SETADCXX` + `_llk_pack_mop_config_` pair that selects a *sub-rectangle* of
DEST to pack (MoE-gate ops write only col0 of rows 0–7 of face 0). Emule is not
a faithful hardware model of the pack engine, but it **captures these two config
calls into thread-local state** and `pack_dst_to_buf` honors them — enough to
let the kernel run without rewriting its `op.hpp` body.

In-tree: the `__emule_pack_x_end` / `__emule_pack_face_r_dim` /
`__emule_pack_num_faces` thread-local state and `__emule_reset_pack_subrect()`
in `api/compute/common.h` (the "Layer-1.5 pack-subrect state" block, landed by
PR #121). `compute_kernel_hw_startup` resets it so a sub-rectangle config can't
leak into the next kernel on the same host thread.

### layer-2 — LLK calls

The low-level kernel API the PoR API expands into: `llk_unpack_AB`,
`llk_math_eltwise_*`, `llk_pack_*`, `llk_math_reduce_*`, the `ckernel::`
init/config calls. A single layer-1 op typically expands to a sequence of
~10–15 LLK calls. When a kernel's `op.hpp` body drops to this level (calls LLKs
directly rather than tile-ops), emule needs an **LLK shim** providing the named
function surface — usually no-op or thin (e.g. `llk_math_binary_api.h`,
`llk_unpack_AB_api.h`, `llk_math_unary_sfpu.h`, `experimental/
llk_pack_block_api.h`). Faithful here means matching the LLK call's *effect* on
DEST/CB state, not the instruction sequence.

### layer-3 — hardware / instruction model

The lowest rung: `sfpi::` vector types, raw DST register indices, `TTI_*`
hardware instructions, CSR/ADC register state. Emule deliberately does **not**
model this faithfully — there are no separate UNPACK/MATH/PACK TRISC threads
(they collapse to one host thread), DEST is a plain fp32 `float` array, and SFPU
vectors are scalar loops. A kernel that genuinely requires layer-3 fidelity
(reads back a specific CSR, depends on per-lane SFPU semantics) is the hardest
to bring up and usually signals the work belongs at a higher layer instead.

---

## Why the layer matters for bring-up

When a kernel fails to compile or run under emule, the layer it bottoms out at
tells you the cheapest faithful fix:

| Kernel bottoms out at | Emule response |
|---|---|
| layer-1 tile-ops | model the op directly in `api/compute/` (most faithful) |
| layer-1.5 pack config | capture the config into thread-local state, honor it in pack |
| layer-2 LLK calls | add/extend an LLK shim that reproduces the call's effect |
| layer-3 HW instructions | usually re-express the kernel at a higher layer; faithful HW modeling is last resort |

Per the project rule, the goal is always fidelity to the canonical silicon
behavior — pick the highest layer at which emule can reproduce the *effect*,
and avoid parallel/divergent code paths.

---

## Classification tooling

> **Forward note (PR #136, not yet on this base).** A classifier
> (`scripts/classify_kernels.py`) sweeps ttnn and blaze kernels and labels each
> against the emule surface — `layer1` (already modeled), `needs_stub`, or
> `ruled_out` — with the `ruled_out` cases bucketed and mapped onto the layer-2
> / layer-3 distinction above (which LLK or HW symbol blocks them). It
> operationalizes this taxonomy to answer "what should we bring up next?".
>
> This section is a placeholder: it will be filled in (run instructions,
> verdict definitions, the `ruled_out` bucket → layer mapping, `scripts/out/`
> outputs) once PR #136 lands on a shared base. The taxonomy above stands on its
> own regardless.
