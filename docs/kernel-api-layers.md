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

`scripts/classify_kernels.py` operationalizes the taxonomy above. It is a
deterministic, **no-LLM / no-build** classifier: point it at a directory of
kernels or at a specific kernel file and it labels each against the emule
surface, so you can answer "what should we bring up next?" without compiling
anything. It is a per-kernel primitive and stays agnostic to any kernel tree's
layout — an external driver can roll the per-kernel verdicts up however it
likes (per operation, per test, per model). It decides includes the way emule's
JIT does — an include that resolves into `include/jit_hw/` is a shadowed
**layer-1 leaf**; one that falls through to the real source tree (under
`--repo-root`) is recursed and scanned for lower-layer signals.

```bash
# every kernel TU under a tree, with its include search roots (JIT -I order)
python3 scripts/classify_kernels.py --root /path/to/kernels \
    -I /path/to/src/hw/inc -I /path/to/src

# a single kernel (the unit an external driver loops over)
python3 scripts/classify_kernels.py /path/to/kernels/reader.cpp -I /path/to/src
```

### Verdicts

| verdict | meaning |
|---|---|
| `layer1` | modelable by emule today — only high-level `cb_*` / `noc_async_*` / `compute_kernel_api` wrappers in its include graph |
| `needs_stub` | layer-1 by API, but calls a `*_tile`/`_tiles` wrapper emule doesn't ship yet — unblockable with a shim under `include/jit_hw/api/compute/` |
| `ruled_out` | reaches a lower layer emule can't model (see buckets) |

### `ruled_out` buckets → layer

Each `ruled_out` unit carries a `ruleout_buckets` set and a `ruleout_primary`
(its hardest blocker), mapped onto the layer stack defined above (ordered
hardest → easiest to bring up):

| bucket | layer | meaning |
|--------|-------|---------|
| `sfpi_intrinsics` | layer-3 | hand-written SFPU vector code (`sfpi::`, `vFloat/vInt`) |
| `dst_register` | layer-3 | raw DST register indexing (`dst_reg[]`) |
| `hw_instructions` | layer-3 | raw HW-instruction / MOP macros (`TTI_*`, `mop_run`, `TT_LLK_`) |
| `llk_headers` | layer-2 | reaches a real LLK/SFPU header tree (`tt-llk/`, `ckernel_sfpu_*`) |
| `unshadowed_ckernel` | layer-2 | uses a `ckernel::x` emule doesn't provide — shimmable |
| `unshadowed_llk_call` | layer-2 | calls an `llk_x()` emule doesn't shadow — most tractable (add a jit_hw shim) |

`llk_`/`ckernel::` usage is **gated against what emule actually provides in
`jit_hw`** — a standard kernel calling a *shadowed* wrapper (e.g.
`llk_pack_relu_config`, `ckernel::PoolType`) stays `layer1`. The report's
**shimmable** set is the units ruled out *only* by unshadowed `llk_`/`ckernel`
symbols (a single jit_hw shim unblocks each).

### Outputs

Writes to `--out-dir` (default `scripts/out/`, gitignored — the reports
reference a specific source checkout): `kernels_manifest.json` (per-kernel
verdicts, buckets, reasons, `kernel_files`, `missing_compute_stubs`) and
`kernels_CLASSIFICATION.md` (the human report with the layer/bucket breakdown
and the shimmable set). The console prints a summary.

### "What does emule provide?" — single source of truth

The classifier's denominators — which `*_tile` wrappers, `llk_*` shims, and
`ckernel::` members emule ships in `jit_hw` — come from the authoritative
file/symbol index [`.claude/references/structure.yaml`](../.claude/references/structure.yaml),
the same index `scripts/find_symbol.py` queries. The index is regenerated and
kept in sync with the headers by `scripts/gen_structure.py` (a pre-commit hook +
the Structure Index CI gate run `--check`), so "what emule provides" stays
consistent across the classifier, `find_symbol.py`, and the
[compute-llk-bringup](../.claude/skills/compute-llk-bringup/SKILL.md) workflow.

For a single-symbol / single-include probe — *without* running the full sweep —
use the early-detect modes (backed by the shared `scripts/emule_surface.py`):

```bash
python3 scripts/find_symbol.py --supports <op>_tile          # layer1 / needs_stub / ruled_out for one symbol
python3 scripts/find_symbol.py --shadows  api/compute/foo.h   # does emule shadow this include?
python3 scripts/find_symbol.py --supports <op>_tile --exit-status   # exit 0 iff layer1 (shell branching)
```

`--supports` is the fast "does the shim already exist *and* would a kernel using
it run?" check when bringing up a `ruled_out`/`needs_stub` unit.
