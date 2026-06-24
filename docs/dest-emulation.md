# DEST Register Emulation in tt-emule

How tt-emule emulates the **DEST register file** — the compute engine's tile
accumulator. Read this before debugging a compute-numerics bug, touching the
acquire/pack lifecycle, or reasoning about DST capacity / `DEST_AUTO_LIMIT` /
the `DST_ACCUM_MODE` accumulation flag.

On silicon, DEST is a ~64 KB register file the MATH engine writes and the PACK
engine drains, holding up to 16 bf16 tiles in the default mode and 8 in
fp32-accumulate (`DST_ACCUM_MODE != 0`, equivalent to
`ComputeConfig.fp32_dest_acc_en`). Emule backs it with a plain `float` array
and performs the nfaces↔row-major layout conversion at the pack/unpack boundary;
the accumulation flag governs which kernel-visible slot count is active (see §3).

Companion doc: [tilize-untilize-pack.md](tilize-untilize-pack.md) (the full
nfaces / pack-untilize pipeline this references), [cb-emulation.md](cb-emulation.md)
(where packed tiles land).

---

## 1. Emulation model

There are no separate UNPACK / MATH / PACK TRISC threads in emule — all three
collapse onto one host thread (the `PACK(x) x` / `MATH(x) x` / `UNPACK(x) x`
macros are identity). DEST is a **per-compute-thread `float` array, always fp32
row-major**, regardless of the configured HW data format. It is a member of
`ComputeThreadCtx` (the per-thread state tier — see
[state-tiers.md](state-tiers.md)), reached via the typed accessor:

```cpp
// include/jit_hw/api/compute/common.h — capacity constants
static constexpr uint32_t __EMULE_DST_TILES      = 16;    // bf16 SyncFull ceiling
static constexpr uint32_t __EMULE_DST_TILES_FP32 = 8;     // fp32 SyncFull ceiling
static constexpr uint32_t __EMULE_TILE_ELEMS     = 1024;  // 32x32 floats/tile

// include/jit_hw/internal/emule_thread_ctx.h — storage
struct ComputeThreadCtx /* ... */ { float dst[16][1024]; bool dst_fresh[16]; /* ... */ };
// reached as __emule_compute_ctx().dst[idst][elem]
```

A single fp32 store per element is a unified internal representation: bf16 inputs
are widened on unpack, int32/uint16 are bit-punned through the float slot,
block-float is decoded on unpack. **Precision loss happens only at the pack /
unpack boundary**, never during intermediate compute — a deliberate simplification
(silicon would round at each MATH step).

---

## 2. Lifecycle: acquire / commit / wait / release

`include/jit_hw/api/compute/reg_api.h`:

- `tile_regs_acquire()` — **zeros all active DST slots** and marks them *fresh*.
  (Silicon leaves DEST undefined here; emule zeros it so ops can `+=` safely.)
- `tile_regs_commit()` / `tile_regs_wait()` / `tile_regs_release()` — no-ops
  (the MATH↔PACK handoff is vacuous on one thread).

The host-side `DstRegisterFile` (`include/tt_emule/dst_register_file.hpp`) wraps
a real `IDLE/ACQUIRED/COMMITTED/PACKING` state machine with mutex/CV; it is used
by the standalone wrapper, not the JIT kernel path.

**Fresh/dirty tracking** (`common_globals.h`): because acquire zeros DEST,
`reduce_tile<MAX/MIN>` starting from 0 would clamp all-negative inputs. So a
per-slot *fresh* flag (`__emule_dst_take_fresh`) lets the first write to a slot
overwrite rather than accumulate; any write marks the slot dirty
(`__emule_dst_mark_dirty`).

---

## 3. Capacity and `DEST_AUTO_LIMIT`

`__emule_dst_check(slot, caller)` aborts on out-of-bounds access; the bound is
`__emule_dst_active_tiles()` = **8 if `DST_ACCUM_MODE != 0` (fp32) else 16**
(common.h). Capacity depends on **two** settings:

| `dst_full_sync_en` \ accum | bf16 | fp32 |
|---|---|---|
| SyncFull | 16 | 8 |
| SyncHalf | 8 | 4 |

The kernel-facing `DEST_AUTO_LIMIT` (`tt-metal/.../kernel_lib/dest_helpers.hpp`)
resolves exactly this table. **Both the compute kernel and any cooperating
dataflow kernel of one program must agree on it** (e.g. the multi-core H-reduce
interleaves input tiles in chunks of `DEST_AUTO_LIMIT`). Emule's runner therefore
threads the program's resolved `ComputeConfig` into the compute TU's JIT defines,
mirroring silicon `genfiles.cpp`:

```cpp
// emulated_program_runner.cpp build_kernel_defines() — for COMPUTE kernels
defines["DST_ACCUM_MODE"]       = cc->fp32_dest_acc_en ? "1" : "0";
defines["ENABLE_FP32_DEST_ACC"] = cc->fp32_dest_acc_en ? "1" : "0";
defines["DST_SYNC_FULL"]        = cc->dst_full_sync_en ? "1" : "0";
```

`__EMULE_DST_TILES` (16 / 8) is a generous SyncFull *backing-store* ceiling; the
per-program `DEST_AUTO_LIMIT` is the value kernels actually chunk against.

---

## 4. nfaces ↔ row-major

DEST is row-major; CBs store tiles in **nfaces** (face-row-major: four 16×16
faces). Unpack/pack permute via constexpr LUTs in
`include/jit_hw/api/compute/nfaces.h`:

- `rowmajor_to_nfaces[1024]` / `nfaces_to_rowmajor[1024]` — full-tile maps.
- `tile_rc_to_nfaces(r, c, th, tw)` — the general tile-shape-aware map (single
  source of truth): handles thin (`th<32`) **and** narrow (`tw=16`) tiles via
  `face_r_dim`/`num_faces_c` derived from the CB's `get_tile_r_dim`/`get_tile_c_dim`.
- `tile_rm_to_nfaces(i, rows)` — thin wrapper over `tile_rc_to_nfaces` (width
  always 32); `rows >= 32` take the cached full-tile LUT.
- `tile_rows_from_pagesize(page, elem_bytes)` — derive row count from CB page.

The full pack/unpack pipeline (block-float codecs, thin tiles, L1-accumulation,
pack ReLU) is documented in
[tilize-untilize-pack.md](tilize-untilize-pack.md); this doc only covers the DEST
side.

---

## 5. What's intentionally simplified

- No SRC registers, no UNPACK/MATH/PACK pipeline parallelism — one host thread.
- DEST math is fp32 throughout; no per-op HW rounding.
- `tile_regs_commit/wait/release` are no-ops; acquire zeroing + fresh/dirty
  flags substitute for the silicon MATH↔PACK double-buffer handoff.

---

## 6. Where to read silicon canonical

| Topic | Canonical |
|---|---|
| DEST capacity / `get_dest_max_tiles` | `tt-metal/tt_metal/tt-llk/tt_llk_*/common/inc/ckernel.h` |
| `DEST_AUTO_LIMIT` / mode resolution | `tt-metal/ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp` |
| ComputeConfig (fp32_dest_acc_en, dst_full_sync_en) | `tt-metal/.../kernel_types.hpp` |
| Compute tile APIs | `tt-metal/tt_metal/hw/inc/api/compute/*.h` |
