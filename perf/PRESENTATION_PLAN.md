<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->
# emule vs silicon — perf comparison presentation plan

A thorough plan for a deck comparing tt-emule (software emulation, host, slow
dispatch) against Wormhole B0 silicon. Covers the narrative, the full matrix of
parameters to vary, a slide-by-slide graph spec (with exact axes + data status),
and the silicon-side runs still needed to complete each comparison.

All measurements use tt-metal's own `sweep_framework` e2e_perf or the portable
microbenches in this dir, so the *same code path* runs on both backends. Numbers
below are what we've measured so far (WH B0 silicon, git SHA 8b9517627da).

---

## 1. The thesis (what the deck argues)

**emule is a *functional* emulator — it runs real Tensix kernels on the host to
get correct results without hardware. On every steady-state speed axis silicon
wins (46–500× for eltwise, 1000–3500× for matmul), because emule pays a fixed
per-program launch overhead plus serial per-core/per-element host work. emule
only wins wall-clock in two narrow regimes: cold-start light workloads (<~55 ops
from a powered-off chip) and problems that exceed silicon's capacity. Its cost
model also *ignores* things silicon cares about (dtype width, L1 vs DRAM) while
*reflecting* things it emulates faithfully (SFPU math complexity). Conclusion:
use emule for correctness-without-silicon, not for perf estimation.**

Three take-home slides: (a) silicon wins everywhere on speed, by how much and
why; (b) emule's cost model — what drives it, what it ignores; (c) the narrow
regimes where emule is the right tool.

---

## 2. Parameter matrix — vary as much as possible

| Axis | Values to sweep | Why it matters | Status |
|---|---|---|---|
| **Op** | exp, gelu, relu, i0 (SFPU spread); add/mul (binary); sum/max/mean (reduce); softmax; transpose; tilize/untilize; matmul; layernorm/rms_norm; SDPA | different op *classes* have different overhead/compute ratios | eltwise+matmul+reduce+dm done; **norm/SDPA todo** |
| **Op class** | fixed-floor vs per-tile-scaling | emule cost is bimodal | done (emule) |
| **Size** | 1 tile → 1B elements (side 1..1024) | overhead floor → compute-bound knee | emule done to 1B; **silicon only to 2.4M** |
| **dtype (storage)** | bf16, bf8_b, (bf4, fp32) | silicon = bandwidth; emule ignores it | emule bf16/bf8 done; **silicon todo** |
| **Memory location** | DRAM→DRAM, DRAM→L1, L1→DRAM, L1→L1 | silicon L1≫DRAM; emule flat | emule done; **silicon todo** |
| **Interleaved vs sharded** | interleaved vs width/height-sharded | layout cost | partial (via core-scaling) |
| **Core count** | 1, 2, 4, 8, 16, 32, 64 | isolates per-program vs per-core cost | emule done; **silicon todo** |
| **MathFidelity** | LoFi, HiFi2, HiFi3, HiFi4 (matmul) | multi-pass bf16 cost; faithfulness | **todo both** |
| **Lifecycle / ops-per-session** | N = 1..500; warm vs cold (tt-smi -r) | dispatch/bringup amortization | done both (warm + cold) |
| **Fusion** | K separate ops vs 1 fused program | emule per-program floor rewards fusion | **todo** |
| **Batch** | B = 1, 8, 32 at fixed per-batch shape | batch-dim handling | **todo** |
| **Metric** | e2e wall-clock; throughput Gelem/s; GFLOP/s; ratio; roofline | different lenses on same data | done (recast tooling exists) |

Legend: "done" = both backends measured; "emule done" = emule only, needs
silicon; "todo" = not yet run.

---

## 3. Slide-by-slide graph spec

Each entry: **graph → x-axis / y-axis → the point it makes → data status.**
Ideal form for comparison slides: overlay both backends + a ratio panel. (Add
tt-sim as a 3rd line everywhere if that backend comes online.)

### Act I — the headline: silicon wins on speed

1. **Per-op steady-state, eltwise** — `emule_vs_silicon_exp.png`, `_gelu.png`
   - x: total elements (log) · y: median e2e ms (log) + ratio panel
   - Point: emule flat ~90–115ms; silicon 0.2→3.8ms; **46–500× gap**, narrowing
     with size. Same shape for gelu → not op-specific.
   - Status: **done** (exp + gelu).

2. **matmul — the compute-bound extreme** — `emule_vs_silicon_matmul.png`
   - x: square side (log) · y: median ms (log) **and** GFLOP/s (log)
   - Point: **1000–1900×** per-op, **~3500×** peak (38.5 TFLOP/s vs 11 GFLOP/s).
     The op the matrix engines exist for is emulation's worst case.
   - Status: **done** to S=1024 (emule) / S=4096 (silicon).

3. **Throughput / roofline** — `throughput_vs_size.png`
   - x: total elements (log) · y: achieved Gelem/s (log)
   - Point: same S-curve, silicon shifted ~2.5 decades left (efficient at
     ~250–1000× smaller size). Silicon curve still rising at 2.4M.
   - Status: emule to 1B; **needs silicon large-tensor sweep to show its ceiling.**

### Act II — emule's cost model (what "emule perf" actually is)

4. **Bimodal op classes** — `emule_ops_scaling.png` + bar `emule_ops.png`
   - x: tiles (log) · y: median ms (log), one line per op, colored by class
   - Point: fixed-floor ops (unary/binary/softmax/transpose) flat ~140ms from 1
     tile; per-tile ops (matmul/reduce/tilize) rise from ~0.6ms. Cost = program
     structure, not data volume.
   - Status: **done** (emule; this is an emule-characterization slide).

5. **Core-count decomposition** — `emule_core_scaling.png`
   - x: cores 1→64 (fixed 1 tile/core, log2) · y: median ms (log)
   - Point: exp flat ~140ms across all core counts → fixed cost is **per-program**,
     not per-core; reduce rises 1→213ms → serial per-core emulation. Silicon
     would be flat for both (concurrent).
   - Status: emule **done**; **add silicon overlay** (should be flat/low for both).

6. **SFPU op complexity** — `emule_sfpu_complexity.png`
   - x: min ms (log) · y: op (sorted)
   - Point: emule reflects per-element math — relu 89ms … erf 219ms … **i0
     1420ms (16×)**. Expensive activations cost real emule time.
   - Status: emule **done**; **add silicon** (expect ~flat — HW SFPU is fixed-rate).

### Act III — what emule ignores that silicon doesn't

7. **dtype: bf16 vs bf8_b** — `emule_exp_dtype.png` (+ silicon)
   - x: total elements (log) · y: median ms (log), line per dtype
   - Point: emule bf16≈bf8 (overhead-bound); silicon bf8 faster (½ the bytes).
     Shows emule doesn't model bandwidth.
   - Status: emule **done**; **needs silicon dtype slice** to make the contrast.

8. **Operand location: L1 vs DRAM** — `emule_exp_memcfg.png` (+ silicon)
   - x: total elements (log) · y: median ms (log), line per {in→out} combo
   - Point: emule's 4 combos overlap; silicon L1≫DRAM. emule doesn't model L1
     bandwidth advantage.
   - Status: emule **done**; **needs silicon memcfg slice.**

### Act IV — the narrow regimes where emule wins

9. **One-shot lifecycle, warm** — `oneshot_crossover.png`
   - x: ops per session N (linear) · y: total wall-clock s
   - Point: emule 2.70s + 152ms·N vs silicon flat ~2.6s → **no crossover, silicon
     wins at every N** on a warm device.
   - Status: **done** both.

10. **One-shot lifecycle, cold** — `oneshot_cold_crossover.png`
    - x: ops per session N (linear) · y: total wall-clock s (incl. `tt-smi -r`)
    - Point: from a powered-off chip silicon pays ~11s bringup (the reset is
      ~8.5s) → **crossover N\* ≈ 55**: below it emule finishes a run-once job
      sooner. The one wall-clock win.
    - Status: **done** both (silicon cold at N=1; see gap #C).

11. **Capacity wall** (conceptual / small demo)
    - Point: tensor > silicon ~12GB DRAM → silicon OOMs, emule runs it in host
      RAM. emule "wins" by being able to run it at all.
    - Status: **todo** — small demo: run an op at ~7B elems on emule, show silicon
      OOM size.

### Act V — implications

12. **Optimization headroom** (analysis slide, not a sweep)
    - matmul.h already AVX2+FMA but compute path is **single-threaded**; host peak
      ~3–6 TFLOP/s vs WH 38.5 → **~10× is the hard floor** for faithful emulation;
      ~100× is recoverable via threading the per-core loop + AVX-512. Faithfulness
      constraint: parallelize over independent output tiles (preserve K-order).
    - Status: analysis done; a "projected optimized emule" line could be added.

### Optional new experiments (strengthen the deck)

13. **MathFidelity matmul** — x: LoFi/HiFi2/3/4 · y: ms, both backends. Does
    emule model the multi-pass cost? Faithfulness + perf. **todo.**
14. **Fusion** — x: chain length K · y: total ms, fused vs separate, both
    backends. emule's per-program floor should reward fusion far more. **todo.**
15. **Real-model composite ops** — layernorm, rms_norm, SDPA, embedding, conv at
    a representative size, both backends. Bridges microbench → model relevance.
    **todo.**
16. **Batch scaling** — x: batch B · y: ms, both backends. **todo.**

---

## 4. Silicon runs still needed (prioritized — you run on aus-wh-10)

All portable scripts already on the branch; pull + run with the normal tt-metal
env (no emule vars), scp CSVs back.

- **P0 — silicon large-tensor exp** → completes slides 3 (roofline ceiling) & the
  capacity story. `bench_eltwise_unary.py --op exp --side-tiles 32 128 256 512 768 1024`.
- **P0 — silicon core-count** → completes slide 5 (should be flat for both).
  `bench_core_scaling.py`.
- **P1 — silicon dtype slice** → slide 7. `plot_dtype.py <exp sweep json>`.
- **P1 — silicon L1/DRAM slice** → slide 8. `plot_memcfg.py <exp sweep json>`.
- **P1 — silicon SFPU complexity** → slide 6 (expect flat). Port the exp_extra
  op list into a small bench.
- **P2 — silicon matmul MathFidelity, fusion, norm/SDPA, batch** → slides 13–16.
- **C — cold-start at more N** (optional): silicon cold is one N=1 point flat-
  extrapolated on the measured warm slope; a couple more `--reset-first --ops K`
  runs would make slide 10's silicon line measured rather than projected.

---

## 5. Presentation structure (suggested flow)

1. Title + one-sentence thesis (§1).
2. Methodology: sweep_framework, same code path both sides, e2e_perf definition,
   faithfulness note (§ how measured).
3. Act I (slides 1–3): silicon wins on speed — eltwise, matmul, roofline.
4. Act II (slides 4–6): *why* — emule's cost model (bimodal, per-program vs
   per-core, SFPU complexity).
5. Act III (slides 7–8): what emule ignores (dtype, L1/DRAM) — caution for using
   emule as a perf proxy.
6. Act IV (slides 9–11): where emule wins (cold-start crossover, capacity).
7. Act V (slide 12 + 13–16 if run): implications & optimization headroom.
8. Summary: the three take-homes.

---

## 6. Caveats to state on the deck (credibility)

- e2e_perf includes a `ttnn.to_torch` readback and is single-run/uncached; both
  backends measured identically, so comparisons are fair.
- emule per-op timings are noisy (~1.5–3× jitter); use medians/mins over many
  iters. Large-matmul especially (S=1024 swung 120–193ms).
- Silicon `device open` (~0.9s) measured warm; cold bringup is the `tt-smi -r`
  (~8.5s), stated explicitly in the cold slide.
- Silicon topology (2-device WH) vs emule mock (N150 single) differ; fine for
  order-of-magnitude comparisons, note it for exactness.
- emule wins on *capacity/availability*, not throughput — say so plainly.
