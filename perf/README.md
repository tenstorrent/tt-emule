<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->
# emule vs silicon — perf comparison harness

One script (`bench.py`) runs a fixed set of ttnn ops across the axes we care
about and writes one tidy CSV per graph. The **same script** runs on both
backends, so the comparison is apples-to-apples:

- **emule** — `perf/run_emule.sh` (sets the emulator env, tags rows `emule`)
- **silicon** — run `bench.py` directly in a normal tt-metal ttnn env on the
  device box (auto-tagged `silicon`)

`plot.py` then overlays whatever CSVs are present in `perf/data/` and writes
`perf/graphs/<graph>.png`, with a ratio sub-panel on line graphs.

**Metric:** e2e host-observed latency — `perf_counter` around the op call plus a
`synchronize_device`. This is the only metric directly comparable across both
backends without a profiler build.

**Reference silicon:** Wormhole B0 **n150** (single chip; matches emule's default
arch). No CCL/multichip ops in this pass.

## The graphs

| graph | op(s) | swept axis | pinned | shows |
|---|---|---|---|---|
| `size_exp` | exp | size (1→1024²) | bf16, DRAM→DRAM | overhead floor → compute knee |
| `size_matmul` | matmul | size | HiFi4, bf16 | compute-bound extreme |
| `opspread` | 8-op roster | (fixed side=8) | bf16, DRAM | which classes diverge |
| `sfpu` | relu/gelu/exp/erf/sqrt | (fixed side=8) | bf16, DRAM | emule tracks math cost, silicon flat |
| `dtype_exp` | exp | size × {bf16, bf8_b} | DRAM | emule ignores dtype width |
| `memcfg_exp` | exp | size × {DRAM,L1}² | bf16 | emule ignores L1≫DRAM |
| `layout_add` | add | size × {interleaved, sharded} | bf16 | layout cost |
| `cores_exp` | exp | cores 1→64 (1 tile/core) | bf16, sharded | per-program vs per-core (serial emule) |
| `fidelity_matmul` | matmul | LoFi→HiFi4 (side=8) | bf16 | emule ignores fidelity passes |
| `datamov` | transpose, tilize | size | bf16 | NoC-bound vs memcpy |
| `composite` | layernorm, softmax, sdpa | size / seq | bf16 | multi-kernel per-program floor |

## Run it

### emule (this box)
```bash
cd tt-emule
TT_METAL_DIR=/path/to/tt-metal perf/run_emule.sh --graph all
# or a subset:  perf/run_emule.sh --graph size_matmul fidelity_matmul --iters 5
```
Requires a **current** `build_emule` (see note below). Clear the JIT cache after a
rebuild: `rm -rf /tmp/tt_emule_jit_cache_$(id -u)`.

### silicon (device box, e.g. aus-wh-10)
Copy `perf/bench.py` over, then in your normal tt-metal ttnn env (no emule vars):
```bash
python bench.py --graph all --outdir perf/data
```
It auto-tags rows `silicon` (override with `--backend silicon`). scp the
`perf/data/*__silicon.csv` files back into this tree's `perf/data/`.

### plot (either box, once CSVs are gathered)
```bash
python perf/plot.py            # all graphs found
python perf/plot.py size_exp   # just one
```

## Options
- `--iters N` timed iterations per point (default 8), `--warmup N` (default 2)
- `--graph all | <names…>`, `--list` to list graphs
- `--backend emule|silicon` to override the auto-detected label

## Extending
Silicon can run bigger than emule (or n150's DRAM ceiling): edit `SIDES` /
`MM_SIDES` / `SEQS` at the top of `bench.py`. Add an op by adding a branch to
`build_op()` and a graph entry to `graphs()`; add its plot spec to `PLOT` in
`plot.py`.

## Prerequisite: build_emule must match the source
The JIT-compiled kernels use the current `include/jit_hw/` headers and link
against `build_emule`'s runtime lib. If the build predates header changes (new
`__emule_*` symbols), ops fail at load with `undefined symbol: __emule_*`. Rebuild
against the current pin + tt-emule main (see `../BUILD_GUIDE.md`) and clear the
JIT cache.
