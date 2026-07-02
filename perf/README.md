<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->
# perf — emule vs silicon single-op wall-clock sweeps

Portable harness for comparing **end-to-end wall-clock** time of a single
ttnn op across a size sweep, on emule (host, slow dispatch) vs silicon (real
device). The python benchmark is backend-agnostic — the *same file* runs on
both; only the environment differs, and each run is tagged with a `--backend`
label so the CSVs merge and the curves stay distinguishable.

## Files

| File | Role |
|---|---|
| `bench_eltwise_unary.py` | Portable sweep+timer. Opens a device, runs one eltwise unary op over a size sweep, writes a CSV. No emule-specific env — the caller sets it. |
| `run_bench_emule.sh` | emule env wrapper (slow dispatch, mock cluster, `TT_METAL_EMULE_MODE=1`). Forces `--backend emule-<arch>`. |
| `plot_bench.py` | Reads ≥1 CSVs, plots median wall-clock vs total elements (log-log), + an emule/silicon ratio panel when two backends share sizes. matplotlib only — no device needed. |

## Method

For each size we run `--warmup` untimed iterations (the first pays JIT-compile
/ program-cache cost), then `--iters` timed iterations. Each timed sample wraps
**both** the op call and a `synchronize_device` in `perf_counter`, so it
measures op *completion*, not async dispatch return (matters on silicon
fast-dispatch; free on emule slow-dispatch). Headline number is the median.

Sizes are square `[1,1,S*32,S*32]` tensors, swept by `--side-tiles S` — total
elements `(S*32)^2`, tile count `S*S`.

## Run on emule (here)

```bash
TT_METAL_DIR=/localdev/mkamran/emule/tt-metal \
  perf/run_bench_emule.sh --op exp --side-tiles 1 2 4 8 16 32 64 128 256 \
  --out /tmp/emule_exp.csv
```

`ARCH=blackhole` selects the Blackhole mock cluster. `--op` also accepts
`gelu`, `sqrt`, `sigmoid`, `relu`.

## Run on silicon (portable — no emule env)

On a box with a real device and the normal tt-metal python env
(`TT_METAL_HOME`, `PYTHONPATH` → the silicon `_ttnn.so`), run the **same**
script directly — do **not** set any `TT_METAL_EMULE_MODE` / slow-dispatch /
mock-cluster vars:

```bash
python perf/bench_eltwise_unary.py --op exp \
  --side-tiles 1 2 4 8 16 32 64 128 256 \
  --backend silicon-n150 --out silicon_exp.csv
```

## Plot / compare

```bash
python perf/plot_bench.py emule_exp.csv silicon_exp.csv --out compare.png
```

## Finding (emule, wormhole, exp bf16)

emule per-op wall-clock is **dispatch-overhead-bound**: a flat ~85ms floor
(JIT dispatch + slow-dispatch program launch + 64-core host program
emulation) that swamps per-element compute up to ~10–40M elements. The
**compute-bound** regime only appears above ~40M elements, where time rises
roughly linearly (≈180ms at 151M elements).

This *inverts* the naive expectation that emule wins at small sizes for lack
of device open/close overhead: emule's fixed per-program cost is *larger* than
silicon fast-dispatch (µs-scale), so on wall-clock silicon is expected to win
across this entire range, with emule's flat floor only ever crossing at
extreme sizes. Confirm against real silicon data with the steps above.
