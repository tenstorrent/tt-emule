<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->
# perf — emule vs silicon single-op perf sweeps

Compare end-to-end per-op time of a single ttnn op across a size sweep, emule
(host, slow dispatch) vs silicon (real device), using **tt-metal's own
`sweep_framework`** for the measurement so both sides run the identical upstream
code path. Post-process the framework's JSON into a size→time curve and plot.

## Files

| File | Role |
|---|---|
| `run_sweep_emule.sh` | Drives the upstream two-stage sweep (`sweeps_parameter_generator.py` → `sweeps_runner.py --perf`) under the emule env + the accommodations emule needs. |
| `sweep_to_curve.py` | Reduces a `sweep_framework` results JSON to a size→time curve CSV (slices to one dtype + in/out buffer; aggregates `e2e_perf_ms` by total elements). |
| `plot_bench.py` | Plots one or more curve CSVs (log-log time vs total elements) + an emule/silicon ratio panel. matplotlib only, no device. |
| `bench_eltwise_unary.py`, `run_bench_emule.sh` | **Secondary** lightweight microbench (not the sweep_framework). Only used to push far past the sweep suite's max size — see "Reaching the compute-bound regime". |
| `emule_exp_curve.csv` | emule wormhole `exp` bf16 baseline curve (reduced from a `nightly` sweep run; the multi-MB raw framework JSON is not committed). |

## Method

`sweep_framework` runs the op across a shape × dtype × memory-config grid and
records `e2e_perf_ms` per vector (its `stop_measuring_time` returns nanoseconds;
the exporter divides by 1e6, so the stored value really is ms). The e2e region
wraps the op **and** host readback (`ttnn.to_torch`), single-run / uncached.
`sweep_to_curve.py` slices to one config (default `BFLOAT16, DRAM→DRAM`) so the
only free axis is size, then aggregates by total elements.

## Run on emule (here)

```bash
# one-time: the runner needs the progress-bar lib
/opt/ttmlir-toolchain/venv/bin/python -m pip install enlighten

TT_METAL_DIR=/localdev/mkamran/emule/tt-metal \
  perf/run_sweep_emule.sh eltwise.unary.exp.exp nightly
```

Emule needs three accommodations the script sets for you (a silicon box needs
none of them): `ARCH_NAME=wormhole_b0` (the runner rejects the box default
`compute`), and `TT_SMI_RESET_COMMAND=true` + `TT_SMI_RESET_FALLBACK_COMMAND=none`
(no device/tt-smi to reset — point the reset hook at a no-op).

## Run on silicon (upstream, unmodified)

On a box with a real device and the normal tt-metal python env, run the same two
upstream commands directly — no emule env, no accommodations:

```bash
cd $TT_METAL_HOME/tests/sweep_framework
python sweeps_parameter_generator.py --module-name eltwise.unary.exp.exp --suite-name nightly --tag perfrun
python sweeps_runner.py --module-name eltwise.unary.exp.exp --suite-name nightly \
    --vector-source vectors_export --result-dest results_export --perf --main-proc-verbose --tag perfrun
```

## Reduce + plot / compare

```bash
python perf/sweep_to_curve.py <emule results>.json  --backend emule-wormhole --op exp --out emule_exp_curve.csv
python perf/sweep_to_curve.py <silicon results>.json --backend silicon-n150  --op exp --out silicon_exp_curve.csv
python perf/plot_bench.py emule_exp_curve.csv silicon_exp_curve.csv --out compare.png
```

## Finding (emule, wormhole, exp bf16)

Across the `nightly` suite (2K → 2.4M elements) emule's `e2e_perf` is **flat and
dispatch-overhead-bound** — ~80–150ms with heavy jitter, no scaling trend. The
per-op cost is dominated by fixed program-launch overhead (JIT dispatch +
slow-dispatch launch + 64-core host program emulation), not per-element compute.

This inverts the naive expectation that emule wins at small sizes for lack of
device open/close overhead: emule's fixed per-program cost is *larger* than
silicon fast-dispatch (µs-scale), so on wall-clock silicon is expected to win
across this range. Confirm against real silicon data with the steps above.

### Reaching the compute-bound regime

The `nightly` suite tops out at ~2.4M elements, entirely inside emule's overhead
floor. The secondary microbench (`run_bench_emule.sh`, pure ttnn — also portable
to silicon) pushes further and shows emule only becomes **compute-bound above
~40M elements**, rising ~linearly to ~180ms at 151M. Use it when you need the
tail the sweep suite can't reach:

```bash
TT_METAL_DIR=/localdev/mkamran/emule/tt-metal \
  perf/run_bench_emule.sh --op exp --side-tiles 32 64 128 192 256 384 --out /tmp/emule_big.csv
```
