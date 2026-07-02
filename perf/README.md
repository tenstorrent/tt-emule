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
| `emule_exp_curve.csv`, `silicon_exp_curve.csv` | reduced `exp` bf16 curves for emule (WH host) and silicon (WH-B0), both from a `nightly` sweep at git SHA `8b9517627da`. Raw multi-MB framework JSONs are not committed. |
| `emule_gelu_curve.csv`, `silicon_gelu_curve.csv` | same for `gelu` (companion op — confirms the finding is not exp-specific). |
| `emule_vs_silicon_exp.png`, `emule_vs_silicon_gelu.png` | overlay plots (left: both curves; right: emule/silicon ratio). |

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

## Finding (emule vs silicon WH-B0, exp bf16, DRAM→DRAM)

Overlay: `emule_vs_silicon_exp.png` (both from `nightly`, same git SHA
`8b9517627da`). Curves: `emule_exp_curve.csv`, `silicon_exp_curve.csv`.

- **emule** is **flat and dispatch-overhead-bound** — ~90–115ms with heavy
  jitter, no scaling trend across 2K → 2.4M elements. The per-op cost is fixed
  program-launch overhead (JIT dispatch + slow-dispatch launch + 64-core host
  program emulation), not per-element compute.
- **silicon** shows real **compute scaling** — a ~0.2ms dispatch floor at tiny
  sizes rising to ~3.8ms at 2.4M elements (1176× size → ~18× time; sub-linear
  because the small end is also floor-bound).
- **silicon wins everywhere**, by ~393× at <10K elements narrowing to ~46× at
  >1M. The gap *shrinks* with size (emule flat, silicon rising).

This **refutes** the initial hypothesis that emule wins at small sizes for lack
of device open/close overhead. It is exactly backwards: emule is relatively
*worst* at small sizes, where its fixed ~90ms program overhead is compared
against silicon's ~0.2ms. emule's fixed per-program cost (slow dispatch + host
emulation) dwarfs silicon fast-dispatch, so there is no small-size regime where
emule wins; the disadvantage only narrows as silicon's compute term grows.

(The lone ~400ms spikes on both curves at ~86K elements are first-run JIT/compile
outliers, not a size effect — the exporter records a single uncached run per
vector.)

**gelu confirms it.** The companion `gelu` sweep (`emule_vs_silicon_gelu.png`)
reproduces the exact shape: emule flat ~90–115ms, silicon ~0.24ms → ~2.95ms,
silicon winning ~502× at <10K narrowing to ~28× at >1M. The finding is not
exp-specific — the emule floor is the fixed program-launch cost, independent of
which SFPU op runs. (gelu shows the same ~86K first-run compile spike on both
sides.)

### Reaching the compute-bound regime

The `nightly` suite tops out at ~2.4M elements, entirely inside emule's overhead
floor. The secondary microbench (`run_bench_emule.sh`, pure ttnn — also portable
to silicon) pushes further and shows emule only becomes **compute-bound above
~40M elements**, rising ~linearly to ~180ms at 151M. Note that silicon keeps
scaling too, so any true emule/silicon crossover (if one exists) is far beyond
the measured range and would require both curves out in the compute-bound tail.
Use it when you need the tail the sweep suite can't reach:

```bash
TT_METAL_DIR=/localdev/mkamran/emule/tt-metal \
  perf/run_bench_emule.sh --op exp --side-tiles 32 64 128 192 256 384 --out /tmp/emule_big.csv
```

## One-shot lifecycle: is there ANY regime where emule wins on speed?

The per-op metric excludes device bringup. `oneshot_lifecycle.py` measures the
opposite — full run-once process wall-clock (`import ttnn` + open + first-op
compile + N warm ops + close) vs ops-per-session N — to test whether emule's
cheaper mock bringup lets it win end-to-end for small N. Fits and overlay:
`oneshot_crossover.py` → `oneshot_crossover.png`.

Result (exp, side=8 tiles, WH; silicon device **warm**):

| phase | emule | silicon |
|---|---|---|
| `import ttnn` | ~1690ms | ~1494ms |
| device open | ~201ms | ~914ms |
| first-op compile | ~871ms | ~66ms |
| close | ~0ms | ~115ms |
| warm per-op | ~160ms | ~0.14ms |

Fits: **emule `total(N) ≈ 2.70s + 152ms·N`**, **silicon `total(N) ≈ 2.60s`
(flat)**. **No positive crossover — silicon wins at every N ≥ 1.** emule's
device-open *is* ~713ms cheaper (mock vs real init), but that saving is almost
exactly cancelled by emule's ~805ms-slower first-op JIT compile (host-compiling
the emulated kernel), so the floors are equal (~2.6–2.7s) and emule's per-op
cost loses from N=1 up.

Caveat: silicon `open` here (~0.9s) is a **warm** re-open. A truly cold device
(post `tt-smi -r`: PCIe enum + firmware boot + DRAM training) can be seconds to
tens of seconds; that would raise silicon's floor and let emule win the
first-boot one-shot at low N.

**Measuring the true cold start.** `oneshot_lifecycle.py --reset-first` resets
the chip (chip "off") *before* `open()`, so `open_ms` captures the full cold
bringup. Each process is only cold once, so use `--ops 1` (every invocation
resets). The row records `reset_ms` (the power-off) separately and
`total_with_reset_ms` = the full "cold → open → op → close" workload. On silicon:

```bash
# one row per fresh cold start; tune --reset-cmd for your box
for i in 1 2 3; do
  python perf/oneshot_lifecycle.py --backend silicon-wh-cold --op exp --ops 1 \
      --reset-first --reset-cmd "tt-smi -r" --out silicon_cold.csv
done
```

Then compare against emule using the reset-inclusive column:

```bash
python perf/oneshot_crossover.py perf/emule_oneshot.csv silicon_cold.csv \
    --col total_with_reset_ms --out cold_crossover.png
```

(emule has no cold state — its mock `open` is a constant ~0.2s — so its
`total_with_reset_ms` equals its `total_ms`.)

### Bottom line — where emule actually beats silicon

On **every speed axis** — per-op, tensor size, dtype, memory config, and
one-shot lifecycle on a warm device — **silicon wins**. emule's advantages are
**capacity/availability**, not speed: a tensor larger than silicon's ~12GB DRAM
(silicon OOMs, emule runs it in host RAM), or needing more chips than are
physically present / no hardware at all.
