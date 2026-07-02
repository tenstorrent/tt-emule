# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Plot one or more perf CSVs (from bench_eltwise_unary.py) on a shared axis.

  python perf/plot_bench.py emule.csv silicon.csv --out compare.png

Left panel : median wall-clock (ms) vs total elements, log-log, one line per
             backend. Where two backends overlap, their crossover is the size
             at which silicon's fixed dispatch overhead is amortized.
Right panel: emule/silicon time ratio vs size (only when >=2 backends share
             sizes) — >1 means emule slower, <1 means emule faster.

matplotlib only; no ttnn / device needed, so this runs anywhere.
"""

import argparse
import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(paths):
    """path list -> {backend: {total_elements: median_ms}} plus meta."""
    series = defaultdict(dict)
    meta = {"op": set(), "dtype": set()}
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                be = row["backend"]
                n = int(row["total_elements"])
                series[be][n] = float(row["median_ms"])
                meta["op"].add(row.get("op", "?"))
                meta["dtype"].add(row.get("dtype", "?"))
    return series, meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csvs", nargs="+", help="one or more bench CSVs")
    p.add_argument("--out", default="perf_compare.png")
    p.add_argument("--title", default=None)
    args = p.parse_args(argv or sys.argv[1:])

    series, meta = load(args.csvs)
    if not series:
        print("no data", file=sys.stderr)
        return 1

    op = "/".join(sorted(meta["op"]))
    dtype = "/".join(sorted(meta["dtype"]))
    title = args.title or f"eltwise unary '{op}' ({dtype}) — end-to-end wall clock"

    have_ratio = len(series) >= 2
    fig, axes = plt.subplots(1, 2 if have_ratio else 1, figsize=(13 if have_ratio else 7, 5))
    ax = axes[0] if have_ratio else axes

    for be in sorted(series):
        xs = sorted(series[be])
        ys = [series[be][x] for x in xs]
        ax.plot(xs, ys, marker="o", label=be)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total elements")
    ax.set_ylabel("median wall-clock per op (ms)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()

    if have_ratio:
        backends = sorted(series)
        # ratio = (backend containing 'emule') / (other), else backends[0]/backends[1]
        emu = next((b for b in backends if "emule" in b), backends[0])
        sil = next((b for b in backends if b != emu), backends[1])
        common = sorted(set(series[emu]) & set(series[sil]))
        rax = axes[1]
        if common:
            ratio = [series[emu][x] / series[sil][x] for x in common]
            rax.plot(common, ratio, marker="s", color="tab:red")
            rax.axhline(1.0, color="k", ls="--", alpha=0.6)
            rax.set_xscale("log")
            rax.set_yscale("log")
            rax.set_xlabel("total elements")
            rax.set_ylabel(f"time ratio  {emu} / {sil}")
            rax.set_title("emule slower (>1)  vs  emule faster (<1)")
            rax.grid(True, which="both", ls=":", alpha=0.5)
        else:
            rax.text(0.5, 0.5, "no shared sizes\nbetween backends", ha="center", va="center")
            rax.axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}  ({', '.join(sorted(series))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
