# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Achieved throughput vs problem size, from median-time curve CSVs.

Recasts the size->time curves (bench_eltwise_unary.py or sweep_to_curve.py
output: columns backend,total_elements,median_ms) into achieved throughput:

    elem/s = total_elements / (median_ms / 1000)

For an elementwise op this is the "useful work rate" — it rises out of the
fixed-overhead floor and plateaus at the backend's compute/bandwidth ceiling, so
the plateau height is the honest peak and the knee is where overhead stops
dominating. (For a FLOP axis, scale by flops-per-element: exp ~ O(10) flops/elem
on most impls; Gelem/s x that ~ GFLOP/s.)

  python perf/plot_throughput.py emule_exp_large.csv silicon_exp_curve.csv --out tput.png
"""
import argparse
import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", default="throughput.png")
    ap.add_argument("--flops-per-elem", type=float, default=1.0,
                    help="scale elem/s -> FLOP/s (default 1 = report elem/s)")
    ap.add_argument("--title", default="achieved throughput vs size (elementwise exp)")
    args = ap.parse_args(argv or sys.argv[1:])

    series = defaultdict(dict)  # backend -> {elems: Gunit/s}
    for path in args.csvs:
        for r in csv.DictReader(open(path)):
            n = int(r["total_elements"]); ms = float(r["median_ms"])
            if ms > 0:
                series[r["backend"]][n] = (n / (ms / 1e3)) * args.flops_per_elem / 1e9

    unit = "GFLOP/s" if args.flops_per_elem != 1.0 else "Gelem/s"
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for be in sorted(series):
        xs = sorted(series[be]); ys = [series[be][x] for x in xs]
        ax.plot(xs, ys, marker="o", ms=4, label=be)
        peak = max(ys)
        ax.axhline(peak, color="gray", ls=":", alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total elements")
    ax.set_ylabel(f"achieved throughput ({unit})")
    ax.set_title(args.title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    for be in sorted(series):
        ys = series[be]
        top = max(ys.values())
        print(f"  {be}: peak {top:.3f} {unit} (at {max(ys, key=ys.get):,} elems)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
