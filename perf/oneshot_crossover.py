# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Plot one-shot lifecycle total(N ops) per backend and find the crossover.

Consumes CSVs from oneshot_lifecycle.py (one row per --ops value). For each
backend it fits total_ms ≈ a + b·N (a = fixed bringup+compile floor, b = warm
per-op cost) by least squares, plots the measured points + fit lines, and — when
two backends are present — reports the crossover N where their totals are equal:

    N* = (a_emule - a_silicon) / (b_silicon - b_emule)

Below N* the higher-floor backend (silicon) is slower end-to-end, so emule wins;
above N* the per-op term dominates and silicon wins.

  python perf/oneshot_crossover.py emule_oneshot.csv silicon_oneshot.csv --out crossover.png
"""

import argparse
import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(paths, col):
    series = defaultdict(dict)  # backend -> {ops: chosen time_ms}
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                # fall back to total_ms when the requested column is absent
                # (older CSVs) or blank — for those, reset_ms is 0 anyway.
                v = row.get(col) or row.get("total_ms")
                series[row["backend"]][int(row["ops"])] = float(v)
    return series


def linfit(xs, ys):
    """Least-squares a + b·x. Returns (a, b)."""
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return sy / n, 0.0
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return a, b


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csvs", nargs="+")
    p.add_argument("--out", default="oneshot_crossover.png")
    p.add_argument("--title", default="one-shot lifecycle: total wall-clock vs ops-per-session")
    p.add_argument(
        "--col",
        default="total_ms",
        help="which time column to fit: total_ms (process) or "
        "total_with_reset_ms (full cold: chip-off reset included). Default total_ms.",
    )
    args = p.parse_args(argv or sys.argv[1:])

    series = load(args.csvs, args.col)
    fits = {}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for be in sorted(series):
        xs = sorted(series[be])
        ys = [series[be][x] / 1e3 for x in xs]  # -> seconds
        a, b = linfit(xs, ys)
        fits[be] = (a, b)
        (line,) = ax.plot(xs, ys, marker="o", ls="", label=f"{be} (measured)")
        xf = [0, max(xs)]
        ax.plot(xf, [a + b * x for x in xf], ls="--", color=line.get_color(),
                label=f"{be} fit: {a:.2f}s + {b*1e3:.0f}ms·N")

    ax.set_xlabel("ops per session (N)")
    ax.set_ylabel("total process wall-clock (s)")
    ax.set_title(args.title)
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_ylim(bottom=0)

    msg = None
    backends = sorted(fits)
    if len(backends) >= 2:
        b1, b2 = backends[0], backends[1]
        a1, s1 = fits[b1]; a2, s2 = fits[b2]
        if s1 != s2:
            nstar = (a1 - a2) / (s2 - s1)
            if nstar > 0:
                yc = a1 + s1 * nstar
                ax.axvline(nstar, color="k", ls=":", alpha=0.6)
                ax.plot([nstar], [yc], "k*", ms=14)
                lo = b1 if a1 < a2 else b2   # lower floor wins for small N
                hi = b2 if a1 < a2 else b1
                msg = (f"crossover N* ≈ {nstar:.0f} ops: "
                       f"below it {lo} is faster end-to-end, above it {hi} wins")
            else:
                msg = "no positive crossover — one backend dominates at all N"
    ax.legend()
    if msg:
        ax.text(0.5, -0.16, msg, transform=ax.transAxes, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")
    for be, (a, b) in fits.items():
        print(f"  {be}: total(N) ≈ {a:.3f}s + {b*1e3:.1f}ms·N")
    if msg:
        print(f"  {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
