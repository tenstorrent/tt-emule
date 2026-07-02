# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Slice a sweep_framework results JSON into per-dtype size->time curves.

The eltwise sweeps run each shape at multiple input dtypes (bf16, bfloat8_b).
This plots median e2e_perf_ms vs total elements, one line per dtype, at a fixed
in/out buffer (default DRAM->DRAM), to see whether dtype moves wall-clock on a
backend. (emule is overhead-bound so expect ~overlap; silicon moves less data
with bf8 so expect a real gap.)

  python perf/plot_dtype.py <results>.json --backend emule-wh --op exp --out dtype.png
"""
import argparse
import json
import math
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def params(r):
    o = {}
    for p in r.get("op_params_set") or []:
        v = p.get("param_value_json")
        if v is None:
            v = p.get("param_value_text")
        o[p["param_name"]] = v
    return o


def e2e(r):
    for m in r.get("metrics") or []:
        if m.get("metric_name") == "e2e_perf_ms":
            return m.get("metric_value")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_json")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--op", required=True)
    ap.add_argument("--in-buffer", default="DRAM", choices=["DRAM", "L1"])
    ap.add_argument("--out-buffer", default="DRAM", choices=["DRAM", "L1"])
    ap.add_argument("--out", default="dtype.png")
    args = ap.parse_args(argv or sys.argv[1:])

    by = defaultdict(lambda: defaultdict(list))  # dtype -> {elems: [ms]}
    for r in json.load(open(args.results_json)):
        if not r.get("success"):
            continue
        p = params(r)
        if p.get("input_a_memory_config.data.buffer_type") != args.in_buffer:
            continue
        if p.get("output_memory_config.data.buffer_type") != args.out_buffer:
            continue
        shape = p.get("input_shape")
        t = e2e(r)
        dt = p.get("input_a_dtype")
        if not isinstance(shape, list) or t is None or dt is None:
            continue
        by[dt][math.prod(shape)].append(float(t))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for dt in sorted(by):
        pts = by[dt]
        xs = sorted(pts)
        ys = [statistics.median(pts[x]) for x in xs]
        ax.plot(xs, ys, marker="o", ms=4, label=dt.split(".")[-1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total elements")
    ax.set_ylabel("median e2e per op (ms)")
    ax.set_title(f"{args.backend} {args.op} — e2e vs size by dtype "
                 f"({args.in_buffer}→{args.out_buffer})")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(title="input dtype")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    for dt in sorted(by):
        allms = [m for x in by[dt] for m in by[dt][x]]
        print(f"  {dt.split('.')[-1]:>10}: median {statistics.median(allms):7.2f}ms "
              f"over {len(by[dt])} sizes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
