# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Slice a sweep_framework results JSON into the 4 in->out memory-config curves.

The eltwise sweeps run every shape at all four {DRAM,L1}x{DRAM,L1} in/out buffer
combos. This plots median e2e_perf_ms vs total elements, one line per combo, so
you can see whether operand location (L1 vs DRAM) moves the wall-clock — and by
how much — on a given backend.

  python perf/plot_memcfg.py <results>.json --backend emule-wh --op exp --out memcfg.png
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
    ap.add_argument("--dtype", default="DataType.BFLOAT16")
    ap.add_argument("--out", default="memcfg.png")
    args = ap.parse_args(argv or sys.argv[1:])

    combos = defaultdict(lambda: defaultdict(list))  # (in,out) -> {elems: [ms]}
    for r in json.load(open(args.results_json)):
        if not r.get("success"):
            continue
        p = params(r)
        if p.get("input_a_dtype") != args.dtype:
            continue
        shape = p.get("input_shape")
        t = e2e(r)
        if not isinstance(shape, list) or t is None:
            continue
        key = (p.get("input_a_memory_config.data.buffer_type"),
               p.get("output_memory_config.data.buffer_type"))
        combos[key][math.prod(shape)].append(float(t))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for key in sorted(combos):
        pts = combos[key]
        xs = sorted(pts)
        ys = [statistics.median(pts[x]) for x in xs]
        ax.plot(xs, ys, marker="o", ms=4, label=f"{key[0]} → {key[1]}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total elements")
    ax.set_ylabel("median e2e per op (ms)")
    ax.set_title(f"{args.backend} {args.op} ({args.dtype.split('.')[-1].lower()}) "
                 f"— e2e vs size by operand location")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(title="input → output buffer")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    # summary: median across sizes per combo
    for key in sorted(combos):
        allms = [m for x in combos[key] for m in combos[key][x]]
        print(f"  {key[0]:>4} -> {key[1]:<4}: median {statistics.median(allms):7.2f}ms "
              f"over {len(combos[key])} sizes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
