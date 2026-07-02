# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Reduce a tt-metal sweep_framework results JSON to a size->time scaling curve.

The sweep_framework (tests/sweep_framework) runs an op across a shape × dtype ×
memory-config grid and exports per-vector records with an `e2e_perf_ms` metric.
This tool slices that grid to one config (dtype + in/out buffer) so the only
free axis is tensor size, aggregates by total elements, and writes a CSV in the
same schema perf/plot_bench.py consumes (backend, op, dtype, total_elements,
median_ms, ...). Run it on both the emule and silicon result JSONs, then plot.

  python perf/sweep_to_curve.py \
      /path/to/results_export/eltwise_<digest>_<ts>.json \
      --backend emule-wormhole --op exp --out emule_exp_curve.csv

Note on units: sweep_framework's stop_measuring_time returns nanoseconds; the
exporter divides by 1e6, so the stored `e2e_perf_ms` really is milliseconds.
The e2e region includes host readback (ttnn.to_torch) and is single-run
(uncached), matching how the framework reports it.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict


def params(record):
    """op_params_set -> flat {name: value} dict (json value preferred over text)."""
    out = {}
    for p in record.get("op_params_set") or []:
        v = p.get("param_value_json")
        if v is None:
            v = p.get("param_value_text")
        if v is None:
            v = p.get("param_value_numeric")
        out[p["param_name"]] = v
    return out


def e2e_ms(record):
    for m in record.get("metrics") or []:
        if m.get("metric_name") == "e2e_perf_ms":
            return m.get("metric_value")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_json", nargs="+", help="sweep_framework results file(s)")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--op", required=True, help="op label for the CSV (e.g. exp)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="DataType.BFLOAT16")
    ap.add_argument("--in-buffer", default="DRAM", choices=["DRAM", "L1"])
    ap.add_argument("--out-buffer", default="DRAM", choices=["DRAM", "L1"])
    args = ap.parse_args(argv or sys.argv[1:])

    by_elems = defaultdict(list)  # total_elements -> [e2e_ms, ...]
    kept = skipped = 0
    for path in args.results_json:
        with open(path) as f:
            records = json.load(f)
        for r in records:
            if not r.get("success"):
                skipped += 1
                continue
            p = params(r)
            if p.get("input_a_dtype") != args.dtype:
                continue
            if p.get("input_a_memory_config.data.buffer_type") != args.in_buffer:
                continue
            if p.get("output_memory_config.data.buffer_type") != args.out_buffer:
                continue
            shape = p.get("input_shape")
            t = e2e_ms(r)
            if not isinstance(shape, list) or t is None:
                skipped += 1
                continue
            by_elems[math.prod(shape)].append(float(t))
            kept += 1

    if not by_elems:
        print("no matching records for that config slice", file=sys.stderr)
        return 1

    rows = []
    for n in sorted(by_elems):
        s = by_elems[n]
        rows.append(
            {
                "backend": args.backend,
                "op": args.op,
                "dtype": args.dtype.split(".")[-1].lower(),
                "total_elements": n,
                "num_tiles": n // (32 * 32),
                "samples": len(s),
                "median_ms": statistics.median(s),
                "min_ms": min(s),
                "mean_ms": statistics.fmean(s),
                "std_ms": statistics.pstdev(s) if len(s) > 1 else 0.0,
            }
        )

    fields = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(
        f"[sweep_to_curve] backend={args.backend} slice={args.dtype.split('.')[-1]}/"
        f"{args.in_buffer}->{args.out_buffer}: kept {kept}, {len(rows)} size points "
        f"(skipped {skipped}) -> {args.out}"
    )
    for r in rows:
        print(
            f"  elems={r['total_elements']:>9} tiles={r['num_tiles']:>5} "
            f"n={r['samples']:>2} median={r['median_ms']:9.3f}ms min={r['min_ms']:9.3f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
