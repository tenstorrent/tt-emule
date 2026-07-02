# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Portable single-op wall-clock sweep for emule-vs-silicon perf comparison.

This script is backend-agnostic: it opens a ttnn device, runs one eltwise
unary op across a sweep of tensor sizes, and records end-to-end wall-clock
time (dispatch + execute + device sync) per invocation. It does NOT set any
emule-specific environment — the caller does that. Run it unchanged on:

  * emule   — via perf/run_bench_emule.sh (sets TT_METAL_EMULE_MODE etc.)
  * silicon — with the normal tt-metal python env (TT_METAL_HOME + PYTHONPATH),
              no emule vars, a real /dev/tenstorrent device.

Pass a distinct --backend label per run so the CSVs can be merged and the
two curves distinguished at plot time.

Metric: for each size we run --warmup untimed iterations (the first absorbs
JIT compile / program cache population), then --iters timed iterations. Each
timed iteration wraps the op call AND a device synchronize in perf_counter,
so the recorded time reflects op completion, not just async dispatch return
(critical on silicon fast-dispatch; a no-op cost on emule slow-dispatch).
We report the median across timed iterations as the headline number.
"""

import argparse
import csv
import statistics
import sys
import time

import torch
import ttnn

# op name -> ttnn callable. All are elementwise unary (one input tile in,
# one tile out) so the size sweep is a clean total-elements axis.
OPS = {
    "exp": ttnn.exp,
    "gelu": ttnn.gelu,
    "sqrt": ttnn.sqrt,
    "sigmoid": ttnn.sigmoid,
    "relu": ttnn.relu,
}

DTYPES = {
    "bfloat16": (ttnn.bfloat16, torch.bfloat16),
    "float32": (ttnn.float32, torch.float32),
}


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op", choices=sorted(OPS), default="exp")
    p.add_argument(
        "--backend",
        required=True,
        help="label written into every CSV row (e.g. 'emule', 'silicon-n150')",
    )
    p.add_argument("--out", required=True, help="CSV output path")
    p.add_argument(
        "--side-tiles",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="sweep points. Each S produces a square [1,1,S*32,S*32] tensor, "
        "i.e. S*S tiles / (S*32)^2 elements.",
    )
    p.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    p.add_argument("--warmup", type=int, default=3, help="untimed iterations")
    p.add_argument("--iters", type=int, default=15, help="timed iterations")
    p.add_argument("--device-id", type=int, default=0)
    return p.parse_args(argv)


TILE = 32


def bench_one(dev, op_fn, ttnn_dtype, torch_dtype, side, warmup, iters):
    """Return per-invocation timing stats (ms) for one square size."""
    h = w = side * TILE
    host = torch.randn(1, 1, h, w, dtype=torch_dtype)
    x = ttnn.from_torch(host, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=dev)

    # Warmup: first call pays JIT-compile / program-cache cost; discard.
    for _ in range(warmup):
        op_fn(x)
    ttnn.synchronize_device(dev)

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        op_fn(x)
        ttnn.synchronize_device(dev)
        samples.append((time.perf_counter() - t0) * 1e3)  # ms

    ttnn.deallocate(x)
    return {
        "height": h,
        "width": w,
        "total_elements": h * w,
        "num_tiles": side * side,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
    }


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    op_fn = OPS[args.op]
    ttnn_dtype, torch_dtype = DTYPES[args.dtype]

    print(
        f"[bench] backend={args.backend} op={args.op} dtype={args.dtype} "
        f"warmup={args.warmup} iters={args.iters} sizes(side_tiles)={args.side_tiles}",
        flush=True,
    )

    dev = ttnn.open_device(device_id=args.device_id)
    rows = []
    try:
        for side in args.side_tiles:
            r = bench_one(
                dev, op_fn, ttnn_dtype, torch_dtype, side, args.warmup, args.iters
            )
            r.update(
                backend=args.backend,
                op=args.op,
                dtype=args.dtype,
                warmup=args.warmup,
                iters=args.iters,
            )
            rows.append(r)
            print(
                f"[bench] side={side:>3} tiles={r['num_tiles']:>5} "
                f"elems={r['total_elements']:>9} "
                f"median={r['median_ms']:8.3f}ms min={r['min_ms']:8.3f}ms",
                flush=True,
            )
    finally:
        ttnn.close_device(dev)

    fields = [
        "backend", "op", "dtype", "height", "width", "total_elements",
        "num_tiles", "warmup", "iters", "median_ms", "min_ms", "mean_ms", "std_ms",
    ]
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"[bench] wrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
