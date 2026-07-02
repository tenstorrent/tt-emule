# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Portable square-matmul wall-clock sweep (FLOP-heavy companion to the eltwise bench).

matmul is O(N^3) compute over O(N^2) data, so unlike elementwise exp it is
genuinely compute/arithmetic-bound — the right op for a FLOP-vs-time / roofline
view. Sweeps square A[S,S] @ B[S,S] over side S (in tiles), measures warm median
wall-clock (warmup absorbs JIT compile), and records FLOPs = 2*S^3.

Backend-agnostic (import ttnn + torch only) — same file runs on emule (via env
wrapper) and silicon (normal tt-metal env, no emule vars), tagged by --backend.
"""
import argparse
import csv
import statistics
import sys
import time

import torch
import ttnn

TILE = 32


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--side-tiles", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--device-id", type=int, default=0)
    return p.parse_args(argv)


def bench_one(dev, side, warmup, iters):
    s = side * TILE
    a = ttnn.from_torch(torch.randn(s, s, dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.from_torch(torch.randn(s, s, dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    for _ in range(warmup):
        ttnn.matmul(a, b)
    ttnn.synchronize_device(dev)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ttnn.matmul(a, b)
        ttnn.synchronize_device(dev)
        samples.append((time.perf_counter() - t0) * 1e3)
    ttnn.deallocate(a); ttnn.deallocate(b)
    flops = 2.0 * s * s * s
    med = statistics.median(samples)
    return {
        "M": s, "N": s, "K": s, "total_flops": flops,
        "total_elements": s * s,  # output size, for shared-axis plotting
        "median_ms": med, "min_ms": min(samples),
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "gflops": flops / (med / 1e3) / 1e9,
    }


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    print(f"[matmul] backend={args.backend} sizes(side_tiles)={args.side_tiles} "
          f"warmup={args.warmup} iters={args.iters}", flush=True)
    dev = ttnn.open_device(device_id=args.device_id)
    rows = []
    try:
        for side in args.side_tiles:
            r = bench_one(dev, side, args.warmup, args.iters)
            r.update(backend=args.backend, op="matmul", dtype="bfloat16",
                     side_tiles=side, warmup=args.warmup, iters=args.iters)
            rows.append(r)
            print(f"[matmul] S={r['M']:>5} flops={r['total_flops']:.3e} "
                  f"median={r['median_ms']:9.2f}ms  {r['gflops']:8.3f} GFLOP/s", flush=True)
    finally:
        ttnn.close_device(dev)
    fields = ["backend", "op", "dtype", "side_tiles", "M", "N", "K", "total_flops",
              "total_elements", "warmup", "iters", "median_ms", "min_ms", "mean_ms",
              "std_ms", "gflops"]
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader(); wr.writerows(rows)
    print(f"[matmul] wrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
