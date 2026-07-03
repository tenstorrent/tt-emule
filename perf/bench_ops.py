# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Broad single-op perf survey: time many ttnn op CLASSES at a fixed size.

Where bench_eltwise_unary / bench_matmul sweep one op across size, this sweeps
across op *types* at one size, to see which classes are cheap vs expensive on a
backend (unary vs binary vs reduction vs softmax vs norm vs data-movement vs
matmul). Warm median wall-clock per op call (warmup absorbs JIT compile).

Backend-agnostic (import ttnn + torch). Each op is guarded — an unsupported op
records an error row instead of aborting the run.

  python perf/bench_ops.py --backend emule-wh --side-tiles 16 --out ops.csv
"""
import argparse
import csv
import statistics
import sys
import time

import torch
import ttnn

TILE = 32


def _t(dev, h, w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(torch.randn(1, 1, h, w, dtype=torch.bfloat16),
                           dtype=dtype, layout=layout, device=dev)


def build_ops(dev, side):
    """name -> (klass, thunk). thunk() runs the op once."""
    h = w = side * TILE
    x = _t(dev, h, w)
    y = _t(dev, h, w)
    xr = ttnn.from_torch(torch.randn(1, 1, h, w, dtype=torch.bfloat16),
                         dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    sq = _t(dev, h, w)  # square for matmul
    ops = {
        # unary (SFPU)
        "unary.exp": ("unary", lambda: ttnn.exp(x)),
        "unary.gelu": ("unary", lambda: ttnn.gelu(x)),
        "unary.sqrt": ("unary", lambda: ttnn.sqrt(x)),
        "unary.sigmoid": ("unary", lambda: ttnn.sigmoid(x)),
        "unary.relu": ("unary", lambda: ttnn.relu(x)),
        # binary
        "binary.add": ("binary", lambda: ttnn.add(x, y)),
        "binary.mul": ("binary", lambda: ttnn.multiply(x, y)),
        "binary.sub": ("binary", lambda: ttnn.subtract(x, y)),
        # reduction
        "reduce.sum_w": ("reduce", lambda: ttnn.sum(x, dim=-1)),
        "reduce.max_w": ("reduce", lambda: ttnn.max(x, dim=-1)),
        "reduce.mean_w": ("reduce", lambda: ttnn.mean(x, dim=-1)),
        # fused / multipass
        "fused.softmax": ("fused", lambda: ttnn.softmax(x, dim=-1)),
        # data movement
        "dm.transpose": ("datamov", lambda: ttnn.transpose(x, -2, -1)),
        "dm.tilize": ("datamov", lambda: ttnn.tilize(xr)),
        "dm.untilize": ("datamov", lambda: ttnn.untilize(x)),
        # matmul
        "matmul.sq": ("matmul", lambda: ttnn.matmul(sq, sq)),
    }
    return ops


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--side-tiles", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args(argv or sys.argv[1:])

    h = w = args.side_tiles * TILE
    print(f"[ops] backend={args.backend} size={h}x{w} ({args.side_tiles**2} tiles) "
          f"warmup={args.warmup} iters={args.iters}", flush=True)
    dev = ttnn.open_device(device_id=args.device_id)
    rows = []
    try:
        ops = build_ops(dev, args.side_tiles)
        for name, (klass, thunk) in ops.items():
            try:
                for _ in range(args.warmup):
                    thunk()
                ttnn.synchronize_device(dev)
                s = []
                for _ in range(args.iters):
                    t0 = time.perf_counter()
                    thunk()
                    ttnn.synchronize_device(dev)
                    s.append((time.perf_counter() - t0) * 1e3)
                r = dict(backend=args.backend, op=name, op_class=klass,
                         side_tiles=args.side_tiles, total_elements=h * w,
                         median_ms=statistics.median(s), min_ms=min(s),
                         std_ms=statistics.pstdev(s) if len(s) > 1 else 0.0, error="")
                print(f"[ops] {name:<18} {klass:<8} median={r['median_ms']:8.2f}ms "
                      f"min={r['min_ms']:8.2f}ms", flush=True)
            except Exception as e:
                r = dict(backend=args.backend, op=name, op_class=klass,
                         side_tiles=args.side_tiles, total_elements=h * w,
                         median_ms=float("nan"), min_ms=float("nan"), std_ms=0.0,
                         error=f"{type(e).__name__}: {str(e)[:120]}")
                print(f"[ops] {name:<18} {klass:<8} ERROR {r['error']}", flush=True)
            rows.append(r)
    finally:
        ttnn.close_device(dev)

    fields = ["backend", "op", "op_class", "side_tiles", "total_elements",
              "median_ms", "min_ms", "std_ms", "error"]
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader(); wr.writerows(rows)
    print(f"[ops] wrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
