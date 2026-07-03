# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Does emule op time scale with the number of cores the program spans?

Holds per-core work FIXED (1 tile/core) and varies the core grid via height-
sharding, so the only thing changing is how many cores the op runs on. If emule
emulates the core grid serially, time should rise ~linearly with core count; if
the fixed-floor cost is per-program (independent of cores), time stays flat.

Runs a fixed-floor op (exp) and a per-tile-scaling op (reduce) for contrast.
Backend-agnostic (import ttnn + torch); on silicon this instead shows the real
parallel hardware (time ~flat as cores grow, since they run concurrently).

  python perf/bench_core_scaling.py --backend emule-wh --out core_scaling.csv
"""
import argparse
import csv
import statistics
import sys
import time

import torch
import ttnn

TILE = 32


def grid_for(c):
    """Rectangular CoreRangeSet covering exactly c cores on an 8-wide grid."""
    if c <= 8:
        end = ttnn.CoreCoord(c - 1, 0)
    else:
        assert c % 8 == 0, "c>8 must be a multiple of 8"
        end = ttnn.CoreCoord(7, c // 8 - 1)
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), end)})


def sharded_tensor(dev, cores, tiles_per_core):
    """Height-shard a [1,1,cores*tpc*32, 32] tensor: tpc tiles on each of `cores`."""
    sh = cores * tiles_per_core * TILE
    w = TILE
    spec = ttnn.ShardSpec(grid_for(cores), (tiles_per_core * TILE, w),
                          ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                           ttnn.BufferType.L1, spec)
    t = torch.randn(1, 1, sh, w, dtype=torch.bfloat16)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)


def time_op(dev, fn, x, warmup, iters):
    for _ in range(warmup):
        fn(x)
    ttnn.synchronize_device(dev)
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(x)
        ttnn.synchronize_device(dev)
        s.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(s), min(s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cores", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--tiles-per-core", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args(argv or sys.argv[1:])

    print(f"[core] backend={args.backend} tiles/core={args.tiles_per_core} "
          f"cores={args.cores}", flush=True)
    dev = ttnn.open_device(device_id=args.device_id)
    rows = []
    ops = {"exp": lambda x: ttnn.exp(x), "reduce_sum": lambda x: ttnn.sum(x, dim=-1)}
    try:
        for c in args.cores:
            x = sharded_tensor(dev, c, args.tiles_per_core)
            for opname, fn in ops.items():
                try:
                    med, mn = time_op(dev, fn, x, args.warmup, args.iters)
                    err = ""
                except Exception as e:
                    med = mn = float("nan"); err = f"{type(e).__name__}: {str(e)[:100]}"
                rows.append(dict(backend=args.backend, op=opname, cores=c,
                                 tiles_per_core=args.tiles_per_core,
                                 total_tiles=c * args.tiles_per_core,
                                 median_ms=med, min_ms=mn, error=err))
                print(f"[core] cores={c:>3} {opname:<11} median={med:8.2f}ms "
                      f"min={mn:8.2f}ms {err}", flush=True)
            ttnn.deallocate(x)
    finally:
        ttnn.close_device(dev)

    fields = ["backend", "op", "cores", "tiles_per_core", "total_tiles",
              "median_ms", "min_ms", "error"]
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader(); wr.writerows(rows)
    print(f"[core] wrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
