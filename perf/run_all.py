# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""One-shot: run the whole emule-vs-silicon experiment battery on ONE backend.

Self-contained (only `import ttnn` + torch + stdlib). Opens the device once and
runs every experiment used in the comparison deck, writing one CSV per
experiment into --outdir, prefixed by --backend. Schemas match the emule CSVs in
this dir so the existing plotters overlay directly.

Run on silicon (aus-wh-10) with the normal tt-metal env (TT_METAL_HOME +
PYTHONPATH to the silicon _ttnn.so), NO emule vars:

    python run_all.py --backend silicon-wh --outdir /tmp/sil

then scp /tmp/sil/*.csv back. The same file also runs on emule (via the emule
env) for method-matched parity.

Experiments (each guarded; a failure writes an error row and continues):
  exp_size, core_scaling, dtype, memcfg, sfpu, matmul, matmul_fidelity,
  chain_tax, composite, batch_matmul
"""
import argparse
import csv
import os
import statistics
import sys
import time

import torch
import ttnn

TILE = 32


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True, help="label + CSV prefix, e.g. silicon-wh")
    p.add_argument("--outdir", default=".")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--quick", action="store_true", help="smaller sweeps for a fast smoke")
    p.add_argument("--only", default="", help="comma-list of experiments to run (default all)")
    return p.parse_args(argv or sys.argv[1:])


class Ctx:
    def __init__(self, dev, args):
        self.dev = dev
        self.args = args
        self.rows = {}

    def timed(self, fn, warmup=None, iters=None):
        wu = self.args.warmup if warmup is None else warmup
        it = self.args.iters if iters is None else iters
        for _ in range(wu):
            fn()
        ttnn.synchronize_device(self.dev)
        s = []
        for _ in range(it):
            t0 = time.perf_counter()
            fn()
            ttnn.synchronize_device(self.dev)
            s.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(s), min(s)

    def T(self, *shape, dt=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mc=None):
        t = torch.rand(*shape, dtype=torch.bfloat16) + 0.5
        kw = dict(dtype=dt, layout=layout, device=self.dev)
        if mc is not None:
            kw["memory_config"] = mc
        return ttnn.from_torch(t, **kw)

    def write(self, name, rows, fields):
        path = os.path.join(self.args.outdir, f"{self.args.backend}_{name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"  -> wrote {path} ({len(rows)} rows)", flush=True)


# ---------------------------------------------------------------- experiments
def exp_size(c):
    """exp size sweep (steady-state + roofline). Schema matches emule_exp_large.csv-ish."""
    sides = [1, 2, 4, 8, 16, 32] if c.args.quick else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024]
    rows = []
    for side in sides:
        h = w = side * TILE
        x = c.T(1, 1, h, w)
        try:
            med, mn = c.timed(lambda: ttnn.exp(x))
            err = ""
        except Exception as e:
            med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:90]}"
        rows.append(dict(backend=c.args.backend, op="exp", dtype="bfloat16",
                         side_tiles=side, total_elements=h * w, num_tiles=side * side,
                         median_ms=med, min_ms=mn, error=err))
        print(f"  exp side={side:>4} elems={h*w:>10} median={med:9.2f}ms {err}", flush=True)
        ttnn.deallocate(x)
    c.write("exp_size", rows, list(rows[0].keys()))


def _grid(cores):
    if cores <= 8:
        end = ttnn.CoreCoord(cores - 1, 0)
    else:
        end = ttnn.CoreCoord(7, cores // 8 - 1)
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), end)})


def core_scaling(c):
    counts = [1, 2, 4, 8] if c.args.quick else [1, 2, 4, 8, 16, 32, 64]
    rows = []
    for cores in counts:
        spec = ttnn.ShardSpec(_grid(cores), (TILE, TILE), ttnn.ShardOrientation.ROW_MAJOR)
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
        for op, fn in [("exp", ttnn.exp), ("reduce_sum", lambda x: ttnn.sum(x, dim=-1))]:
            try:
                x = c.T(1, 1, cores * TILE, TILE, mc=mc)
                med, mn = c.timed(lambda: fn(x)); err = ""
                ttnn.deallocate(x)
            except Exception as e:
                med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:90]}"
            rows.append(dict(backend=c.args.backend, op=op, cores=cores, tiles_per_core=1,
                             total_tiles=cores, median_ms=med, min_ms=mn, error=err))
            print(f"  cores={cores:>3} {op:<11} median={med:9.2f}ms {err}", flush=True)
    c.write("core_scaling", rows, list(rows[0].keys()))


def dtype(c):
    sides = [1, 4, 16] if c.args.quick else [1, 4, 16, 64, 128]
    dts = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b}
    rows = []
    for dname, dv in dts.items():
        for side in sides:
            h = w = side * TILE
            try:
                x = c.T(1, 1, h, w, dt=dv)
                med, mn = c.timed(lambda: ttnn.exp(x)); err = ""
                ttnn.deallocate(x)
            except Exception as e:
                med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:90]}"
            rows.append(dict(backend=c.args.backend, dtype=dname, total_elements=h * w,
                             median_ms=med, min_ms=mn, error=err))
            print(f"  dtype={dname:<10} side={side:>4} median={med:9.2f}ms {err}", flush=True)
    c.write("dtype", rows, list(rows[0].keys()))


def memcfg(c):
    # L1-resident tensors are capacity-limited -> keep sizes small.
    sides = [1, 2, 4] if c.args.quick else [1, 2, 4, 8]
    combos = [("DRAM", "DRAM"), ("DRAM", "L1"), ("L1", "DRAM"), ("L1", "L1")]
    bt = {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}
    rows = []
    for ib, ob in combos:
        for side in sides:
            h = w = side * TILE
            try:
                x = c.T(1, 1, h, w, mc=bt[ib])
                med, mn = c.timed(lambda: ttnn.exp(x, memory_config=bt[ob])); err = ""
                ttnn.deallocate(x)
            except Exception as e:
                med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:90]}"
            rows.append(dict(backend=c.args.backend, in_buf=ib, out_buf=ob,
                             total_elements=h * w, median_ms=med, min_ms=mn, error=err))
            print(f"  {ib}->{ob} side={side:>3} median={med:9.2f}ms {err}", flush=True)
    c.write("memcfg", rows, list(rows[0].keys()))


def sfpu(c):
    side = 64 if c.args.quick else 256  # compute-bound size
    h = w = side * TILE
    x = c.T(1, 1, h, w)
    ops = {"relu": ttnn.relu, "abs": ttnn.abs, "square": ttnn.square, "sqrt": ttnn.sqrt,
           "rsqrt": ttnn.rsqrt, "exp": ttnn.exp, "exp2": ttnn.exp2, "log": ttnn.log,
           "sigmoid": ttnn.sigmoid, "tanh": ttnn.tanh, "gelu": ttnn.gelu, "silu": ttnn.silu,
           "sin": ttnn.sin, "cos": ttnn.cos, "erf": ttnn.erf, "i0": ttnn.i0}
    rows = []
    for name, fn in ops.items():
        try:
            med, mn = c.timed(lambda: fn(x), warmup=2, iters=4); err = ""
        except Exception as e:
            med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:90]}"
        rows.append(dict(backend=c.args.backend, name=name, side=side,
                         median_ms=med, min_ms=mn, error=err))
        print(f"  sfpu {name:<10} median={med:9.2f}ms {err}", flush=True)
    ttnn.deallocate(x)
    c.write("sfpu", rows, list(rows[0].keys()))


def matmul(c):
    sides = [1, 4, 16, 32] if c.args.quick else [1, 2, 4, 8, 16, 32, 64, 128]
    rows = []
    for side in sides:
        s = side * TILE
        a = c.T(1, 1, s, s); b = c.T(1, 1, s, s)
        try:
            med, mn = c.timed(lambda: ttnn.matmul(a, b)); err = ""
        except Exception as e:
            med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:90]}"
        flops = 2.0 * s * s * s
        rows.append(dict(backend=c.args.backend, op="matmul", dtype="bfloat16", side_tiles=side,
                         M=s, N=s, K=s, total_flops=flops, total_elements=s * s,
                         median_ms=med, min_ms=mn,
                         gflops=(flops / (med / 1e3) / 1e9 if med == med else float("nan")),
                         error=err))
        print(f"  matmul S={s:>5} median={med:9.2f}ms {err}", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)
    c.write("matmul", rows, list(rows[0].keys()))


def matmul_fidelity(c):
    order = ["LoFi", "HiFi2", "HiFi3", "HiFi4"]
    fids = {n: getattr(ttnn.MathFidelity, n) for n in order if hasattr(ttnn.MathFidelity, n)}
    rows = []
    for side in ([16] if c.args.quick else [16, 32]):
        s = side * TILE
        a = c.T(1, 1, s, s); b = c.T(1, 1, s, s)
        for fn, fv in fids.items():
            try:
                cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=fv, math_approx_mode=False)
                med, mn = c.timed(lambda: ttnn.matmul(a, b, compute_kernel_config=cfg)); err = ""
            except Exception as e:
                med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:90]}"
            rows.append(dict(backend=c.args.backend, fidelity=fn, side=side, S=s,
                             median_ms=med, min_ms=mn, error=err))
            print(f"  fidelity S={s} {fn:<6} median={med:9.2f}ms {err}", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)
    c.write("matmul_fidelity", rows, list(rows[0].keys()))


def chain_tax(c):
    x = c.T(1, 1, 512, 512)
    rows = []
    for K in ([1, 2, 4, 8] if c.args.quick else [1, 2, 4, 8, 16]):
        def chain(K=K):
            y = x
            for _ in range(K):
                y = ttnn.exp(y)
            return y
        try:
            med, mn = c.timed(chain); err = ""
        except Exception as e:
            med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:90]}"
        rows.append(dict(backend=c.args.backend, chain_len=K, median_ms=med, min_ms=mn,
                         per_op_ms=(mn / K if mn == mn else float("nan")), error=err))
        print(f"  chain K={K:>2} total_min={mn:9.2f}ms", flush=True)
    c.write("chain_tax", rows, list(rows[0].keys()))


def composite(c):
    side = 16
    h = w = side * TILE
    x = c.T(1, 1, h, w); wt = c.T(1, 1, 1, w); bs = c.T(1, 1, 1, w)
    specs = {
        "silu": lambda: ttnn.silu(x),
        "layer_norm": lambda: ttnn.layer_norm(x, epsilon=1e-5),
        "layer_norm_w": lambda: ttnn.layer_norm(x, epsilon=1e-5, weight=wt, bias=bs),
        "rms_norm": lambda: ttnn.rms_norm(x, epsilon=1e-5),
        "rms_norm_w": lambda: ttnn.rms_norm(x, epsilon=1e-5, weight=wt),
    }
    rows = []
    for name, fn in specs.items():
        try:
            med, mn = c.timed(fn); err = ""
        except Exception as e:
            med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:90]}"
        rows.append(dict(backend=c.args.backend, op=name, median_ms=med, min_ms=mn, error=err))
        print(f"  composite {name:<14} median={med:9.2f}ms {err}", flush=True)
    # embedding
    try:
        vocab, dim = 4096, 512
        wtab = ttnn.from_torch(torch.rand(vocab, dim, dtype=torch.bfloat16),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=c.dev)
        idx = ttnn.from_torch(torch.randint(0, vocab, (1, 256), dtype=torch.int32),
                              dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=c.dev)
        med, mn = c.timed(lambda: ttnn.embedding(idx, wtab)); err = ""
    except Exception as e:
        med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:90]}"
    rows.append(dict(backend=c.args.backend, op="embedding", median_ms=med, min_ms=mn, error=err))
    print(f"  composite {'embedding':<14} median={med:9.2f}ms {err}", flush=True)
    # sdpa
    try:
        q = c.T(1, 8, 512, 64); k = c.T(1, 8, 512, 64); v = c.T(1, 8, 512, 64)
        med, mn = c.timed(lambda: ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True))
        err = ""
    except Exception as e:
        med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:90]}"
    rows.append(dict(backend=c.args.backend, op="sdpa_512x64x8h", median_ms=med, min_ms=mn, error=err))
    print(f"  composite {'sdpa':<14} median={med:9.2f}ms {err}", flush=True)
    c.write("composite", rows, list(rows[0].keys()))


def batch_matmul(c):
    rows = []
    for B in ([1, 4, 16] if c.args.quick else [1, 4, 16, 64]):
        a = c.T(B, 1, 256, 256); b = c.T(B, 1, 256, 256)
        try:
            med, mn = c.timed(lambda: ttnn.matmul(a, b)); err = ""
        except Exception as e:
            med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:90]}"
        rows.append(dict(backend=c.args.backend, batch=B, median_ms=med, min_ms=mn, error=err))
        print(f"  batch B={B:>3} median={med:9.2f}ms {err}", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)
    c.write("batch_matmul", rows, list(rows[0].keys()))


def transfer(c):
    """Host<->device transfer vs size. emule = host memcpy; silicon = PCIe.
    The one place emule may WIN a sub-operation."""
    sides = [4, 16, 64] if c.args.quick else [4, 16, 64, 128, 256, 512]
    rows = []
    for side in sides:
        h = w = side * TILE
        host = torch.rand(1, 1, h, w, dtype=torch.bfloat16) + 0.5
        # H2D: time from_torch (host tensor already built)
        try:
            med, mn = c.timed(lambda: ttnn.from_torch(host, dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=c.dev)); err = ""
        except Exception as e:
            med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:80]}"
        rows.append(dict(backend=c.args.backend, dir="H2D_from_torch", total_elements=h * w,
                         median_ms=med, min_ms=mn, error=err))
        # D2H: time to_torch (device tensor already built)
        x = c.T(1, 1, h, w)
        try:
            med, mn = c.timed(lambda: ttnn.to_torch(x)); err = ""
        except Exception as e:
            med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:80]}"
        rows.append(dict(backend=c.args.backend, dir="D2H_to_torch", total_elements=h * w,
                         median_ms=med, min_ms=mn, error=err))
        print(f"  transfer side={side:>4} elems={h*w:>9}", flush=True)
        ttnn.deallocate(x)
    c.write("transfer", rows, list(rows[0].keys()))


def datamov(c):
    side = 16
    h = w = side * TILE
    x = c.T(1, 1, h, w)
    xr = c.T(1, 1, h, w, layout=ttnn.ROW_MAJOR_LAYOUT)
    specs = {
        "concat_w": lambda: ttnn.concat([x, x], dim=-1),
        "concat_h": lambda: ttnn.concat([x, x], dim=-2),
        "permute": lambda: ttnn.permute(x, (0, 1, 3, 2)),
        "reshape": lambda: ttnn.reshape(x, (1, 1, w, h)),
        "repeat": lambda: ttnn.repeat(x, ttnn.Shape([1, 2, 1, 1])),
        "pad": lambda: ttnn.pad(x, [(0, 0), (0, 0), (0, TILE), (0, TILE)], value=0.0),
        "slice": lambda: ttnn.slice(x, [0, 0, 0, 0], [1, 1, h // 2, w // 2]),
        "typecast_bf8": lambda: ttnn.typecast(x, ttnn.bfloat8_b),
        "clone": lambda: ttnn.clone(x),
        "to_layout_rm": lambda: ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT),
        "to_layout_tile": lambda: ttnn.to_layout(xr, ttnn.TILE_LAYOUT),
    }
    rows = []
    for name, fn in specs.items():
        try:
            med, mn = c.timed(fn); err = ""
        except Exception as e:
            med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:90]}"
        rows.append(dict(backend=c.args.backend, op=name, median_ms=med, min_ms=mn, error=err))
        print(f"  datamov {name:<14} median={med:9.2f}ms {err}", flush=True)
    c.write("datamov", rows, list(rows[0].keys()))


def index_ops(c):
    side = 16
    h = w = side * TILE
    x = c.T(1, 1, h, w)
    specs = {
        "argmax_w": lambda: ttnn.argmax(x, dim=-1),
        "topk32_w": lambda: ttnn.topk(x, 32, dim=-1, largest=True, sorted=True),
        "sort_w": lambda: ttnn.sort(x, dim=-1),
        "max_w": lambda: ttnn.max(x, dim=-1),
    }
    rows = []
    for name, fn in specs.items():
        try:
            med, mn = c.timed(fn); err = ""
        except Exception as e:
            med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:90]}"
        rows.append(dict(backend=c.args.backend, op=name, median_ms=med, min_ms=mn, error=err))
        print(f"  index {name:<12} median={med:9.2f}ms {err}", flush=True)
    c.write("index_ops", rows, list(rows[0].keys()))


def sharded_vs_interleaved(c):
    """exp at fixed size: interleaved DRAM vs height-sharded across 64 cores."""
    rows = []
    for side in ([8, 16] if c.args.quick else [8, 16, 32, 64]):
        h = w = side * TILE
        # interleaved
        try:
            x = c.T(1, 1, h, w, mc=ttnn.DRAM_MEMORY_CONFIG)
            med, mn = c.timed(lambda: ttnn.exp(x)); err = ""
            ttnn.deallocate(x)
        except Exception as e:
            med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:80]}"
        rows.append(dict(backend=c.args.backend, layout="interleaved_dram",
                         total_elements=h * w, median_ms=med, min_ms=mn, error=err))
        # height-sharded across 64 cores (needs h divisible by 64 tiles)
        try:
            ntiles_h = side * side  # total tiles = side^2; shard height across 64 cores
            spec = ttnn.ShardSpec(_grid(64), (max(1, side * side // 64) * TILE, w),
                                  ttnn.ShardOrientation.ROW_MAJOR)
            mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
            x = c.T(1, 1, h, w, mc=mc)
            med, mn = c.timed(lambda: ttnn.exp(x)); err = ""
            ttnn.deallocate(x)
        except Exception as e:
            med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:80]}"
        rows.append(dict(backend=c.args.backend, layout="sharded_l1_64core",
                         total_elements=h * w, median_ms=med, min_ms=mn, error=err))
        print(f"  shard vs interleaved side={side}", flush=True)
    c.write("sharded_vs_interleaved", rows, list(rows[0].keys()))


def broadcast(c):
    side = 16
    h = w = side * TILE
    x = c.T(1, 1, h, w)
    y = c.T(1, 1, h, w)
    row = c.T(1, 1, 1, w)
    specs = {
        "add_full": lambda: ttnn.add(x, y),
        "add_row_bcast": lambda: ttnn.add(x, row),
        "add_scalar": lambda: ttnn.add(x, 1.0),
        "mul_scalar": lambda: ttnn.multiply(x, 2.0),
    }
    rows = []
    for name, fn in specs.items():
        try:
            med, mn = c.timed(fn); err = ""
        except Exception as e:
            med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:90]}"
        rows.append(dict(backend=c.args.backend, op=name, median_ms=med, min_ms=mn, error=err))
        print(f"  bcast {name:<14} median={med:9.2f}ms {err}", flush=True)
    c.write("broadcast", rows, list(rows[0].keys()))


def conv2d(c):
    rows = []
    cfgs = [(64, 64, 32, 32, 3), (128, 128, 16, 16, 3)]  # (Cin,Cout,H,W,K)
    for cin, cout, ih, iw, ks in cfgs:
        try:
            inp = ttnn.from_torch(torch.rand(1, ih, iw, cin, dtype=torch.bfloat16),
                                  dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=c.dev)
            wt = ttnn.from_torch(torch.rand(cout, cin, ks, ks, dtype=torch.bfloat16),
                                 dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=c.dev)
            def run():
                return ttnn.conv2d(input_tensor=inp, weight_tensor=wt, device=c.dev,
                                   in_channels=cin, out_channels=cout,
                                   batch_size=1, input_height=ih, input_width=iw,
                                   kernel_size=(ks, ks), stride=(1, 1), padding=(1, 1))
            med, mn = c.timed(run, warmup=1, iters=3); err = ""
        except Exception as e:
            med = mn = float("nan"); err = f"{type(e).__name__}:{str(e)[:120]}"
        rows.append(dict(backend=c.args.backend, cin=cin, cout=cout, ih=ih, iw=iw, k=ks,
                         median_ms=med, min_ms=mn, error=err))
        print(f"  conv2d {cin}->{cout} {ih}x{iw} k{ks} median={med:9.2f}ms {err}", flush=True)
    c.write("conv2d", rows, list(rows[0].keys()))


EXPERIMENTS = {
    "exp_size": exp_size, "core_scaling": core_scaling, "dtype": dtype, "memcfg": memcfg,
    "sfpu": sfpu, "matmul": matmul, "matmul_fidelity": matmul_fidelity,
    "chain_tax": chain_tax, "composite": composite, "batch_matmul": batch_matmul,
    "transfer": transfer, "datamov": datamov, "index_ops": index_ops,
    "sharded_vs_interleaved": sharded_vs_interleaved, "broadcast": broadcast, "conv2d": conv2d,
}


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    os.makedirs(args.outdir, exist_ok=True)
    todo = [x.strip() for x in args.only.split(",") if x.strip()] or list(EXPERIMENTS)
    print(f"[run_all] backend={args.backend} outdir={args.outdir} experiments={todo}", flush=True)
    dev = ttnn.open_device(device_id=args.device_id)
    c = Ctx(dev, args)
    try:
        for name in todo:
            print(f"\n=== {name} ===", flush=True)
            try:
                EXPERIMENTS[name](c)
            except Exception as e:
                print(f"  !! {name} aborted: {type(e).__name__}: {str(e)[:160]}", flush=True)
    finally:
        ttnn.close_device(dev)
    print("\n[run_all] DONE", flush=True)


if __name__ == "__main__":
    main()
