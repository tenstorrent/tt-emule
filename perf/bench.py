# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
emule-vs-silicon perf collector.

Runs a set of ttnn ops across the axes we care about (size, dtype, memory
location, layout/sharding, core count, math fidelity) and writes one tidy
long-format CSV per graph. The SAME script runs on both backends:

  * emule   : run under perf/run_emule.sh (sets TT_METAL_EMULE_MODE=1 etc.)
  * silicon : run in a normal tt-metal ttnn env (no emule vars) on the device box

Backend is auto-detected from TT_METAL_EMULE_MODE and can be overridden with
--backend. Output goes to perf/data/<graph>__<backend>.csv. Send the silicon
CSVs back and perf/plot.py overlays whatever is present.

Metric: e2e host-observed latency (perf_counter around the op + a
synchronize_device). This is the honest wall-clock and the only metric that is
directly comparable across both backends without a profiler build.

Examples:
  python perf/bench.py --graph all
  python perf/bench.py --graph size_matmul fidelity_matmul --iters 5
  python perf/bench.py --list
"""
import argparse
import csv
import os
import statistics
import sys
import time
import traceback

import ttnn
import torch

TILE = 32

# ---------------------------------------------------------------------------
# CSV schema — one unified long format so every graph merges/plots the same way.
# Each graph fills the relevant subset of axis columns; the rest stay blank.
# ---------------------------------------------------------------------------
FIELDNAMES = [
    "backend", "graph", "op",
    "side", "elements", "dtype", "in_mem", "out_mem", "layout",
    "sharded", "cores", "fidelity", "seq",
    "flops", "iters", "ms_min", "ms_med", "ms_mean", "ms_p90", "error",
]

DTYPES = {
    "bf16": ttnn.bfloat16,
    "bf8_b": ttnn.bfloat8_b,
    "fp32": ttnn.float32,
}
MEMS = {
    "dram": ttnn.DRAM_MEMORY_CONFIG,
    "l1": ttnn.L1_MEMORY_CONFIG,
}


def fidelity_config(name):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, name),
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def time_call(fn, device, warmup, iters):
    """fn() performs one op and returns its output tensor. Returns ms stats."""
    for _ in range(warmup):
        out = fn()
        ttnn.synchronize_device(device)
        _try_dealloc(out)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e3)
        _try_dealloc(out)
    samples.sort()
    return {
        "ms_min": round(samples[0], 4),
        "ms_med": round(statistics.median(samples), 4),
        "ms_mean": round(statistics.fmean(samples), 4),
        "ms_p90": round(samples[min(len(samples) - 1, int(0.9 * len(samples)))], 4),
    }


def _try_dealloc(t):
    try:
        ttnn.deallocate(t)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# tensor builders
# ---------------------------------------------------------------------------
def _rand(shape, dtype, layout, mem, device):
    t = torch.randn(*shape, dtype=torch.float32)
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=mem)


def sq(side):
    """square shape in tiles -> element dims"""
    return [side * TILE, side * TILE]


def height_shard(shape, cores, device):
    """height-sharded mem config across `cores` cores (best-effort)."""
    grid = device.core_grid  # ttnn.CoreGrid
    x = min(cores, grid.x)
    y = max(1, (cores + grid.x - 1) // grid.x)
    y = min(y, grid.y)
    ncores = x * y
    shard_h = shape[0] // ncores
    shard_h = max(TILE, (shard_h // TILE) * TILE)
    return ttnn.create_sharded_memory_config(
        shape=(shard_h, shape[1]),
        core_grid=ttnn.CoreGrid(y=y, x=x),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    ), ncores


# ---------------------------------------------------------------------------
# op runners: each returns (callable, derived) where derived augments the row.
# ---------------------------------------------------------------------------
def build_op(op, p, device):
    dt = DTYPES[p.get("dtype", "bf16")]
    in_mem = MEMS[p.get("in_mem", "dram")]
    out_mem = MEMS[p.get("out_mem", "dram")]
    layout = ttnn.TILE_LAYOUT

    if op in ("exp", "relu", "gelu", "erf", "sqrt"):
        shape = sq(p["side"])
        x = _rand(shape, dt, layout, in_mem, device)
        f = getattr(ttnn, op)
        return (lambda: f(x, memory_config=out_mem)), {"elements": shape[0] * shape[1]}

    if op == "add":
        shape = sq(p["side"])
        # optional sharded operands
        if p.get("sharded"):
            base = _rand(shape, dt, layout, ttnn.DRAM_MEMORY_CONFIG, device)
            smem, ncores = height_shard(shape, p.get("cores", 8), device)
            a = ttnn.to_memory_config(base, smem)
            b = ttnn.to_memory_config(_rand(shape, dt, layout, ttnn.DRAM_MEMORY_CONFIG, device), smem)
            return (lambda: ttnn.add(a, b, memory_config=smem)), {"elements": shape[0] * shape[1], "cores": ncores}
        a = _rand(shape, dt, layout, in_mem, device)
        b = _rand(shape, dt, layout, in_mem, device)
        return (lambda: ttnn.add(a, b, memory_config=out_mem)), {"elements": shape[0] * shape[1]}

    if op == "matmul":
        s = p["side"] * TILE
        a = _rand([s, s], dt, layout, in_mem, device)
        b = _rand([s, s], dt, layout, in_mem, device)
        ckc = fidelity_config(p.get("fidelity", "HiFi4"))
        return (lambda: ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=out_mem)), \
               {"elements": s * s, "flops": 2 * s * s * s}

    if op == "softmax":
        shape = sq(p["side"])
        x = _rand(shape, dt, layout, in_mem, device)
        return (lambda: ttnn.softmax(x, dim=-1)), {"elements": shape[0] * shape[1]}

    if op == "transpose":
        shape = sq(p["side"])
        x = _rand(shape, dt, layout, in_mem, device)
        return (lambda: ttnn.transpose(x, -2, -1)), {"elements": shape[0] * shape[1]}

    if op == "tilize":
        shape = sq(p["side"])
        t = torch.randn(*shape, dtype=torch.float32)
        x = ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem)
        return (lambda: ttnn.tilize(x, memory_config=out_mem)), {"elements": shape[0] * shape[1]}

    if op == "layernorm":
        shape = sq(p["side"])
        x = _rand(shape, dt, layout, in_mem, device)
        return (lambda: ttnn.layer_norm(x, epsilon=1e-5)), {"elements": shape[0] * shape[1]}

    if op == "sdpa":
        seq = p["seq"]
        b, nh, dh = 1, 8, 128
        shape = [b, nh, seq, dh]
        q = _rand(shape, dt, layout, in_mem, device)
        k = _rand(shape, dt, layout, in_mem, device)
        v = _rand(shape, dt, layout, in_mem, device)
        return (lambda: ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)), \
               {"elements": b * nh * seq * dh, "flops": 2 * 2 * b * nh * seq * seq * dh}

    raise ValueError(f"unknown op {op}")


# ---------------------------------------------------------------------------
# graph definitions: each yields a list of param dicts (rows to run).
# Everything not being swept is pinned to the defaults (bf16, tile, dram->dram,
# HiFi4, single-core-equivalent).
# ---------------------------------------------------------------------------
SIDES = [1, 2, 4, 8, 16, 32]          # tiles/side -> 32..1024 elems/side (up to ~1M)
MM_SIDES = [1, 2, 4, 8, 16, 32]       # matmul is O(n^3); keep <=1024 (extend on silicon)
SEQS = [128, 256, 512, 1024, 2048]
CORE_COUNTS = [1, 2, 4, 8, 16, 32, 64]

CORE_OPS = ["exp", "add", "matmul", "softmax", "transpose", "tilize", "layernorm"]
SFPU_OPS = ["relu", "gelu", "exp", "erf", "sqrt"]


def graphs():
    g = {}

    # 1. size scaling, unary (exp)
    g["size_exp"] = [dict(op="exp", side=s) for s in SIDES]

    # 2. size scaling, matmul (HiFi4)
    g["size_matmul"] = [dict(op="matmul", side=s, fidelity="HiFi4") for s in MM_SIDES]

    # 3. op-class spread at a fixed medium size (side=8 -> 256x256)
    g["opspread"] = [dict(op=o, side=8) for o in CORE_OPS] + [dict(op="sdpa", seq=512)]

    # 4. SFPU math-complexity spread at fixed size
    g["sfpu"] = [dict(op=o, side=8) for o in SFPU_OPS]

    # 5. dtype: exp across sizes, bf16 vs bf8_b
    g["dtype_exp"] = [dict(op="exp", side=s, dtype=d) for d in ("bf16", "bf8_b") for s in SIDES]

    # 6. memory location: exp across sizes, the four {in->out} combos
    g["memcfg_exp"] = [dict(op="exp", side=s, in_mem=i, out_mem=o)
                       for i in ("dram", "l1") for o in ("dram", "l1") for s in SIDES]

    # 7. layout: add, interleaved vs height-sharded across sizes
    g["layout_add"] = ([dict(op="add", side=s, sharded=False) for s in SIDES]
                       + [dict(op="add", side=s, sharded=True, cores=8) for s in SIDES])

    # 8. core scaling: fixed 1 tile/core, sweep core count (exp, height-sharded)
    g["cores_exp"] = [dict(op="exp", side=c, sharded=True, cores=c) for c in CORE_COUNTS]

    # 9. math fidelity: fixed-size matmul across LoFi..HiFi4
    g["fidelity_matmul"] = [dict(op="matmul", side=8, fidelity=f)
                            for f in ("LoFi", "HiFi2", "HiFi3", "HiFi4")]

    # 10. data movement: transpose + tilize across sizes
    g["datamov"] = ([dict(op="transpose", side=s) for s in SIDES]
                    + [dict(op="tilize", side=s) for s in SIDES])

    # 11. composite / model ops: layernorm + softmax across sizes, sdpa across seq
    g["composite"] = ([dict(op="layernorm", side=s) for s in SIDES]
                      + [dict(op="softmax", side=s) for s in SIDES]
                      + [dict(op="sdpa", seq=q) for q in SEQS])

    return g


# ---------------------------------------------------------------------------
def detect_backend():
    if os.environ.get("TT_METAL_EMULE_MODE") == "1":
        return "emule"
    return "silicon"


def run_graph(name, rows, device, backend, warmup, iters, outdir):
    path = os.path.join(outdir, f"{name}__{backend}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        for p in rows:
            row = {k: "" for k in FIELDNAMES}
            row.update(backend=backend, graph=name, op=p["op"],
                       side=p.get("side", ""), dtype=p.get("dtype", "bf16"),
                       in_mem=p.get("in_mem", "dram"), out_mem=p.get("out_mem", "dram"),
                       layout="tile", sharded=int(bool(p.get("sharded", False))),
                       cores=p.get("cores", ""), fidelity=p.get("fidelity", ""),
                       seq=p.get("seq", ""), iters=iters)
            label = ",".join(f"{k}={v}" for k, v in p.items() if k != "op")
            try:
                fn, derived = build_op(p["op"], p, device)
                row.update(derived)
                row.update(time_call(fn, device, warmup, iters))
                print(f"  [{name}] {p['op']:10s} {label:40s} "
                      f"med={row['ms_med']:>10} ms  min={row['ms_min']} ms")
            except Exception as e:
                row["error"] = repr(e)
                print(f"  [{name}] {p['op']:10s} {label:40s} ERROR: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            w.writerow(row)
            fh.flush()
    print(f"  -> wrote {path}")
    return path


def main():
    all_graphs = graphs()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", nargs="+", default=["all"],
                    help=f"graphs to run (default all). choices: all, {', '.join(all_graphs)}")
    ap.add_argument("--backend", choices=["emule", "silicon"], default=None,
                    help="override backend label (default: auto from TT_METAL_EMULE_MODE)")
    ap.add_argument("--iters", type=int, default=8, help="timed iterations per point")
    ap.add_argument("--warmup", type=int, default=2, help="warmup iterations per point")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--list", action="store_true", help="list graphs and exit")
    args = ap.parse_args()

    if args.list:
        for k, v in all_graphs.items():
            print(f"{k:18s} {len(v)} points")
        return

    backend = args.backend or detect_backend()
    selected = list(all_graphs) if args.graph == ["all"] else args.graph
    unknown = [g for g in selected if g not in all_graphs]
    if unknown:
        ap.error(f"unknown graph(s): {unknown}. choices: all, {', '.join(all_graphs)}")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"backend={backend}  iters={args.iters}  warmup={args.warmup}  graphs={selected}")

    device = ttnn.open_device(device_id=0)
    try:
        for name in selected:
            print(f"\n=== {name} ===")
            run_graph(name, all_graphs[name], device, backend, args.warmup, args.iters, args.outdir)
    finally:
        ttnn.close_device(device)
    print("\ndone.")


if __name__ == "__main__":
    main()
