# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Measure the full lifecycle wall-clock of a run-once ttnn workload.

The steady-state per-op sweep (bench_eltwise_unary.py / sweep_framework)
deliberately EXCLUDES device bringup. This harness measures the opposite:
everything a one-shot process pays — `import ttnn`, device open, program
JIT-compile (first op), N steady-state ops, device close — as a single
end-to-end wall-clock, broken into phases.

Purpose: find the cold-start crossover. Total time is roughly

    total(N) ≈ bringup + compile + N · per_op_warm

Silicon has tiny per_op_warm (µs) but large `bringup` (real device init is
seconds); emule has large per_op_warm (~90ms) but small `bringup` (mock open
~0.2s). So for small N the bringup term dominates and emule can win end-to-end;
above some N the per-op term dominates and silicon wins. Run this at several
--ops values on each backend and compare total_ms to locate that N.

Backend-agnostic: only `import ttnn` + torch + stdlib. Runs unchanged on
silicon (normal tt-metal env, no emule vars). The very first line captures a
timestamp BEFORE importing ttnn so the import cost is part of the measurement.
"""

import time

# Capture process-relative t0 as early as possible — import ttnn is a real,
# backend-dependent cost of a one-shot run, so it must be inside the window.
_T0 = time.perf_counter()

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
import shlex  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402
import ttnn  # noqa: E402

_T_IMPORT = time.perf_counter()

OPS = {
    "exp": ttnn.exp,
    "gelu": ttnn.gelu,
    "sqrt": ttnn.sqrt,
    "sigmoid": ttnn.sigmoid,
    "relu": ttnn.relu,
}
TILE = 32


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True, help="label for the CSV row")
    p.add_argument("--op", choices=sorted(OPS), default="exp")
    p.add_argument("--ops", type=int, default=1, help="number of op invocations this run")
    p.add_argument("--side-tiles", type=int, default=8, help="square tensor side in tiles")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--out", default=None, help="CSV to append one row to (created w/ header)")
    p.add_argument(
        "--reset-first",
        action="store_true",
        help="TRUE COLD START (silicon only): run --reset-cmd before opening so "
        "open() pays full firmware-boot + DRAM-training. Only the FIRST process "
        "after a reset is cold, so use with --ops 1 (each invocation resets).",
    )
    p.add_argument(
        "--reset-cmd",
        default="tt-smi -r",
        help="reset command for --reset-first (default 'tt-smi -r'; try "
        "'tt-smi -r 0', 'tt-smi -r all', or prefix 'sudo ' as your box needs).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    op_fn = OPS[args.op]
    import_ms = (_T_IMPORT - _T0) * 1e3

    # TRUE COLD START: reset the chip so the following open() pays the full
    # cold bringup (PCIe re-enum + firmware boot + DRAM training), not a warm
    # re-open. The reset is the "power off" transition; open() below is then
    # the cold "power on". reset_ms is reported separately so you can see both.
    reset_ms = 0.0
    if args.reset_first:
        cmd = shlex.split(args.reset_cmd)
        print(f"[oneshot] COLD START: running reset '{args.reset_cmd}' ...", flush=True)
        t = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        reset_ms = (time.perf_counter() - t) * 1e3
        if r.returncode != 0:
            sys.stderr.write(
                f"[oneshot] reset command failed (rc={r.returncode}):\n{r.stderr}\n"
                "Adjust --reset-cmd for your box (e.g. 'tt-smi -r 0', add 'sudo ').\n"
            )
            sys.exit(2)

    t = time.perf_counter()
    dev = ttnn.open_device(device_id=args.device_id)
    ttnn.synchronize_device(dev)
    open_ms = (time.perf_counter() - t) * 1e3

    h = w = args.side_tiles * TILE
    host = torch.randn(1, 1, h, w, dtype=torch.bfloat16)
    x = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    # Op #1 is COLD — it pays the program JIT-compile. Ops #2..N are WARM
    # (program cache hit) — steady-state per-op cost.
    per_op = []
    for i in range(max(1, args.ops)):
        t = time.perf_counter()
        op_fn(x)
        ttnn.synchronize_device(dev)
        per_op.append((time.perf_counter() - t) * 1e3)

    compile_ms = per_op[0]
    warm = per_op[1:]
    warm_mean_ms = statistics.fmean(warm) if warm else float("nan")
    ops_total_ms = sum(per_op)

    ttnn.deallocate(x)
    t = time.perf_counter()
    ttnn.close_device(dev)
    close_ms = (time.perf_counter() - t) * 1e3

    # total_ms = the process (import->open->ops->close), the "warm re-open" view.
    # total_with_reset_ms adds the reset (chip power-off), i.e. the full
    # "cold device -> open -> op -> close" workload you actually pay from off.
    total_ms = import_ms + open_ms + ops_total_ms + close_ms
    total_with_reset_ms = reset_ms + total_ms
    bringup_ms = import_ms + open_ms  # size/op-independent one-time cost

    row = {
        "backend": args.backend,
        "op": args.op,
        "ops": args.ops,
        "side_tiles": args.side_tiles,
        "cold": int(args.reset_first),
        "reset_ms": round(reset_ms, 3),
        "import_ms": round(import_ms, 3),
        "open_ms": round(open_ms, 3),        # COLD bringup when --reset-first
        "compile_ms": round(compile_ms, 3),   # op #1 (cold)
        "warm_mean_ms": round(warm_mean_ms, 3),  # ops #2..N mean
        "ops_total_ms": round(ops_total_ms, 3),
        "close_ms": round(close_ms, 3),
        "bringup_ms": round(bringup_ms, 3),
        "total_ms": round(total_ms, 3),
        "total_with_reset_ms": round(total_with_reset_ms, 3),
    }

    print(
        f"[oneshot] backend={row['backend']} op={row['op']} ops={row['ops']} "
        f"side={row['side_tiles']} cold={bool(args.reset_first)}",
        flush=True,
    )
    for k in ("reset_ms", "import_ms", "open_ms", "compile_ms", "warm_mean_ms",
              "close_ms", "bringup_ms", "total_ms", "total_with_reset_ms"):
        print(f"[oneshot]   {k:>19} = {row[k]:10.3f}", flush=True)

    if args.out:
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                wr.writeheader()
            wr.writerow(row)
        print(f"[oneshot] appended row -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
