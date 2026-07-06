# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Cold-start lifecycle benchmark — the one regime where emule can beat silicon.

Measures total wall-clock to run a run-once job of N ops *from cold*, and sweeps
N to find the crossover. "Cold" means the fixed bring-up cost is paid fresh every
time — the thing a warm steady-state bench (bench.py) deliberately hides:

  * emule   : fresh process + CLEARED JIT cache -> pays import + first-time kernel
              JIT-compile + execute. No warmup, no reuse.
  * silicon : `tt-smi -r` (full board reset ~8.5s) + device bring-up + first
              program compile + execute. Pass it via --pre-cmd "tt-smi -r".

Each (N, rep) is a FRESH subprocess so nothing is reused between points. The
outer driver times the whole subprocess (python start + import + open + N ops +
close) and, on emule, wipes the JIT cache first; on silicon it runs --pre-cmd
first and times it as the reset cost. total_cold = reset + subprocess.

The inner process prints one INNER_JSON line decomposing its own time
(device-open vs op-execute) so the total can be split into bring-up vs compute.

Backend auto-detected from TT_METAL_EMULE_MODE (override with --backend).
Output: perf/data/coldstart__<backend>.csv.

  # emule (fresh cache wiped automatically each point):
  EMULE_SCRIPT=perf/bench_cold.py perf/run_emule.sh --op exp
  # silicon (on the n150 box, normal ttnn env, at the same pin):
  python bench_cold.py --op exp --pre-cmd "tt-smi -r" --outdir perf/data
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time

# same-dir import of the op builders (reuse exactly the bench.py definitions)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIELDNAMES = [
    "backend", "op", "side", "seq", "ops_n", "reps",
    "total_med_s", "total_min_s", "reset_med_s", "proc_med_s",
    "open_med_ms", "opsexec_med_ms", "error",
]

N_LIST = [1, 2, 4, 8, 16, 32, 64, 128]


def jit_cache_dir():
    return f"/tmp/tt_emule_jit_cache_{os.getuid()}"


def detect_backend():
    return "emule" if os.environ.get("TT_METAL_EMULE_MODE") == "1" else "silicon"


# ---------------------------------------------------------------------------
# inner: this process IS the cold workload. Open device, run N ops, exit.
# ---------------------------------------------------------------------------
def run_inner(op, side, seq, n):
    import ttnn
    from bench import build_op

    t0 = time.perf_counter()
    device = ttnn.open_device(device_id=0)
    t_open = time.perf_counter()

    p = dict(op=op, side=side, seq=seq)
    fn, _ = build_op(op, p, device)  # first fn() call below triggers the JIT compile

    t_ops0 = time.perf_counter()
    for _ in range(n):
        out = fn()
        ttnn.synchronize_device(device)
        try:
            ttnn.deallocate(out)
        except Exception:
            pass
    t_ops1 = time.perf_counter()

    ttnn.close_device(device)
    print("INNER_JSON " + json.dumps({
        "open_ms": (t_open - t0) * 1e3,
        "ops_ms": (t_ops1 - t_ops0) * 1e3,
    }), flush=True)


# ---------------------------------------------------------------------------
# outer: for each N, spawn `reps` fresh cold subprocesses and time them.
# ---------------------------------------------------------------------------
def run_outer(args):
    backend = args.backend or detect_backend()
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"coldstart__{backend}.csv")
    n_list = args.n or N_LIST

    print(f"backend={backend}  op={args.op}  side={args.side}  reps={args.reps}  "
          f"N={n_list}  pre_cmd={args.pre_cmd!r}  clear_cache={backend == 'emule' and not args.no_clear}")

    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        for n in n_list:
            totals, resets, procs, opens, opsms = [], [], [], [], []
            err = ""
            for rep in range(args.reps):
                # --- cold prep ---
                if backend == "emule" and not args.no_clear:
                    subprocess.run(["rm", "-rf", jit_cache_dir()], check=False)
                t_reset = 0.0
                if args.pre_cmd:
                    r0 = time.perf_counter()
                    subprocess.run(args.pre_cmd, shell=True, check=False)
                    t_reset = time.perf_counter() - r0

                # --- timed cold subprocess ---
                t0 = time.perf_counter()
                proc = subprocess.run(
                    [sys.executable, os.path.abspath(__file__), "--inner",
                     "--op", args.op, "--side", str(args.side), "--seq", str(args.seq),
                     "--ops", str(n)],
                    capture_output=True, text=True, env=os.environ.copy())
                t_proc = time.perf_counter() - t0

                open_ms = ops_ms = float("nan")
                for line in proc.stdout.splitlines():
                    if line.startswith("INNER_JSON "):
                        d = json.loads(line[len("INNER_JSON "):])
                        open_ms, ops_ms = d["open_ms"], d["ops_ms"]
                if proc.returncode != 0:
                    err = f"inner rc={proc.returncode}"
                    sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")

                totals.append(t_reset + t_proc)
                resets.append(t_reset)
                procs.append(t_proc)
                opens.append(open_ms)
                opsms.append(ops_ms)
                print(f"  N={n:4d} rep={rep}  total={t_reset + t_proc:7.3f}s  "
                      f"(reset={t_reset:5.2f}s proc={t_proc:6.3f}s open={open_ms:7.1f}ms ops={ops_ms:8.1f}ms)")

            def med(xs):
                xs = [x for x in xs if x == x]  # drop nan
                return round(statistics.median(xs), 4) if xs else ""

            w.writerow({
                "backend": backend, "op": args.op, "side": args.side, "seq": args.seq,
                "ops_n": n, "reps": args.reps,
                "total_med_s": med(totals), "total_min_s": round(min(totals), 4),
                "reset_med_s": med(resets), "proc_med_s": med(procs),
                "open_med_ms": med(opens), "opsexec_med_ms": med(opsms), "error": err,
            })
            fh.flush()
    print(f"-> wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--op", default="exp", help="op to run (default exp — light workload shows the win)")
    ap.add_argument("--side", type=int, default=8, help="square side in tiles")
    ap.add_argument("--seq", type=int, default=512, help="seq len (sdpa only)")
    ap.add_argument("--ops", type=int, default=1, help="[inner] ops to run in this process")
    ap.add_argument("--n", type=int, nargs="+", default=None, help="[outer] N values to sweep")
    ap.add_argument("--reps", type=int, default=3, help="[outer] cold repetitions per N (median)")
    ap.add_argument("--pre-cmd", default="", help="[outer] shell cmd before each cold rep (silicon: 'tt-smi -r')")
    ap.add_argument("--no-clear", action="store_true", help="[outer] do NOT wipe the emule JIT cache")
    ap.add_argument("--backend", choices=["emule", "silicon"], default=None)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "data"))
    args = ap.parse_args()

    if args.inner:
        run_inner(args.op, args.side, args.seq, args.ops)
    else:
        run_outer(args)


if __name__ == "__main__":
    main()
