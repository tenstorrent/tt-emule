# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Plot emule-vs-silicon perf graphs from the CSVs bench.py produced.

Reads perf/data/<graph>__<backend>.csv for every backend present and writes
perf/graphs/<graph>.png. Line graphs get a ratio sub-panel (emule_ms /
silicon_ms = "x times silicon is faster") wherever both backends have the same
point. Graphs with only one backend still plot (single series).

  python perf/plot.py                 # all graphs found in perf/data
  python perf/plot.py size_exp datamov
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATADIR = os.path.join(os.path.dirname(__file__), "data")
OUTDIR = os.path.join(os.path.dirname(__file__), "graphs")

BACKEND_STYLE = {"emule": dict(color="#c1440e", marker="o"),
                 "silicon": dict(color="#1f6feb", marker="s")}

# kind: line (numeric x) | bar (categorical x)
# series: column(s) that distinguish lines within a backend (None => backend only)
PLOT = {
    "size_exp":        dict(kind="line", x="elements", xlabel="elements", series=None,
                            title="exp — size scaling"),
    "size_matmul":     dict(kind="line", x="elements", xlabel="elements (N=M=K side²)", series=None,
                            title="matmul — size scaling"),
    "opspread":        dict(kind="bar", x="op", series=None, title="op-class spread (side=8)"),
    "sfpu":            dict(kind="bar", x="op", series=None, title="SFPU math complexity (side=8)",
                            sort_by_emule=True),
    "dtype_exp":       dict(kind="line", x="elements", xlabel="elements", series=["dtype"],
                            title="exp — dtype (bf16 vs bf8_b)"),
    "memcfg_exp":      dict(kind="line", x="elements", xlabel="elements", series=["in_mem", "out_mem"],
                            title="exp — memory location (in→out)"),
    "layout_add":      dict(kind="line", x="elements", xlabel="elements", series=["sharded"],
                            title="add — interleaved vs sharded"),
    "cores_exp":       dict(kind="line", x="cores", xlabel="cores (1 tile/core)", series=None,
                            title="exp — core scaling"),
    "fidelity_matmul": dict(kind="bar", x="fidelity", series=None, title="matmul — MathFidelity (side=8)",
                            order=["LoFi", "HiFi2", "HiFi3", "HiFi4"]),
    "datamov":         dict(kind="line", x="elements", xlabel="elements", series=["op"],
                            title="data movement — transpose / tilize"),
    "composite":       dict(kind="line", x="elements", xlabel="elements", series=["op"],
                            title="composite / model ops"),
}


def load(graph):
    """returns backend -> list[row] for all CSVs of this graph."""
    out = defaultdict(list)
    for path in glob.glob(os.path.join(DATADIR, f"{graph}__*.csv")):
        backend = os.path.basename(path)[len(graph) + 2:-4]
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r.get("error"):
                    continue
                out[backend].append(r)
    return out


def series_key(row, cols):
    if not cols:
        return ""
    return "/".join(str(row[c]) for c in cols)


def fnum(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def plot_line(graph, spec, data):
    xcol = spec["x"]
    scols = spec["series"]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    # collect per (backend, series) -> {x: med}
    curves = defaultdict(dict)
    seriesset = set()
    for backend, rows in data.items():
        for r in rows:
            x = fnum(r[xcol]); y = fnum(r["ms_med"])
            if x is None or y is None:
                continue
            s = series_key(r, scols)
            seriesset.add(s)
            curves[(backend, s)][x] = y

    linestyles = ["-", "--", ":", "-."]
    smap = {s: linestyles[i % len(linestyles)] for i, s in enumerate(sorted(seriesset))}
    for (backend, s), pts in sorted(curves.items()):
        xs = sorted(pts)
        ys = [pts[x] for x in xs]
        st = BACKEND_STYLE.get(backend, {})
        lbl = backend + (f" [{s}]" if s else "")
        ax.plot(xs, ys, linestyle=smap[s], label=lbl, **st)

    # ratio panel: emule/silicon at matched (series, x)
    for s in sorted(seriesset):
        e = curves.get(("emule", s), {})
        si = curves.get(("silicon", s), {})
        common = sorted(set(e) & set(si))
        if common:
            axr.plot(common, [e[x] / si[x] for x in common], linestyle=smap[s],
                     color="#555", marker="d", label=(s or "ratio"))
    axr.axhline(1.0, color="k", lw=0.6, ls=":")

    ax.set_xscale("log"); ax.set_yscale("log")
    axr.set_xscale("log"); axr.set_yscale("log")
    ax.set_ylabel("median e2e latency (ms)")
    axr.set_ylabel("emule / silicon\n(×silicon faster)")
    axr.set_xlabel(spec.get("xlabel", xcol))
    ax.set_title(spec["title"])
    ax.legend(fontsize=8, ncol=2); ax.grid(True, which="both", alpha=0.3)
    axr.grid(True, which="both", alpha=0.3)
    if len(seriesset) > 1:
        axr.legend(fontsize=7, ncol=3)
    _save(graph, fig)


def plot_bar(graph, spec, data):
    xcol = spec["x"]
    # x categories
    cats = {}
    for backend, rows in data.items():
        for r in rows:
            cats.setdefault(r[xcol], {})[backend] = fnum(r["ms_med"])
    order = spec.get("order")
    if order:
        keys = [k for k in order if k in cats]
    elif spec.get("sort_by_emule"):
        keys = sorted(cats, key=lambda k: cats[k].get("emule", cats[k].get("silicon", 0)))
    else:
        keys = sorted(cats, key=lambda k: cats[k].get("emule", cats[k].get("silicon", 0)))
    backends = sorted({b for v in cats.values() for b in v})
    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(backends)
    width = 0.8 / max(n, 1)
    for i, b in enumerate(backends):
        xs = [j + i * width for j in range(len(keys))]
        ys = [cats[k].get(b, 0) for k in keys]
        ax.bar(xs, ys, width=width, label=b, color=BACKEND_STYLE.get(b, {}).get("color"))
    ax.set_xticks([j + width * (n - 1) / 2 for j in range(len(keys))])
    ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("median e2e latency (ms)")
    ax.set_title(spec["title"])
    ax.legend(); ax.grid(True, axis="y", which="both", alpha=0.3)
    _save(graph, fig)


def _save(graph, fig):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"{graph}.png")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graphs", nargs="*", help="graphs to plot (default: all present in perf/data)")
    args = ap.parse_args()

    present = sorted({os.path.basename(p).split("__")[0]
                      for p in glob.glob(os.path.join(DATADIR, "*__*.csv"))})
    todo = args.graphs or present
    for g in todo:
        if g not in PLOT:
            print(f"skip {g}: no plot spec"); continue
        data = load(g)
        if not data:
            print(f"skip {g}: no data"); continue
        spec = PLOT[g]
        (plot_line if spec["kind"] == "line" else plot_bar)(g, spec, data)


if __name__ == "__main__":
    main()
