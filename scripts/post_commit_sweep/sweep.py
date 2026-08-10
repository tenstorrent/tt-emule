#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""tt-emule post-commit pass-rate sweep — manifest expander + sharded runner.

Runs the same ttnn test set tt-metal runs in its per-arch post-commit CI lane
(`tests/pipeline_reorg/ttnn_sanity_tests.yaml`) under the emule backend, writing
per-entry JUnit XML + logs for `parse_sweep_results.py`. The unit of work (and
of sharding, and of one XML file) is the manifest *entry* — the faithful "what
metal runs" granularity. Successor to the one-off `arminale/bh-ci-sweep`
pipeline; arch-parameterized and runnable both in CI and locally.

Subcommands:
  expand  resolve + audit the arch's post-commit entry list (prints JSON).
  run     execute this shard's entries under the emule env (round-robin shard).
They are split so each CI shard job re-expands cheaply with no shared state.

The `run` default is **hybrid**: pass 1 no-forked (fast), then re-run only
entries that left no usable XML with `--forked`. `--forked` forces forking
everywhere (no retry); `--no-retry` is pass 1 only. See `entry_needs_retry` and
the README/docs for the measured ~0.8s/test fork cost that motivates it.

`--asan` runs the same entry set with the emule sanitizers armed, to surface
ASAN violations in tt-metal's kernels rather than to measure a pass rate. It
implies `--forked` and adds `-s`/`-v`; findings are extracted from the logs by
`scripts/asan_sweep/parse_asan_results.py`. See docs/asan-nightly-sweep.md.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write("ERROR: PyYAML required (pip install pyyaml)\n")
    raise


# `sku` selects WHICH manifest entries run; `mesh_device`/`cluster_desc` set how
# emule presents the device (mirrors scripts/run_ttnn_pytests_<arch>.sh) — two
# independent axes, hence wormhole's wh_n300_civ2 lane on the single-device N150.
ARCH_CONFIG = {
    "blackhole": {
        "sku": "bh_p150b_civ2",
        "mesh_device": "P100",
        "cluster_desc": "blackhole_P100.yaml",
    },
    "wormhole": {
        "sku": "wh_n300_civ2",
        "mesh_device": "N150",
        "cluster_desc": "wormhole_N150.yaml",
    },
}

EMULE_ENV = {
    "TT_METAL_EMULE_MODE": "1",
    "TT_METAL_SLOW_DISPATCH_MODE": "1",
}

# Per-entry wallclock backstop = 3× the manifest SKU timeout, capped — emule is
# heavier than silicon, so 3× gives headroom; the cap stops a runaway pinning a
# runner (a capped-out entry shows as "truncated" in the report).
WALLCLOCK_MULT = 3
WALLCLOCK_CAP = 5400
DEFAULT_PER_TEST_TIMEOUT = 300  # injected only when the manifest cmd lacks one

ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Live per-entry violation counter. Same grammar as
# scripts/asan_sweep/parse_asan_results.py — that report stays the authority;
# this only surfaces hits in the job log while the run is still in progress.
ASAN_LINE_RE = re.compile(r"\[ASAN ERROR\]\s+([^:]+):")


def slug(name: str) -> str:
    """Kebab-case a manifest entry name. MUST match parse_sweep_results.slugify."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def split_env_prefix(tokens):
    """Walk leading KEY=value tokens; return (env_dict, remaining_tokens)."""
    env = {}
    i = 0
    while i < len(tokens) and ENV_ASSIGN_RE.match(tokens[i]):
        k, _, v = tokens[i].partition("=")
        env[k] = v
        i += 1
    return env, tokens[i:]


def strip_x_flag(args):
    """Remove pytest's `-x` (stop-on-first-failure), incl. combined short flags
    like `-xv` / `-vx`. Returns (new_args, was_stripped)."""
    out, stripped = [], False
    for tok in args:
        if tok == "-x":
            stripped = True
            continue
        if len(tok) > 1 and tok.startswith("-") and not tok.startswith("--") and "x" in tok[1:]:
            new = "-" + tok[1:].replace("x", "")
            stripped = True
            if new != "-":
                out.append(new)
            continue
        out.append(tok)
    return out, stripped


def has_flag(args, flag):
    """True if `flag` appears as `--flag` or `--flag=...`."""
    return any(a == flag or a.startswith(flag + "=") for a in args)


def normalise_args(args):
    """Canonical multiset of args for byte-equal comparison: expand
    `--flag=value` into `--flag value`, then sort."""
    out = []
    for tok in args:
        if tok.startswith("--") and "=" in tok:
            k, _, v = tok.partition("=")
            out += [k, v]
        else:
            out.append(tok)
    return sorted(out)


def expand_entries(manifest_path: Path, arch: str):
    """Return (entries, audit) for the arch's post-commit lane.

    Each entry records the parsed manifest cmd (src_exe/src_env/src_args), the
    transformed emit_args, wallclock, and which transforms were applied. audit:
    {"ok": bool, "reasons": {slug: [str, ...]}}.
    """
    cfg = ARCH_CONFIG[arch]
    sku = cfg["sku"]
    with manifest_path.open() as f:
        manifest = yaml.safe_load(f)

    entries, reasons = [], {}
    audit_ok = True
    for e in manifest:
        if e.get("merge_gate", False):
            continue
        skus = e.get("skus") or {}
        if sku not in skus:
            continue

        name = e["name"]
        s = slug(name)
        src_tokens = shlex.split(e.get("cmd", ""))
        src_env, src_rest = split_env_prefix(src_tokens)
        if not src_rest:
            reasons[s] = ["empty cmd after env prefix"]
            audit_ok = False
            continue
        src_exe, src_args = src_rest[0], src_rest[1:]
        is_pytest = "pytest" in src_exe

        timeout_min = int(skus[sku].get("timeout", 30))
        wallclock = min(WALLCLOCK_MULT * timeout_min * 60, WALLCLOCK_CAP)

        # Note: --forked is NOT injected here — it is a per-RUN choice (the
        # hybrid runner leads with no-forked and only retries failed entries
        # with --forked). run_entry() adds it when asked.
        emit_args = list(src_args)
        x_stripped = coce_injected = timeout_injected = False
        if is_pytest:
            emit_args, x_stripped = strip_x_flag(emit_args)
            if not has_flag(emit_args, "--continue-on-collection-errors"):
                emit_args = ["--continue-on-collection-errors"] + emit_args
                coce_injected = True
            if not has_flag(emit_args, "--timeout"):
                emit_args = ["--timeout", str(DEFAULT_PER_TEST_TIMEOUT)] + emit_args
                timeout_injected = True

        rec = {
            "name": name, "slug": s, "is_pytest": is_pytest,
            "src_exe": src_exe, "src_env": src_env, "src_args": src_args,
            "emit_args": emit_args, "wallclock": wallclock,
            "x_stripped": x_stripped, "coce_injected": coce_injected,
            "timeout_injected": timeout_injected,
        }
        r = audit_entry(rec)
        if r:
            reasons[s] = r
            audit_ok = False
        entries.append(rec)

    if not entries:
        raise SystemExit(f"ERROR: no entries match (arch={arch}, sku={sku}) in {manifest_path}")
    return entries, {"ok": audit_ok, "reasons": reasons}


def audit_entry(rec):
    """Prove emit_args == src_args modulo the allowed transformations. Returns a
    list of failure reasons (empty == PASS)."""
    if not rec["is_pytest"]:
        return []  # non-pytest entries pass through verbatim
    canonical = list(rec["emit_args"])

    def drop(flag):
        if flag in canonical:
            canonical.remove(flag)
            return True
        return False

    reasons = []
    if rec["coce_injected"] and not drop("--continue-on-collection-errors"):
        reasons.append("claimed --continue-on-collection-errors injection but flag absent")
    if rec["timeout_injected"]:
        # remove the injected `--timeout N` pair
        if "--timeout" in canonical:
            i = canonical.index("--timeout")
            del canonical[i:i + 2]
        else:
            reasons.append("claimed --timeout injection but flag absent")

    # Reverse the -x strip on the source side so the comparison is apples-to-apples.
    cmp_src, _ = strip_x_flag(rec["src_args"]) if rec["x_stripped"] else (rec["src_args"], False)
    if normalise_args(canonical) != normalise_args(cmp_src):
        reasons.append(
            f"args differ after reversing transforms: "
            f"src={normalise_args(cmp_src)} emitted={normalise_args(canonical)}")
    return reasons


def select_shard(entries, shard_index, shard_count):
    """Round-robin shard selection (1-based index). Round-robin (not contiguous
    blocks) so heavy entries distribute across shards instead of bunching."""
    return [e for i, e in enumerate(entries)
            if i % shard_count == (shard_index - 1)]


def build_runtime_env(tt_metal_dir: Path, build_dir: Path, arch: str, pytest_bin: Path,
                      asan: bool = False):
    """The full environment for an emule pytest invocation: build/runtime layout
    + emule env + arch variant env. Mirrors scripts/run_ttnn_pytests_<arch>.sh
    and snapshots/bh_sanity/env_setup.sh.

    `asan` arms the emule sanitizers for the whole shard. Core dumps are left
    suppressed (TT_METAL_EMULE_ASAN_ALLOW_CORE stays unset) — an ASAN abort
    dumps a multi-GB core, which fills a runner's disk within a few entries."""
    cfg = ARCH_CONFIG[arch]
    cluster_examples = tt_metal_dir / "tt_metal/third_party/umd/tests/cluster_descriptor_examples"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(tt_metal_dir / "ttnn"), str(tt_metal_dir / "tools"),
        str(build_dir / "lib"), str(tt_metal_dir), env.get("PYTHONPATH", ""),
    ]).rstrip(os.pathsep)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(build_dir / "lib"), env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    env["TT_METAL_HOME"] = str(tt_metal_dir)
    env["TT_METAL_RUNTIME_ROOT"] = str(tt_metal_dir)
    env.update(EMULE_ENV)
    if asan:
        env["TT_METAL_EMULE_ASAN"] = "1"
    env["MESH_DEVICE"] = cfg["mesh_device"]
    env["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(cluster_examples / cfg["cluster_desc"])
    # Make the toolchain pytest's dir first on PATH (manifest cmds call `pytest`
    # bare, and any inner subprocess should resolve the same binary).
    env["PATH"] = os.pathsep.join([str(pytest_bin.parent), env.get("PATH", "")])
    return env


IMPORT_ERR_MODULE_RE = re.compile(r"No module named '([^']+)'")
COLLECT_FILTER_TIMEOUT = 600  # seconds for the collect-only pre-pass


def collect_filter(rec, *, tt_metal_dir, runtime_env, pytest_bin):
    """Pre-pass: find test files in this entry that fail to IMPORT.

    pytest 9.0.3's `--continue-on-collection-errors` is broken — a single
    ImportError aborts the entire run (collected items, ran ZERO). So we run a
    fast `--collect-only` first (imports only; no device init, ~seconds), parse
    the files that error, and `--ignore` them on the real run. They are recorded
    in <slug>.excluded.json so the parser still reports them as excluded files.

    Returns a list of {file, missing_module} dicts (relative paths).
    """
    exe = str(pytest_bin) if rec["src_exe"] == "pytest" else rec["src_exe"]
    # emit_args already carries --continue-on-collection-errors (so collect lists
    # the good items); --forked is pointless when only collecting.
    args = [a for a in rec["emit_args"] if a != "--forked"]
    argv = [exe, "--collect-only", "-q"] + args
    env = dict(runtime_env)
    env.update(rec["src_env"])
    try:
        proc = subprocess.run(argv, cwd=str(tt_metal_dir), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=COLLECT_FILTER_TIMEOUT)
        out = proc.stdout.decode("utf-8", "ignore")
    except subprocess.TimeoutExpired:
        print(f"# collect-only timed out after {COLLECT_FILTER_TIMEOUT}s — skipping pre-filter")
        return []

    bad = []
    # Each "ERROR collecting <file>" block is followed (somewhere) by the
    # missing-module line; pair them per block by splitting on the marker.
    blocks = re.split(r"ERROR collecting ", out)
    for blk in blocks[1:]:
        fm = re.match(r"(\S+\.py)", blk)
        if not fm:
            continue
        mm = IMPORT_ERR_MODULE_RE.search(blk)
        bad.append({"file": fm.group(1), "missing_module": mm.group(1) if mm else "(see log)"})
    # De-dup by file.
    seen, uniq = set(), []
    for b in bad:
        if b["file"] not in seen:
            seen.add(b["file"])
            uniq.append(b)
    return uniq


def run_entry(rec, *, tt_metal_dir, out_dir, runtime_env, pytest_bin,
              forked=False, dry_run=False, asan=False):
    """Execute one entry: `timeout <wallclock> <exe> [--forked] <args> --junitxml=`
    with stdout+stderr to <out>/<slug>.log. `forked` toggles per-test process
    isolation. `asan` adds the two pytest flags the sanitizer report needs to
    survive (see below). Returns the entry rc."""
    s = rec["slug"]
    xml_path = out_dir / f"{s}.xml"
    log_path = out_dir / f"{s}.log"
    excluded_path = out_dir / f"{s}.excluded.json"

    env = dict(runtime_env)
    env.update(rec["src_env"])  # per-entry source env (rare) layered last

    # Pre-filter import-error files (pytest-9.0.3 collection-error bug). On a
    # forked retry the sidecar already exists — reuse it instead of re-collecting.
    excluded = []
    if rec["is_pytest"] and not dry_run:
        cached = None
        if excluded_path.is_file():
            try:
                cached = json.loads(excluded_path.read_text())
            except (OSError, json.JSONDecodeError):
                # Unreadable sidecar — fall through and regenerate rather than
                # silently skipping the collect-only pre-pass, which would
                # reintroduce the pytest-9.0.3 collection-abort bug (and leave
                # the bad sidecar in place for future runs).
                cached = None
        if cached is None:
            excluded = collect_filter(rec, tt_metal_dir=tt_metal_dir,
                                      runtime_env=env, pytest_bin=pytest_bin)
            excluded_path.write_text(json.dumps(excluded, indent=2) + "\n")
        else:
            excluded = cached
        if excluded:
            print(f"# pre-filter: ignoring {len(excluded)} import-error file(s): "
                  + ", ".join(e["file"] for e in excluded))

    exe = str(pytest_bin) if rec["src_exe"] == "pytest" else rec["src_exe"]
    # Two ways an import-error file reaches pytest, both must be neutralized:
    #   1. As an explicit positional target `file.py` or `file.py::node` — here
    #      --ignore is OVERRIDDEN by the explicit arg ("found no collectors"),
    #      so we must DROP the matching positional tokens.
    #   2. Collected via a directory target (e.g. `tests/.../eltwise`) — here
    #      there is no positional to drop, so --ignore=<file> is what excludes it.
    bad_files = {e["file"] for e in excluded}
    run_args = [a for a in rec["emit_args"]
                if a not in bad_files and not any(a.startswith(bf + "::") for bf in bad_files)]
    forked_args = ["--forked"] if (forked and "--forked" not in run_args) else []
    ignore_args = [f"--ignore={f}" for f in bad_files]
    # Both ASAN flags are load-bearing, not cosmetic:
    #   -s  pytest's fd-level capture points fd 2 at a temp file it only drains
    #       during teardown. An [ASAN ERROR] report ends in abort(), so teardown
    #       never runs and the entire report — header, kernel identity, backtrace
    #       — is discarded; the log shows only "Fatal Python error: Aborted".
    #       Measured on this stack: 0 reports captured without -s, 1 with it.
    #   -v  prints each nodeid before its test runs, which is what lets
    #       parse_asan_results.py attribute a log region to a specific test.
    asan_args = [f for f in ("-s", "-v") if f not in run_args] if (asan and rec["is_pytest"]) else []
    argv = ["timeout", str(rec["wallclock"]), exe] + forked_args + asan_args + ignore_args + run_args
    if rec["is_pytest"]:
        argv.append(f"--junitxml={xml_path}")

    print(f"::group::{rec['name']}")
    print(f"# slug:      {s}")
    print(f"# wallclock: {rec['wallclock']}s  --forked: {'on' if forked else 'off'}")
    print(f"# cmd:       {' '.join(shlex.quote(a) for a in argv)}")
    sys.stdout.flush()
    if dry_run:
        print("# (dry-run: not executed)")
        print("::endgroup::")
        return 0

    # Tee the child's output: to the artifact log AND to this job's stdout, so a
    # run can be watched live. Writing only to the file (the previous behaviour)
    # left the job log showing nothing but the command line for up to two hours,
    # which makes a running sweep impossible to follow and an ASAN hit invisible
    # until the artifact is downloaded.
    asan_hits = Counter()
    with log_path.open("wb") as logf:
        proc = subprocess.Popen(argv, cwd=str(tt_metal_dir), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        pending = 0
        for raw in proc.stdout:
            logf.write(raw)
            line = raw.decode("utf-8", "replace").rstrip("\n")
            m = ASAN_LINE_RE.search(line)
            if m:
                asan_hits[m.group(1).strip()] += 1
            sys.stdout.write(line + "\n")
            # Block-buffered on a pipe, so flush periodically (and immediately on a
            # violation) or "live" output arrives in multi-minute bursts.
            pending += 1
            if m is not None or pending >= 100:
                sys.stdout.flush()
                pending = 0
        proc.wait()
    sys.stdout.flush()
    print(f"rc={proc.returncode} (entry={rec['name']})")
    print("::endgroup::")
    # Printed OUTSIDE the group so the run reads as a scannable timeline even with
    # every entry collapsed: one line per entry saying whether it tripped anything.
    if asan_hits:
        detail = ", ".join(f"{c} x{n}" for c, n in asan_hits.most_common())
        print(f"::error::{rec['name']}: {sum(asan_hits.values())} ASAN hit(s) — {detail}")
    else:
        print(f"  {rec['name']}: no ASAN hits (rc={proc.returncode})")
    sys.stdout.flush()
    return proc.returncode


def entry_needs_retry(xml_path: Path) -> bool:
    """An entry needs a --forked retry if its no-forked run left no usable XML:
    a C-level crash/hang (or wallclock SIGTERM) kills the shared pytest process
    before it writes JUnit, so the XML is missing, malformed, or has 0
    testcases. A normal run with test *failures* still writes a full XML."""
    if not xml_path.is_file():
        return True
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return True
    suites = root.findall("testsuite") if root.tag != "testsuite" else [root]
    total = sum(len(s.findall("testcase")) for s in suites if s.tag == "testsuite")
    return total == 0


# -- Subcommands -------------------------------------------------------------

def cmd_expand(args):
    manifest = Path(args.manifest)
    entries, audit = expand_entries(manifest, args.arch)
    payload = {
        "arch": args.arch,
        "sku": ARCH_CONFIG[args.arch]["sku"],
        "manifest": str(manifest),
        "count": len(entries),
        "audit": audit,
        "entries": [
            {k: e[k] for k in ("name", "slug", "is_pytest", "wallclock",
                               "x_stripped", "coce_injected", "timeout_injected")}
            for e in entries
        ],
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not audit["ok"]:
        sys.stderr.write("AUDIT FAIL — see `audit.reasons` above\n")
        return 2
    print(f"\nAUDIT PASS — {len(entries)} entries for arch={args.arch} "
          f"(sku={payload['sku']})", file=sys.stderr)
    return 0


def cmd_run(args):
    manifest = Path(args.manifest)
    tt_metal_dir = Path(args.tt_metal_dir).resolve()
    build_dir = Path(args.build_dir).resolve() if args.build_dir else tt_metal_dir / "build_emule"
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pytest_bin = Path(args.pytest_bin)

    entries, audit = expand_entries(manifest, args.arch)
    if not audit["ok"] and not args.allow_audit_fail:
        sys.stderr.write("AUDIT FAIL — refusing to run (pass --allow-audit-fail to override):\n")
        sys.stderr.write(json.dumps(audit["reasons"], indent=2) + "\n")
        return 2

    # Optional SWEEP_ONLY filter (comma-separated slugs) — debug / targeted reruns.
    only = args.only or os.environ.get("SWEEP_ONLY", "")
    if only:
        wanted = {x.strip() for x in only.split(",") if x.strip()}
        entries = [e for e in entries if e["slug"] in wanted]
        if not entries:
            sys.stderr.write(f"ERROR: SWEEP_ONLY={only!r} matched no entries\n")
            return 2

    shard = select_shard(entries, args.shard_index, args.shard_count)
    if not shard:
        sys.stderr.write("ERROR: shard selected zero entries\n")
        return 2
    asan = args.asan or os.environ.get("SWEEP_ASAN") == "1"
    runtime_env = build_runtime_env(tt_metal_dir, build_dir, args.arch, pytest_bin, asan=asan)

    # Modes (see module docstring): hybrid (default) = no-forked then forked
    # retry of failed entries; --forked = fork everything; --no-retry = pass 1 only.
    #
    # ASAN implies forked-all. The sanitizers abort() on the FIRST violation, so
    # without per-test forking one finding ends the entry's process and hides
    # every later one in the same file. Forked, only the child dies: the parent
    # runs the remaining tests and still writes a complete JUnit XML, so one
    # entry can report many findings — and the pass-2 retry becomes moot.
    force_forked = args.forked or asan or os.environ.get("SWEEP_FORKED") == "1"
    do_retry = not (args.no_retry or os.environ.get("SWEEP_NO_RETRY") == "1") and not force_forked
    mode = "forked-all" if force_forked else ("hybrid" if do_retry else "no-forked")

    print(f"== post-commit sweep run ==")
    print(f"  arch:        {args.arch} (sku={ARCH_CONFIG[args.arch]['sku']})")
    print(f"  asan:        {'ON (TT_METAL_EMULE_ASAN=1)' if asan else 'off'}")
    print(f"  tt_metal:    {tt_metal_dir}")
    print(f"  build_dir:   {build_dir}")
    print(f"  out_dir:     {out_dir}")
    print(f"  shard:       {args.shard_index} of {args.shard_count}")
    print(f"  mode:        {mode}")
    print(f"  entries:     {len(shard)} of {len(entries)} selected")
    for e in shard:
        print(f"    - {e['slug']}")
    print()
    sys.stdout.flush()

    # Pass 1.
    print(f"--- pass 1 ({'--forked' if force_forked else 'no-forked'}) ---")
    for rec in shard:
        run_entry(rec, tt_metal_dir=tt_metal_dir, out_dir=out_dir,
                  runtime_env=runtime_env, pytest_bin=pytest_bin,
                  forked=force_forked, dry_run=args.dry_run, asan=asan)

    # Pass 2 — forked retry of entries that produced no usable XML.
    if do_retry and not args.dry_run:
        retry = [rec for rec in shard if entry_needs_retry(out_dir / f"{rec['slug']}.xml")]
        if retry:
            print(f"\n--- pass 2 (--forked retry of {len(retry)} entry/entries with no usable XML) ---")
            for rec in retry:
                print(f"  retrying: {rec['slug']}")
            for rec in retry:
                run_entry(rec, tt_metal_dir=tt_metal_dir, out_dir=out_dir,
                          runtime_env=runtime_env, pytest_bin=pytest_bin,
                          forked=True, dry_run=False, asan=asan)
        else:
            print("\n--- pass 2: no entries needed a forked retry ---")

    # A nonzero entry rc is EXPECTED (the suite is not 100% green) — the sweep is
    # report-only, so we never propagate test failures as the process exit code.
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common_manifest = dict(help="path to tt-metal tests/pipeline_reorg/ttnn_sanity_tests.yaml")

    pe = sub.add_parser("expand", help="resolve + audit the arch's post-commit entry list")
    pe.add_argument("--arch", required=True, choices=sorted(ARCH_CONFIG))
    pe.add_argument("--manifest", required=True, **common_manifest)
    pe.add_argument("--out", default=None, help="also write the JSON payload here")
    pe.set_defaults(func=cmd_expand)

    pr = sub.add_parser("run", help="execute this shard's entries under emule")
    pr.add_argument("--arch", required=True, choices=sorted(ARCH_CONFIG))
    pr.add_argument("--manifest", required=True, **common_manifest)
    pr.add_argument("--tt-metal-dir", required=True)
    pr.add_argument("--build-dir", default=None, help="default <tt-metal-dir>/build_emule")
    pr.add_argument("--out-dir", required=True, help="per-entry XML + log destination")
    pr.add_argument("--shard-index", type=int, default=1)
    pr.add_argument("--shard-count", type=int, default=1)
    pr.add_argument("--only", default=None,
                    help="comma-separated entry slugs to run (also via SWEEP_ONLY env)")
    pr.add_argument("--pytest-bin", default="/opt/ttmlir-toolchain/venv/bin/pytest")
    pr.add_argument("--forked", action="store_true",
                    help="force --forked on every entry, no retry (also via SWEEP_FORKED=1). "
                         "Default is the hybrid: no-forked pass 1 + --forked retry of failed entries.")
    pr.add_argument("--no-retry", action="store_true",
                    help="no-forked pass only, skip the --forked retry (also via SWEEP_NO_RETRY=1)")
    pr.add_argument("--asan", action="store_true",
                    help="arm the emule sanitizers (TT_METAL_EMULE_ASAN=1) and add the -s/-v "
                         "pytest flags their report needs to survive an abort(). Implies "
                         "--forked. Also via SWEEP_ASAN=1.")
    pr.add_argument("--dry-run", action="store_true", help="print commands, don't execute")
    pr.add_argument("--allow-audit-fail", action="store_true")
    pr.set_defaults(func=cmd_run)

    args = ap.parse_args()
    if args.cmd == "run":
        if args.shard_count < 1 or not (1 <= args.shard_index <= args.shard_count):
            ap.error(f"--shard-index must be in [1, {args.shard_count}]")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
