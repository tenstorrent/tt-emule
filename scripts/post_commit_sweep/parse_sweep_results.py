#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Parse + classify + report the post-commit sweep results.

Reads the per-entry JUnit XML + logs produced by `sweep.py run` (across all
shards, merged into one directory) and emits:

  --out-exec     exec.md   — leadership-readable headline + top gaps
  --out-dev      dev.md    — full per-file / per-class detail for triage
  --out-headline headline.json — machine-readable headline (for trend/alerting)
  --out-summary  summary.md — compact block for the GitHub Actions job summary

The blocker-class taxonomy + per-testcase classification are ported from the
original sweep parser on `arminale/bh-ci-sweep` (`snapshots/bh_sanity/
parse_and_report.py`); the IO/labels here are arch-parameterized so the same
parser serves blackhole, wormhole, and any future lane.
"""

import argparse
import collections
import datetime
import glob
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def slugify(name: str) -> str:
    """Kebab-case a manifest entry name. MUST match sweep.py's slug()."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# -- Truncation detection (entries whose XML never landed) -------------------

def detect_truncated_entries(xml_dir: str, expected_slugs):
    """Flag entries that ran but produced no usable XML (wallclock SIGTERM,
    malformed partial write, or zero testcases). expected_slugs: {slug: name}."""
    p = Path(xml_dir)
    truncated = []
    for slug, name in expected_slugs.items():
        xml_path = p / f"{slug}.xml"
        log_path = p / f"{slug}.log"
        rec = {"name": name, "slug": slug, "xml_path": str(xml_path),
               "log_path": str(log_path), "reason": None}
        if not xml_path.is_file():
            rec["reason"] = "XML missing — likely wallclock SIGTERM mid-run"
            truncated.append(rec)
            continue
        try:
            root = ET.parse(xml_path).getroot()
            suites = root.findall("testsuite") if root.tag != "testsuite" else [root]
            total = sum(len(s.findall("testcase")) for s in suites if s.tag == "testsuite")
            if total == 0:
                rec["reason"] = "XML has zero testcases — ran but recorded none"
                truncated.append(rec)
        except ET.ParseError as exc:
            rec["reason"] = f"XML malformed — partial write before SIGTERM ({exc})"
            truncated.append(rec)
    return truncated


# -- Parser ------------------------------------------------------------------

COLLECTION_ERROR_MARKERS = ("collection failure", "ImportError", "ModuleNotFoundError")
MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")

# Blocker-class taxonomy. Order matters — first match wins.
BLOCKER_CLASS_PATTERNS = [
    (r"signal 11|signal 6|SIGSEGV|SIGABRT|CRASHED", "Setup crash"),
    (r"jit_compile_kernel.*failed|use of undeclared identifier|"
     r"compiler failed \(exit", "Missing op / unjitted symbol"),
    (r"Need at least \d+ devices|Need at least \d+ cores|requires_grid_size",
     "Multi-host / grid gate"),
    (r"AttributeError|ModuleNotFoundError|undefined symbol|TypeError.*has no attribute",
     "Mock-API drift"),
    (r"PCC|ATOL|assert.*tensor|allclose|expected.*got", "Numeric drift (PCC/ATOL)"),
]


def assign_blocker_class(text):
    if not text:
        return "Unclassified", None
    for pat, label in BLOCKER_CLASS_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            return label, m.group(0)
    return "Unclassified", None


def is_collection_error(err_element) -> bool:
    if err_element is None:
        return False
    msg = (err_element.get("message", "") or "").lower()
    text = err_element.text or ""
    if "collection failure" in msg:
        return True
    return any(marker in text for marker in COLLECTION_ERROR_MARKERS)


def _new_counters():
    return {
        "P": 0, "F": 0, "E": 0, "S": 0, "X": 0,
        "fail_sample": None, "error_sample": None,
        "skip_samples": [], "xfail_samples": [],
        "fail_class_counts": collections.Counter(),
    }


def parse_xml(path: str):
    """Per-file counters + sample messages from one JUnit XML.
    Collection errors are stashed under '__excluded_files__'."""
    root = ET.parse(path).getroot()
    suites = root.findall("testsuite") if root.tag != "testsuite" else [root]
    by_file = collections.defaultdict(_new_counters)
    excluded_files = []

    for ts in suites:
        if ts.tag != "testsuite":
            continue
        for tc in ts.findall("testcase"):
            cls = tc.get("classname", "")
            name = tc.get("name", "")
            parts = cls.split(".")
            leaf = cls
            for j in range(len(parts) - 1, -1, -1):
                if parts[j].startswith("test_"):
                    leaf = "/".join(parts[: j + 1]) + ".py"
                    break

            fail = tc.find("failure")
            err = tc.find("error")
            skip = tc.find("skipped")

            if err is not None and is_collection_error(err):
                file_path = (name.replace(".", "/") + ".py") if not cls else leaf
                m = MISSING_MODULE_RE.search(err.text or "")
                excluded_files.append({
                    "file": file_path,
                    "missing_module": m.group(1) if m else "(see log)",
                    "message": (err.get("message", "") or "")[:200],
                })
                continue

            if fail is not None:
                by_file[leaf]["F"] += 1
                msg_full = (fail.get("message", "") or "") + " " + (fail.text or "")
                klass, _ = assign_blocker_class(msg_full)
                by_file[leaf]["fail_class_counts"][klass] += 1
                if by_file[leaf]["fail_sample"] is None:
                    by_file[leaf]["fail_sample"] = (
                        name, fail.get("message", "") or "", (fail.text or "").strip()[:500])
            elif err is not None:
                by_file[leaf]["E"] += 1
                if by_file[leaf]["error_sample"] is None:
                    by_file[leaf]["error_sample"] = (
                        name, err.get("message", "") or "", (err.text or "").strip()[:500])
            elif skip is not None:
                msg = skip.get("message", "") or ""
                if skip.get("type", "") == "pytest.xfail":
                    by_file[leaf]["X"] += 1
                    by_file[leaf]["xfail_samples"].append((name, msg))
                else:
                    by_file[leaf]["S"] += 1
                    by_file[leaf]["skip_samples"].append((name, msg))
            else:
                by_file[leaf]["P"] += 1

    by_file["__excluded_files__"] = excluded_files  # type: ignore[assignment]
    return by_file


def parse_xml_dir(xml_path: str):
    """Merge per-file counters across every XML in a directory (one per entry).

    Also merges any <slug>.excluded.json sidecars written by sweep.py's
    collect-only pre-filter (import-error files it had to --ignore so the
    pytest-9.0.3 collection bug wouldn't abort the whole entry)."""
    p = Path(xml_path)
    files = [str(p)] if p.is_file() else sorted(glob.glob(f"{xml_path}/*.xml"))
    merged = collections.defaultdict(_new_counters)
    all_excluded, excluded_by_entry = [], {}
    for f in files:
        try:
            by_file = parse_xml(f)
        except ET.ParseError:
            # A malformed shard XML (partial write before SIGTERM) shouldn't sink
            # the whole report; detect_truncated_entries flags it separately.
            continue
        entry_slug = Path(f).stem
        entry_excluded = by_file.pop("__excluded_files__", [])
        # Merge the pre-filter sidecar (import-error files --ignore'd before run).
        sidecar = Path(f).with_name(f"{entry_slug}.excluded.json")
        if sidecar.is_file():
            try:
                entry_excluded = list(entry_excluded) + json.loads(sidecar.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        if entry_excluded:
            # De-dup by file within the entry.
            seen, deduped = set(), []
            for ex in entry_excluded:
                if ex["file"] not in seen:
                    seen.add(ex["file"])
                    deduped.append(ex)
            excluded_by_entry[entry_slug] = deduped
            all_excluded.extend({**ex, "entry_slug": entry_slug} for ex in deduped)
        for leaf, c in by_file.items():
            slot = merged[leaf]
            for k in ("P", "F", "E", "S", "X"):
                slot[k] += c[k]
            slot["fail_sample"] = slot["fail_sample"] or c["fail_sample"]
            slot["error_sample"] = slot["error_sample"] or c["error_sample"]
            slot["skip_samples"].extend(c["skip_samples"])
            slot["xfail_samples"].extend(c["xfail_samples"])
            slot["fail_class_counts"].update(c["fail_class_counts"])
    return dict(merged), files, all_excluded, excluded_by_entry


# -- Classifier + skip categorisation ---------------------------------------

def bucket(c):
    P, F, E = c["P"], c["F"], c["E"]
    non_skip = P + F + E
    if non_skip == 0:
        return "ALL_SKIP"
    if P == non_skip:
        return "ALL_PASS"
    if P == 0:
        return "ALL_FAIL"
    return "PARTIAL"


SKIP_REASON_PATTERNS = [
    (r"Requested more devices than available", "Multi-device (>1 devices required)"),
    (r"Need at least \d+ devices", "Multi-device (>1 devices required)"),
    (r"Need at least \d+ cores", "Grid-size gate (more cores required)"),
    (r"requires_grid_size", "Explicit grid-size marker"),
    (r"skip_for_blackhole|not.*supported on blackhole", "Skipped for Blackhole"),
    (r"skip_for_wormhole|not.*supported on wormhole", "Skipped for Wormhole"),
    (r"slow dispatch", "Slow-dispatch gate"),
    (r"front padding", "Front-padding not supported (ttnn.pad)"),
    (r"bfloat8_b.*row major|row major.*bfloat8_b", "BFLOAT8_B + row-major layout unsupported"),
    (r"tile layout", "Tile-layout shape restriction"),
    (r"repeat dim must be", "torch.repeat dim restriction"),
    (r"illegal shard config", "Illegal shard config (open issue)"),
    (r"#\d{4,5}", "Open ticket marker"),
    (r"WH_EPSILON", "WH-specific epsilon"),
]


def categorise_skip(msg):
    for pat, label in SKIP_REASON_PATTERNS:
        if re.search(pat, msg, re.I):
            return label
    return "Misc / per-parametrization"


# -- Headline metrics --------------------------------------------------------

def headline_metrics(by_file, excluded_files=()):
    P = sum(c["P"] for c in by_file.values())
    F = sum(c["F"] for c in by_file.values())
    E = sum(c["E"] for c in by_file.values())
    S = sum(c["S"] for c in by_file.values())
    X = sum(c["X"] for c in by_file.values())
    C = len(excluded_files)
    total = P + F + E + S + X
    executed = total - S - X
    dominant_crash = max((c["E"] for c in by_file.values()), default=0)

    def pct(n, d):
        return (n / d * 100.0) if d else 0.0

    buckets = collections.Counter(bucket(c) for c in by_file.values())
    files_total = sum(buckets.values())
    testable_files = files_total - buckets["ALL_SKIP"]

    return {
        "P": P, "F": F, "E": E, "S": S, "X": X, "C": C,
        "total": total, "executed": executed, "dominant_crash": dominant_crash,
        "pass_of_total": pct(P, total),
        "pass_of_executed": pct(P, executed),
        "pass_of_executed_ex_crash": pct(P, executed - dominant_crash) if executed > dominant_crash else 0.0,
        "files_all_pass": buckets["ALL_PASS"],
        "files_partial": buckets["PARTIAL"],
        "files_all_fail": buckets["ALL_FAIL"],
        "files_all_skip": buckets["ALL_SKIP"],
        "files_total": files_total,
        "files_pass_of_total": pct(buckets["ALL_PASS"], files_total),
        "files_pass_of_testable": pct(buckets["ALL_PASS"], testable_files),
    }


def blocker_class_totals(by_file):
    """Per-class failing-testcase totals (per-testcase, not per-file-sample)."""
    totals = collections.Counter()
    for c in by_file.values():
        totals.update(c.get("fail_class_counts") or {})
    return totals


def fmt_pct(x):
    return f"{x:.1f}%"


# -- Derived narrative -------------------------------------------------------

def derive_top_gaps(by_file, truncated, excluded_files):
    class_to_count = blocker_class_totals(by_file)
    class_to_sample_files = collections.defaultdict(list)
    for leaf, c in by_file.items():
        if c["F"] == 0:
            continue
        cc = c.get("fail_class_counts") or collections.Counter()
        if cc:
            dom = cc.most_common(1)[0][0]
            if len(class_to_sample_files[dom]) < 3:
                class_to_sample_files[dom].append((leaf, c["F"]))

    gaps = []
    for cls, n in class_to_count.most_common(3):
        files = class_to_sample_files[cls]
        sample = ", ".join(f"`{Path(f).name}` ({fc})" for f, fc in files[:2])
        gaps.append((f"Class **{cls}** — {n} failing testcases", f"Top affected files: {sample}"))
    if truncated:
        names = ", ".join(t["name"] for t in truncated[:3])
        gaps.append((f"**{len(truncated)} entries unmeasured** — hit the per-entry wallclock, no XML",
                     f"Affected: {names}{' …' if len(truncated) > 3 else ''}"))
    if excluded_files and len(excluded_files) >= 10:
        by_mod = collections.Counter(e["missing_module"] for e in excluded_files)
        mods = ", ".join(f"`{m}` ({n})" for m, n in by_mod.most_common(2))
        gaps.append((f"**{len(excluded_files)} files excluded** — venv lacks test deps",
                     f"Top missing modules: {mods}"))
    return gaps[:3]


def derive_top_next_steps(gaps, truncated):
    fix_class_map = {
        "Setup crash": ("triage the SIGABRT/segfault root cause; isolate via --forked", "~1–2 days"),
        "Numeric drift (PCC/ATOL)": ("audit the mock for the dominant op family; align with silicon's compute path (see /memory-debug)", "~1–2 weeks per op family"),
        "Missing op / unjitted symbol": ("add the missing op shim (see /compute-llk-bringup)", "~1 day per op"),
        "Multi-host / grid gate": ("not in scope — single-device emule cannot satisfy", "n/a"),
        "Mock-API drift": ("rebase mock against current tt-metal pin (see /uplift)", "~1 day"),
        "Unclassified": ("triage per-file; classify into the taxonomy then prioritise", "~few hours per file"),
    }
    steps, seen = [], set()
    for title, _ in gaps:
        for cls, (action, effort) in fix_class_map.items():
            if cls in title and (cls, action) not in seen:
                seen.add((cls, action))
                steps.append(f"**{cls}**: {action} ({effort})")
                break
    if truncated:
        steps.append(f"**Wallclock-truncated entries**: re-run the {len(truncated)} truncated "
                     "entries with extended budget; isolate crashes via --forked")
    return steps[:3]


KERNEL_FAIL_RE = re.compile(r"compiler failed \(exit \d+\) for kernel: (\S+)")
CPP_ERROR_RE = re.compile(r":\d+:\d+:\s+(?:fatal\s+)?error:\s+([^\n]{5,180})")


def extract_compile_errors(log_dir, top_n=5):
    """Scan the per-entry .log files for JIT compile failures and cluster them.

    Two signals (the sweep writes one log per manifest entry):
      1. The raw C++ compiler line `<file>:<line>:<col>: error: <msg>` when the
         toolchain's stderr is captured — clustered by normalized message.
      2. Otherwise the `jit_compile_kernel: compiler failed ... for kernel:
         <path>` RuntimeError emule raises — clustered by kernel basename. (In
         emule the underlying compiler stderr is often not in the pytest log, so
         this kernel-path clustering is the actionable root-cause signal.)

    Returns the top-N (label, count) pairs."""
    err_counts = collections.Counter()
    for lp in sorted(glob.glob(f"{log_dir}/*.log")):
        try:
            text = Path(lp).read_text(errors="ignore")
        except OSError:
            continue
        for m in CPP_ERROR_RE.finditer(text):
            e = m.group(1)
            e = re.sub(r"0x[0-9a-fA-F]+", "ADDR", e)
            e = re.sub(r"\b\d+\b", "N", e)
            e = re.split(r"\s+(?:in|when|with|note)\s+", e)[0]
            e = e.split(";")[0][:160].strip()
            err_counts[e] += 1
        # Kernel-path clustering — count distinct (kernel) per failing test so a
        # widely-shared broken kernel rises to the top.
        for m in KERNEL_FAIL_RE.finditer(text):
            kernel = m.group(1).split("/")[-1]
            err_counts[f"compiler failed for kernel: {kernel}"] += 1
    return err_counts.most_common(top_n)


def derive_recent_landmarks(tt_emule_dir, n=5):
    try:
        r = subprocess.run(["git", "-C", str(tt_emule_dir), "log", "-n", str(n), "--format=%h %s"],
                           capture_output=True, text=True, timeout=10)
        return [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
    except Exception:
        return []


# -- Report writers ----------------------------------------------------------

def _scope_line(arch, variant):
    return (f"**Scope:** {arch} post-commit ttnn lane under the tt-emule backend "
            f"({variant}); one invocation per manifest entry, sharded.")


def write_headline_json(out_path, *, arch, variant, sku, pin_sha, metrics,
                        by_file, truncated, when):
    cls_totals = blocker_class_totals(by_file)
    payload = {
        "generated_utc": when,
        "arch": arch, "variant": variant, "sku": sku,
        "tt_metal_pin": pin_sha,
        "tests": {
            "collected": metrics["total"],
            "executed": metrics["executed"],
            "passing": metrics["P"],
            "failing": metrics["F"],
            "errors": metrics["E"],
            "skipped": metrics["S"],
            "xfail": metrics["X"],
            "excluded_files": metrics["C"],
        },
        "pass_rate": {
            "of_executed_pct": round(metrics["pass_of_executed"], 2),
            "of_total_pct": round(metrics["pass_of_total"], 2),
        },
        "files": {
            "fully_passing": metrics["files_all_pass"],
            "partial": metrics["files_partial"],
            "all_failing": metrics["files_all_fail"],
            "all_skip": metrics["files_all_skip"],
            "total": metrics["files_total"],
        },
        "blocker_classes": dict(cls_totals),
        "truncated_entries": [t["name"] for t in truncated],
    }
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def write_summary(out_path, *, arch, variant, sku, pin_sha, metrics, by_file,
                  truncated, when):
    """Compact block for $GITHUB_STEP_SUMMARY."""
    gaps = derive_top_gaps(by_file, truncated, ())
    lines = [
        f"## tt-emule post-commit sweep — {arch} ({variant})",
        "",
        f"- **Pass rate (of executed): {fmt_pct(metrics['pass_of_executed'])}** "
        f"— {metrics['P']:,} / {metrics['executed']:,}",
        f"- tt-metal pin: `{pin_sha}` · sku: `{sku}` · {when}",
        "",
        "| Collected | Executed | Pass | Fail | Error | Skip | Xfail | Excluded files |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {metrics['total']:,} | {metrics['executed']:,} | {metrics['P']:,} | "
        f"{metrics['F']:,} | {metrics['E']:,} | {metrics['S']:,} | {metrics['X']:,} | {metrics['C']:,} |",
        "",
        f"Files fully passing: **{metrics['files_all_pass']} / {metrics['files_total']}** "
        f"({fmt_pct(metrics['files_pass_of_total'])})",
        "",
    ]
    if truncated:
        lines.append(f"⚠️ **{len(truncated)} entries truncated** (no/empty XML): "
                     + ", ".join(t["name"] for t in truncated))
        lines.append("")
    if gaps:
        lines.append("**Top gaps:**")
        for i, (title, detail) in enumerate(gaps, 1):
            lines.append(f"{i}. {title} — {detail}")
        lines.append("")
    Path(out_path).write_text("\n".join(lines) + "\n")


def write_exec(out_path, *, arch, suite_name, variant, metrics, by_file,
               truncated, excluded_files, tt_emule_dir, when):
    lines = [
        f"# {suite_name} status (exec)",
        _scope_line(arch, variant),
        f"**Snapshot:** {when}.",
        "",
        "## Headline",
        "",
        f"| Metric | {variant} |",
        "|---|---:|",
        f"| **Total tests collected** | **{metrics['total']:,}** |",
        f"| **Total tests executed** (collected − skipped − xfail) | **{metrics['executed']:,}** |",
        f"| Tests passing | {metrics['P']:,} |",
        f"| Passing — share of total collected | {fmt_pct(metrics['pass_of_total'])} |",
        f"| Passing — share of executed | {fmt_pct(metrics['pass_of_executed'])} |",
        f"| Passing — share of executed (excl. setup-crash inflation) | {fmt_pct(metrics['pass_of_executed_ex_crash'])} |",
        f"| Test files fully passing | {metrics['files_all_pass']} / {metrics['files_total']} |",
        f"| Test files fully passing — share of total | {fmt_pct(metrics['files_pass_of_total'])} |",
        f"| Test files fully passing — share of testable | {fmt_pct(metrics['files_pass_of_testable'])} |",
        "",
        f"Single-variant snapshot ({variant}); cross-variant comparison out of scope.",
        "",
        "## Status",
        "",
        "| Bucket | Count | Notes |",
        "|---|---:|---|",
        f"| Passing | {metrics['P']} | sub-tests that returned PASS in the JUnit XML |",
        f"| In-scope failures | {metrics['F']} | numeric / functional failures (PCC/ATOL or assertion) |",
        f"| Setup crashes | {metrics['E']} | E bucket; Class **Setup crash** in the taxonomy |",
        f"| Skipped by design | {metrics['S']} | gates emule can't satisfy (multi-device, grid-size, …) |",
        f"| Xfail / xpass | {metrics['X']} | silicon-side `@pytest.mark.xfail` markers; tracked separately |",
        f"| **Test files excluded (import error)** | **{metrics['C']}** | files that couldn't import (missing venv deps). Per-FILE; **not in the % denominator**. |",
        f"| **Truncated entries** | **{metrics.get('truncated_count', 0)}** | entries with missing/empty XML — likely hit the wallclock backstop. **Not in the headline %.** |",
        "",
    ]

    if excluded_files:
        by_module = collections.Counter(e["missing_module"] for e in excluded_files)
        lines += ["## Test files excluded due to import errors", "",
                  f"{len(excluded_files)} files could not be imported (env gap, not an emule "
                  "failure). Grouped by missing module:", "",
                  "| Missing module | Files affected |", "|---|---:|"]
        for mod, n in by_module.most_common():
            lines.append(f"| `{mod}` | {n} |")
        lines.append("")

    if truncated:
        lines += ["## Truncated entries (wallclock-truncated, missing data)", "",
                  "These entries hit their per-entry wallclock backstop before pytest wrote its "
                  "JUnit XML; they contribute **0** to every count above.", "",
                  "| Entry | Reason | Log path |", "|---|---|---|"]
        for t in truncated:
            log_ref = f"`{t['log_path']}`" if Path(t['log_path']).is_file() else "(no log)"
            lines.append(f"| {t['name']} | {t['reason']} | {log_ref} |")
        lines.append("")

    skip_groups = collections.Counter()
    for c in by_file.values():
        for _, msg in c["skip_samples"]:
            skip_groups[categorise_skip(msg)] += 1
    lines += ["## What \"skipped by design\" covers", ""]
    if not skip_groups:
        lines.append("(no skipped tests recorded)")
    else:
        lines += ["| Reason | Count |", "|---|---:|"]
        for label, n in skip_groups.most_common():
            lines.append(f"| {label} | {n} |")
    lines.append("")

    gaps = derive_top_gaps(by_file, truncated, excluded_files)
    lines += ["## Top 3 gaps", ""]
    if not gaps:
        lines.append("(no failures of note — every file fully passed)")
    else:
        for i, (title, detail) in enumerate(gaps, 1):
            lines += [f"{i}. {title}", f"   {detail}"]
    lines.append("")

    steps = derive_top_next_steps(gaps, truncated)
    lines += ["## Top 3 next steps", ""]
    if not steps:
        lines.append("(no failures to address)")
    else:
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. {step}")
    lines.append("")

    landmarks = derive_recent_landmarks(tt_emule_dir)
    lines += ["## Recent landmarks", ""]
    lines += ([f"- `{ln}`" for ln in landmarks] if landmarks else ["(git log unavailable)"])
    lines += ["", "---", f"*Generated by scripts/post_commit_sweep/parse_sweep_results.py at {when}.*"]
    Path(out_path).write_text("\n".join(lines) + "\n")


def write_dev(out_path, *, arch, suite_name, variant, metrics, by_file, manifest,
              runner, xml_files, log_paths_seen, truncated,
              expected_slugs, excluded_files, excluded_by_entry, when):
    lines = [
        f"# {suite_name} status (dev)",
        _scope_line(arch, variant),
        f"**Snapshot:** {when}.",
        "",
        "## Headline",
        "",
        f"Variant **{variant}** — raw counts: P={metrics['P']:,}  F={metrics['F']:,}  "
        f"E={metrics['E']:,}  S={metrics['S']:,}  X={metrics['X']:,} (xfail)  "
        f"truncated_entries={metrics.get('truncated_count', 0)} of {len(expected_slugs)} expected",
        "",
        f"- **Total tests collected**: {metrics['total']:,}",
        f"- **Total tests executed** (collected − S − X): {metrics['executed']:,}",
        "",
        "Pass-rate denominators:",
        f"- of total: {fmt_pct(metrics['pass_of_total'])} ({metrics['P']:,} / {metrics['total']:,})",
        f"- of executed: {fmt_pct(metrics['pass_of_executed'])} ({metrics['P']:,} / {metrics['executed']:,})",
        f"- of executed excl. dominant crash: {fmt_pct(metrics['pass_of_executed_ex_crash'])}",
        "",
    ]

    rows = [(leaf, c, bucket(c)) for leaf, c in by_file.items()]
    rows.sort(key=lambda r: (r[2], r[0]))

    def tbl(title, predicate, with_sample=False):
        sub = [(l, c, b) for (l, c, b) in rows if predicate(b)]
        out = [f"## {title}", ""]
        if not sub:
            return out + ["(none)", ""]
        if with_sample:
            out += ["| File | P / F / E / S | Class | Sample |", "|---|---|---|---|"]
            for leaf, c, _ in sub:
                sample_text, klass = "", "Unclassified"
                src = c["fail_sample"] or c["error_sample"]
                if src:
                    _, msg, text = src
                    sample_text = (msg or text)[:140]
                    klass, _ = assign_blocker_class(msg or text)
                sample_text = sample_text.replace("|", "\\|").replace("\n", " ")
                out.append(f"| `{leaf}` | {c['P']} / {c['F']} / {c['E']} / {c['S']} | {klass} | {sample_text} |")
        else:
            out += ["| File | P / F / E / S |", "|---|---|"]
            for leaf, c, _ in sub:
                out.append(f"| `{leaf}` | {c['P']} / {c['F']} / {c['E']} / {c['S']} |")
        return out + [""]

    lines += tbl("Fully-passing files", lambda b: b == "ALL_PASS")
    lines += tbl("Partial-pass files", lambda b: b == "PARTIAL", with_sample=True)
    lines += tbl("All-failing files", lambda b: b == "ALL_FAIL", with_sample=True)
    lines += tbl("All-skip files", lambda b: b == "ALL_SKIP")

    # Blocker taxonomy — per-testcase totals.
    cls_totals = blocker_class_totals(by_file)
    cls_samples = collections.defaultdict(list)
    for leaf, c in by_file.items():
        src = c.get("fail_sample") or c.get("error_sample")
        if not src:
            continue
        name, msg, text = src
        klass, _ = assign_blocker_class(msg or text)
        if len(cls_samples[klass]) < 1:
            cls_samples[klass].append((leaf, name, (msg or text)[:200]))
    lines += ["## Blocker taxonomy (per-testcase totals)", "",
              "| Class | Failing testcases | Sample |", "|---|---:|---|"]
    skill_classes = ["Setup crash", "Numeric drift (PCC/ATOL)", "Missing op / unjitted symbol",
                     "Multi-host / grid gate", "Mock-API drift", "Unclassified"]
    for cls in skill_classes:
        n = cls_totals.get(cls, 0)
        if n and cls_samples.get(cls):
            leaf, name, sample = cls_samples[cls][0]
            sample = sample.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {cls} | {n} | `{leaf}::{name}` — {sample} |")
        else:
            lines.append(f"| {cls} | {n} | — |")
    lines.append("")

    # JIT compile-error root causes.
    log_dir = log_paths_seen[0].rsplit("/", 1)[0] if log_paths_seen else None
    top_errs = extract_compile_errors(log_dir, top_n=5) if log_dir else []
    lines += ["## JIT compile-error root causes (top 5)", ""]
    if not top_errs:
        lines.append("(no jit_compile_kernel failures in this snapshot)")
    else:
        lines += ["Behind the **Missing op / unjitted symbol** bucket, the C++ compile errors "
                  "cluster into a few root causes — each unblocks many tests if fixed in "
                  "`include/jit_hw/`.", "", "| Occurrences | Compile error |", "|---:|---|"]
        for err, n in top_errs:
            lines.append(f"| **{n:,}** | `{err.replace('|', chr(92) + '|')}` |")
        total = sum(n for _, n in top_errs)
        jit_tc = cls_totals.get("Missing op / unjitted symbol", 0)
        lines += ["", f"These top-5 account for **{total:,} compile-error occurrences** across the "
                  f"JIT-build logs (one failing testcase can emit several as an include cascades). "
                  f"They drive the {jit_tc:,} testcases in the 'Missing op / unjitted symbol' bucket."]
    lines.append("")

    # Top files by numeric drift.
    pcc_files = sorted(
        ((leaf, (c.get("fail_class_counts") or {}).get("Numeric drift (PCC/ATOL)", 0))
         for leaf, c in by_file.items()),
        key=lambda r: -r[1])
    pcc_files = [(l, n) for l, n in pcc_files if n > 0]
    lines += ["## Top 5 files by Numeric drift (PCC/ATOL) failures", ""]
    if not pcc_files:
        lines.append("(no PCC/ATOL failures in this snapshot)")
    else:
        lines += ["| File | PCC/ATOL failures |", "|---|---:|"]
        for leaf, n in pcc_files[:5]:
            lines.append(f"| `{leaf}` | **{n:,}** |")
    lines.append("")

    # Excluded files.
    lines += ["## Excluded test files (collection errors)", ""]
    if not excluded_files:
        lines.append("(no test files were excluded)")
    else:
        lines.append(f"**{len(excluded_files)} files** could not be imported and did NOT run.")
        lines.append("")
        for slug, files in sorted((excluded_by_entry or {}).items()):
            lines += [f"### Entry: `{slug}` — {len(files)} excluded files", "",
                      "| File | Missing module |", "|---|---|"]
            for ex in files:
                lines.append(f"| `{ex['file'].replace('|', chr(92)+'|')}` | "
                             f"`{ex['missing_module'].replace('|', chr(92)+'|')}` |")
            lines.append("")
    lines.append("")

    # Truncated entries.
    lines += ["## Truncated entries (wallclock-truncated)", ""]
    if not expected_slugs:
        lines.append("Truncation check not performed (no expected entry list supplied).")
    elif not truncated:
        lines.append(f"**OK.** All {len(expected_slugs)} expected entries produced a non-empty XML.")
    else:
        lines += [f"**{len(truncated)} of {len(expected_slugs)} entries truncated.**", "",
                  "| Entry | Slug | Reason | XML | Log |", "|---|---|---|---|---|"]
        for t in truncated:
            xml_ref = f"`{t['xml_path']}`" if Path(t['xml_path']).is_file() else "(missing)"
            log_ref = f"`{t['log_path']}`" if Path(t['log_path']).is_file() else "(missing)"
            lines.append(f"| {t['name']} | `{t['slug']}` | {t['reason']} | {xml_ref} | {log_ref} |")
    lines.append("")

    lines += ["## References", "",
              f"- **Source manifest:** `{manifest}`",
              f"- **Runner:** `{runner}`",
              f"- **XML files parsed ({len(xml_files)}):**"]
    lines += [f"  - `{f}`" for f in xml_files]
    lines += ["", f"*Generated by scripts/post_commit_sweep/parse_sweep_results.py at {when}.*"]
    Path(out_path).write_text("\n".join(lines) + "\n")


# -- Driver -----------------------------------------------------------------

def load_expected_slugs(path):
    """Read the expand JSON payload → {slug: name} of expected entries."""
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    return {e["slug"]: e["name"] for e in data.get("entries", [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml-dir", required=True)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--out-exec", required=True)
    ap.add_argument("--out-dev", required=True)
    ap.add_argument("--out-headline", default=None)
    ap.add_argument("--out-summary", default=None)
    ap.add_argument("--arch", default="blackhole")
    ap.add_argument("--variant-label", default=None, help="default <arch>_emule")
    ap.add_argument("--sku", default="")
    ap.add_argument("--pin-sha", default="unknown")
    ap.add_argument("--suite-name", default=None, help="default <arch>_post_commit")
    ap.add_argument("--manifest", default="(unknown)")
    ap.add_argument("--runner", default="scripts/post_commit_sweep/sweep.py")
    ap.add_argument("--expected-json", default=None,
                    help="expand-subcommand JSON; enables truncation detection")
    ap.add_argument("--tt-emule-dir", default=".")
    args = ap.parse_args()

    variant = args.variant_label or f"{args.arch}_emule"
    suite_name = args.suite_name or f"{args.arch}_post_commit"
    when = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    by_file, xml_files, all_excluded, excluded_by_entry = parse_xml_dir(args.xml_dir)
    log_paths_seen = sorted(glob.glob(f"{args.log_dir}/*.log")) if args.log_dir else []

    expected_slugs = load_expected_slugs(args.expected_json)
    truncated = detect_truncated_entries(args.xml_dir, expected_slugs) if expected_slugs else []

    if not by_file and not truncated and not all_excluded:
        sys.stderr.write("ERROR: no testcases, no excluded files, no truncated entries — nothing to report\n")
        return 2

    metrics = headline_metrics(by_file, excluded_files=all_excluded)
    metrics["truncated_count"] = len(truncated)

    write_exec(args.out_exec, arch=args.arch, suite_name=suite_name, variant=variant,
               metrics=metrics, by_file=by_file,
               truncated=truncated, excluded_files=all_excluded,
               tt_emule_dir=args.tt_emule_dir, when=when)
    write_dev(args.out_dev, arch=args.arch, suite_name=suite_name, variant=variant,
              metrics=metrics, by_file=by_file, manifest=args.manifest, runner=args.runner,
              xml_files=xml_files, log_paths_seen=log_paths_seen,
              truncated=truncated, expected_slugs=expected_slugs,
              excluded_files=all_excluded, excluded_by_entry=excluded_by_entry, when=when)
    if args.out_headline:
        write_headline_json(args.out_headline, arch=args.arch, variant=variant, sku=args.sku,
                            pin_sha=args.pin_sha, metrics=metrics, by_file=by_file,
                            truncated=truncated, when=when)
    if args.out_summary:
        write_summary(args.out_summary, arch=args.arch, variant=variant, sku=args.sku,
                      pin_sha=args.pin_sha, metrics=metrics, by_file=by_file,
                      truncated=truncated, when=when)

    print(f"wrote {args.out_exec}")
    print(f"wrote {args.out_dev}")
    if args.out_headline:
        print(f"wrote {args.out_headline}")
    if args.out_summary:
        print(f"wrote {args.out_summary}")
    print(f"files: {len(by_file)}  P={metrics['P']} F={metrics['F']} E={metrics['E']} "
          f"S={metrics['S']} X={metrics['X']} C={metrics['C']}  "
          f"pass_of_executed={fmt_pct(metrics['pass_of_executed'])}  truncated={len(truncated)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
