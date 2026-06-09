#!/usr/bin/env python3
# DRY-RUN parse + classify + headline + report writer for /status-snapshot
# Phases 5+6+7. Produces throwaway exec + dev reports with a DRY RUN banner.
#
# Inputs:
#   --xml-dir       directory containing per-entry junit XMLs (or a single file)
#   --log-dir       directory containing matching .log files (same basename as xml)
#   --out-exec      exec report path
#   --out-dev       dev report path
#   --suite-name    suite identifier for filename / titles
#   --audit-log     path to the upstream audit_*.log (referenced in dev report)
#   --runner        path to the audited runner script (referenced in dev report)
#   --manifest      path to the source manifest (referenced in dev report)
#   --variant-label single-variant label (BH dryrun uses "bh_emule")
#   --baseline      "silicon-passing" | "none" (default "silicon-passing")

import argparse
import collections
import datetime
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def slugify(name: str) -> str:
    """Kebab-case a manifest entry name. Must match generate.py's slug()."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def parse_expected_entries(audit_log_path: str):
    """Extract the list of selected manifest entries from an audit log.

    Returns a list of {name, slug} dicts in the order they appear.
    """
    text = Path(audit_log_path).read_text()
    entries = []
    for line in text.splitlines():
        m = re.match(r"^--- (.+?) \[(PASS|FAIL)\] ---$", line)
        if m:
            name = m.group(1)
            entries.append({"name": name, "slug": slugify(name)})
    return entries


def detect_truncated_entries(xml_dir: str, expected_entries):
    """Compare expected manifest entries to the XML files actually produced.

    Returns a list of {name, slug, reason, xml_path, log_path} dicts for
    entries that look truncated. Reasons:
      * 'XML missing'      — no XML file at all (wallclock SIGTERM mid-run)
      * 'XML malformed'    — XML exists but parser raises
      * 'XML empty'        — XML parses but has zero testcases

    Returns [] for the path-mode dryrun case where xml_dir is a single file.
    """
    p = Path(xml_dir)
    if p.is_file() or not expected_entries:
        return []
    truncated = []
    for e in expected_entries:
        xml_path = p / f"{e['slug']}.xml"
        log_path = p / f"{e['slug']}.log"
        record = {"name": e["name"], "slug": e["slug"],
                  "xml_path": str(xml_path), "log_path": str(log_path),
                  "reason": None}
        if not xml_path.is_file():
            record["reason"] = "XML missing — likely wallclock SIGTERM mid-run"
            truncated.append(record)
            continue
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            suites = root.findall("testsuite") if root.tag != "testsuite" else [root]
            total = sum(len(s.findall("testcase")) for s in suites if s.tag == "testsuite")
            if total == 0:
                record["reason"] = "XML has zero testcases — entry ran but collected/recorded none"
                truncated.append(record)
        except ET.ParseError as exc:
            record["reason"] = f"XML malformed — partial write before SIGTERM ({exc})"
            truncated.append(record)
    return truncated


# -- Parser (lifted from SKILL.md Phase 5) ----------------------------------

def parse_xml(path: str):
    """Return per-file counters + sample messages from one JUnit XML.

    Bucket discriminator for <skipped> elements:
      * type="pytest.xfail" → X bucket (test marked @pytest.mark.xfail; the
        marker fired — either it failed as expected or it unexpectedly
        passed). Reported separately from real skips since these tracks
        silicon-side known-broken-tests, not arch-gate skips.
      * anything else      → S bucket (regular pytest.skip, multi-device
        gate, grid-size gate, etc.).
    """
    tree = ET.parse(path)
    root = tree.getroot()
    suites = root.findall("testsuite") if root.tag != "testsuite" else [root]

    by_file = collections.defaultdict(lambda: {
        "P": 0, "F": 0, "E": 0, "S": 0, "X": 0,
        "fail_sample": None,
        "error_sample": None,
        "skip_samples": [],
        "xfail_samples": [],
    })

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
            if fail is not None:
                by_file[leaf]["F"] += 1
                if by_file[leaf]["fail_sample"] is None:
                    by_file[leaf]["fail_sample"] = (
                        name, fail.get("message", "") or "",
                        (fail.text or "").strip()[:500])
            elif err is not None:
                by_file[leaf]["E"] += 1
                if by_file[leaf]["error_sample"] is None:
                    by_file[leaf]["error_sample"] = (
                        name, err.get("message", "") or "",
                        (err.text or "").strip()[:500])
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
    return by_file


def parse_xml_dir(xml_path: str):
    """Accept a file or a directory of XMLs; return merged per-file counters."""
    p = Path(xml_path)
    if p.is_file():
        files = [str(p)]
    else:
        files = sorted(glob.glob(f"{xml_path}/*.xml"))
    merged = collections.defaultdict(lambda: {
        "P": 0, "F": 0, "E": 0, "S": 0, "X": 0,
        "fail_sample": None, "error_sample": None,
        "skip_samples": [], "xfail_samples": [],
    })
    for f in files:
        for leaf, counters in parse_xml(f).items():
            slot = merged[leaf]
            for k in ("P", "F", "E", "S", "X"):
                slot[k] += counters[k]
            if slot["fail_sample"] is None:
                slot["fail_sample"] = counters["fail_sample"]
            if slot["error_sample"] is None:
                slot["error_sample"] = counters["error_sample"]
            slot["skip_samples"].extend(counters["skip_samples"])
            slot["xfail_samples"].extend(counters["xfail_samples"])
    return dict(merged), files


# -- Classifier --------------------------------------------------------------

def bucket(counters):
    # X (xfail) is treated like S for bucketing: it's not a pass and not a
    # fail under emule's control — it's a silicon-side known-broken-test
    # marker firing. A file with only P + X + S is ALL_PASS for our purposes.
    P, F, E = counters["P"], counters["F"], counters["E"]
    non_skip = P + F + E
    if non_skip == 0:
        return "ALL_SKIP"
    if P == non_skip:
        return "ALL_PASS"
    if P == 0:
        return "ALL_FAIL"
    return "PARTIAL"


# -- Skip-reason categorisation ---------------------------------------------

SKIP_REASON_PATTERNS = [
    (r"Need at least \d+ devices", "Multi-device (>1 devices required)"),
    (r"Need at least \d+ cores",   "Grid-size gate (more cores required)"),
    (r"requires_grid_size",         "Explicit grid-size marker"),
    (r"skip_for_blackhole|not.*supported on blackhole",
                                    "Skipped for Blackhole"),
    (r"skip_for_wormhole|not.*supported on wormhole",
                                    "Skipped for Wormhole"),
]


def categorise_skip(msg):
    for pat, label in SKIP_REASON_PATTERNS:
        if re.search(pat, msg, re.I):
            return label
    return "Misc / per-parametrization"


# -- Blocker-class assignment (default 5 from SKILL.md) ---------------------

BLOCKER_CLASS_PATTERNS = [
    (r"signal 11|signal 6|SIGSEGV|SIGABRT|CRASHED",      "Setup crash"),
    (r"jit_compile_kernel.*failed|use of undeclared identifier|"
     r"compiler failed \(exit", "Missing op / unjitted symbol"),
    (r"Need at least \d+ devices|Need at least \d+ cores|requires_grid_size",
                                                          "Multi-host / grid gate"),
    (r"AttributeError|ModuleNotFoundError|undefined symbol|TypeError.*has no attribute",
                                                          "Mock-API drift"),
    (r"PCC|ATOL|assert.*tensor|allclose|expected.*got",  "Numeric drift (PCC/ATOL)"),
]


def assign_blocker_class(text):
    if not text:
        return "Unclassified", None
    for pat, label in BLOCKER_CLASS_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            return label, m.group(0)
    return "Unclassified", None


# -- Headline metrics --------------------------------------------------------

def headline_metrics(by_file):
    P = sum(c["P"] for c in by_file.values())
    F = sum(c["F"] for c in by_file.values())
    E = sum(c["E"] for c in by_file.values())
    S = sum(c["S"] for c in by_file.values())
    X = sum(c["X"] for c in by_file.values())
    total = P + F + E + S + X
    # Executed = the set of tests whose pass/fail is emule's responsibility.
    # Excludes both regular skips (gates emule can't satisfy by design) and
    # xfails (silicon-side known-broken-test markers).
    executed = total - S - X

    # Dominant-crash detection: the file with the largest E count.
    dominant_crash = max((c["E"] for c in by_file.values()), default=0)

    def pct(n, d):
        return (n / d * 100.0) if d else 0.0

    buckets = collections.Counter(bucket(c) for c in by_file.values())
    files_total = sum(buckets.values())
    testable_files = files_total - buckets["ALL_SKIP"]

    return {
        "P": P, "F": F, "E": E, "S": S, "X": X,
        "total": total, "executed": executed,
        "dominant_crash": dominant_crash,
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


# -- Report writers ----------------------------------------------------------

BANNER = (
    "> ⚠️  **DRY RUN — DO NOT SHIP.** This report was generated from a "
    "single-file pipeline-validation run, not a full BH post-commit sweep. "
    "Numbers and bucket assignments here are smoke-test signal only.\n"
)


def fmt_pct(x):
    return f"{x:.1f}%"


def write_exec(out_path, *, suite_name, variant, metrics, by_file,
               manifest, audit_log, runner, baseline, truncated=()):
    when = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# {suite_name} status (exec) — DRY RUN",
        BANNER,
        f"**Scope:** dry-run of one test file under emule ({variant}).  ",
        f"**Snapshot:** {when}.",
        "",
        "## Headline",
        "",
        f"| Metric | {variant} |",
        "|---|---:|",
        f"| Tests passing | {metrics['P']} |",
        f"| Passing — share of total collected | {fmt_pct(metrics['pass_of_total'])} |",
        f"| Passing — share of executed | {fmt_pct(metrics['pass_of_executed'])} |",
        f"| Passing — share of executed (excluding setup-crash inflation) | {fmt_pct(metrics['pass_of_executed_ex_crash'])} |",
        f"| Test files fully passing | {metrics['files_all_pass']} / {metrics['files_total']} |",
        f"| Test files fully passing — share of total | {fmt_pct(metrics['files_pass_of_total'])} |",
        f"| Test files fully passing — share of testable | {fmt_pct(metrics['files_pass_of_testable'])} |",
    ]
    if baseline == "silicon-passing":
        # For the dry-run the "in-gate entries" abstraction collapses to
        # files-in-this-run, so this metric is mostly N/A.
        lines.append(
            f"| Fraction of in-gate entries fully green | "
            f"{metrics['files_all_pass']} / {metrics['files_total']} "
            f"({fmt_pct(metrics['files_pass_of_total'])}) |")
    lines.append("")

    lines.append(f"Single-variant run ({variant}); cross-variant comparison N/A in a dry-run.")
    lines.append("")

    lines += [
        "## Status",
        "",
        f"| Bucket | Count | Notes |",
        "|---|---:|---|",
        f"| Passing | {metrics['P']} | sub-tests that returned PASS in the JUnit XML |",
        f"| In-scope failures | {metrics['F']} | numerical / functional failures with PCC/ATOL or assertion |",
        f"| Setup crashes | {metrics['E']} | E bucket; would be Class **Setup crash** in the blocker taxonomy |",
        f"| Skipped by design | {metrics['S']} | gates emule can't satisfy by design (multi-device, grid-size, etc.); see breakdown below |",
        f"| Xfail / xpass | {metrics['X']} | silicon-side `@pytest.mark.xfail` markers fired; tracked separately from real skips and from emule's pass/fail accounting |",
        f"| **Truncated entries** | **{metrics.get('truncated_count', 0)}** | manifest entries with missing/empty XML — likely hit the per-entry wallclock backstop. **These entries contribute 0 to every other bucket and are NOT in the headline % above.** See breakdown below. |",
        "",
    ]

    if truncated:
        lines += [
            "## Truncated entries (wallclock-truncated, missing data)",
            "",
            "These entries hit their per-entry wallclock backstop before pytest could write its JUnit XML. They contribute **0** to every count above — the metric over-counts the coverage of the rest, but only if these entries had silently disappeared. They have not — they are listed here so the reader can apply a correction in their head.",
            "",
            "| Entry | Reason | Log path |",
            "|---|---|---|",
        ]
        for t in truncated:
            log_ref = f"`{t['log_path']}`" if Path(t['log_path']).is_file() else "(no log)"
            lines.append(f"| {t['name']} | {t['reason']} | {log_ref} |")
        lines.append("")

    # Skip breakdown
    skip_groups = collections.Counter()
    for c in by_file.values():
        for _, msg in c["skip_samples"]:
            skip_groups[categorise_skip(msg)] += 1
    lines += ["## What \"skipped by design\" covers", ""]
    if not skip_groups:
        lines.append("(No skipped tests in this dry-run.)")
    else:
        lines += ["| Reason | Count |", "|---|---:|"]
        for label, n in skip_groups.most_common():
            lines.append(f"| {label} | {n} |")
    lines.append("")

    # Top 3 gaps + next steps (placeholder shape so the verify checks the section)
    lines += [
        "## Top 3 gaps",
        "",
        "1. *(dry-run; populated in full-run report from the largest blocker-class buckets)*",
        "2. *(dry-run; populated in full-run report)*",
        "3. *(dry-run; populated in full-run report)*",
        "",
        "## Top 3 next steps",
        "",
        "1. *(dry-run; populated after eyeballing the full-run results)*",
        "2. *(dry-run)*",
        "3. *(dry-run)*",
        "",
        "## Variant readiness",
        "",
        f"Single-variant dry-run ({variant}). Full-run report will compare across "
        "any added variants (e.g. P150 vs P100). No readiness signal from N=1.",
        "",
        "## Recent landmarks",
        "",
        "- *(dry-run; populated from recent commit log in the full-run report)*",
        "",
        "---",
        f"*Generated by snapshots/bh_sanity_dryrun/parse_and_report.py at {when}.*",
    ]
    Path(out_path).write_text("\n".join(lines) + "\n")


def write_dev(out_path, *, suite_name, variant, metrics, by_file,
              manifest, audit_log, runner, baseline, xml_files,
              log_paths_seen, truncated=(), expected_entries=()):
    when = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# {suite_name} status (dev) — DRY RUN",
        BANNER,
        f"**Scope:** dry-run of one test file under emule ({variant}).  ",
        f"**Snapshot:** {when}.",
        "",
        "## Headline",
        "",
        f"Variant **{variant}** — raw counts: "
        f"P={metrics['P']}  F={metrics['F']}  E={metrics['E']}  "
        f"S={metrics['S']}  X={metrics['X']} (xfail)  "
        f"truncated_entries={metrics.get('truncated_count', 0)} of {metrics.get('expected_entries', 0)} expected",
        "",
        f"Pass-rate denominators (see SKILL.md Phase 6 formulas):",
        f"- of total: {fmt_pct(metrics['pass_of_total'])}",
        f"- of executed: {fmt_pct(metrics['pass_of_executed'])}",
        f"- of executed excl. dominant crash: {fmt_pct(metrics['pass_of_executed_ex_crash'])}",
        "",
    ]

    # File-level tables
    rows = [(leaf, c, bucket(c)) for leaf, c in by_file.items()]
    rows.sort(key=lambda r: (r[2], r[0]))

    def tbl(title, predicate, with_sample=False):
        sub = [(l, c, b) for (l, c, b) in rows if predicate(b)]
        out = [f"## {title}", ""]
        if not sub:
            out.append("(none)")
            out.append("")
            return out
        if with_sample:
            out += ["| File | P / F / E / S | Class | Sample |",
                    "|---|---|---|---|"]
            for leaf, c, b in sub:
                sample_text = ""
                cls = "Unclassified"
                if c["fail_sample"]:
                    name, msg, text = c["fail_sample"]
                    sample_text = (msg or text)[:140]
                    cls, _ = assign_blocker_class(msg or text)
                elif c["error_sample"]:
                    name, msg, text = c["error_sample"]
                    sample_text = (msg or text)[:140]
                    cls, _ = assign_blocker_class(msg or text)
                sample_text = sample_text.replace("|", "\\|").replace("\n", " ")
                out.append(f"| `{leaf}` | {c['P']} / {c['F']} / {c['E']} / {c['S']} | {cls} | {sample_text} |")
        else:
            out += ["| File | P / F / E / S |", "|---|---|"]
            for leaf, c, _ in sub:
                out.append(f"| `{leaf}` | {c['P']} / {c['F']} / {c['E']} / {c['S']} |")
        out.append("")
        return out

    lines += tbl("Fully-passing files", lambda b: b == "ALL_PASS")
    lines += tbl("Partial-pass files", lambda b: b == "PARTIAL", with_sample=True)
    lines += tbl("All-failing files", lambda b: b == "ALL_FAIL", with_sample=True)
    lines += tbl("All-skip files", lambda b: b == "ALL_SKIP")

    # Blocker taxonomy summary
    cls_counts = collections.Counter()
    cls_samples = collections.defaultdict(list)
    for leaf, c in by_file.items():
        for sample_key in ("fail_sample", "error_sample"):
            sample = c.get(sample_key)
            if not sample:
                continue
            name, msg, text = sample
            cls, evidence = assign_blocker_class(msg or text)
            cls_counts[cls] += 1
            if len(cls_samples[cls]) < 2:
                cls_samples[cls].append((leaf, name, (msg or text)[:200]))
    lines += ["## Blocker taxonomy (5-class default from SKILL.md)", ""]
    skill_classes = [
        "Setup crash",
        "Numeric drift (PCC/ATOL)",
        "Missing op / unjitted symbol",
        "Multi-host / grid gate",
        "Mock-API drift",
    ]
    lines += ["| Class | Observed in dry-run | Sample |", "|---|---:|---|"]
    for cls in skill_classes + (["Unclassified"] if "Unclassified" in cls_counts else []):
        n = cls_counts.get(cls, 0)
        if n and cls_samples[cls]:
            leaf, name, sample = cls_samples[cls][0]
            sample = sample.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {cls} | {n} | `{leaf}::{name}` — {sample} |")
        else:
            lines.append(f"| {cls} | {n} | — |")
    lines.append("")

    # Skip categorisation
    skip_groups = collections.Counter()
    skip_samples = collections.defaultdict(list)
    for leaf, c in by_file.items():
        for name, msg in c["skip_samples"]:
            cat = categorise_skip(msg)
            skip_groups[cat] += 1
            if len(skip_samples[cat]) < 1:
                skip_samples[cat].append((leaf, name, msg[:200]))
    lines += ["## Skip-reason analysis", ""]
    if not skip_groups:
        lines.append("(no skipped tests in this dry-run)")
    else:
        lines += ["| Reason | Count | Sample |", "|---|---:|---|"]
        for label, n in skip_groups.most_common():
            leaf, name, msg = skip_samples[label][0]
            msg = msg.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {label} | {n} | `{leaf}::{name}` — {msg} |")
    lines.append("")

    # Xfail-specific section. xfail-marked tests are tracked separately
    # because they reflect silicon-side known-broken-test markers, not
    # emule's pass/fail accounting. Surfaced for transparency but excluded
    # from the `% passing` metric (treated like skips).
    xfail_count = sum(c["X"] for c in by_file.values())
    lines += ["## Xfail / xpass breakdown", ""]
    if xfail_count == 0:
        lines.append("(no xfail markers fired in this dry-run)")
    else:
        per_file_x = sorted(
            ((leaf, c["X"], c["xfail_samples"]) for leaf, c in by_file.items() if c["X"]),
            key=lambda r: -r[1],
        )
        lines.append(f"**Total xfail/xpass:** {xfail_count} — tracked separately from skips and from emule's pass/fail accounting.")
        lines.append("")
        lines += ["| File | xfail count | Sample message |", "|---|---:|---|"]
        for leaf, n, samples in per_file_x:
            if samples:
                _, msg = samples[0]
                msg = msg.replace("|", "\\|").replace("\n", " ")[:200]
            else:
                msg = "—"
            lines.append(f"| `{leaf}` | {n} | {msg} |")
    lines.append("")

    # Coverage analysis: in dry-run mode, this collapses to "what file did we
    # run?" The full-run version cross-references the manifest + the audit log.
    lines += [
        "## Coverage analysis",
        "",
        "Dry-run scope: one test file from the BH post-commit `ttnn data movement group`. "
        "The full-run report cross-references every selected manifest entry against "
        "the audited runner's emitted invocation.",
        "",
    ]

    # Truncated entries section — present even when empty, so the reader
    # always sees that the check happened.
    lines += [
        "## Truncated entries (wallclock-truncated)",
        "",
    ]
    if not expected_entries:
        lines.append("Truncation check not performed (no `--expected-from-audit` argument). "
                     "In dry-run mode the parser reads the XML files it finds and trusts that "
                     "the harness produced them all.")
    elif not truncated:
        lines.append(f"**OK.** All {len(expected_entries)} expected entries produced a non-empty XML. "
                     "No data loss from wallclock SIGTERM.")
    else:
        lines.append(f"**{len(truncated)} of {len(expected_entries)} entries hit a wallclock truncation.** "
                     "Each one's XML was missing, malformed, or empty — the entry ran but couldn't "
                     "report results. These entries contribute 0 to the headline counts; correct "
                     "for them when interpreting the metric.")
        lines.append("")
        lines += ["| Entry | Slug | Reason | XML | Log |",
                  "|---|---|---|---|---|"]
        for t in truncated:
            xml_ref = f"`{t['xml_path']}`" if Path(t['xml_path']).is_file() else "(missing)"
            log_ref = f"`{t['log_path']}`" if Path(t['log_path']).is_file() else "(missing)"
            lines.append(f"| {t['name']} | `{t['slug']}` | {t['reason']} | {xml_ref} | {log_ref} |")
    lines.append("")

    # Variant readiness
    lines += [
        "## Variant readiness",
        "",
        "N/A in dry-run mode (single variant). Full-run report adds cross-variant "
        "delta + regression flags.",
        "",
    ]

    # Next steps
    lines += [
        "## Top suggested next steps",
        "",
        "1. Promote `harness.sh` / `parse_and_report.py` / `verify.sh` from "
        "`snapshots/bh_sanity_dryrun/` into `snapshots/bh_sanity/`, generalised "
        "to walk a per-entry XML directory.",
        "2. Run the full 14-entry sweep via the promoted harness wrapping "
        "`snapshots/bh_sanity/run_bh_sanity.sh`.",
        "3. Re-run `parse_and_report.py` on the full-sweep XML dir to produce "
        "the real exec + dev reports.",
        "",
    ]

    # References
    audit_path = Path(audit_log)
    audit_state = "AUDIT PASS" if audit_path.is_file() and "AUDIT PASS" in audit_path.read_text() else "AUDIT FAIL or missing"
    lines += [
        "## References",
        "",
        f"- **Source manifest:** `{manifest}`",
        f"- **Audited runner:** `{runner}`",
        f"- **Audit log:** `{audit_log}` — {audit_state}",
        f"- **XML files parsed ({len(xml_files)}):**",
    ]
    for f in xml_files:
        lines.append(f"  - `{f}`")
    lines.append(f"- **Log files referenced ({len(log_paths_seen)}):**")
    for f in log_paths_seen:
        lines.append(f"  - `{f}`")
    lines.append("")
    lines.append(f"*Generated by snapshots/bh_sanity_dryrun/parse_and_report.py at {when}.*")
    Path(out_path).write_text("\n".join(lines) + "\n")


# -- Driver -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-dir", required=True)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--out-exec", required=True)
    ap.add_argument("--out-dev", required=True)
    ap.add_argument("--suite-name", default="bh_sanity_dryrun")
    ap.add_argument("--variant-label", default="bh_emule")
    ap.add_argument("--audit-log", required=True)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--baseline", default="silicon-passing")
    ap.add_argument("--expected-from-audit", default=None,
                    help="If set, treat this audit log as the source of truth "
                         "for which entries should appear; entries with missing "
                         "or empty XML get flagged as TRUNCATED.")
    args = ap.parse_args()

    by_file, xml_files = parse_xml_dir(args.xml_dir)

    log_paths_seen = []
    if args.log_dir:
        log_paths_seen = sorted(glob.glob(f"{args.log_dir}/*.log"))

    expected_entries = []
    truncated = []
    if args.expected_from_audit:
        expected_entries = parse_expected_entries(args.expected_from_audit)
        truncated = detect_truncated_entries(args.xml_dir, expected_entries)

    if not by_file and not truncated:
        print("ERROR: no testcases parsed from XML and no truncated entries", file=sys.stderr)
        sys.exit(2)

    metrics = headline_metrics(by_file)
    metrics["truncated_count"] = len(truncated)
    metrics["expected_entries"] = len(expected_entries)
    write_exec(args.out_exec,
               suite_name=args.suite_name, variant=args.variant_label,
               metrics=metrics, by_file=by_file,
               manifest=args.manifest, audit_log=args.audit_log,
               runner=args.runner, baseline=args.baseline,
               truncated=truncated)
    write_dev(args.out_dev,
              suite_name=args.suite_name, variant=args.variant_label,
              metrics=metrics, by_file=by_file,
              manifest=args.manifest, audit_log=args.audit_log,
              runner=args.runner, baseline=args.baseline,
              xml_files=xml_files, log_paths_seen=log_paths_seen,
              truncated=truncated, expected_entries=expected_entries)
    print(f"wrote {args.out_exec}")
    print(f"wrote {args.out_dev}")
    print(f"files: {len(by_file)}  P={metrics['P']} F={metrics['F']} E={metrics['E']} S={metrics['S']} X={metrics['X']}  truncated_entries={len(truncated)}")


if __name__ == "__main__":
    main()
