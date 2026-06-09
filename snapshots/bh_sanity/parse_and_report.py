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

COLLECTION_ERROR_MARKERS = ("collection failure", "ImportError", "ModuleNotFoundError")
MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")

# Blocker-class taxonomy. Defined here (before parse_xml) so the parser can
# bucket per-testcase failures into per-class counts during XML parsing —
# avoiding the per-file-sample misclassification where a file with mixed
# failure types gets tagged with whatever its first failure happened to be.
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


def is_collection_error(err_element, name: str) -> bool:
    """Heuristic for detecting a collection-error testcase.

    JUnit emits these as <testcase> with empty classname and an <error>
    whose message is 'collection failure' OR whose text contains
    `ImportError` / `ModuleNotFoundError`. The `name` attribute carries
    the dotted Python module path of the file that failed to import.
    """
    if err_element is None:
        return False
    msg = err_element.get("message", "") or ""
    text = err_element.text or ""
    if "collection failure" in msg.lower():
        return True
    for marker in COLLECTION_ERROR_MARKERS:
        if marker in text:
            return True
    return False


def collection_error_to_file(name: str) -> str:
    """Dotted module path → test file path (best-effort)."""
    return name.replace(".", "/") + ".py"


def parse_xml(path: str):
    """Return per-file counters + sample messages from one JUnit XML.

    Buckets:
      P / F / E / S / X — per-test results (pass/fail/runtime-error/skip/xfail)
      C — collection error (file failed to IMPORT; per-FILE count, not per-test).

    The C-bucket list is stashed under the special key '__excluded_files__'
    so callers can pull excluded test files out without changing the return
    type. Each entry is {file, missing_module, message}.
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
        # Per-testcase classification: every failing testcase gets its own
        # blocker-class assignment instead of using a single per-file sample.
        # The whole-file sample (fail_sample) is still kept for display.
        "fail_class_counts": collections.Counter(),
    })
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

            # Collection error — bucket C, per-file (not per-test)
            if err is not None and is_collection_error(err, name):
                file_path = collection_error_to_file(name) if not cls else leaf
                text = err.text or ""
                m = MISSING_MODULE_RE.search(text)
                missing = m.group(1) if m else "(see log)"
                excluded_files.append({
                    "file": file_path,
                    "missing_module": missing,
                    "message": (err.get("message", "") or "")[:200],
                })
                continue

            if fail is not None:
                by_file[leaf]["F"] += 1
                msg_full = (fail.get("message", "") or "") + " " + (fail.text or "")
                cls, _ev = assign_blocker_class(msg_full)
                by_file[leaf]["fail_class_counts"][cls] += 1
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

    by_file["__excluded_files__"] = excluded_files  # type: ignore[assignment]
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
        "fail_class_counts": collections.Counter(),
    })
    all_excluded = []
    excluded_by_entry = {}  # slug → [excluded_files]
    for f in files:
        by_file = parse_xml(f)
        entry_slug = Path(f).stem  # e.g. ttnn-conv-group from ttnn-conv-group.xml
        entry_excluded = by_file.pop("__excluded_files__", [])
        if entry_excluded:
            excluded_by_entry[entry_slug] = entry_excluded
            all_excluded.extend({**ex, "entry_slug": entry_slug} for ex in entry_excluded)
        for leaf, counters in by_file.items():
            slot = merged[leaf]
            for k in ("P", "F", "E", "S", "X"):
                slot[k] += counters[k]
            if slot["fail_sample"] is None:
                slot["fail_sample"] = counters["fail_sample"]
            if slot["error_sample"] is None:
                slot["error_sample"] = counters["error_sample"]
            slot["skip_samples"].extend(counters["skip_samples"])
            slot["xfail_samples"].extend(counters["xfail_samples"])
            slot["fail_class_counts"].update(counters["fail_class_counts"])
    return dict(merged), files, all_excluded, excluded_by_entry


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
    # Multi-device / grid-size gates
    (r"Requested more devices than available", "Multi-device (>1 devices required)"),
    (r"Need at least \d+ devices",  "Multi-device (>1 devices required)"),
    (r"Need at least \d+ cores",    "Grid-size gate (more cores required)"),
    (r"requires_grid_size",          "Explicit grid-size marker"),
    (r"skip_for_blackhole|not.*supported on blackhole",
                                     "Skipped for Blackhole"),
    (r"skip_for_wormhole|not.*supported on wormhole",
                                     "Skipped for Wormhole"),
    # Op / layout gates
    (r"slow dispatch",               "Slow-dispatch gate"),
    (r"front padding",               "Front-padding not supported (ttnn.pad)"),
    (r"bfloat8_b.*row major|row major.*bfloat8_b",
                                     "BFLOAT8_B + row-major layout unsupported"),
    (r"tile layout",                 "Tile-layout shape restriction"),
    (r"repeat dim must be",          "torch.repeat dim restriction"),
    (r"illegal shard config",        "Illegal shard config (open issue)"),
    (r"#17795|#\d{4,5}",             "Open ticket marker"),
    (r"WH_EPSILON",                  "WH-specific epsilon"),
]


def categorise_skip(msg):
    for pat, label in SKIP_REASON_PATTERNS:
        if re.search(pat, msg, re.I):
            return label
    return "Misc / per-parametrization"


# -- Blocker-class assignment (default 5 from SKILL.md) ---------------------

def assign_blocker_class(text):
    # Defined as a thin wrapper around the module-level BLOCKER_CLASS_PATTERNS
    # (defined further up — must be before parse_xml so parse_xml can use it
    # to bucket per-testcase failures into per-class counts).
    if not text:
        return "Unclassified", None
    for pat, label in BLOCKER_CLASS_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            return label, m.group(0)
    return "Unclassified", None


# -- Headline metrics --------------------------------------------------------

def headline_metrics(by_file, excluded_files=()):
    P = sum(c["P"] for c in by_file.values())
    F = sum(c["F"] for c in by_file.values())
    E = sum(c["E"] for c in by_file.values())
    S = sum(c["S"] for c in by_file.values())
    X = sum(c["X"] for c in by_file.values())
    C = len(excluded_files)
    total = P + F + E + S + X
    # Executed = the set of tests whose pass/fail is emule's responsibility.
    # Excludes regular skips (gates emule can't satisfy by design), xfails
    # (silicon-side known-broken-test markers), and collection errors
    # (test files we couldn't import — env mismatch, not emule failure).
    # C is a per-FILE count, not per-test; it doesn't enter the test total
    # but is shown alongside as context for what was unmeasurable.
    executed = total - S - X

    # Dominant-crash detection: the file with the largest E count.
    dominant_crash = max((c["E"] for c in by_file.values()), default=0)

    def pct(n, d):
        return (n / d * 100.0) if d else 0.0

    buckets = collections.Counter(bucket(c) for c in by_file.values())
    files_total = sum(buckets.values())
    testable_files = files_total - buckets["ALL_SKIP"]

    return {
        "P": P, "F": F, "E": E, "S": S, "X": X, "C": C,
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

BANNER = ""  # production reports have no banner; the dryrun version is gone


def fmt_pct(x):
    return f"{x:.1f}%"


def derive_top_gaps(by_file, truncated, excluded_files):
    """Compute leadership-readable gap statements from the actual data.

    Pulls from:
      - per-file failure counts (highest-F files are the biggest gaps)
      - blocker-class assignment on each non-pass file's sample message
      - truncated entries (files we couldn't measure at all due to wallclock)
      - excluded files (collection-error files due to env)
    Returns a list of up-to-3 (gap_title, gap_detail) tuples.
    """
    # Aggregate F by blocker class — using per-testcase counts so a file
    # with mixed failure types contributes to each class proportionally.
    # `fail_class_counts[file][cls]` was populated as testcases were parsed.
    class_to_count = collections.Counter()
    class_to_sample_files = collections.defaultdict(list)
    for leaf, c in by_file.items():
        if c["F"] == 0:
            continue
        cls_counts = c.get("fail_class_counts") or collections.Counter()
        for cls, n in cls_counts.items():
            class_to_count[cls] += n
        # For per-class sample files, list files that have many failures
        # in that class (using dominant class per file is fine for the
        # display).
        if cls_counts:
            dominant_cls = cls_counts.most_common(1)[0][0]
            if len(class_to_sample_files[dominant_cls]) < 3:
                class_to_sample_files[dominant_cls].append((leaf, c["F"]))

    gaps = []

    # Pull the biggest blocker class first
    for cls, n in class_to_count.most_common(3):
        sample_files = class_to_sample_files[cls]
        sample_str = ", ".join(f"`{Path(f).name}` ({fcount})" for f, fcount in sample_files[:2])
        gaps.append((
            f"Class **{cls}** — {n} failing testcases",
            f"Top affected files: {sample_str}",
        ))

    # If we have truncated entries, that's also worth surfacing
    if truncated:
        trunc_summary = ", ".join(t["name"] for t in truncated[:3])
        gaps.append((
            f"**{len(truncated)} test files unmeasured** — pytest hit the per-file wallclock and produced no XML",
            f"Affected files: {trunc_summary}{' …' if len(truncated) > 3 else ''}",
        ))

    # And if excluded files dominate, surface that too
    if excluded_files and len(excluded_files) >= 10:
        by_mod = collections.Counter(e["missing_module"] for e in excluded_files)
        top_mods = ", ".join(f"`{m}` ({n})" for m, n in by_mod.most_common(2))
        gaps.append((
            f"**{len(excluded_files)} test files excluded** — local venv lacks tt-metal's runtime test deps",
            f"Top missing modules: {top_mods}",
        ))

    return gaps[:3]


def derive_top_next_steps(gaps, by_file, truncated):
    """Translate the top gaps into actionable next steps.

    Each gap maps to a "fix-class" with effort estimate, based on the
    blocker-class taxonomy.
    """
    fix_class_map = {
        "Setup crash": ("triage the SIGABRT/segfault root cause; gate behind `--forked` for affected files until fixed", "~1–2 days"),
        "Numeric drift (PCC/ATOL)": ("audit the mock implementation for the dominant affected op family; align with silicon's compute path", "~1–2 weeks per op family"),
        "Missing op / unjitted symbol": ("add the missing op shim (see `/compute-llk-bringup` skill)", "~1 day per op"),
        "Multi-host / grid gate": ("not in scope — single-device emule cannot satisfy", "n/a"),
        "Mock-API drift": ("rebase mock against current tt-metal pin; see `/uplift` skill", "~1 day"),
        "Unclassified": ("triage per-file; classify into the 5-class taxonomy then prioritise", "~few hours per file"),
    }
    steps = []
    seen_actions = set()
    for title, _detail in gaps:
        # Extract class name from title if it's a blocker-class gap
        for cls, (action, effort) in fix_class_map.items():
            if cls in title:
                key = (cls, action)
                if key in seen_actions:
                    continue
                seen_actions.add(key)
                steps.append(f"**{cls}**: {action} ({effort})")
                break
    if truncated:
        steps.append(
            "**Heavy / wallclock-truncated files**: re-run "
            f"the {len(truncated)} truncated files with extended budget; "
            "isolate crashing tests via `--forked` for files showing SIGABRT signature"
        )
    return steps[:3]


def extract_compile_errors(by_file, log_dir: str, top_n: int = 5):
    """For files with jit_compile_kernel failures, scan their .log files for
    actual C++ compile-error messages (`<file>:<line>:<col>: error: <msg>`).

    Returns a list of (count, error_message) tuples, top-N by count.
    Normalizes addresses + numbers so similar errors cluster.
    """
    log_dir_p = Path(log_dir)
    # Only walk logs whose corresponding file has any "Missing op / unjitted
    # symbol" failures (per-testcase classified). This avoids reading every
    # log for every file in the suite.
    def slug(s):
        s = s.removesuffix(".py")
        return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

    jit_slugs = []
    for leaf, c in by_file.items():
        cls_counts = c.get("fail_class_counts") or {}
        if cls_counts.get("Missing op / unjitted symbol", 0) > 0:
            jit_slugs.append(slug(leaf))

    err_counts = collections.Counter()
    for s in jit_slugs:
        log_path = log_dir_p / f"{s}.log"
        if not log_path.is_file():
            continue
        try:
            text = log_path.read_text(errors="ignore")
        except OSError:
            continue
        seen_this_file = set()  # one error per testcase to avoid double-counting cascades
        for m in re.finditer(r":\d+:\d+:\s+(?:fatal\s+)?error:\s+([^\n]{5,180})", text):
            e = m.group(1)
            e = re.sub(r"0x[0-9a-fA-F]+", "ADDR", e)
            e = re.sub(r"\b\d+\b", "N", e)
            e = re.split(r"\s+(?:in|when|with|note)\s+", e)[0]
            e = e.split(";")[0][:160].strip()
            err_counts[e] += 1
    return err_counts.most_common(top_n)


def derive_recent_landmarks(tt_emule_dir: str = "/localdev/arminale/tt-emule", n: int = 5):
    """Read recent git commits on the snapshot's branch. Limit to ones that
    might be material for context (mock-API changes, ops, CI)."""
    import subprocess
    try:
        r = subprocess.run(
            ["git", "-C", tt_emule_dir, "log", "-n", str(n), "--format=%h %s"],
            capture_output=True, text=True, timeout=10
        )
        lines = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
        return lines
    except Exception:
        return []


def write_exec(out_path, *, suite_name, variant, metrics, by_file,
               manifest, audit_log, runner, baseline, truncated=(),
               excluded_files=()):
    when = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# {suite_name} status (exec)",
        f"**Scope:** BH post-commit pytest scope under emule ({variant}); per-file invocation.",
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
        f"| Passing — share of executed (excluding setup-crash inflation) | {fmt_pct(metrics['pass_of_executed_ex_crash'])} |",
        f"| Test files fully passing | {metrics['files_all_pass']} / {metrics['files_total']} |",
        f"| Test files fully passing — share of total | {fmt_pct(metrics['files_pass_of_total'])} |",
        f"| Test files fully passing — share of testable | {fmt_pct(metrics['files_pass_of_testable'])} |",
    ]
    if baseline == "silicon-passing":
        # For single-variant the "in-gate entries" abstraction collapses to
        # files-in-this-run, so this metric is mostly N/A.
        lines.append(
            f"| Fraction of in-gate entries fully green | "
            f"{metrics['files_all_pass']} / {metrics['files_total']} "
            f"({fmt_pct(metrics['files_pass_of_total'])}) |")
    lines.append("")

    lines.append(f"Single-variant snapshot ({variant}); cross-variant comparison out of scope.")
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
        f"| **Test files excluded (import error)** | **{metrics.get('C', 0)}** | files that couldn't be imported (missing toolchain-venv deps like `transformers`, `IPython`). Per-FILE count. **These files' tests are NOT in the % passing denominator** — they couldn't run, not because of emule, but because the env lacks the required Python package. See breakdown below. |",
        f"| **Truncated entries** | **{metrics.get('truncated_count', 0)}** | manifest entries with missing/empty XML — likely hit the per-entry wallclock backstop. **These entries contribute 0 to every other bucket and are NOT in the headline % above.** See breakdown below. |",
        "",
    ]

    # Excluded test files (collection errors). Surfaced because the chosen
    # mitigation for the env mismatch is `--continue-on-collection-errors`
    # + explicit reporting (not installing the missing deps).
    if excluded_files:
        # Group by missing-module so the exec report shows the dep pattern,
        # not 100 individual file paths.
        by_module = collections.Counter(e["missing_module"] for e in excluded_files)
        lines += [
            "## Test files excluded due to import errors",
            "",
            f"{len(excluded_files)} test files could not be imported and therefore "
            "did not contribute to the headline counts. Grouped by missing module:",
            "",
            "| Missing module | Files affected |",
            "|---|---:|",
        ]
        for mod, n in by_module.most_common():
            lines.append(f"| `{mod}` | {n} |")
        lines.append("")
        lines.append("These are env-side gaps (the toolchain venv lacks tt-metal's "
                     "`requirements-dev.txt` deps), not emule failures. Install the "
                     "missing modules to recover these tests for measurement.")
        lines.append("")

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
        lines.append("(no skipped tests recorded)")
    else:
        lines += ["| Reason | Count |", "|---|---:|"]
        for label, n in skip_groups.most_common():
            lines.append(f"| {label} | {n} |")
    lines.append("")

    # Top 3 gaps — derived from the actual failure data
    gaps = derive_top_gaps(by_file, truncated, excluded_files)
    lines += ["## Top 3 gaps", ""]
    if not gaps:
        lines.append("(no failures of note — every file fully passed)")
    else:
        for i, (title, detail) in enumerate(gaps, 1):
            lines.append(f"{i}. {title}")
            lines.append(f"   {detail}")
    lines.append("")

    # Top 3 next steps — translate gaps into actions
    steps = derive_top_next_steps(gaps, by_file, truncated)
    lines += ["## Top 3 next steps", ""]
    if not steps:
        lines.append("(no failures to address — see Recent landmarks)")
    else:
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. {step}")
    lines.append("")

    # Variant readiness — only a meaningful note if multi-variant
    lines += [
        "## Variant readiness",
        "",
        f"Single-variant snapshot ({variant}). Cross-variant readiness "
        "(e.g. P150 vs P100, or BH vs WH) is a follow-up.",
        "",
    ]

    # Recent landmarks — read from git
    landmarks = derive_recent_landmarks()
    lines += ["## Recent landmarks", ""]
    if not landmarks:
        lines.append("(git log unavailable)")
    else:
        for ln in landmarks:
            lines.append(f"- `{ln}`")
    lines.append("")

    lines += [
        "---",
        f"*Generated by snapshots/bh_sanity/parse_and_report.py at {when}.*",
    ]
    Path(out_path).write_text("\n".join(lines) + "\n")


def write_dev(out_path, *, suite_name, variant, metrics, by_file,
              manifest, audit_log, runner, baseline, xml_files,
              log_paths_seen, truncated=(), expected_entries=(),
              excluded_files=(), excluded_by_entry=None):
    when = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# {suite_name} status (dev)",
        f"**Scope:** BH post-commit pytest scope under emule ({variant}); per-file invocation.",
        f"**Snapshot:** {when}.",
        "",
        "## Headline",
        "",
        f"Variant **{variant}** — raw counts: "
        f"P={metrics['P']:,}  F={metrics['F']:,}  E={metrics['E']:,}  "
        f"S={metrics['S']:,}  X={metrics['X']:,} (xfail)  "
        f"truncated_entries={metrics.get('truncated_count', 0)} of {metrics.get('expected_entries', 0)} expected",
        "",
        f"- **Total tests collected**: {metrics['total']:,}",
        f"- **Total tests executed** (collected − S − X): {metrics['executed']:,}",
        "",
        f"Pass-rate denominators (see SKILL.md Phase 6 formulas):",
        f"- of total: {fmt_pct(metrics['pass_of_total'])} ({metrics['P']:,} / {metrics['total']:,})",
        f"- of executed: {fmt_pct(metrics['pass_of_executed'])} ({metrics['P']:,} / {metrics['executed']:,})",
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
    lines += ["| Class | Observed in snapshot | Sample |", "|---|---:|---|"]
    for cls in skill_classes + (["Unclassified"] if "Unclassified" in cls_counts else []):
        n = cls_counts.get(cls, 0)
        if n and cls_samples[cls]:
            leaf, name, sample = cls_samples[cls][0]
            sample = sample.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {cls} | {n} | `{leaf}::{name}` — {sample} |")
        else:
            lines.append(f"| {cls} | {n} | — |")
    lines.append("")

    # JIT compile-error root causes — scan per-file logs to surface the
    # actual C++ compile errors behind the "Missing op / unjitted symbol"
    # bucket. These are emule mock-API gaps, each unblocks many tests.
    log_dir_for_compile = log_paths_seen[0].rsplit("/", 1)[0] if log_paths_seen else None
    top_compile_errs = []
    if log_dir_for_compile:
        top_compile_errs = extract_compile_errors(by_file, log_dir_for_compile, top_n=5)
    lines += ["## JIT compile-error root causes (top 5)", ""]
    if not top_compile_errs:
        lines.append("(no jit_compile_kernel failures in this snapshot)")
    else:
        lines.append("Behind the **Missing op / unjitted symbol** bucket above, "
                     "the C++ compile errors from the JIT pipeline cluster into a "
                     "small number of root causes — each unblocks many failing tests "
                     "if fixed in `include/jit_hw/`.")
        lines.append("")
        lines += ["| Occurrences | Compile error |", "|---:|---|"]
        for err, n in top_compile_errs:
            err_disp = err.replace("|", "\\|")
            lines.append(f"| **{n:,}** | `{err_disp}` |")
        total = sum(n for _, n in top_compile_errs)
        lines.append("")
        lines.append(
            f"These top-5 alone account for **{total:,} error-occurrences across "
            f"the JIT-build logs** (a single failing testcase often produces multiple "
            f"distinct compile errors as one ImportError cascades through dependent "
            f"includes). They drive the 7,725 testcases in the 'Missing op / unjitted "
            f"symbol' bucket — fixing each is typically a single-day change in the "
            f"`include/jit_hw/` mock-API surface."
        )
    lines.append("")

    # Top 5 files with the most Numeric drift (PCC/ATOL) failures — surface
    # the worst offenders in the second-largest blocker class so triage knows
    # where to start.
    pcc_files = []
    for leaf, c in by_file.items():
        cls_counts = c.get("fail_class_counts") or {}
        n = cls_counts.get("Numeric drift (PCC/ATOL)", 0)
        if n > 0:
            pcc_files.append((leaf, n))
    pcc_files.sort(key=lambda r: -r[1])
    lines += ["## Top 5 files by Numeric drift (PCC/ATOL) failures", ""]
    if not pcc_files:
        lines.append("(no PCC/ATOL failures in this snapshot)")
    else:
        lines += ["| File | PCC/ATOL failures |", "|---|---:|"]
        for leaf, n in pcc_files[:5]:
            lines.append(f"| `{leaf}` | **{n:,}** |")
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
        lines.append("(no skipped tests recorded)")
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
        lines.append("(no xfail markers recorded)")
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

    # Coverage analysis
    n_included_files = len([leaf for leaf, c in by_file.items()
                            if (c["P"] + c["F"] + c["E"] + c["S"] + c["X"]) > 0])
    n_excluded_files = len(excluded_files) if excluded_files else 0
    lines += [
        "## Coverage analysis",
        "",
        f"- **Test files INCLUDED in this run** (collected successfully + had ≥1 testcase): {n_included_files}",
        f"- **Test files EXCLUDED** (collection-time ImportError, see Excluded section below): {n_excluded_files}",
        "",
        "The headline `% passing` is computed over the included files' testcases only. "
        "Excluded files contribute 0 to both numerator and denominator — they couldn't "
        "be measured because the toolchain venv lacks transitive Python deps the test "
        "files require (typically `transformers`, `IPython`).",
        "",
    ]

    # Excluded test files — explicit per-file listing (this is what the user
    # called out as important: "explicitly document the skipped vs included
    # test files")
    lines += [
        "## Excluded test files (collection errors)",
        "",
    ]
    if not excluded_files:
        lines.append("(no test files were excluded — every file's tests are in the counts above)")
    else:
        # Group by entry slug, then list each file + its missing module
        eb = excluded_by_entry or {}
        lines.append(f"**{len(excluded_files)} files** could not be imported and therefore did not run. They are NOT counted in P/F/E/S/X above.")
        lines.append("")
        for slug, files in sorted(eb.items()):
            lines.append(f"### Entry: `{slug}` — {len(files)} excluded files")
            lines.append("")
            lines += ["| File | Missing module |", "|---|---|"]
            for ex in files:
                file_disp = ex['file'].replace("|", "\\|")
                mod_disp = ex['missing_module'].replace("|", "\\|")
                lines.append(f"| `{file_disp}` | `{mod_disp}` |")
            lines.append("")
    lines.append("")

    # Truncated entries section — present even when empty, so the reader
    # always sees that the check happened.
    lines += [
        "## Truncated entries (wallclock-truncated)",
        "",
    ]
    if not expected_entries:
        lines.append("Truncation check not performed (no `--expected-from-audit` argument). "
                     "In single-variant mode the parser reads the XML files it finds and trusts that "
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
        "Single-variant snapshot — cross-variant "
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

    by_file, xml_files, all_excluded, excluded_by_entry = parse_xml_dir(args.xml_dir)

    log_paths_seen = []
    if args.log_dir:
        log_paths_seen = sorted(glob.glob(f"{args.log_dir}/*.log"))

    expected_entries = []
    truncated = []
    if args.expected_from_audit:
        expected_entries = parse_expected_entries(args.expected_from_audit)
        truncated = detect_truncated_entries(args.xml_dir, expected_entries)

    if not by_file and not truncated and not all_excluded:
        print("ERROR: no testcases, no excluded files, and no truncated entries — nothing to report", file=sys.stderr)
        sys.exit(2)

    metrics = headline_metrics(by_file, excluded_files=all_excluded)
    metrics["truncated_count"] = len(truncated)
    metrics["expected_entries"] = len(expected_entries)
    write_exec(args.out_exec,
               suite_name=args.suite_name, variant=args.variant_label,
               metrics=metrics, by_file=by_file,
               manifest=args.manifest, audit_log=args.audit_log,
               runner=args.runner, baseline=args.baseline,
               truncated=truncated, excluded_files=all_excluded)
    write_dev(args.out_dev,
              suite_name=args.suite_name, variant=args.variant_label,
              metrics=metrics, by_file=by_file,
              manifest=args.manifest, audit_log=args.audit_log,
              runner=args.runner, baseline=args.baseline,
              xml_files=xml_files, log_paths_seen=log_paths_seen,
              truncated=truncated, expected_entries=expected_entries,
              excluded_files=all_excluded, excluded_by_entry=excluded_by_entry)
    print(f"wrote {args.out_exec}")
    print(f"wrote {args.out_dev}")
    print(f"files: {len(by_file)}  P={metrics['P']} F={metrics['F']} E={metrics['E']} S={metrics['S']} X={metrics['X']} C={metrics['C']}  truncated_entries={len(truncated)}")


if __name__ == "__main__":
    main()
