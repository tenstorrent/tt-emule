#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Issue #35 stability classifier.

Given a directory tree of pytest JUnit XML outputs from N D2M regression
runs (one subdir per run, conventionally `runN/`), produce a per-test
verdict:

  stable_pass  — passed in every run                        → remove from allowlist
  stable_fail  — failed in every run                        → demote `flaky:` → plain
  flaky        — mixed                                      → keep `flaky:`
  absent       — never collected (test removed or skipped)  → likely stale

Optional `--only-flaky` filters to entries in the allowlist that carry
the `flaky:` prefix. Useful for sizing the surviving flake set after
applying Fix 1.

Doesn't auto-edit the allowlist — emits a sed patch on `--patch-out`
for human review.

Reuses _node_id() and parse_xml() shape from
.github/scripts/classify-d2m-results.py — kept here as a separate copy
to avoid putting investigation tooling on the import path of the
production classifier.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def _node_id(tc: ET.Element, xml_stem: str) -> str | None:
    name = tc.get("name", "")
    if not name:
        return None
    file_attr = tc.get("file", "")
    if file_attr:
        return f"{os.path.basename(file_attr)}::{name}"
    classname = tc.get("classname", "")
    if classname:
        return f"{classname.rsplit('.', 1)[-1]}.py::{name}"
    if xml_stem:
        return f"{xml_stem}.py::{name}"
    return None


def parse_run(xml_dir: Path) -> tuple[set[str], set[str]]:
    """Return (passed_node_ids, failed_node_ids) for one run."""
    passed: set[str] = set()
    failed: set[str] = set()
    for xml_path in sorted(xml_dir.glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as e:
            print(f"WARNING: cannot parse {xml_path}: {e}", file=sys.stderr)
            continue
        for tc in root.iter("testcase"):
            nid = _node_id(tc, xml_path.stem)
            if nid is None:
                continue
            if tc.find("skipped") is not None:
                continue
            if tc.find("failure") is not None or tc.find("error") is not None:
                failed.add(nid)
            else:
                passed.add(nid)
    return passed, failed


def parse_allowlist(path: Path) -> tuple[list[str], set[str]]:
    """Return (entries_with_prefix_stripped, flaky_entries)."""
    entries: list[str] = []
    flaky: set[str] = set()
    if not path.exists():
        return entries, flaky
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("flaky:"):
            entry = line[len("flaky:") :].strip()
            flaky.add(entry)
            entries.append(entry)
        else:
            entries.append(line)
    return entries, flaky


_GLOB_RE = re.compile(r"[*?]")


def _pattern_to_regex(pat: str) -> re.Pattern[str]:
    out: list[str] = []
    for ch in pat:
        if ch == "*":
            out.append(".*")
        elif ch == "?":
            out.append(".")
        else:
            out.append(re.escape(ch))
    return re.compile("^" + "".join(out) + "$")


def classify(
    runs: list[tuple[set[str], set[str]]], universe: set[str]
) -> dict[str, dict]:
    """Per node-id verdict given a list of (pass, fail) sets across runs."""
    verdict: dict[str, dict] = {}
    for nid in universe:
        p = sum(1 for ps, _ in runs if nid in ps)
        f = sum(1 for _, fs in runs if nid in fs)
        if p + f == 0:
            v = "absent"
        elif f == 0:
            v = "stable_pass"
        elif p == 0:
            v = "stable_fail"
        else:
            v = "flaky"
        verdict[nid] = {"pass": p, "fail": f, "runs": p + f, "verdict": v}
    return verdict


def expand(entry: str, universe: set[str]) -> set[str]:
    if _GLOB_RE.search(entry):
        rx = _pattern_to_regex(entry)
        return {n for n in universe if rx.match(n)}
    return {entry} if entry in universe else set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-root", type=Path, required=True,
                    help="Directory containing run1/, run2/, ... subdirs of *.xml")
    ap.add_argument("--allowlist", type=Path, required=True)
    ap.add_argument("--only-flaky", action="store_true",
                    help="Restrict report to allowlist entries with `flaky:` prefix")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--patch-out", type=Path, default=None,
                    help="Write a sed-script patch suggestion (do not auto-apply)")
    args = ap.parse_args()

    run_dirs = sorted(d for d in args.xml_root.iterdir()
                      if d.is_dir() and d.name.startswith("run"))
    if not run_dirs:
        print(f"ERROR: no run*/ subdirs under {args.xml_root}", file=sys.stderr)
        return 2

    runs = [parse_run(d) for d in run_dirs]
    universe = set().union(*(p | f for p, f in runs))

    entries, flaky_entries = parse_allowlist(args.allowlist)
    targets = sorted(flaky_entries) if args.only_flaky else entries

    verdict = classify(runs, universe)

    # Resolve each allowlist entry (which may be a glob) to its members and
    # aggregate the verdict across members.
    rows: list[tuple[str, str, int, int, list[str]]] = []
    for entry in targets:
        members = expand(entry, universe)
        if not members:
            rows.append((entry, "absent", 0, 0, []))
            continue
        # Collapse member verdicts. If all stable_pass: stable_pass; if any flaky:
        # flaky; if all stable_fail: stable_fail; mixed pass/fail across members
        # counts as flaky.
        per = [verdict[m]["verdict"] for m in members]
        if all(v == "stable_pass" for v in per):
            v_entry = "stable_pass"
        elif all(v == "stable_fail" for v in per):
            v_entry = "stable_fail"
        elif all(v == "absent" for v in per):
            v_entry = "absent"
        else:
            v_entry = "flaky"
        total_pass = sum(verdict[m]["pass"] for m in members)
        total_fail = sum(verdict[m]["fail"] for m in members)
        rows.append((entry, v_entry, total_pass, total_fail, sorted(members)))

    # Markdown table
    print(f"## D2M stability report — N={len(run_dirs)} runs, allowlist={args.allowlist.name}\n")
    counts = defaultdict(int)
    for _, v, _, _, _ in rows:
        counts[v] += 1
    summary = ", ".join(f"{counts[k]} {k}" for k in
                        ("stable_pass", "stable_fail", "flaky", "absent")
                        if counts[k])
    print(f"Summary: {summary}\n")
    print("| entry | verdict | pass | fail |")
    print("|---|---|---:|---:|")
    for entry, v, p, f, _ in rows:
        print(f"| `{entry}` | {v} | {p} | {f} |")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {entry: {"verdict": v, "pass": p, "fail": f, "members": ms}
             for entry, v, p, f, ms in rows},
            indent=2))
        print(f"\nJSON: {args.json_out}", file=sys.stderr)

    if args.patch_out:
        lines: list[str] = ["# Allowlist mutations suggested by d2m_stability_report.py\n"]
        for entry, v, _, _, _ in rows:
            if v == "stable_pass":
                lines.append(f"# delete:    flaky:{entry}\n")
            elif v == "stable_fail":
                lines.append(f"# demote:    flaky:{entry}  →  {entry}\n")
            elif v == "absent":
                lines.append(f"# investigate (stale-or-renamed): {entry}\n")
        args.patch_out.write_text("".join(lines))
        print(f"Patch suggestions: {args.patch_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
