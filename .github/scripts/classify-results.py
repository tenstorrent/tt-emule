#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Classify gtest XML results against .github/known-failures.txt.

Reads per-test JUnit XML files produced by gtest (one per
run_regression.sh test invocation, captured via GTEST_XML_DIR), diffs the
observed pass/fail set against the allowlist, and writes a markdown summary
to $GITHUB_STEP_SUMMARY (or stdout if unset).

Exits 0 iff there are no new failures, no newly-passing entries, and no
stale allowlist entries. Exits 1 otherwise.

With no --allowlist, the allowlist is empty: every failure is a new failure
(zero-tolerance, all tests must pass). Used by arches with no known failures.

Usage:
    classify-results.py --xml-dir <dir> [--allowlist <file>] [--build-dir <dir>]
"""

import argparse
import fnmatch
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_xml(xml_dir: Path) -> tuple[set[str], dict[str, str]]:
    """Walk *.xml under xml_dir, return (passed, failed_with_binary_hint).

    passed                — set of "Suite.Test" names that passed.
    failed_with_binary    — map "Suite.Test" -> source XML file stem (which
                            roughly identifies the test binary / regression
                            entry it came from). Used for repro hints.
    """
    passed: set[str] = set()
    failed: dict[str, str] = {}

    if not xml_dir.is_dir():
        print(f"WARNING: XML dir does not exist: {xml_dir}", file=sys.stderr)
        return passed, failed

    for xml_path in sorted(xml_dir.glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as e:
            print(f"WARNING: cannot parse {xml_path.name}: {e}", file=sys.stderr)
            continue
        for tc in root.iter("testcase"):
            suite = tc.get("classname", "")
            name = tc.get("name", "")
            if not suite or not name:
                continue
            fqname = f"{suite}.{name}"
            # gtest emits <failure>, <error>, or <skipped> children for non-pass.
            # Disabled tests show up as status="notrun".
            has_failure = (
                tc.find("failure") is not None or tc.find("error") is not None
            )
            is_skipped = tc.find("skipped") is not None
            if is_skipped:
                continue
            if has_failure:
                failed[fqname] = xml_path.stem
            else:
                passed.add(fqname)
    return passed, failed


def expand_allowlist(
    entries: list[str], universe: set[str]
) -> tuple[dict[str, set[str]], list[str]]:
    """Match each allowlist pattern against the observed test universe.

    Returns (matches, stale) where:
      matches  — map pattern -> set of fqnames it matched in the universe
      stale    — patterns that matched nothing
    """
    matches: dict[str, set[str]] = {}
    stale: list[str] = []
    for pat in entries:
        if any(c in pat for c in "*?[]"):
            hits = {fq for fq in universe if fnmatch.fnmatchcase(fq, pat)}
        else:
            hits = {pat} if pat in universe else set()
        matches[pat] = hits
        if not hits:
            stale.append(pat)
    return matches, stale


def read_allowlist(path: Path) -> tuple[list[str], set[str]]:
    """Parse the allowlist. Entries prefixed with `flaky:` are returned with
    the prefix stripped and also collected into a `flaky` set so the caller
    can suppress newly-passing/stale errors for those entries (a flake
    documented elsewhere is expected to flip between PASS and FAIL).
    """
    if not path.is_file():
        return [], set()
    entries: list[str] = []
    flaky: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("flaky:"):
            pat = line[len("flaky:"):].strip()
            entries.append(pat)
            flaky.add(pat)
        else:
            entries.append(line)
    return entries, flaky


def write_summary(
    out_path: Path | None,
    new_failures: dict[str, str],
    newly_passing: set[str],
    stale_entries: list[str],
    passed_count: int,
    failed_count: int,
    build_dir: str | None,
) -> None:
    lines: list[str] = []
    has_problems = bool(new_failures) or bool(newly_passing) or bool(stale_entries)
    status = "❌ Regression failed" if has_problems else "✅ Regression passed"
    lines.append(f"# {status}")
    lines.append("")
    lines.append(f"- **{passed_count}** tests passed")
    lines.append(f"- **{failed_count}** tests failed ({failed_count - len(new_failures)} expected, {len(new_failures)} new)")
    # Surface all three allowlist-health counts up front. Each is an independent
    # gate (any non-zero count fails the job). Showing them here in the summary
    # means a reader cannot miss one of the three problem categories — important
    # because the per-category sections below are only rendered when non-empty,
    # and a reader who scrolls only to the first ❌/⚠️ section may not realize a
    # second one is below.
    new_marker = "❌" if new_failures else "✅"
    np_marker = "⚠️" if newly_passing else "✅"
    stale_marker = "⚠️" if stale_entries else "✅"
    lines.append(f"- {new_marker} **{len(new_failures)}** new failures (failed but not allowlisted)")
    lines.append(f"- {np_marker} **{len(newly_passing)}** newly-passing allowlist entries (allowlisted but now passing)")
    lines.append(f"- {stale_marker} **{len(stale_entries)}** stale allowlist entries (matched no test in this run)")
    lines.append("")

    if new_failures:
        lines.append("## ❌ New failures")
        lines.append("")
        lines.append("These tests failed and are NOT in `known-failures.txt`.")
        lines.append("Either fix the regression or, if expected, add to the allowlist.")
        lines.append("")
        for fq, src in sorted(new_failures.items()):
            lines.append(f"### `{fq}`")
            if build_dir:
                lines.append("")
                lines.append("Reproduce locally:")
                lines.append("```bash")
                lines.append(
                    f'BUILD_DIR={build_dir} bash run_regression.sh  # full suite'
                )
                lines.append(
                    f'# or run the gtest binary directly with:'
                )
                lines.append(
                    f'# --gtest_filter="{fq}"'
                )
                lines.append("```")
            lines.append("")
            lines.append(f"_Source: `{src}.xml`_")
            lines.append("")

    if newly_passing:
        lines.append("## ⚠️ Newly-passing allowlist entries")
        lines.append("")
        lines.append(
            "These tests are listed in `known-failures.txt` but now PASS. "
            "Remove them from the allowlist."
        )
        lines.append("")
        for fq in sorted(newly_passing):
            lines.append(f"- `{fq}`")
        lines.append("")

    if stale_entries:
        lines.append("## ⚠️ Stale allowlist entries")
        lines.append("")
        lines.append(
            "These patterns in `known-failures.txt` matched NO test in this run "
            "(test removed, renamed, or binary missing). Update the allowlist."
        )
        lines.append("")
        for pat in stale_entries:
            lines.append(f"- `{pat}`")
        lines.append("")

    if not has_problems:
        lines.append("All test outcomes match expectations.")
        lines.append("")

    summary = "\n".join(lines)
    if out_path is not None:
        with out_path.open("a") as f:
            f.write(summary)
            f.write("\n")
    print(summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", required=True, type=Path)
    parser.add_argument("--allowlist", default=None, type=Path)
    parser.add_argument("--build-dir", default=None)
    args = parser.parse_args()

    passed, failed = parse_xml(args.xml_dir)
    universe = passed | set(failed.keys())

    # A run that produced no test results at all is never a pass: a missing test
    # binary or a regression script that aborted before any gtest ran would
    # otherwise classify green (no failures observed). Fail loudly instead.
    if not universe:
        print(f"::error::no test results found in {args.xml_dir} — regression did not run")
        return 1

    # No --allowlist => empty allowlist => zero-tolerance (all tests must pass).
    if args.allowlist is None:
        allowlist_patterns, flaky_patterns = [], set()
    else:
        allowlist_patterns, flaky_patterns = read_allowlist(args.allowlist)
    matches, stale = expand_allowlist(allowlist_patterns, universe)

    # Flatten matched fqnames across all patterns; track which came via flaky entries.
    allowlisted_fqnames: set[str] = set()
    flaky_fqnames: set[str] = set()
    for pat, hits in matches.items():
        allowlisted_fqnames |= hits
        if pat in flaky_patterns:
            flaky_fqnames |= hits

    # Flaky entries: stale only if they matched neither pass nor fail.
    flaky_patterns_seen = {pat for pat in flaky_patterns if matches.get(pat)}
    stale = [pat for pat in stale if pat not in flaky_patterns_seen]

    new_failures = {fq: src for fq, src in failed.items() if fq not in allowlisted_fqnames}
    newly_passing = (allowlisted_fqnames & passed) - flaky_fqnames

    summary_path_env = os.environ.get("GITHUB_STEP_SUMMARY")
    summary_path = Path(summary_path_env) if summary_path_env else None

    write_summary(
        out_path=summary_path,
        new_failures=new_failures,
        newly_passing=newly_passing,
        stale_entries=stale,
        passed_count=len(passed),
        failed_count=len(failed),
        build_dir=args.build_dir,
    )

    if new_failures or newly_passing or stale:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
