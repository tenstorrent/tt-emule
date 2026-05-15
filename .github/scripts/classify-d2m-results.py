#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Classify pytest JUnit XML results against .github/known-failures-d2m.txt.

run_d2m_regression.sh, when invoked with D2M_XML_DIR set, writes one
pytest-junit XML per test file. This script walks those XMLs, builds a set of
"<test_file.py>::<test_name>" node-ids, diffs against the allowlist, and
fails if there are new failures, newly-passing entries, or stale entries.

Exits 0 iff all three buckets are empty.

Usage:
    classify-d2m-results.py --xml-dir <dir> --allowlist <file> [--build-dir <dir>]
"""

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _pattern_to_regex(pat: str) -> re.Pattern[str]:
    """Compile an allowlist pattern. `*` and `?` are wildcards; everything
    else (including `[` and `]`, common in pytest parametrize ids) is literal.
    """
    out = []
    for ch in pat:
        if ch == "*":
            out.append(".*")
        elif ch == "?":
            out.append(".")
        else:
            out.append(re.escape(ch))
    return re.compile("".join(out) + r"\Z")


def _node_id(tc: ET.Element, xml_stem: str) -> str | None:
    """Build "<file.py>::<test_name>" from a pytest <testcase>.

    pytest junit emits:
      <testcase classname="test.python.golden.test_metal_matmul"
                name="test_matmul_ttnn_shapes_double_buffered"
                file="test/python/golden/test_metal_matmul.py" ...>

    Prefer the `file` attribute (yields the developer-friendly node-id), fall
    back to deriving the file from `classname` last segment, fall back to the
    XML file stem (set to ${tf%.py} by run_d2m_regression.sh).
    """
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


def parse_xml(xml_dir: Path) -> tuple[set[str], dict[str, str]]:
    """Walk *.xml under xml_dir, return (passed, failed_with_xml_hint)."""
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
            node_id = _node_id(tc, xml_path.stem)
            if node_id is None:
                continue
            # pytest emits <skipped> for skipped tests; we don't count those.
            if tc.find("skipped") is not None:
                continue
            has_failure = (
                tc.find("failure") is not None or tc.find("error") is not None
            )
            if has_failure:
                failed[node_id] = xml_path.stem
            else:
                passed.add(node_id)
    return passed, failed


def expand_allowlist(
    entries: list[str], universe: set[str]
) -> tuple[dict[str, set[str]], list[str]]:
    matches: dict[str, set[str]] = {}
    stale: list[str] = []
    for pat in entries:
        if "*" in pat or "?" in pat:
            regex = _pattern_to_regex(pat)
            hits = {fq for fq in universe if regex.match(fq)}
        else:
            hits = {pat} if pat in universe else set()
        matches[pat] = hits
        if not hits:
            stale.append(pat)
    return matches, stale


def read_allowlist(path: Path) -> list[str]:
    if not path.is_file():
        return []
    entries: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


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
    status = "❌ D2M regression failed" if has_problems else "✅ D2M regression passed"
    lines.append(f"# {status}")
    lines.append("")
    lines.append(f"- **{passed_count}** tests passed")
    expected = failed_count - len(new_failures)
    lines.append(
        f"- **{failed_count}** tests failed ({expected} expected, {len(new_failures)} new)"
    )
    lines.append("")

    if new_failures:
        lines.append("## ❌ New failures")
        lines.append("")
        lines.append("These tests failed and are NOT in `.github/known-failures-d2m.txt`.")
        lines.append("Fix the regression or, if expected, add to the allowlist.")
        lines.append("")
        for fq, src in sorted(new_failures.items()):
            lines.append(f"### `{fq}`")
            lines.append("")
            lines.append("Reproduce locally:")
            lines.append("```bash")
            lines.append(f"bash run_d2m_regression.sh  # full suite")
            lines.append(f"# or just this test:")
            lines.append(f"pytest tt-mlir/test/python/golden/{fq}")
            lines.append("```")
            lines.append("")
            lines.append(f"_Source: `{src}.xml`_")
            lines.append("")

    if newly_passing:
        lines.append("## ⚠️ Newly-passing allowlist entries")
        lines.append("")
        lines.append(
            "These tests are listed in `known-failures-d2m.txt` but now PASS. "
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
            "These patterns in `known-failures-d2m.txt` matched NO test in this run "
            "(test removed, renamed, or its file failed to even start). Update the allowlist."
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
    parser.add_argument("--allowlist", required=True, type=Path)
    parser.add_argument("--build-dir", default=None)
    args = parser.parse_args()

    passed, failed = parse_xml(args.xml_dir)
    universe = passed | set(failed.keys())

    allowlist_patterns = read_allowlist(args.allowlist)
    matches, stale = expand_allowlist(allowlist_patterns, universe)

    allowlisted_fqnames: set[str] = set()
    for hits in matches.values():
        allowlisted_fqnames |= hits

    new_failures = {fq: src for fq, src in failed.items() if fq not in allowlisted_fqnames}
    newly_passing = allowlisted_fqnames & passed

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
