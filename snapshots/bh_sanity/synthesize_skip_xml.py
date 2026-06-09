#!/usr/bin/env python3
# Synthesize a placeholder JUnit XML for a file we're choosing to skip.
#
# Usage: python3 synthesize_skip_xml.py <test_file_rel_path>
#
# Reads the file's expected test count from file_classification.yaml,
# emits N <testcase><failure>…</failure></testcase> entries under the
# file's canonical classname. The failure message makes it crystal
# clear this is a placeholder (not a real test result) so a future
# reader of the dev report sees the asterisk.

import re
import sys
from pathlib import Path

import yaml

CLASSIFICATION = Path("/localdev/arminale/tt-emule/snapshots/bh_sanity/file_classification.yaml")
OUT_DIR = Path("/localdev/arminale/tt-emule/snapshots/bh_sanity/bh_emule")

PLACEHOLDER_MESSAGE = (
    "PLACEHOLDER: file was not actually run. "
    "emule has no implementation for this op family — the prior partial "
    "run showed every test failing. Skipped here to save the ~30-45 min "
    "wallclock cost of re-running with the same expected outcome."
)


def slug(path: str) -> str:
    s = path.removesuffix(".py")
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <test_file_rel_path>", file=sys.stderr)
        sys.exit(2)
    rel = sys.argv[1]

    with CLASSIFICATION.open() as f:
        cls = yaml.safe_load(f)
    found = None
    for r in cls.get("run", []):
        if r["file"] == rel:
            found = r
            break
    if not found:
        print(f"file not in classification: {rel}", file=sys.stderr)
        sys.exit(2)

    n = found["collected"]
    # classname = dotted-module path, matching what pytest would emit
    classname = rel.replace("/", ".").removesuffix(".py")
    file_slug = slug(rel)
    out_xml = OUT_DIR / f"{file_slug}.xml"
    out_log = OUT_DIR / f"{file_slug}.log"

    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<testsuites name="pytest tests">',
        f'<testsuite name="pytest" errors="0" failures="{n}" skipped="0" tests="{n}" time="0.0" '
        f'timestamp="2026-06-08T00:00:00+00:00" hostname="placeholder">',
    ]
    for i in range(1, n + 1):
        parts.append(
            f'<testcase classname="{classname}" name="placeholder_{i:04d}" time="0.0">'
            f'<failure message="placeholder: would have failed">'
            f'{PLACEHOLDER_MESSAGE}'
            f'</failure>'
            f'</testcase>'
        )
    parts.append('</testsuite>')
    parts.append('</testsuites>')

    out_xml.write_text("\n".join(parts) + "\n")

    out_log.write_text(
        f"=== PLACEHOLDER ===\n"
        f"file: {rel}\n"
        f"expected tests: {n}\n"
        f"reason: emule has no implementation for this op family; the prior\n"
        f"        partial run showed every test failing. This XML was\n"
        f"        synthesized by snapshots/bh_sanity/synthesize_skip_xml.py\n"
        f"        to skip the ~30-45 min wallclock cost of re-running with\n"
        f"        the same expected outcome.\n"
        f"\n"
        f"NOT a real pytest log. The XML has {n} failure testcases for\n"
        f"metric-counting purposes; see the XML directly for details.\n"
    )

    print(f"wrote {out_xml} ({n} failure testcases)")
    print(f"wrote {out_log} (placeholder note)")


if __name__ == "__main__":
    main()
