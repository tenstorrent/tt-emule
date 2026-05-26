#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Triage orchestrator — classify a nightly run, then drive effects.

Effects (in order):
  1. Per finding: upsert a tracking GitHub issue (deduped by signature hash).
  2. Per workflow: replace the AUTO-TRIAGE region of the matching open
     ``auto/uplift-{metal,mlir}`` PR with the current finding list (no-op
     if no such PR exists).

Honours ``DRY_RUN=1`` end-to-end via the gh wrappers.

CLI:
  triage.py --workflow {metal,d2m} --run-id <id> --conclusion {success,failure}
            --artifacts-dir <path> --patterns <yaml>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make triage_lib importable when called as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from triage_lib import artifacts as art_lib
from triage_lib.classify import Finding, classify
from triage_lib import gh as gh_lib  # noqa: F401 (registers DRY_RUN behaviour)


REPO = os.environ.get("GITHUB_REPOSITORY", "tenstorrent/tt-emule")


def _run_url(run_id: str) -> str:
    return f"https://github.com/{REPO}/actions/runs/{run_id}"


def _findings_to_md(findings: list[Finding], run_id: str) -> str:
    if not findings:
        return "_no findings — nightly green_"
    out: list[str] = [
        f"_Updated by [nightly run {run_id}]({_run_url(run_id)})_",
        "",
    ]
    for f in findings:
        out.append(f"### `{f.signature}` — {f.category}")
        out.append("")
        if f.summary:
            out.append(f.summary.strip())
            out.append("")
        if f.suggested_fix:
            out.append("**Suggested next steps**")
            out.append(f.suggested_fix.strip())
            out.append("")
        out.append(f"_Pattern: `{f.pattern_id}` · Source: `{f.source}`_")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _issue_body(finding: Finding, run_id: str) -> str:
    lines = [
        f"**First detected**: [nightly run {run_id}]({_run_url(run_id)})",
        f"**Category**: `{finding.category}`",
        f"**Pattern**: `{finding.pattern_id}`",
        f"**Source**: `{finding.source}`",
        f"**Signature**: `{finding.signature}`",
        "",
    ]
    if finding.summary:
        lines += ["### Summary", finding.summary.strip(), ""]
    if finding.upstream_link:
        lines += [f"**Upstream**: {finding.upstream_link}", ""]
    if finding.evidence:
        lines += [
            "### Evidence",
            "```",
            finding.evidence.strip(),
            "```",
            "",
        ]
    if finding.suggested_fix:
        lines += ["### Suggested next steps", finding.suggested_fix.strip(), ""]
    lines.append("---")
    lines.append("_Filed automatically by `.github/workflows/triage-and-uplift.yml`. Comment to add manual context — agent will continue posting follow-ups under the same signature._")
    return "\n".join(lines).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workflow", required=True, choices=("metal", "d2m"))
    p.add_argument("--run-id", required=True)
    p.add_argument("--conclusion", required=True, choices=("success", "failure", "cancelled", "timed_out", "skipped"))
    p.add_argument("--artifacts-dir", required=True, type=Path)
    p.add_argument("--patterns", required=True, type=Path)
    args = p.parse_args(argv)

    # On success there are no failures to triage; the uplift job advances pins.
    if args.conclusion == "success":
        print("[triage] nightly succeeded; skipping triage classification.")
        return 0

    inputs = art_lib.collect(args.artifacts_dir, workflow=args.workflow)
    findings = classify(inputs, args.patterns, workflow=args.workflow)
    print(f"[triage] classified: {len(findings)} finding(s) from {args.workflow} run {args.run_id}")
    for f in findings:
        print(f"  - {f.signature} [{f.category}] {f.pattern_id}")

    # Lazy import — keeps the triage_issues / uplift_pr modules out of the path
    # when this script is just inspecting findings for tests.
    from triage_issues import upsert  # type: ignore
    from uplift_pr import comment  # type: ignore

    extra_labels = ["d2m"] if args.workflow == "d2m" else []
    for f in findings:
        title = f.issue_title or f"[auto-triage {f.signature}] {f.category}: {f.pattern_id}"
        # GitHub title cap = 256 chars
        if len(title) > 250:
            title = title[:247] + "…"
        upsert(
            signature=f.signature,
            title=title,
            body=_issue_body(f, args.run_id),
            labels=[*extra_labels, *f.labels],
            run_url=_run_url(args.run_id),
            repo=REPO,
        )

    comment(args.workflow, _findings_to_md(findings, args.run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
