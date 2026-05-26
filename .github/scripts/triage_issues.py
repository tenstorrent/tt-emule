#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Upsert tracking issues for triage findings.

A finding's stable signature hash goes into the issue title; subsequent
findings with the same hash bump the existing issue with a comment rather
than create a duplicate.

Usable as a CLI (for ad-hoc use) and as a library called by ``triage.py``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from triage_lib import gh


def upsert(
    *,
    signature: str,
    title: str,
    body: str,
    labels: list[str],
    run_url: str,
    repo: str | None = None,
) -> dict[str, Any]:
    """Either create a new issue or comment on an existing one with the same signature.

    Returns the issue dict (number, url, ...).
    """
    # Ensure base labels exist (idempotent). Each call only attempts what's used.
    for lbl in labels:
        gh.ensure_label(lbl, repo=repo)

    existing = gh.issue_list(label="auto-triage", query=f'in:title "{signature}"', repo=repo)
    if existing:
        issue = existing[0]
        gh.issue_comment(
            issue["number"],
            f"Still failing in [nightly run]({run_url}).",
            repo=repo,
        )
        return issue
    return gh.issue_create(title=title, body=body, labels=["auto-triage", *labels], repo=repo)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--signature", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--body", required=True)
    p.add_argument("--labels", default="", help="comma-separated")
    p.add_argument("--run-url", required=True)
    p.add_argument("--repo", default=None)
    args = p.parse_args(argv)

    labels = [l for l in args.labels.split(",") if l]
    issue = upsert(
        signature=args.signature,
        title=args.title,
        body=args.body,
        labels=labels,
        run_url=args.run_url,
        repo=args.repo,
    )
    print(issue.get("url", ""), file=sys.stdout)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
    raise SystemExit(_main())
