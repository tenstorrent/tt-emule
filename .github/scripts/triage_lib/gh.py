# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Thin wrappers around the ``gh`` CLI.

All mutating calls honour ``DRY_RUN=1`` from the environment: they log what
they would have done and return deterministic fake responses.  This makes
local end-to-end smoke tests safe to run against the real repo.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from typing import Any


def _dry_run() -> bool:
    return os.environ.get("DRY_RUN") == "1"


def _run(args: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    cmd = ["gh", *args]
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def gh_json(args: list[str]) -> Any:
    """Run ``gh <args>`` and JSON-parse stdout."""
    result = _run(args)
    if not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def issue_list(label: str, query: str, repo: str | None = None) -> list[dict[str, Any]]:
    args = [
        "issue", "list",
        "--state", "open",
        "--label", label,
        "--search", query,
        "--json", "number,title,url,body,state,labels",
        "--limit", "20",
    ]
    if repo:
        args += ["--repo", repo]
    return gh_json(args) or []


def issue_create(title: str, body: str, labels: list[str], repo: str | None = None) -> dict[str, Any]:
    if _dry_run():
        print(f"[DRY_RUN] gh issue create --title {title!r} --label {','.join(labels)}", file=sys.stderr)
        print("[DRY_RUN] body:\n" + body, file=sys.stderr)
        return {"number": 0, "url": "dry-run://issue/0"}
    args = ["issue", "create", "--title", title, "--body", body]
    for lbl in labels:
        args += ["--label", lbl]
    if repo:
        args += ["--repo", repo]
    result = _run(args)
    # gh prints the URL on stdout; extract number from the URL.
    url = result.stdout.strip().splitlines()[-1]
    number = int(url.rsplit("/", 1)[-1])
    return {"number": number, "url": url}


def issue_comment(number: int, body: str, repo: str | None = None) -> None:
    if _dry_run():
        print(f"[DRY_RUN] gh issue comment {number} (body {len(body)} chars)", file=sys.stderr)
        return
    args = ["issue", "comment", str(number), "--body", body]
    if repo:
        args += ["--repo", repo]
    _run(args)


def pr_list(label: str, repo: str | None = None) -> list[dict[str, Any]]:
    args = [
        "pr", "list",
        "--state", "open",
        "--label", label,
        "--json", "number,headRefName,title,body,url,labels",
        "--limit", "10",
    ]
    if repo:
        args += ["--repo", repo]
    return gh_json(args) or []


def pr_create(branch: str, base: str, title: str, body: str, labels: list[str], repo: str | None = None) -> dict[str, Any]:
    if _dry_run():
        print(f"[DRY_RUN] gh pr create --head {branch} --base {base} --title {title!r}", file=sys.stderr)
        return {"number": 0, "url": f"dry-run://pr/{branch}"}
    args = ["pr", "create", "--base", base, "--head", branch, "--title", title, "--body", body]
    for lbl in labels:
        args += ["--label", lbl]
    if repo:
        args += ["--repo", repo]
    result = _run(args)
    url = result.stdout.strip().splitlines()[-1]
    number = int(url.rsplit("/", 1)[-1])
    return {"number": number, "url": url}


def pr_edit(number: int, *, title: str | None = None, body: str | None = None, repo: str | None = None) -> None:
    if _dry_run():
        print(f"[DRY_RUN] gh pr edit {number} title={title!r} body=<{len(body or '')} chars>", file=sys.stderr)
        return
    args = ["pr", "edit", str(number)]
    if title is not None:
        args += ["--title", title]
    if body is not None:
        args += ["--body", body]
    if repo:
        args += ["--repo", repo]
    _run(args)


def api(path: str, *, paginate: bool = False) -> Any:
    args = ["api", path]
    if paginate:
        args.append("--paginate")
    return gh_json(args)


def compare_status(repo: str, old: str, new: str) -> str:
    """Return the .status field of GitHub's compare API.

    Possible values: ``identical | ahead | behind | diverged`` (or ``unknown``
    if the API returns something we don't recognise / fails).
    """
    try:
        data = api(f"/repos/{repo}/compare/{old}...{new}")
    except subprocess.CalledProcessError:
        return "unknown"
    return (data or {}).get("status", "unknown")


def ensure_label(name: str, *, color: str = "ededed", description: str = "", repo: str | None = None) -> None:
    """Idempotently create a label.  ``gh label create -f`` is the idiom."""
    if _dry_run():
        print(f"[DRY_RUN] gh label create {name}", file=sys.stderr)
        return
    args = ["label", "create", name, "-c", color, "-d", description, "-f"]
    if repo:
        args += ["--repo", repo]
    _run(args, check=False)  # don't fail if label already exists with different colour
