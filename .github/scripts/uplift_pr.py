#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Manage the long-lived auto/uplift-{metal,mlir} branches and PRs.

Two independent paths:

  * --workflow metal → edits only tt-metal-pin.txt, pushes auto/uplift-metal
  * --workflow d2m   → edits only tt-mlir-pin.txt,  pushes auto/uplift-mlir

Subcommands:
  advance   — bump the pin to the SHA recorded in the nightly artifact and
              create/refresh the PR.
  comment   — replace the PR body's <!-- AUTO-TRIAGE --> region with the
              current findings.  No-op if no PR exists for the path.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from triage_lib import gh, pins


# --------------------------------------------------------------------- paths

REPO = os.environ.get("GITHUB_REPOSITORY", "tenstorrent/tt-emule")

PATHS = {
    "metal": {
        "branch":     "auto/uplift-metal",
        "pin_file":   "tt-metal-pin.txt",
        "label":      "uplift-metal",
        "upstream":   "tenstorrent/tt-metal",
        "title_fmt":  "[uplift] tt-metal-pin → {short}",
        "artifact_sha_file": "tt-metal-pin-resolved.txt",
    },
    "d2m": {
        "branch":     "auto/uplift-mlir",
        "pin_file":   "tt-mlir-pin.txt",
        "label":      "uplift-mlir",
        "upstream":   "tenstorrent/tt-mlir",
        "title_fmt":  "[uplift] tt-mlir-pin → {short}",
        "artifact_sha_file": "tt-mlir-pin-resolved.txt",
    },
}

# Managed PR-body regions.
TRIAGE_MARKERS = ("<!-- AUTO-TRIAGE-START -->", "<!-- AUTO-TRIAGE-END -->")
UPLIFT_MARKERS = ("<!-- AUTO-UPLIFT-PR-START -->", "<!-- AUTO-UPLIFT-PR-END -->")
HISTORY_MARKERS = ("<!-- AUTO-HISTORY-START -->", "<!-- AUTO-HISTORY-END -->")


# ----------------------------------------------------------------- helpers

def _path_cfg(workflow: str) -> dict[str, str]:
    if workflow not in PATHS:
        raise ValueError(f"unknown workflow: {workflow!r}")
    return PATHS[workflow]


def _read_resolved_sha(artifacts_dir: Path, fname: str) -> str:
    for p in artifacts_dir.rglob(fname):
        return p.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"{fname} not found under {artifacts_dir}")


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True)


def _replace_region(body: str, start: str, end: str, new_content: str) -> str:
    """Replace text between two HTML-comment markers, creating the region if absent.

    The markers themselves are preserved.
    """
    pattern = re.compile(
        re.escape(start) + r".*?" + re.escape(end),
        re.DOTALL,
    )
    block = f"{start}\n{new_content.strip()}\n{end}"
    if pattern.search(body):
        return pattern.sub(block, body)
    return (body.rstrip() + "\n\n" + block + "\n") if body else block + "\n"


def find_open_pr(workflow: str) -> dict[str, Any] | None:
    cfg = _path_cfg(workflow)
    prs = gh.pr_list(label=cfg["label"], repo=REPO)
    for pr in prs:
        if pr.get("headRefName") == cfg["branch"]:
            return pr
    return None


# ---------------------------------------------------------------- commands

def advance(workflow: str, run_id: str, artifacts_dir: Path) -> int:
    cfg = _path_cfg(workflow)
    new_sha = _read_resolved_sha(artifacts_dir, cfg["artifact_sha_file"])

    # Ensure local branch is up to date with the latest base (main).
    _git(["fetch", "origin", "main"])
    # Either reset the existing branch to origin/main or create it from origin/main.
    if _git(["rev-parse", "--verify", f"origin/{cfg['branch']}"], check=False).returncode == 0:
        _git(["checkout", "-B", cfg["branch"], f"origin/{cfg['branch']}"])
    else:
        _git(["checkout", "-B", cfg["branch"], "origin/main"])

    old_sha = pins.read_pin(cfg["pin_file"])
    if old_sha == new_sha:
        print(f"[uplift] {cfg['pin_file']} already at {new_sha[:8]}; refreshing PR body only.")
        _refresh_pr(workflow, old_sha=old_sha, new_sha=new_sha, run_id=run_id)
        return 0

    # Refuse to go backward / sideways. compare_status returns "ahead" iff `new`
    # is a strict descendant of `old`.
    status = gh.compare_status(cfg["upstream"], old_sha, new_sha)
    if status not in ("ahead", "identical"):
        print(
            f"[uplift] refusing advance: {cfg['upstream']} compare {old_sha[:8]}..{new_sha[:8]} = {status!r}",
            file=sys.stderr,
        )
        return 0

    pins.write_pin(cfg["pin_file"], new_sha)
    _git(["add", cfg["pin_file"]])
    msg = f"[uplift] {cfg['pin_file']}: {old_sha[:7]} → {new_sha[:7]}"
    _git(["-c", "user.email=tt-emule-bot@tenstorrent.com",
          "-c", "user.name=tt-emule-bot",
          "commit", "-m", msg])

    if os.environ.get("DRY_RUN") == "1":
        print(f"[DRY_RUN] would: git push --force-with-lease origin {cfg['branch']}")
    else:
        _git(["push", "--force-with-lease", "origin", cfg["branch"]])

    _refresh_pr(workflow, old_sha=old_sha, new_sha=new_sha, run_id=run_id)
    return 0


def _refresh_pr(workflow: str, *, old_sha: str, new_sha: str, run_id: str) -> None:
    cfg = _path_cfg(workflow)
    pr = find_open_pr(workflow)
    title = cfg["title_fmt"].format(short=new_sha[:7])
    body = _render_pr_body(
        workflow=workflow,
        old_sha=old_sha,
        new_sha=new_sha,
        run_id=run_id,
        prev_body=pr["body"] if pr else "",
    )
    if pr is None:
        gh.pr_create(
            branch=cfg["branch"],
            base="main",
            title=title,
            body=body,
            labels=["uplift", "automation", cfg["label"]],
            repo=REPO,
        )
    else:
        gh.pr_edit(pr["number"], title=title, body=body, repo=REPO)


def _render_pr_body(*, workflow: str, old_sha: str, new_sha: str, run_id: str, prev_body: str) -> str:
    cfg = _path_cfg(workflow)
    run_url = f"https://github.com/{REPO}/actions/runs/{run_id}"
    compare_url = f"https://github.com/{cfg['upstream']}/compare/{old_sha}...{new_sha}"
    intro = (
        "This PR bumps the upstream pin to the SHA that the most recent nightly\n"
        "completed successfully against. Humans review and merge.\n\n"
        "_Managed by `.github/workflows/triage-and-uplift.yml`. Edits inside the\n"
        "marked regions below will be overwritten on the next nightly._\n\n"
    )
    uplift_block = (
        f"**Last advanced via** [nightly run {run_id}]({run_url}).\n\n"
        f"| pin | old | new |\n"
        f"|---|---|---|\n"
        f"| `{cfg['pin_file']}` | `{old_sha}` | `{new_sha}` |\n\n"
        f"Upstream compare: {compare_url}\n"
    )
    body = (intro
            + _replace_region("", *UPLIFT_MARKERS, uplift_block)
            + "\n"
            + _replace_region("", *TRIAGE_MARKERS, "_no findings — nightly green_")
            + "\n"
            + _replace_region("", *HISTORY_MARKERS, _history_entry(old_sha, new_sha, run_url, prev_body)))
    return body


def _history_entry(old_sha: str, new_sha: str, run_url: str, prev_body: str) -> str:
    # Preserve prior history lines if present.
    history_lines: list[str] = []
    if prev_body:
        m = re.search(
            re.escape(HISTORY_MARKERS[0]) + r"(.*?)" + re.escape(HISTORY_MARKERS[1]),
            prev_body,
            re.DOTALL,
        )
        if m:
            history_lines = [l.strip() for l in m.group(1).strip().splitlines() if l.strip()]
    import datetime
    today = datetime.date.today().isoformat()
    new_line = f"- {today}: advance {old_sha[:7]} → {new_sha[:7]} ([run]({run_url}))"
    history_lines.insert(0, new_line)
    # Cap to 20 entries
    history_lines = history_lines[:20]
    return "\n".join(history_lines)


def comment(workflow: str, findings_md: str) -> int:
    pr = find_open_pr(workflow)
    if pr is None:
        print(f"[uplift] no open PR for workflow={workflow}; comment is a no-op.")
        return 0
    new_body = _replace_region(pr["body"] or "", *TRIAGE_MARKERS, findings_md.strip() or "_no findings_")
    gh.pr_edit(pr["number"], body=new_body, repo=REPO)
    print(f"[uplift] refreshed triage region on PR #{pr['number']}")
    return 0


# ---------------------------------------------------------------------- CLI

def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_adv = sub.add_parser("advance")
    p_adv.add_argument("--workflow", required=True, choices=("metal", "d2m"))
    p_adv.add_argument("--run-id", required=True)
    p_adv.add_argument("--artifacts-dir", required=True, type=Path)

    p_cmt = sub.add_parser("comment")
    p_cmt.add_argument("--workflow", required=True, choices=("metal", "d2m"))
    p_cmt.add_argument("--findings-md", required=True, help="Markdown to put inside the AUTO-TRIAGE region")

    args = p.parse_args(argv)
    if args.cmd == "advance":
        return advance(args.workflow, args.run_id, args.artifacts_dir)
    if args.cmd == "comment":
        return comment(args.workflow, args.findings_md)
    return 2


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(_main())
