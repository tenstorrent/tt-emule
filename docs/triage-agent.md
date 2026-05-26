<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Nightly Triage + Uplift Agent — ops notes

This document is the operational reference for the automation defined in
`.github/workflows/triage-and-uplift.yml`. The plan that motivates it lives
in the PR description; this page is what you read when you're on call and the
agent does something surprising.

## What it does

After either nightly workflow completes — **Nightly Metal Upstream Regression**
(07:00 UTC) or **Nightly D2M Upstream** (08:00 UTC) — the agent fires.

- **On failure**: pattern-matches the build log, regression log, and classify
  step's markdown summary, then:
  1. Files or bumps a tracking GitHub issue per *signature* (a hash of the
     failure category + key captured tokens).
  2. Replaces the `<!-- AUTO-TRIAGE -->` region of the matching open uplift
     PR's body with the current finding list.
- **On success**: advances the corresponding pin file
  (`tt-metal-pin.txt` for the Metal nightly, `tt-mlir-pin.txt` for D2M) on a
  long-lived branch (`auto/uplift-metal` or `auto/uplift-mlir`) and opens or
  refreshes the PR. Humans review and merge.

The two paths are **independent**. Metal nightly never touches `tt-mlir-pin.txt`
and vice versa. (`tt-metal-pin.txt` runs ahead of tt-mlir's internal
`TT_METAL_VERSION` by design.)

## Required setup (admin)

Before the agent is useful in production, an admin must:

1. **Create the bot PAT.** Classic PAT, `repo` scope, owned by a service
   account (e.g. `tt-emule-bot`). Store as repo secret `UPLIFT_BOT_PAT`.
   Reason: PRs opened by the default `GITHUB_TOKEN` do not trigger downstream
   `pr-*-regression.yml` workflows — the PAT path bypasses that recursion
   guard so the uplift PR's CI runs automatically.
2. **Create the labels** the agent uses. `triage_issues.py` and `uplift_pr.py`
   call `gh label create -f` so they're auto-created on first use, but pick
   readable colours up front:
   - `auto-triage` — set on every issue the agent creates
   - `uplift` — set on both uplift PRs
   - `automation` — set on every PR the agent creates
   - `uplift-metal`, `uplift-mlir` — path discriminators
   - `upstream-blocker` — added when a pattern flags an upstream change
   - `flake` — added when a pattern flags a known flake
3. **Provision the kill switch.** Set the repo variable
   `TRIAGE_AGENT_DISABLED=1` initially. Both jobs short-circuit when this is
   non-empty. Unset it once you've validated the agent in dry-run.

## How to dry-run

You can verify everything end-to-end without touching the live repo:

```bash
# Pick a recent nightly run id
NIGHTLY_ID=$(gh run list --workflow nightly-d2m-upstream.yml --limit 1 \
  --json databaseId -q '.[0].databaseId')

# Download artifacts
mkdir -p _triage_dryrun && cd _triage_dryrun
gh run download "$NIGHTLY_ID" --repo tenstorrent/tt-emule \
  --dir _art --pattern 'nightly-*regression-results-*'

# Dry-run the agent (no GitHub writes happen)
DRY_RUN=1 python3 /path/to/tt-emule/.github/scripts/triage.py \
  --workflow d2m \
  --run-id "$NIGHTLY_ID" \
  --conclusion failure \
  --artifacts-dir _art \
  --patterns /path/to/tt-emule/.github/scripts/triage-patterns.yaml
```

`DRY_RUN=1` is honoured by every `gh` mutation wrapper in
`triage_lib/gh.py`. The script logs the actions it *would* take.

## How to re-trigger against a historic run

```bash
gh workflow run triage-and-uplift.yml \
  -f run_id=<historic-nightly-run-id> \
  -f nightly=d2m \
  -f conclusion=failure
```

Useful for back-testing a pattern change against the last 30 days of artifacts.

## When the agent does something wrong

### "It opened a duplicate issue!"
Check `triage_lib/signatures.py::normalize`. The hash treats two strings as
equal only after path/address/timestamp stripping. If a new failure mode is
producing slightly different captures across runs, extend the normaliser and
add a regression test in `tests/triage/test_signatures.py`.

### "It posted findings on the wrong uplift PR!"
The path is decided in the workflow YAML from
`github.event.workflow_run.name`. A Metal nightly always lands on
`auto/uplift-metal`; a D2M nightly on `auto/uplift-mlir`. If the wrong one is
being touched, check that the nightly workflow's `name:` field hasn't
changed.

### "It advanced the pin to a SHA we wanted to skip!"
The agent refuses backward / sideways moves but does not refuse "ahead but
broken". If today's main tip is known-bad, the agent will happily advance.
Two ways to recover:
- Close the PR; the next successful nightly will reopen it at the new SHA.
- Set `TRIAGE_AGENT_DISABLED=1` while you work around upstream.

### "The agent isn't firing"
1. Check `vars.TRIAGE_AGENT_DISABLED` is not `1`.
2. Check the nightly workflow's name matches what the agent's
   `workflow_run.workflows` filter expects.
3. Check `UPLIFT_BOT_PAT` is set (uplift job only — triage will still run).

## Files

| Path | Role |
|---|---|
| `.github/workflows/triage-and-uplift.yml` | Trigger + orchestration |
| `.github/scripts/triage.py` | Entry point — classify, then dispatch effects |
| `.github/scripts/uplift_pr.py` | Manages `auto/uplift-*` branches and PRs |
| `.github/scripts/triage_issues.py` | Upserts deduped tracking issues |
| `.github/scripts/triage-patterns.yaml` | Pattern library (data) |
| `.github/scripts/triage-patterns.schema.json` | Pattern schema |
| `.github/scripts/triage_lib/` | Shared helpers (classify, gh, pins, render, signatures, artifacts) |
| `tests/triage/` | pytest suite |

## What this PR does NOT do

These are intentional non-goals (see plan):

- No auto-approve or auto-merge of uplift PRs.
- No auto-closing of tracking issues — humans close.
- No allowlist edits — the agent surfaces drift; you edit
  `.github/known-failures-*.txt`.
- No PRs / issues against upstream repos.
- No LLM classification. (TT_CHAT / `ai_summary` was considered and rejected.
  If the unmatched-failure bucket grows we'll revisit with a different
  external LLM in a separate PR.)
- No Slack integration in this PR — that's Phase 3, after the in-flight
  Slack-alerts PR lands.
