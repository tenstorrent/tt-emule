# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Locate the inputs the classifier scans inside a downloaded artifact tree.

The nightly workflows upload artifacts named ``nightly-*regression-results-*``
that, when extracted, contain:

  <root>/<artifact-name>/__w/_temp/{regression.log, d2m-regression.log}
                        and the gtest/pytest XML files
                        and the resolved-SHA text files

This module gives the rest of the agent a stable interface that abstracts
the on-disk layout: it walks the download root, finds the relevant files,
and exposes them via an ``ArtifactSet`` dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class ArtifactSet:
    """Inputs scanned by the classifier for a single nightly run."""

    # The "build log" — pulled from the workflow run's job log (see triage.py).
    # When the build job succeeded but a test failed, this may be empty / None.
    build_log: str | None = None

    # The "regression log" — the artifact's regression.log / d2m-regression.log.
    regression_log: str | None = None

    # The markdown $GITHUB_STEP_SUMMARY produced by the classify step.
    classify_summary: str | None = None

    # Resolved SHAs from the artifact's pin-resolved text files.
    mlir_sha: str | None = None
    metal_sha: str | None = None

    # Absolute paths to per-test XML files (gtest or pytest junit format).
    xml_paths: list[Path] = field(default_factory=list)


def collect(artifacts_dir: Path, *, workflow: str) -> ArtifactSet:
    """Walk ``artifacts_dir`` and return an ``ArtifactSet``.

    ``workflow`` is ``"metal"`` or ``"d2m"`` — controls which log filename to
    look for (``regression.log`` vs ``d2m-regression.log``) and which
    pin-resolved file to read.
    """
    if workflow not in ("metal", "d2m"):
        raise ValueError(f"unknown workflow: {workflow!r}")
    log_name = "regression.log" if workflow == "metal" else "d2m-regression.log"

    out = ArtifactSet()
    for path in _walk(artifacts_dir):
        name = path.name
        if name == log_name:
            out.regression_log = _safe_read(path)
        elif name == "tt-mlir-pin-resolved.txt":
            out.mlir_sha = _safe_read(path, single_line=True)
        elif name == "tt-metal-pin-resolved.txt":
            out.metal_sha = _safe_read(path, single_line=True)
        elif name.endswith(".xml"):
            out.xml_paths.append(path)
        elif name == "build_log.txt":
            # Workflow-YAML drops `gh run view <run_id> --log` here so the
            # build-log patterns (e.g. UMD_VIRTUAL_OVERRIDE_DRIFT) can match.
            out.build_log = _safe_read(path)
        elif name == "classify_summary.md":
            # Workflow-YAML drops the classify step's $GITHUB_STEP_SUMMARY here
            # if extractable; otherwise this stays None and the ALLOWLIST_* and
            # NEW_FAILURES patterns will simply not fire.
            out.classify_summary = _safe_read(path)
    return out


def _walk(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    return (p for p in root.rglob("*") if p.is_file())


def _safe_read(path: Path, *, single_line: bool = False) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return text.strip() if single_line else text
