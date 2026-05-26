# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pattern-driven classifier for nightly run logs.

Inputs come from :mod:`triage_lib.artifacts`. Output is a list of
:class:`Finding` records (one per matched pattern, deduped on signature).

The pattern library lives in ``.github/scripts/triage-patterns.yaml`` and
follows the schema documented at ``.github/scripts/triage-patterns.schema.json``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import signatures
from .artifacts import ArtifactSet
from .render import render

# Source identifiers — matched against the YAML pattern's ``sources`` list.
SOURCES = ("build_log", "regression_log", "classify_summary")


@dataclass
class Finding:
    """One classifier output."""

    pattern_id: str
    category: str
    source: str
    captures: dict[str, Any]
    evidence: str
    signature: str
    summary: str
    suggested_fix: str = ""
    upstream_link: str = ""
    issue_title: str = ""
    labels: list[str] = field(default_factory=list)


def classify(inputs: ArtifactSet, patterns_path: Path | str, *, workflow: str) -> list[Finding]:
    """Return findings produced by running every loaded pattern against ``inputs``.

    Findings are deduplicated by ``(source, signature)``; at most one finding
    per pair is emitted. Patterns run in descending ``priority`` order so the
    most specific patterns "consume" their evidence first.
    """
    patterns = _load_patterns(patterns_path)
    text_by_source = {
        "build_log": inputs.build_log or "",
        "regression_log": inputs.regression_log or "",
        "classify_summary": inputs.classify_summary or "",
    }

    seen: set[tuple[str, str]] = set()
    findings: list[Finding] = []

    for pat in sorted(patterns, key=lambda p: -p.get("priority", 0)):
        for source in pat.get("sources", []):
            if source not in text_by_source:
                continue
            text = text_by_source[source]
            if not text:
                continue
            match = re.search(pat["match"], text, re.MULTILINE | re.DOTALL)
            if not match:
                continue
            captures = _collect_captures(pat, text, match)
            if _under_threshold(pat, captures):
                continue
            sig = _compute_signature(pat, captures)
            key = (source, sig)
            if key in seen:
                continue
            seen.add(key)
            evidence = _evidence_window(text, match, pat.get("after_match_window", 5))
            ctx = {
                **captures,
                "signature_hash": sig,
                "upstream_link": pat.get("upstream_link", ""),
                "workflow": workflow,
            }
            summary = render(pat.get("summary_template", ""), ctx) if pat.get("summary_template") else ""
            issue_title = render(pat.get("issue_title_template", ""), ctx) if pat.get("issue_title_template") else ""
            suggested_fix = render(pat.get("suggested_fix", ""), ctx) if pat.get("suggested_fix") else ""
            findings.append(Finding(
                pattern_id=pat["id"],
                category=pat["category"],
                source=source,
                captures=captures,
                evidence=evidence,
                signature=sig,
                summary=summary,
                suggested_fix=suggested_fix,
                upstream_link=pat.get("upstream_link", ""),
                issue_title=issue_title,
                labels=list(pat.get("labels", [])),
            ))
    return findings


def _load_patterns(patterns_path: Path | str) -> list[dict[str, Any]]:
    with open(patterns_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "patterns" not in data:
        raise ValueError(f"{patterns_path}: expected top-level mapping with 'patterns' key")
    return list(data["patterns"])


def _collect_captures(pat: Mapping[str, Any], text: str, primary_match: re.Match[str]) -> dict[str, Any]:
    captures: dict[str, Any] = {}
    # Include named groups from the main ``match`` regex itself.
    captures.update({k: v for k, v in primary_match.groupdict().items() if v is not None})
    for spec in pat.get("capture", []):
        regex = re.compile(spec["regex"], re.MULTILINE)
        if spec.get("multi"):
            captures[spec["name"]] = [_extract_capture(m, spec["name"]) for m in regex.finditer(text)]
        else:
            m = regex.search(text)
            if m is not None:
                captures[spec["name"]] = _extract_capture(m, spec["name"])
    return captures


def _extract_capture(m: re.Match[str], name: str) -> str:
    """Pick the best representative value out of ``m`` for a capture named ``name``.

    Preference order:
      1. A named group matching ``name`` exactly (e.g. ``capture.name: foo`` and
         ``(?P<foo>...)`` in the regex).
      2. The single named group, if the regex has exactly one (this handles the
         common case where the capture name is plural but the inner group is
         singular, e.g. ``failures`` ↔ ``(?P<failure>...)``).
      3. Numbered group 1.
      4. Group 0 (whole match).
    """
    gd = m.groupdict()
    if name in gd and gd[name]:
        return gd[name]
    non_empty = [v for v in gd.values() if v]
    if len(non_empty) == 1:
        return non_empty[0]
    try:
        v = m.group(1)
        if v:
            return v
    except IndexError:
        pass
    return m.group(0)


def _under_threshold(pat: Mapping[str, Any], captures: Mapping[str, Any]) -> bool:
    threshold = pat.get("threshold")
    if not threshold:
        return False
    field_name = threshold.get("field")
    min_count = int(threshold.get("min_count", 1))
    if field_name not in captures:
        return True
    value = captures[field_name]
    count = len(value) if isinstance(value, (list, tuple)) else 1
    return count < min_count


def _compute_signature(pat: Mapping[str, Any], captures: Mapping[str, Any]) -> str:
    fixed = pat.get("fixed_signature")
    if fixed:
        return f"trg-{signatures.signature(pat['category'], {'_': fixed})[4:]}"
    groups = pat.get("signature_groups", [])
    sig_captures = {}
    for g in groups:
        val = captures.get(g)
        if isinstance(val, list):
            val = ",".join(sorted(map(str, val)))
        sig_captures[g] = str(val) if val is not None else ""
    return signatures.signature(pat["category"], sig_captures)


def _evidence_window(text: str, match: re.Match[str], lines_after: int) -> str:
    """Return ``lines_after`` lines of context following ``match``."""
    lines = text.splitlines()
    if not lines:
        return match.group(0)
    # Find the line index of the match
    pre = text[:match.start()].count("\n")
    end = min(len(lines), pre + 1 + max(0, lines_after))
    return "\n".join(lines[pre:end])
