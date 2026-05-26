# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stable signature hashing for triage findings.

A signature collapses many concrete failure instances into one identifier
so the issue-upsert step can dedup.  The hash is deliberately short
(12 hex chars) so it fits in issue titles and is easy to grep for.
"""

from __future__ import annotations

import hashlib
import re
from typing import Mapping


# Normalisation passes applied to capture values before hashing. Order matters
# (apply the more specific ones first). Each entry is (regex, replacement).
_NORMALIZE_RULES: list[tuple[re.Pattern[str], str]] = [
    # GitHub Actions absolute workspace path: /__w/<repo>/<repo>/...
    (re.compile(r"/__w/[^/]+/[^/]+/"), "<WS>/"),
    # Common runner home roots
    (re.compile(r"/opt/runner/_work/[^/]+/[^/]+/"), "<WS>/"),
    (re.compile(r"/home/runner/work/[^/]+/[^/]+/"), "<WS>/"),
    # Hex pointer addresses
    (re.compile(r"0x[0-9a-fA-F]{6,}"), "0xADDR"),
    # File:line:column suffixes  (must come after path stripping)
    (re.compile(r":\d+:\d+"), ":L:C"),
    # Standalone trailing line numbers in stack frames
    (re.compile(r"\bline \d+\b"), "line L"),
    # ISO-8601 timestamps
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "<TS>"),
    # Collapse repeated whitespace
    (re.compile(r"\s+"), " "),
]


def normalize(value: str) -> str:
    """Return a stable, run-independent form of ``value``.

    Applied to every captured group before hashing so that the same logical
    failure produces the same signature across runners, dates and absolute
    paths.
    """
    out = value
    for pat, repl in _NORMALIZE_RULES:
        out = pat.sub(repl, out)
    return out.strip()


def signature(category: str, captures: Mapping[str, str]) -> str:
    """Return the canonical signature hash for a finding.

    The hash format is ``trg-<12 hex>``; it is derived from
    ``"<category>|k1=normalize(v1)|k2=normalize(v2)|..."`` with keys sorted.
    """
    parts = [category]
    for key in sorted(captures.keys()):
        parts.append(f"{key}={normalize(str(captures[key]))}")
    canonical = "|".join(parts)
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=6).hexdigest()
    return f"trg-{digest}"
