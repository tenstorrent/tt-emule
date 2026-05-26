# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for uplift_pr — focused on pure-Python helpers (no `gh` calls).

The integration parts (compare_status, pr_list, push, etc.) are exercised in
the dry-run smoke test on a real artifact tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))

# Imported after path tweak.
from uplift_pr import _replace_region, _history_entry, TRIAGE_MARKERS  # type: ignore


def test_replace_region_inserts_when_missing():
    body = "hello"
    out = _replace_region(body, *TRIAGE_MARKERS, "test content")
    assert "<!-- AUTO-TRIAGE-START -->" in out
    assert "<!-- AUTO-TRIAGE-END -->" in out
    assert "test content" in out
    assert "hello" in out


def test_replace_region_preserves_text_outside_markers():
    body = (
        "before\n"
        f"{TRIAGE_MARKERS[0]}\n"
        "old\n"
        f"{TRIAGE_MARKERS[1]}\n"
        "after"
    )
    out = _replace_region(body, *TRIAGE_MARKERS, "new")
    assert "before" in out
    assert "after" in out
    assert "new" in out
    assert "old" not in out


def test_replace_region_idempotent():
    body = ""
    once = _replace_region(body, *TRIAGE_MARKERS, "x")
    twice = _replace_region(once, *TRIAGE_MARKERS, "x")
    assert once.strip() == twice.strip()


def test_history_entry_caps_at_20():
    prev = "<!-- AUTO-HISTORY-START -->\n" + "\n".join(
        f"- old {i}" for i in range(25)
    ) + "\n<!-- AUTO-HISTORY-END -->"
    out = _history_entry("a" * 40, "b" * 40, "https://example/run/1", prev)
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) <= 20
    # newest entry is at the top
    assert lines[0].startswith("- ") and "a" * 7 in lines[0] and "b" * 7 in lines[0]
