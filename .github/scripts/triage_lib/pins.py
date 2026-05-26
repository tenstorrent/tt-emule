# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read / write the two pin files at repo root.

The on-disk format mirrors what the CI shell already parses:
``grep -vE '^\\s*(#|$)' | head -1 | awk '{print $1}'`` — i.e. the first
non-comment, non-blank line is the SHA. Header comments are preserved on
write.
"""

from __future__ import annotations

from pathlib import Path


def read_pin(path: str | Path) -> str:
    """Return the SHA stored in ``path`` (first non-comment, non-blank line)."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # First token only — matches the awk '{print $1}' behaviour.
            return stripped.split()[0]
    raise ValueError(f"no SHA found in {path}")


def write_pin(path: str | Path, sha: str) -> None:
    """Replace the SHA line in ``path`` with ``sha``, preserving header comments.

    Strictly preserves comment / blank lines that appear *before* the existing
    SHA line. Anything after the SHA line is dropped (the file should only
    contain one SHA — agents that need to append metadata should put it in
    comments above the SHA).
    """
    sha = sha.strip()
    if not _looks_like_sha(sha):
        raise ValueError(f"refusing to write non-SHA value: {sha!r}")

    header: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                header.append(line if line.endswith("\n") else line + "\n")
                continue
            break  # first non-header line ends the header

    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(header)
        fh.write(sha + "\n")


def _looks_like_sha(value: str) -> bool:
    return len(value) == 40 and all(c in "0123456789abcdef" for c in value.lower())
