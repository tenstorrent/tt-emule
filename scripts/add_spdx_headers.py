#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Idempotent SPDX header inserter for tt-emule.

Walks the working tree (relative to repo root), and prepends a REUSE 3.0
SPDX header to every source file that doesn't already contain
`SPDX-License-Identifier`. Comment marker is chosen per file extension.
Shebang lines on shell / python scripts are preserved at line 1.

Usage:
    scripts/add_spdx_headers.py [--check] [--dry-run]

    --check    Exit non-zero if any in-scope file lacks a header. No edits.
    --dry-run  Print what would change without modifying files.

The inserter is intentionally conservative:
    - Skips files outside the source set (data, build outputs, vendored).
    - Skips files already containing the SPDX-License-Identifier marker.
    - Errors on unexpected paths under build/ or generated/.
"""

import argparse
import os
import sys
from pathlib import Path

COPYRIGHT = "© 2026 Tenstorrent USA, Inc."
LICENSE = "Apache-2.0"

# Extension → (line comment prefix, supports_shebang).
# `cmake-comment` is treated identically to `# `, listed here for clarity.
EXT_RULES = {
    ".cpp":  ("//", False),
    ".cc":   ("//", False),
    ".cxx":  ("//", False),
    ".c":    ("//", False),
    ".hpp":  ("//", False),
    ".hh":   ("//", False),
    ".h":    ("//", False),
    ".py":   ("#",  True),
    ".sh":   ("#",  True),
    ".bash": ("#",  True),
    ".cmake": ("#", False),
    ".yml":  ("#",  False),
    ".yaml": ("#",  False),
}

BASENAME_RULES = {
    "CMakeLists.txt": ("#", False),
}

# Path prefixes (POSIX-style, repo-relative) excluded from the walk.
SKIP_PREFIXES = (
    ".git/",
    ".github/known-failures",  # data, not source
    "build",                   # build_*, build/
    "generated/",
    ".claude/",
    "docs/",                   # covered by LICENSE-DOCS, not per-file SPDX
)

SPDX_MARKER = "SPDX-License-Identifier"


def comment_prefix_for(path: Path) -> tuple[str, bool] | None:
    """Return (prefix, supports_shebang) for the file, or None to skip."""
    if path.name in BASENAME_RULES:
        return BASENAME_RULES[path.name]
    return EXT_RULES.get(path.suffix)


def header_block(prefix: str) -> str:
    return (
        f"{prefix} SPDX-FileCopyrightText: {COPYRIGHT}\n"
        f"{prefix}\n"
        f"{prefix} SPDX-License-Identifier: {LICENSE}\n"
    )


def should_skip(rel_posix: str) -> bool:
    for p in SKIP_PREFIXES:
        if rel_posix.startswith(p):
            return True
    return False


def insert_header(path: Path, prefix: str, supports_shebang: bool) -> bool:
    """Insert header into `path`. Returns True if the file was modified."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Binary or non-UTF-8 — skip silently.
        return False

    if SPDX_MARKER in text:
        return False

    lines = text.splitlines(keepends=True)
    header = header_block(prefix) + "\n"  # blank line after header

    if supports_shebang and lines and lines[0].startswith("#!"):
        new_text = lines[0] + header + "".join(lines[1:])
    else:
        new_text = header + text

    path.write_text(new_text, encoding="utf-8")
    return True


def walk(repo_root: Path):
    """Yield (relpath, abspath) for in-scope files tracked by git."""
    import subprocess
    out = subprocess.check_output(
        ["git", "ls-files"], cwd=str(repo_root), text=True
    )
    for line in out.splitlines():
        rel = line.strip()
        if not rel or should_skip(rel):
            continue
        abspath = repo_root / rel
        if not abspath.is_file():
            continue
        rule = comment_prefix_for(abspath)
        if rule is None:
            continue
        yield rel, abspath, rule


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if any file lacks SPDX")
    parser.add_argument("--dry-run", action="store_true", help="print but don't modify")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    missing: list[str] = []
    modified: list[str] = []

    for rel, abspath, (prefix, supports_shebang) in walk(repo_root):
        try:
            text = abspath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if SPDX_MARKER in text:
            continue
        missing.append(rel)
        if not args.check and not args.dry_run:
            if insert_header(abspath, prefix, supports_shebang):
                modified.append(rel)

    if args.check:
        if missing:
            print(f"FAIL: {len(missing)} file(s) missing SPDX header:", file=sys.stderr)
            for m in missing:
                print(f"  {m}", file=sys.stderr)
            return 1
        print("OK: all in-scope files have SPDX headers.")
        return 0

    if args.dry_run:
        print(f"Would insert SPDX headers into {len(missing)} file(s):")
        for m in missing:
            print(f"  {m}")
        return 0

    print(f"Inserted SPDX headers into {len(modified)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
