#!/usr/bin/env python3
"""Query `.claude/references/structure.yaml` for files / symbols.

The structured form lets you do kind- and path-aware lookups that plain
grep can't. The `summary` field on every file entry is preserved
verbatim from the legacy STRUCTURE.md, so falling back to it always
matches what a `grep` would have surfaced.

Usage:
  find_symbol.py <query>                          # name match (exact) + summary contains
  find_symbol.py --kind class <query>             # filter symbol matches by kind
  find_symbol.py --path-prefix include/jit_hw/    # restrict to a subtree
  find_symbol.py --list-paths                     # one path per line
  find_symbol.py --list-symbols [--kind enum]     # one symbol per line (filtered)
  find_symbol.py --summary <query>                # substring search against summary only

Output is TSV: `<path>\\t<kind>\\t<name>` for symbol matches,
              `<path>\\t-\\t<summary>` for summary-only matches.

The script is intentionally dependency-light: stdlib + PyYAML (already a
build/test dep). It can be invoked directly or piped through other CLI tools.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write("find_symbol.py: PyYAML missing — pip install PyYAML\n")
    sys.exit(2)


DEFAULT_INDEX = Path(__file__).resolve().parent.parent / ".claude" / "references" / "structure.yaml"


def load_index(path: Path) -> list[dict]:
    with path.open() as f:
        data = yaml.safe_load(f)
    return data["files"]


def emit_symbol(path: str, kind: str, name: str) -> None:
    print(f"{path}\t{kind}\t{name}")


def emit_summary(path: str, summary: str) -> None:
    print(f"{path}\t-\t{summary}")


def filter_path(files: list[dict], prefix: str | None) -> list[dict]:
    if not prefix:
        return files
    return [f for f in files if f["path"].startswith(prefix)]


def cmd_list_paths(files: list[dict]) -> None:
    for f in files:
        print(f["path"])


def cmd_list_symbols(files: list[dict], kind: str | None) -> None:
    for f in files:
        for s in f.get("symbols", []):
            if kind and s["kind"] != kind:
                continue
            emit_symbol(f["path"], s["kind"], s["name"])


def cmd_summary(files: list[dict], query: str) -> None:
    for f in files:
        if query in f.get("summary", ""):
            emit_summary(f["path"], f.get("summary", ""))


def cmd_query(files: list[dict], query: str, kind: str | None) -> None:
    """Exact-name match across symbols; substring match against summary fallback."""
    sym_hits = 0
    for f in files:
        for s in f.get("symbols", []):
            if kind and s["kind"] != kind:
                continue
            if s["name"] == query:
                emit_symbol(f["path"], s["kind"], s["name"])
                sym_hits += 1
    # Always also surface summary hits — symbol parsing is best-effort, so the
    # summary catches the cases where the structured form mis-classifies
    # (e.g. enums recorded as 'function'). Suppress duplicates by path.
    seen = set()
    if sym_hits:
        seen = {f["path"] for f in files for s in f.get("symbols", [])
                if s["name"] == query and (not kind or s["kind"] == kind)}
    for f in files:
        if f["path"] in seen:
            continue
        if query in f.get("summary", ""):
            emit_summary(f["path"], f.get("summary", ""))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("query", nargs="?", help="Symbol name or summary substring")
    p.add_argument("--kind", help="Filter symbol matches by kind (class/struct/enum/namespace/function/macro/extern_c/fwd/global/using)")
    p.add_argument("--path-prefix", help="Restrict to files whose path starts with this prefix")
    p.add_argument("--list-paths", action="store_true", help="Print every path in the index")
    p.add_argument("--list-symbols", action="store_true", help="Print every symbol (optionally filtered by --kind)")
    p.add_argument("--summary", action="store_true", help="Substring search against the summary field only")
    p.add_argument("--index", default=str(DEFAULT_INDEX), help=f"Path to structure.yaml (default: {DEFAULT_INDEX})")
    args = p.parse_args()

    files = load_index(Path(args.index))
    files = filter_path(files, args.path_prefix)

    if args.list_paths:
        cmd_list_paths(files)
        return 0
    if args.list_symbols:
        cmd_list_symbols(files, args.kind)
        return 0

    if not args.query:
        p.error("query is required (or use --list-paths / --list-symbols)")

    if args.summary:
        cmd_summary(files, args.query)
    else:
        cmd_query(files, args.query, args.kind)
    return 0


if __name__ == "__main__":
    sys.exit(main())
