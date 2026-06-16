#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Generate / maintain `.claude/references/structure.yaml`.

`structure.yaml` is the authoritative file/symbol index of emule's own
`src/` + `include/` tree (queried by `scripts/find_symbol.py`, and consumed by
`scripts/classify_kernels.py` as the "what does emule provide?" source of
truth). This script keeps the `path` / `section` / `symbols` of every entry in
sync with the source, so the index can't silently drift.

Design (see also `.claude/CLAUDE.md`):
  - `symbols`  — REGENERATED from the source by a top-level C++ declaration
                 scanner (see `extract_symbols`). Completeness of the NAMES is
                 the load-bearing property: the classifier's tile / llk_ /
                 ckernel:: denominators come from these names. `kind` is
                 best-effort.
  - `summary`  — EDITORIAL prose, PRESERVED verbatim. Never auto-generated. A
                 new file with no summary gets a `TODO:` sentinel and makes
                 `--check` fail, demanding a human sentence (the reliable
                 `find_symbol` fallback must never be auto-faked).
  - ordering   — existing entry order is preserved; new files are inserted into
                 their section block (path-sorted); deleted files are dropped.

Modes:
  gen_structure.py --check     exit 1 (+ unified diff) if the index is stale
  gen_structure.py --write     regenerate in place
  gen_structure.py --validate  cross-check completeness (summary backticks ⊆
                               symbols; no non-artifact symbol lost vs current)

Dependency-light: stdlib + PyYAML (already a build/test dep).
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write("gen_structure.py: PyYAML missing — pip install PyYAML\n")
    sys.exit(2)

REPO = Path(__file__).resolve().parent.parent
INDEX = REPO / ".claude" / "references" / "structure.yaml"
TODO_SENTINEL = "TODO: one-line summary (auto-inserted by gen_structure.py — replace me)"


# ── source enumeration ───────────────────────────────────────────────────────
def iter_source_files() -> list[str]:
    """Repo-relative paths of every indexed source file: src/*.cpp + include/**.{h,hpp}."""
    paths: list[str] = []
    for p in sorted((REPO / "src").rglob("*.cpp")):
        paths.append(p.relative_to(REPO).as_posix())
    for ext in ("*.h", "*.hpp"):
        for p in (REPO / "include").rglob(ext):
            paths.append(p.relative_to(REPO).as_posix())
    return paths


def section_of(path: str) -> str:
    """Directory of `path` with a trailing slash (e.g. `include/jit_hw/api/`)."""
    d = path.rsplit("/", 1)[0]
    return d + "/"


# ── symbol extraction ────────────────────────────────────────────────────────
_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_comments(text: str) -> str:
    # Drop block comments first (may span lines), then line comments. String
    # literals are not stripped — the decl scanner keys off line starts, where
    # stray `//` inside a string at column 0 does not occur in this tree.
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))


# Leading decl noise consumed before the real declarator: attributes, storage /
# inline / linkage specifiers, the emule `ALWI`/`FORCE_INLINE` macros, etc.
_LEADING = (
    r"(?:(?:\[\[[^\]]*\]\]|ALWI|FORCE_INLINE|inline|static|constexpr|consteval|"
    r"constinit|virtual|explicit|friend|__attribute__\s*\(\([^)]*\)\)|"
    r"thread_local|extern)\s+)*"
)
# A C++ identifier (the maximal run — so `add_binary_tile_init` is never
# truncated to `_init`).
_IDENT = r"[A-Za-z_]\w*"

_RE_NAMESPACE = re.compile(r"^\s*namespace\s+(" + _IDENT + r")\s*\{")
_RE_ENUM = re.compile(r"^\s*enum\s+(?:class\s+|struct\s+)?(" + _IDENT + r")\b")
_RE_RECORD = re.compile(r"^\s*(struct|class|union)\s+(" + _IDENT + r")\b(.*)$")
_RE_MACRO = re.compile(r"^\s*#\s*define\s+(" + _IDENT + r")\b")
_RE_USING = re.compile(
    r"^\s*using\s+(?:(?:typename\s+)?(?:[\w]+::|::)*)?(" + _IDENT + r")\s*[=;]"
)
_RE_EXTERN_C_OPEN = re.compile(r'^\s*extern\s+"C"\s*\{?\s*$')
# Inline `extern "C" <ret> <name>(...)` — a single-line linkage-tagged decl.
_RE_EXTERN_C_FUNC = re.compile(r'^\s*extern\s+"C"\s+.+?\b(' + _IDENT + r")\s*\(")
# A namespace-scope function declaration/definition: optional template head on a
# prior line, leading specifiers, a return type, then NAME( . The return-type
# token before NAME is what defeated the old extractor; we require at least one
# type token (so `NAME(args)` calls without a return type are not matched).
_RE_FUNC = re.compile(
    r"^\s*" + _LEADING
    + r"(?:[\w:<>,\*&\s~]+?\s+|[\w:<>,]+::\s*)"   # return type / qualified head
    + r"(~?" + _IDENT + r")\s*\("
)
# A namespace-scope variable/constant: `constexpr T NAME = ...` / `T NAME;` etc.
# Only recorded at top level; kind `global`.
_RE_GLOBAL = re.compile(
    r"^\s*" + _LEADING
    + r"[\w:<>,\*&\s]+?\s+(" + _IDENT + r")\s*(?:\[[^\]]*\])?\s*(?:=|;|\{)"
)

# Control-flow / non-decl keywords that _RE_FUNC / _RE_GLOBAL could misfire on.
_KW_STOP = {
    "if", "for", "while", "switch", "return", "else", "do", "case", "sizeof",
    "static_assert", "using", "namespace", "template", "typedef", "typename",
    "friend", "enum", "struct", "class", "union", "public", "private",
    "protected", "operator", "throw", "new", "delete", "co_await", "co_return",
    "co_yield", "requires", "concept", "decltype", "static_cast", "const_cast",
    "reinterpret_cast", "dynamic_cast",
}


def extract_symbols(text: str) -> list[dict]:
    """Top-level (namespace-scope) declarations of `text`, in declaration order.

    Brace tracking distinguishes namespace braces (transparent — symbols inside
    a `namespace X {}` are still top-level) from record/function/block braces
    (opaque — their contents are members/locals, not indexed). Deduped by
    (kind, name), first occurrence wins.
    """
    src = _strip_comments(text)
    lines = src.split("\n")

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, name: str) -> None:
        key = (kind, name)
        if name and key not in seen:
            seen.add(key)
            out.append({"kind": kind, "name": name})

    # Brace stack: each frame is "ns" (transparent) or "blk" (opaque).
    stack: list[str] = []
    extern_c_depth = -1            # brace depth at which an `extern "C" {` opened
    pending_template = False       # previous non-blank line was a template head
    paren_depth = 0                # unbalanced `(` carried across lines

    def opaque_depth() -> int:
        return sum(1 for f in stack if f == "blk")

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        # Only match declarations at the start of a statement — never inside a
        # still-open parameter list. A multi-line decl like
        #   void Enqueue(Buffer& b,
        #                bool blocking = true);
        # would otherwise record the continuation `bool blocking = true` as a
        # `global`. `paren_depth` is computed BEFORE this line's parens so the
        # line that OPENS the list is still matched as the function/decl.
        at_top = opaque_depth() == 0 and paren_depth == 0
        paren_depth = max(0, paren_depth + line.count("(") - line.count(")"))

        if at_top:
            m = _RE_NAMESPACE.match(line)
            if m:
                add("namespace", m.group(1))
                stack.append("ns")
                # account for any extra braces opened on the same line
                _push_extra_braces(line, stack, after_first_open=True)
                pending_template = False
                i += 1
                continue

            m = _RE_MACRO.match(line)
            if m:
                add("macro", m.group(1))
                pending_template = False
                i += 1
                continue

            m = _RE_USING.match(line)
            if m:
                add("using", m.group(1))
                pending_template = False
                i += 1
                continue

            if _RE_EXTERN_C_OPEN.match(line):
                # `extern "C" {` — transparent like a namespace; mark its names.
                stack.append("ns")
                extern_c_depth = len(stack)
                pending_template = False
                i += 1
                continue

            mx = _RE_EXTERN_C_FUNC.match(line)
            if mx and mx.group(1) not in _KW_STOP:
                add("extern_c", mx.group(1))
                _update_braces(line, stack)
                pending_template = False
                i += 1
                continue

            m = _RE_ENUM.match(line)
            if m:
                # fwd decl (`enum class X;`) vs definition (`... {`).
                kind = "enum"
                add(kind, m.group(1))
                _update_braces(line, stack)
                pending_template = False
                i += 1
                continue

            m = _RE_RECORD.match(line)
            if m:
                rec_kind, name, rest = m.group(1), m.group(2), m.group(3)
                # forward decl: `struct X;` with no body/base.
                is_fwd = rest.strip().startswith(";") or (";" in rest and "{" not in rest)
                add("fwd" if is_fwd else rec_kind, name)
                _update_braces(line, stack)
                pending_template = False
                i += 1
                continue

            # Consume an inline `template<...>` head; if it spans lines, defer.
            eff, closed = _strip_leading_template(line)
            if not closed:
                pending_template = True
                _update_braces(line, stack)
                i += 1
                continue

            # function decl/def — must precede the global rule (a func sig also
            # ends in `)` then `{` or `;`). Matched on the template-stripped line.
            mf = _RE_FUNC.match(eff)
            if mf and _first_word(eff) not in _KW_STOP and mf.group(1) not in _KW_STOP:
                ec = "extern_c" if (extern_c_depth != -1 and len(stack) >= extern_c_depth) else "function"
                add(ec, mf.group(1))
                _update_braces(line, stack)
                pending_template = False
                i += 1
                continue

            mg = _RE_GLOBAL.match(eff)
            if mg and _first_word(eff) not in _KW_STOP and mg.group(1) not in _KW_STOP:
                ec = "extern_c" if (extern_c_depth != -1 and len(stack) >= extern_c_depth) else "global"
                add(ec, mg.group(1))
                _update_braces(line, stack)
                pending_template = False
                i += 1
                continue

        # default: just track braces.
        _update_braces(line, stack)
        if extern_c_depth != -1 and len(stack) < extern_c_depth:
            extern_c_depth = -1
        pending_template = False
        i += 1

    return out


def _first_word(line: str) -> str:
    m = re.match(r"\s*(" + _IDENT + r")", line)
    return m.group(1) if m else ""


def _strip_leading_template(line: str) -> tuple[str, bool]:
    """If `line` opens with a `template<...>` head, return (remainder, closed).

    `closed` is False when the angle bracket never balances on this line (a
    multi-line template head). Returns (line, True) when there is no template
    head. Indentation is preserved so `^\\s*` anchors still fire.
    """
    m = re.match(r"^(\s*)template\s*<", line)
    if not m:
        return line, True
    indent = m.group(1)
    depth = 1
    i = m.end()
    n = len(line)
    while i < n and depth:
        c = line[i]
        depth += (c == "<") - (c == ">")
        i += 1
    if depth:
        return line, False  # head spans multiple lines
    return indent + line[i:], True


def _push_extra_braces(line: str, stack: list[str], after_first_open: bool) -> None:
    # Count braces on `line`, skipping the first `{` (already pushed by caller).
    seen_first = not after_first_open
    for ch in line:
        if ch == "{":
            if not seen_first:
                seen_first = True
                continue
            stack.append("blk")
        elif ch == "}":
            if stack:
                stack.pop()


def _update_braces(line: str, stack: list[str]) -> None:
    for ch in line:
        if ch == "{":
            stack.append("blk")
        elif ch == "}":
            if stack:
                stack.pop()


# ── YAML emit (byte-stable) ──────────────────────────────────────────────────
class _Sym(dict):
    """A {kind, name} mapping rendered in flow style."""


def _repr_sym(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)


yaml.add_representer(_Sym, _repr_sym)


def dump_index(files: list[dict]) -> str:
    out_files = []
    for f in files:
        e: dict = {"path": f["path"], "section": f["section"]}
        if "summary" in f:
            e["summary"] = f["summary"]
        syms = f.get("symbols")
        if syms:
            e["symbols"] = [_Sym({"kind": s["kind"], "name": s["name"]}) for s in syms]
        out_files.append(e)
    return yaml.dump({"files": out_files}, default_flow_style=False, sort_keys=False,
                     width=200, allow_unicode=True)


# ── build the regenerated index ──────────────────────────────────────────────
def load_existing() -> list[dict]:
    with INDEX.open() as f:
        return yaml.safe_load(f)["files"]


def build_index() -> tuple[list[dict], list[str]]:
    """Return (entries, new_files_without_summary)."""
    existing = load_existing()
    by_path = {e["path"]: e for e in existing}
    order = [e["path"] for e in existing]
    on_disk = set(iter_source_files())

    missing_summary: list[str] = []
    entries: dict[str, dict] = {}

    def make_entry(path: str) -> dict:
        text = (REPO / path).read_text(errors="ignore")
        is_new = path not in by_path
        # Brand-new files must get a human summary (TODO sentinel fails --check).
        # Files already in the index keep their summary verbatim — even an empty
        # one is grandfathered (don't churn / don't fail an existing entry).
        if is_new:
            summary = TODO_SENTINEL
            missing_summary.append(path)
        else:
            summary = by_path[path].get("summary", "")
        e = {"path": path, "section": section_of(path), "symbols": extract_symbols(text)}
        if summary or is_new:
            e["summary"] = summary
        else:
            e["summary"] = ""
        return e

    # Preserve existing order for files still on disk; drop deleted ones.
    final_order: list[str] = []
    for path in order:
        if path in on_disk:
            entries[path] = make_entry(path)
            final_order.append(path)

    # Insert new files into their section block, path-sorted within the block.
    new_paths = sorted(on_disk - set(final_order))
    for path in new_paths:
        entries[path] = make_entry(path)
        sec = section_of(path)
        # find the last index of an entry in the same section; insert after it
        # path-sorted, else append.
        insert_at = len(final_order)
        block = [j for j, p in enumerate(final_order) if entries[p]["section"] == sec]
        if block:
            placed = False
            for j in block:
                if path < final_order[j]:
                    insert_at = j
                    placed = True
                    break
            if not placed:
                insert_at = block[-1] + 1
        final_order.insert(insert_at, path)

    return [entries[p] for p in final_order], missing_summary


# ── modes ────────────────────────────────────────────────────────────────────
def cmd_write() -> int:
    entries, missing = build_index()
    INDEX.write_text(dump_index(entries))
    print(f"gen_structure: wrote {len(entries)} entries to {INDEX.relative_to(REPO)}")
    if missing:
        print(f"  NOTE: {len(missing)} new file(s) need a summary (TODO sentinel inserted):")
        for p in missing:
            print(f"    {p}")
        return 1
    return 0


def cmd_check() -> int:
    entries, missing = build_index()
    want = dump_index(entries)
    have = INDEX.read_text()
    if want == have and not missing:
        print("gen_structure: structure.yaml is up to date.")
        return 0
    if missing:
        sys.stderr.write(
            "gen_structure: new file(s) without a summary — add one and run --write:\n")
        for p in missing:
            sys.stderr.write(f"    {p}\n")
    if want != have:
        sys.stderr.write("gen_structure: structure.yaml is STALE. Run `python3 scripts/gen_structure.py --write`.\n")
        diff = difflib.unified_diff(have.splitlines(), want.splitlines(),
                                    "structure.yaml (current)", "structure.yaml (regenerated)",
                                    lineterm="")
        sys.stderr.write("\n".join(list(diff)[:200]) + "\n")
    return 1


def cmd_validate() -> int:
    """Completeness cross-checks (no source-of-truth file is trusted blindly)."""
    existing = {e["path"]: e for e in load_existing()}
    entries, _ = build_index()
    rc = 0

    # (a) every backticked identifier in a summary should appear in symbols.
    ident_in_ticks = re.compile(r"`([A-Za-z_]\w*)`")
    for e in entries:
        names = {s["name"] for s in e.get("symbols", [])}
        summary = e.get("summary", "")
        ticked = set(ident_in_ticks.findall(summary))
        # Only flag plausible MISSED top-level names. Summaries legitimately
        # mention things that are not this file's top-level symbols: member
        # functions (`endpoint.get_noc_unicast_addr`), cross-file references,
        # and `_init`/`_tile`-style shorthand fragments (leading underscore).
        # Skip those to keep --validate a low-noise advisory.
        missing = {t for t in ticked
                   if len(t) > 2 and t not in names
                   and not t.startswith("_")
                   and not t.endswith(("_t", "_e"))
                   and t not in {"NONE", "uint32_t"}}
        if missing and e.get("symbols"):
            rc = 1
            print(f"[summary≠symbols] {e['path']}: in summary but not symbols: {sorted(missing)}")

    # (b) no non-artifact symbol from the OLD index is lost (catches over-pruning).
    for path, old in existing.items():
        new = next((e for e in entries if e["path"] == path), None)
        if not new:
            continue
        old_names = {s["name"] for s in (old.get("symbols") or [])}
        new_names = {s["name"] for s in (new.get("symbols") or [])}
        # `_init` etc. were artifacts; only flag losses of real-looking names.
        lost = {nm for nm in old_names - new_names if nm != "_init"}
        if lost:
            rc = 1
            print(f"[lost-symbol]   {path}: in old index but not regenerated: {sorted(lost)}")

    if rc == 0:
        print("gen_structure --validate: OK (summaries covered, no real symbol lost).")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="regenerate structure.yaml in place")
    g.add_argument("--check", action="store_true", help="exit 1 (+diff) if stale")
    g.add_argument("--validate", action="store_true", help="completeness cross-checks")
    args = ap.parse_args()
    if args.write:
        return cmd_write()
    if args.check:
        return cmd_check()
    return cmd_validate()


if __name__ == "__main__":
    sys.exit(main())
