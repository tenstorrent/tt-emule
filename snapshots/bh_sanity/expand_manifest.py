#!/usr/bin/env python3
# Expand each BH post-commit manifest entry into a flat list of test files.
#
# Reads tt-metal/tests/pipeline_reorg/ttnn-tests.yaml. For each entry that
# matches the BH post-commit predicate (merge_gate falsy AND bh_p150b_civ2
# in skus), parse its `cmd` into:
#   - explicit ::test_X references → that file goes into the entry's list
#   - directory references → recursive walk for test_*.py files
#
# Output: snapshots/bh_sanity/expanded_manifest.yaml
#   {entry_slug: [{file, source}, ...], ...}
# Plus a flat all_files list for the verifier to cross-check.

import re
import sys
from pathlib import Path

import yaml

MANIFEST = Path("/localdev/arminale/tt-metal/tests/pipeline_reorg/ttnn-tests.yaml")
TT_METAL = Path("/localdev/arminale/tt-metal")
OUT_PATH = Path("/localdev/arminale/tt-emule/snapshots/bh_sanity/expanded_manifest.yaml")
TARGET_SKU = "bh_p150b_civ2"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def expand_cmd(cmd: str):
    """Return list of {file, source} dicts from a single pytest cmd line.

    Heuristic shell parse — pulls tokens that look like test paths:
      - ends in `.py` and starts with `tests/` → explicit file ref
      - contains `::` → explicit `::test_X` form
      - starts with `tests/` and points at a dir → directory ref to expand
    """
    tokens = cmd.split()
    files_set = set()
    out = []
    for tok in tokens:
        # Strip surrounding quotes
        tok = tok.strip("'\"")
        if "::" in tok:
            file_part, _, _ = tok.partition("::")
            if file_part.endswith(".py") and file_part.startswith("tests/"):
                if file_part not in files_set:
                    files_set.add(file_part)
                    out.append({"file": file_part, "source": "explicit"})
        elif tok.endswith(".py") and tok.startswith("tests/"):
            if tok not in files_set:
                files_set.add(tok)
                out.append({"file": tok, "source": "explicit"})
        elif tok.startswith("tests/") and not tok.endswith(".py"):
            # Directory ref — recursively enumerate test_*.py
            dir_abs = TT_METAL / tok
            if not dir_abs.is_dir():
                continue
            for p in sorted(dir_abs.rglob("test_*.py")):
                rel = p.relative_to(TT_METAL).as_posix()
                if rel not in files_set:
                    files_set.add(rel)
                    out.append({"file": rel, "source": f"directory {tok}"})
    return out


def main():
    with MANIFEST.open() as f:
        entries = yaml.safe_load(f)

    expanded = {}
    all_files = set()
    selected = 0
    for e in entries:
        if e.get("merge_gate", False):
            continue
        skus = e.get("skus") or {}
        if TARGET_SKU not in skus:
            continue
        selected += 1
        s = slug(e["name"])
        files = expand_cmd(e.get("cmd", ""))
        expanded[s] = {
            "name": e["name"],
            "files": files,
            "file_count": len(files),
        }
        all_files.update(f["file"] for f in files)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        yaml.safe_dump({
            "manifest": str(MANIFEST),
            "target_sku": TARGET_SKU,
            "selected_entries": selected,
            "total_files_unique": len(all_files),
            "entries": expanded,
            "all_files": sorted(all_files),
        }, f, sort_keys=False, width=200)

    print(f"wrote {OUT_PATH}")
    print(f"selected entries: {selected}")
    print(f"unique test files: {len(all_files)}")
    print()
    print("Per-entry counts:")
    for s, data in expanded.items():
        print(f"  {data['file_count']:4d}  {s}")


if __name__ == "__main__":
    main()
