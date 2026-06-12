#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The emule "surface": what does emule's jit_hw layer provide / shadow?

A single, dependency-light leaf module that answers — from the authoritative
`.claude/references/structure.yaml` index — the question both
`scripts/find_symbol.py` (the `--supports` / `--shadows` early-detect probes)
and `scripts/classify_kernels.py` (the layer-1 classifier denominators) need:

  - which `*_tile(s)` compute wrappers does emule ship?      tile_symbols()
  - which `llk_*` shims does emule shadow?                   llk_symbols()
  - which `ckernel::` members does emule provide?            ckernel_members()
  - does emule shadow this #include?                         is_shadowed(inc)
  - what verdict would a kernel using only <sym> land at?    provides(sym)

Because structure.yaml is regenerated + check-gated by
`scripts/gen_structure.py`, these sets stay in sync with the headers — so
"what emule provides" is consistent across the classifier, find_symbol, and the
compute-llk-bringup workflow.

This module imports nothing from the other scripts (it inlines its own tiny
YAML loader) so it stays a leaf: find_symbol and classify_kernels depend on it,
not the other way round.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("emule_surface: PyYAML missing — pip install PyYAML") from e

REPO = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = REPO / ".claude" / "references" / "structure.yaml"
JITHW_PREFIX = "include/jit_hw/"

# Symbol-name shapes the classifier gates on.
TILE_RE = re.compile(r"^[a-z][a-z0-9_]*_tiles?$")
LLK_RE = re.compile(r"^_?llk_[a-z]\w*$")


class Surface:
    """Resolved emule surface, parsed once from a structure.yaml index."""

    def __init__(self, files: list[dict]):
        self.files = files
        self.jithw = [f for f in files if f["path"].startswith(JITHW_PREFIX)]
        self.jithw_paths = {f["path"] for f in self.jithw}

        self._names: set[str] = set()
        self._owner: dict[str, list[str]] = {}
        ckernel_files = []
        for f in self.jithw:
            syms = f.get("symbols") or []
            if any(s["kind"] == "namespace" and s["name"] == "ckernel" for s in syms):
                ckernel_files.append(f)
            for s in syms:
                self._names.add(s["name"])
                self._owner.setdefault(s["name"], []).append(f["path"])
        # ckernel members: names declared in a file that opens `namespace
        # ckernel` (gates `ckernel::x` references). Over-approximates only when
        # such a file also declares a non-ckernel namespace — acceptable: an
        # over-broad ckernel set never wrongly rules a kernel out.
        self._ckernel: set[str] = set()
        for f in ckernel_files:
            for s in (f.get("symbols") or []):
                if s["kind"] != "namespace":
                    self._ckernel.add(s["name"])

    # ── the three classifier denominators ──────────────────────────────────
    def tile_symbols(self) -> set[str]:
        return {n for n in self._names if TILE_RE.match(n)}

    def llk_symbols(self) -> set[str]:
        return {n for n in self._names if LLK_RE.match(n)}

    def ckernel_members(self) -> set[str]:
        return set(self._ckernel)

    def all_symbols(self) -> set[str]:
        return set(self._names)

    # ── include shadowing ──────────────────────────────────────────────────
    def shadowed_includes(self) -> set[str]:
        """jit_hw-relative include strings emule shadows (`api/compute/x.h`)."""
        return {p[len(JITHW_PREFIX):] for p in self.jithw_paths}

    def _shadow_candidates(self, include: str) -> tuple[str, ...]:
        """Repo-relative paths `include` could resolve to under include/jit_hw.

        Mirrors the classifier's jit_hw-first resolution. Handles bare includes
        (`api/compute/x.h`), ones already written `jit_hw/...`, and ones a caller
        pasted with the full `include/...` prefix (strip it and re-test)."""
        inc = include.strip().lstrip("/")
        stripped = inc[len("include/"):] if inc.startswith("include/") else inc
        return (inc, f"{JITHW_PREFIX}{stripped}", f"include/{stripped}")

    def is_shadowed(self, include: str) -> bool:
        """Does emule shadow `include`? True iff it resolves under include/jit_hw."""
        return any(c in self.jithw_paths for c in self._shadow_candidates(include))

    def shadow_path(self, include: str) -> str | None:
        for c in self._shadow_candidates(include):
            if c in self.jithw_paths:
                return c
        return None

    # ── single-symbol verdict (the --supports probe) ────────────────────────
    def provides(self, symbol: str) -> tuple[str, str]:
        """The verdict a kernel using *only* `symbol` would land at, with the
        symbol class. Mirrors classify_kernels' gating:
          (layer1|needs_stub|ruled_out|unknown, tile|llk|ckernel|provided|unknown)
        """
        if TILE_RE.match(symbol):
            return ("layer1" if symbol in self.tile_symbols() else "needs_stub", "tile")
        if LLK_RE.match(symbol):
            ok = symbol in self._names or symbol.lstrip("_") in {n.lstrip("_") for n in self.llk_symbols()}
            return ("layer1" if ok else "ruled_out", "llk")
        if symbol in self._ckernel:
            return ("layer1", "ckernel")
        if symbol in self._names:
            return ("layer1", "provided")
        return ("unknown", "unknown")

    def owners(self, symbol: str) -> list[str]:
        return self._owner.get(symbol, [])


@lru_cache(maxsize=4)
def _load(index: str) -> Surface:
    with open(index) as f:
        return Surface(yaml.safe_load(f)["files"])


def load(index: str | Path | None = None) -> Surface:
    """Return the (cached) Surface parsed from `index` (default structure.yaml)."""
    return _load(str(index or DEFAULT_INDEX))


if __name__ == "__main__":  # tiny self-report for debugging
    s = load()
    print(f"jit_hw files: {len(s.jithw)}")
    print(f"tile_symbols: {len(s.tile_symbols())}")
    print(f"llk_symbols:  {len(s.llk_symbols())}")
    print(f"ckernel:      {len(s.ckernel_members())}")
