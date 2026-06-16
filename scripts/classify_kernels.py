# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic layer-1 classifier for kernels vs the tt-emule emulator.

Point it at a directory of kernels or at specific kernel files; it labels each
kernel against emule's *layer-1* surface. An external driver can wrap it to roll
results up however it likes (per operation, per test, per model) — this tool is
just the per-kernel classification primitive and stays agnostic to any kernel
tree's layout.

tt-emule models only the *layer-1* kernel API: circular-buffer ops (cb_*), NOC
ops (noc_async_*), the high-level compute_kernel_api wrappers (add_tiles,
matmul_tiles, reduce_tile, copy_tile, <op>_tile SFPU wrappers), tile_regs_*, and
tensor accessors. It does NOT model lower layers: the LLK trees
(tt-llk/tt_llk_*/...), sfpi:: vector intrinsics, TTI_* macros, direct
llk_*/ckernel:: calls, stream/overlay registers, or raw L1 pointer math.

It classifies each kernel as:
  - layer1     : modelable by emule today (only high-level API in its include graph)
  - needs_stub : layer-1 by API, but calls a *_tile/_tiles wrapper not yet present
                 in tt-emule's include/jit_hw/api/compute/ (unblockable via a shim)
  - ruled_out  : pulls a lower-layer header or symbol emule cannot model

Includes are resolved the way emule's JIT does: an include that resolves into
`tt-emule/include/jit_hw` is a shadowed layer-1 leaf; one that falls through to
the real source tree (under --repo-root) is recursed and scanned for lower-layer
signals. Pass the kernel tree's include search roots with -I (JIT -I order).

Pure pattern-matching — no LLM, no build. Examples:

    # every kernel TU under a tree, with its include search roots
    python3 scripts/classify_kernels.py --root /path/to/kernels \\
        -I /path/to/src/hw/inc -I /path/to/src

    # a single kernel (the unit an external driver loops over)
    python3 scripts/classify_kernels.py /path/to/kernels/reader.cpp -I /path/to/src
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── tt-emule layout (the jit_hw shadow is the same for every target) ─────────
SCRIPT_DIR = Path(__file__).resolve().parent
EMULE_REPO = SCRIPT_DIR.parent  # tt-emule/
EMULE_JITHW = EMULE_REPO / "include" / "jit_hw"
EMULE_INCLUDE = EMULE_REPO / "include"

# ── Lower-layer detection (shared, target-independent) ───────────────────────
#
# Calibration note: emule's JIT shadows the STANDARD compute/dataflow stack with
# its include/jit_hw stubs (include order: jit_hw → tt-emule/include → kernel_dir
# → extras), so a generic high-level header like `api/compute/eltwise_binary.h`
# or `llk_unpack_common_api.h` is NOT, by itself, a ruleout signal — when emule
# shadows it the include resolves to a jit_hw leaf. Only CUSTOM lower-layer code
# (which has no emule stub and is reached through the *real* tree) is a hard
# ruleout:
#   - anything under an LLK tree (/tt-llk/, tt_llk_*, /llk_lib/, /llk_api/,
#     /llk_sfpu/, a `/sfpu/` dir)
#   - a direct `ckernel_sfpu_*` / custom `llk_*` include reaching below the
#     api/compute wrapper
#   - a kernel's OWN source using sfpi::/TTI_/dst_reg[]/ckernel::/llk_ directly

# Inherently-custom include *names* (matched on the include string / basename).
_RULED_OUT_INCLUDE_RE = re.compile(
    r"(?:^|/)(?:"
    r"ckernel_sfpu_\w+"
    r"|llk_(?:math|unpack)_\w*(?:sdpa|hadamard|custom_mm|rmsnorm|moe_gate|topk)\w*"
    r")\.h$"
)
# Resolved-path tokens that mark a real LLK / SFPU tree (not a jit_hw shadow).
_RULED_OUT_PATH_TOKENS = (
    "/tt-llk/", "/tt_llk/", "tt_llk_blackhole", "tt_llk_wormhole", "tt_llk_quasar",
    "/llk_lib/", "/llk_api/", "/llk_sfpu/", "/sfpu/",
)

# Lower-layer *symbols* scanned in comment-stripped source of a kernel's OWN files.
#
# Two classes:
#  - HARD: never modeled by emule → unconditional ruleout (vector intrinsics, raw
#    HW-instruction macros, direct DST register indexing).
#  - GATED: `llk_<x>(` calls and `ckernel::<x>` references. emule *does* shadow a
#    curated set of these (e.g. `llk_pack_relu_config`, `ckernel::ReluType`), so a
#    standard kernel that calls one is still layer-1. Only an llk_/ckernel symbol
#    emule does NOT provide is a ruleout. Gated against the jit_hw-provided sets.
# ── ruled_out buckets, ordered hardest → easiest to bring up under emule ─────
# A ruled_out unit's *primary* bucket is the hardest signal it hits (lowest index).
#   sfpi_intrinsics      hand-written SFPU vector code — needs an SFPI emulator
#   dst_register         raw DST register indexing — bypasses the tile abstraction
#   hw_instructions      raw HW-instruction / MOP macros (TTI_*, mop_run, TT_LLK_)
#   llk_headers          reaches a real LLK/SFPU header tree (tt-llk/, ckernel_sfpu_*)
#   unshadowed_ckernel   uses a ckernel::<x> emule doesn't provide — shimmable
#   unshadowed_llk_call  calls an llk_<x>() emule doesn't shadow — most tractable (add a shim)
RULEOUT_BUCKETS = (
    "sfpi_intrinsics", "dst_register", "hw_instructions",
    "llk_headers", "unshadowed_ckernel", "unshadowed_llk_call",
)
_BUCKET_RANK = {b: i for i, b in enumerate(RULEOUT_BUCKETS)}

# The kernel-API abstraction stack this project's headers/PRs refer to informally:
#   layer-1    high-level kernel API emule models (cb_*, noc_async_*, compute_kernel_api
#              wrappers, tile_regs_*, tensor accessors)  → verdict `layer1`
#   layer-1.5  transitional: a few TTI_SETADCXX / _llk_pack_mop_config_ pack-engine
#              config calls captured into emule state (e.g. pack-subrect, PR #121)
#   layer-2    the LLK layer — llk_*/ckernel:: calls just below the API wrappers
#   layer-3    the hardware/instruction model — sfpi:: vectors, raw DST regs, TTI_*
# A ruled_out unit's buckets map onto the layer it reaches below emule's surface:
_BUCKET_LAYER = {
    "sfpi_intrinsics":     "layer-3",   # SFPI vector engine
    "dst_register":        "layer-3",   # raw DST register file
    "hw_instructions":     "layer-3",   # raw HW instructions / MOP
    "llk_headers":         "layer-2",   # reaches the LLK header tree
    "unshadowed_ckernel":  "layer-2",   # ckernel:: member emule lacks
    "unshadowed_llk_call": "layer-2",   # llk_*() emule lacks
}

# (label, pattern, bucket) — the hard signals never modeled by emule.
_HARD_RULED_OUT_SYMBOLS = [
    ("sfpi::", re.compile(r"\bsfpi::"), "sfpi_intrinsics"),
    ("vFloat", re.compile(r"\bvFloat\b"), "sfpi_intrinsics"),
    ("vInt", re.compile(r"\bvInt\b"), "sfpi_intrinsics"),
    ("vUInt", re.compile(r"\bvUInt\b"), "sfpi_intrinsics"),
    ("dst_reg[]", re.compile(r"\bdst_reg\s*\["), "dst_register"),
    ("TTI_*", re.compile(r"\bTTI_[A-Z]"), "hw_instructions"),
    ("TT_LLK macro", re.compile(r"\bTT_LLK_"), "hw_instructions"),
    ("MOP/mop_run", re.compile(r"\bmop_run\s*\("), "hw_instructions"),
]
_LLK_CALL_RE = re.compile(r"\b(_?llk_[a-z]\w*)\s*\(")
_CKERNEL_REF_RE = re.compile(r"\bckernel::(\w+)")

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+["<]([^">]+)[">]', re.MULTILINE)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

# Compute wrapper calls (for needs-stub detection): foo_tile( / foo_tiles(
_TILE_CALL_RE = re.compile(r"\b([a-z][a-z0-9_]*_tiles?)\s*\(")

# *_tile/_tiles names that are structural, not SFPU/compute ops needing a shim.
_TILE_CALL_ALLOWLIST = {
    "cb_wait_front", "cb_pop_front", "cb_reserve_back", "cb_push_back",
    "get_tile", "release_tile", "acquire_tile",
}


def _strip_comments(text: str) -> str:
    return _LINE_COMMENT_RE.sub("", _BLOCK_COMMENT_RE.sub("", text))


def _read_jithw_text() -> str:
    """Concatenated comment-stripped text of every jit_hw header (cached)."""
    global _JITHW_TEXT
    if _JITHW_TEXT is None:
        chunks = []
        if EMULE_JITHW.is_dir():
            for f in list(EMULE_JITHW.rglob("*.h")) + list(EMULE_JITHW.rglob("*.hpp")):
                try:
                    chunks.append(f.read_text(errors="ignore"))
                except OSError:
                    pass
        _JITHW_TEXT = _strip_comments("\n".join(chunks))
    return _JITHW_TEXT


_JITHW_TEXT: str | None = None


def _emule_tile_symbols() -> set[str]:
    """Every `*_tile`/`*_tiles` identifier emule ships *anywhere* in jit_hw —
    compute wrappers, dataflow tile APIs (noc_async_read_tile), debug (print_*_tile).
    Used as the needs_stub denominator."""
    return set(re.findall(r"\b([a-z][a-z0-9_]*_tiles?)\b", _read_jithw_text()))


def _emule_llk_symbols() -> set[str]:
    """Every `llk_*` shim emule provides in jit_hw (gates the `llk_ call` ruleout)."""
    return set(re.findall(r"\b(_?llk_[a-z]\w*)\b", _read_jithw_text()))


def _emule_ckernel_symbols() -> set[str]:
    """Member names emule provides under `namespace ckernel` (gates `ckernel::` refs).

    Best-effort: names emule itself references as `ckernel::<x>`, plus type/function
    names declared inside any `namespace ckernel { ... }` span in jit_hw."""
    text = _read_jithw_text()
    syms = set(re.findall(r"\bckernel::(\w+)", text))
    # Declarations inside `namespace ckernel { ... }` spans (brace-matched).
    for m in re.finditer(r"\bnamespace\s+ckernel\b\s*\{", text):
        i, depth, n = m.end(), 1, len(text)
        while i < n and depth:
            c = text[i]
            depth += (c == "{") - (c == "}")
            i += 1
        body = text[m.end():i]
        syms.update(re.findall(r"\benum\s+(?:class\s+)?(\w+)", body))      # enum names
        syms.update(re.findall(r"\b(?:struct|class)\s+(\w+)", body))        # type names
        syms.update(re.findall(r"\b([a-zA-Z_]\w*)\s*\(", body))             # function names
        # Re-exports of global-scope names into ckernel (emule defines many
        # ckernel-API types at global scope then `using ::Name;` here — e.g.
        # `using ::PoolType; using ::ReduceDim;` in llk_types.h).
        syms.update(re.findall(r"\busing\s+(?:[\w]+::|::)?(\w+)\s*;", body))
        syms.update(re.findall(r"\busing\s+(\w+)\s*=", body))              # type aliases
        # Constants (e.g. `constexpr uint32_t TILE_HEIGHT = 32;`).
        syms.update(re.findall(r"\b(?:constexpr|const)\s+[\w:<>\*&\s]+?\b(\w+)\s*[=;]", body))
    return syms


@dataclass
class EmuleSets:
    """What emule's jit_hw provides — the denominators every heuristic gates on.

    Two sources (selectable via `--surface-source`):
      scan — regex-scan the jit_hw header *text* (this module's `_emule_*`
             functions). The original, self-contained source.
      yaml — read from the authoritative `.claude/references/structure.yaml`
             index (via the shared `emule_surface` module). The source of
             truth, kept in sync by `scripts/gen_structure.py`.
      both — compute both and assert yaml ⊇ scan per set, falling back to the
             union on any shortfall (the silent-denominator-shrink guard).
    """
    tile: set[str]      # *_tile(s) wrappers (needs_stub denominator)
    llk: set[str]       # llk_* shims
    ckernel: set[str]   # ckernel:: members

    @classmethod
    def _from_scan(cls) -> "EmuleSets":
        return cls(tile=_emule_tile_symbols(),
                   llk=_emule_llk_symbols(),
                   ckernel=_emule_ckernel_symbols())

    @classmethod
    def _from_yaml(cls) -> "EmuleSets | None":
        # Degrade to scan if the yaml surface is unavailable. emule_surface
        # raises SystemExit (not Exception) when PyYAML is missing, so catch
        # both to honor the documented fallback rather than hard-exiting.
        try:
            import emule_surface  # leaf module; reads structure.yaml
            s = emule_surface.load()
            return cls(tile=s.tile_symbols(), llk=s.llk_symbols(), ckernel=s.ckernel_members())
        except (Exception, SystemExit) as e:
            sys.stderr.write(f"classify_kernels: --surface-source yaml unavailable ({e}); using scan\n")
            return None

    @classmethod
    def collect(cls, source: str = "yaml") -> "EmuleSets":
        if source == "scan":
            return cls._from_scan()
        yaml_sets = cls._from_yaml()
        if yaml_sets is None:
            return cls._from_scan()
        if source == "yaml":
            return yaml_sets
        # source == "both": guard against a yaml denominator shrinking below the
        # proven scan set (which would wrongly flip kernels to needs_stub /
        # ruled_out). Report any shortfall and fall back to the union.
        scan = cls._from_scan()
        merged = {}
        for field_name in ("tile", "llk", "ckernel"):
            y, sc = getattr(yaml_sets, field_name), getattr(scan, field_name)
            missing = sc - y
            if missing:
                sys.stderr.write(
                    f"classify_kernels[both]: yaml {field_name} missing {len(missing)} "
                    f"scan symbol(s), unioning: {sorted(missing)[:12]}\n")
            merged[field_name] = y | sc
        return cls(**merged)


@dataclass
class Verdict:
    name: str
    verdict: str = "layer1"  # layer1 | needs_stub | ruled_out
    reasons: list[str] = field(default_factory=list)
    ruled_out_includes: list[str] = field(default_factory=list)
    missing_compute_stubs: list[str] = field(default_factory=list)
    kernel_files: list[str] = field(default_factory=list)
    ruleout_buckets: list[str] = field(default_factory=list)  # sorted hardest→easiest

    @property
    def primary_bucket(self) -> str | None:
        return self.ruleout_buckets[0] if self.ruleout_buckets else None

    @property
    def ruleout_layer(self) -> str | None:
        """layer-* the unit reaches below emule's surface (from its primary bucket)."""
        b = self.primary_bucket
        return _BUCKET_LAYER.get(b) if b else None


_EMULE_JITHW_RESOLVED = str(EMULE_JITHW.resolve())


@dataclass
class Classifier:
    """Shared classification core.

    repo_root:     in-tree files under this root are recursed + scanned ("in_repo");
                   anything else (system headers, out-of-tree) is a leaf.
    search_roots:  dirs to resolve a non-jit_hw #include against (JIT -I order).
    emule_syms:    *_tile wrappers emule ships (for the needs_stub verdict).
    """

    repo_root: Path
    search_roots: list[Path]
    emule_syms: set[str] = field(default_factory=set)      # *_tile(s) emule ships
    emule_llk: set[str] = field(default_factory=set)       # llk_* shims emule ships
    emule_ckernel: set[str] = field(default_factory=set)   # ckernel:: members emule ships

    def __post_init__(self) -> None:
        self._repo_resolved = str(self.repo_root.resolve())

    def _resolve_include(self, inc: str, from_dir: Path) -> tuple[Path | None, str]:
        """Resolve an include the way emule's JIT does.

        Returns (path, kind):
          "jithw"    — an emule jit_hw stub (layer-1 leaf, do not recurse)
          "in_repo"  — a file under repo_root (recurse + scan)
          "external" — not found in jit_hw or the repo tree (leaf)
        """
        # 1. emule jit_hw shadows the standard stack first.
        for root in (EMULE_JITHW, EMULE_INCLUDE):
            try:
                rp = (root / inc).resolve()
            except OSError:
                rp = None
            if rp and rp.is_file() and str(rp).startswith(_EMULE_JITHW_RESOLVED):
                return rp, "jithw"
        # 2. relative-to-includer, then the target's search roots.
        for cand in [from_dir / inc] + [root / inc for root in self.search_roots]:
            try:
                rp = cand.resolve()
            except OSError:
                continue
            if rp.is_file() and str(rp).startswith(self._repo_resolved):
                return rp, "in_repo"
        return None, "external"

    def classify(self, name: str, entry: Path, own_dirs: list[Path],
                 seed_siblings: bool = False) -> Verdict:
        """Classify one kernel/op given its entry point.

        own_dirs:      dirs whose files count as the kernel's OWN source (scanned
                       for lower-layer symbols). Vendored/stock headers reached
                       through includes are NOT symbol-scanned — they legitimately
                       name ckernel::/llk_ as the standard surface emule shadows.
        seed_siblings: also enqueue sibling files in entry.parent (for trees that
                       split a kernel across several .hpp/.cpp in one dir).
        """
        v = Verdict(name=name)
        own = [str(d.resolve()) for d in own_dirs]
        visited: set[Path] = set()
        queue: list[Path] = [entry]
        if seed_siblings:
            for sib in sorted(entry.parent.glob("*")):
                if sib.is_file() and sib.suffix in (".hpp", ".h", ".cpp", ".cc"):
                    queue.append(sib)

        tile_calls: set[str] = set()
        buckets: set[str] = set()

        while queue:
            f = queue.pop()
            if f in visited:
                continue
            visited.add(f)
            try:
                rel = f.relative_to(self.repo_root)
            except ValueError:
                rel = f
            v.kernel_files.append(str(rel))
            try:
                raw = f.read_text(errors="ignore")
            except OSError:
                continue
            src = _strip_comments(raw)

            # Lower-layer symbols are only a custom signal in the kernel's OWN
            # files (scanning stock api/* headers would over-ruleout everything).
            is_own = any(str(f.resolve()).startswith(d) for d in own)
            if is_own:
                # Hard signals: never modeled by emule.
                for label, pat, bucket in _HARD_RULED_OUT_SYMBOLS:
                    if pat.search(src):
                        reason = f"symbol {label} in {rel}"
                        if reason not in v.reasons:
                            v.reasons.append(reason)
                        v.verdict = "ruled_out"
                        buckets.add(bucket)
                # Gated: an llk_*( call or ckernel::x ref is only a ruleout when
                # emule does NOT shadow that symbol.
                for sym in {m.group(1) for m in _LLK_CALL_RE.finditer(src)}:
                    if sym not in self.emule_llk and sym.lstrip("_") not in self.emule_llk:
                        reason = f"unshadowed {sym}() in {rel}"
                        if reason not in v.reasons:
                            v.reasons.append(reason)
                        v.verdict = "ruled_out"
                        buckets.add("unshadowed_llk_call")
                for sym in {m.group(1) for m in _CKERNEL_REF_RE.finditer(src)}:
                    if sym not in self.emule_ckernel:
                        reason = f"unshadowed ckernel::{sym} in {rel}"
                        if reason not in v.reasons:
                            v.reasons.append(reason)
                        v.verdict = "ruled_out"
                        buckets.add("unshadowed_ckernel")

            # Compute wrapper calls (for needs-stub).
            for m in _TILE_CALL_RE.finditer(src):
                sym = m.group(1)
                if sym not in _TILE_CALL_ALLOWLIST:
                    tile_calls.add(sym)

            # Walk includes.
            for inc in _INCLUDE_RE.findall(raw):
                if _RULED_OUT_INCLUDE_RE.search(inc) or any(t in inc for t in _RULED_OUT_PATH_TOKENS):
                    if inc not in v.ruled_out_includes:
                        v.ruled_out_includes.append(inc)
                    v.verdict = "ruled_out"
                    buckets.add("llk_headers")
                    continue
                resolved, kind = self._resolve_include(inc, f.parent)
                if kind == "jithw":
                    continue  # emule shadow — layer-1 leaf
                if kind == "in_repo" and resolved is not None:
                    rp = str(resolved)
                    if any(t in rp for t in _RULED_OUT_PATH_TOKENS):
                        if inc not in v.ruled_out_includes:
                            v.ruled_out_includes.append(inc)
                        v.verdict = "ruled_out"
                        buckets.add("llk_headers")
                        continue
                    queue.append(resolved)
                # "external": stock header with no emule stub — leaf, not a ruleout.

        # needs-stub: layer-1 kernel calling a *_tile wrapper emule doesn't ship.
        if v.verdict == "layer1" and self.emule_syms:
            missing = sorted(
                s for s in tile_calls
                if s.endswith(("_tile", "_tiles")) and s not in self.emule_syms
            )
            if missing:
                v.verdict = "needs_stub"
                v.missing_compute_stubs = missing
                v.reasons.append("missing compute shims: " + ", ".join(missing))

        if v.verdict == "ruled_out":
            v.ruleout_buckets = sorted(buckets, key=lambda b: _BUCKET_RANK.get(b, 99))
        if v.verdict == "layer1":
            v.reasons.append("only high-level cb_/noc_/compute_kernel_api in include graph")
        return v


# ── kernel discovery ─────────────────────────────────────────────────────────
# A kernel translation unit: a compute kernel uses `void MAIN` / `#define MAIN`,
# a dataflow kernel uses `kernel_main()`.
_KERNEL_MARKER_RE = re.compile(r"\bkernel_main\s*\(|\bvoid\s+MAIN\b|#define\s+MAIN\b")


def discover_kernels(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """Every kernel TU under `root`: a file with one of `suffixes` whose source
    carries a kernel entry marker."""
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not (p.is_file() and p.suffix in suffixes):
            continue
        try:
            if _KERNEL_MARKER_RE.search(p.read_text(errors="ignore")):
                out.append(p)
        except OSError:
            continue
    return out


def _pick_repo_root(explicit: Path | None, root: Path | None,
                    kernels: list[Path], search_roots: list[Path]) -> Path:
    """The tree boundary for include recursion (in_repo vs external leaf)."""
    if explicit:
        return explicit.resolve()
    if root:
        return root.resolve()
    import os
    cands = [str(k.parent) for k in kernels] + [str(r) for r in search_roots]
    return Path(os.path.commonpath(cands))


def classify_tree(kernels: list[Path], repo_root: Path, search_roots: list[Path],
                  sets: "EmuleSets", seed_siblings: bool = False) -> dict[str, Verdict]:
    """Classify each kernel in `kernels`, keyed by its path relative to repo_root."""
    clf = Classifier(repo_root=repo_root, search_roots=search_roots,
                     emule_syms=sets.tile, emule_llk=sets.llk, emule_ckernel=sets.ckernel)
    results: dict[str, Verdict] = {}
    for kf in kernels:
        try:
            name = kf.relative_to(repo_root).as_posix()
        except ValueError:
            name = kf.as_posix()
        results[name] = clf.classify(name, kf, own_dirs=[kf.parent], seed_siblings=seed_siblings)
    return results


# ── output ───────────────────────────────────────────────────────────────────
def _verdict_counts(verds) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in verds:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    return counts


_BUCKET_MEANING = {
    "sfpi_intrinsics": "hand-written SFPU vector code (needs an SFPI emulator)",
    "dst_register": "raw DST register indexing (bypasses the tile abstraction)",
    "hw_instructions": "raw HW-instruction / MOP macros (TTI_*, mop_run)",
    "llk_headers": "reaches a real LLK/SFPU header tree (tt-llk/, ckernel_sfpu_*)",
    "unshadowed_ckernel": "uses a `ckernel::x` emule doesn't provide — shimmable",
    "unshadowed_llk_call": "calls an `llk_x()` emule doesn't shadow — add a jit_hw shim",
}


def _write_outputs(results: dict, out_dir: Path, source_root: Path,
                   emule_tile_symbols_n: int) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_root": str(source_root),
        "emule_jithw": str(EMULE_JITHW),
        "emule_tile_symbols_found": emule_tile_symbols_n,
        "kernels": {
            n: {
                "verdict": v.verdict,
                "ruleout_layer": v.ruleout_layer,
                "ruleout_primary": v.primary_bucket,
                "ruleout_buckets": v.ruleout_buckets,
                "reasons": v.reasons,
                "ruled_out_includes": v.ruled_out_includes,
                "missing_compute_stubs": v.missing_compute_stubs,
                "kernel_files": v.kernel_files,
            }
            for n, v in sorted(results.items())
        },
    }
    json_path = out_dir / "kernels_manifest.json"
    json_path.write_text(json.dumps(manifest, indent=2))

    counts = _verdict_counts(results.values())
    L = ["# tt-emule layer-1 kernel classification\n",
         f"Generated by `scripts/classify_kernels.py` (deterministic, no LLM). "
         f"Source: `{source_root}`. Manifest: `{json_path.name}`.\n",
         f"## Summary ({len(results)} kernels)\n",
         "| verdict | count |\n|---|---|"]
    for k in ("layer1", "needs_stub", "ruled_out"):
        L.append(f"| {k} | {counts.get(k, 0)} |")

    ruled = [v for v in results.values() if v.verdict == "ruled_out"]
    if ruled:
        bcount: dict[str, int] = {}
        lcount: dict[str, int] = {}
        for v in ruled:
            bcount[v.primary_bucket] = bcount.get(v.primary_bucket, 0) + 1
            lcount[v.ruleout_layer] = lcount.get(v.ruleout_layer, 0) + 1
        L.append("\n### ruled_out by layer\n")
        L.append("| layer | count | what it is |\n|---|---|---|")
        for lyr, desc in (("layer-2", "LLK calls below the API wrappers (llk_*/ckernel::) — often shimmable"),
                          ("layer-3", "hardware/instruction model (sfpi:: vectors, raw DST, TTI_*)")):
            if lcount.get(lyr):
                L.append(f"| {lyr} | {lcount[lyr]} | {desc} |")
        L.append("\n### ruled_out breakdown (by primary blocker, hardest→easiest)\n")
        L.append("| bucket | layer | count | meaning |\n|---|---|---|---|")
        for b in RULEOUT_BUCKETS:
            if bcount.get(b):
                L.append(f"| {b} | {_BUCKET_LAYER[b]} | {bcount[b]} | {_BUCKET_MEANING[b]} |")
        shimmable = sorted(
            n for n, v in results.items() if v.verdict == "ruled_out"
            and set(v.ruleout_buckets) <= {"unshadowed_llk_call", "unshadowed_ckernel"})
        if shimmable:
            L.append(f"\n**Shimmable ({len(shimmable)})** — ruled out *only* by unshadowed "
                     f"llk_/ckernel symbols; a jit_hw shim would unblock them:\n")
            L.append(", ".join(f"`{n}`" for n in shimmable))
    md_path = out_dir / "kernels_CLASSIFICATION.md"
    md_path.write_text("\n".join(L) + "\n")
    return json_path, md_path


def _print_summary(results: dict) -> None:
    counts = _verdict_counts(results.values())
    print("\n=== tt-emule layer-1 kernel classification ===")
    print(f"kernels classified: {len(results)}")
    for k in ("layer1", "needs_stub", "ruled_out"):
        print(f"  {k:11s} {counts.get(k, 0)}")
    ruled = [v for v in results.values() if v.verdict == "ruled_out"]
    if ruled:
        bcount: dict[str, int] = {}
        lcount: dict[str, int] = {}
        for v in ruled:
            bcount[v.primary_bucket] = bcount.get(v.primary_bucket, 0) + 1
            lcount[v.ruleout_layer] = lcount.get(v.ruleout_layer, 0) + 1
        print("  ruled_out by primary blocker (hardest→easiest):")
        for b in RULEOUT_BUCKETS:
            if bcount.get(b):
                print(f"    {b:20s} {bcount[b]:4d}  [{_BUCKET_LAYER[b]}]")
        print("  ruled_out by layer:  " +
              "  ".join(f"{lyr}={lcount[lyr]}" for lyr in ("layer-2", "layer-3") if lcount.get(lyr)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernels", nargs="*", type=Path,
                    help="specific kernel files to classify (the unit an external driver loops over)")
    ap.add_argument("--root", type=Path, default=None,
                    help="classify every kernel TU found under this directory")
    ap.add_argument("-I", "--search-root", action="append", default=[], dest="search_roots",
                    metavar="DIR", help="include search root (repeatable; JIT -I order)")
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="tree boundary for include recursion (default: --root, else the "
                         "common ancestor of the kernels + search roots)")
    ap.add_argument("--suffix", default=".cpp,.cc",
                    help="comma-separated kernel-file suffixes for --root discovery (default: .cpp,.cc)")
    ap.add_argument("--seed-siblings", action="store_true",
                    help="also scan sibling files in each kernel's dir (for kernels split across files)")
    ap.add_argument("--surface-source", choices=("scan", "yaml", "both"), default="yaml",
                    help="where the 'what emule provides' denominators come from: "
                         "`yaml` (structure.yaml, the source of truth — default), "
                         "`scan` (regex-scan jit_hw headers), `both` (yaml ⊇ scan guard).")
    ap.add_argument("--out-dir", default=str(SCRIPT_DIR / "out"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.kernels and not args.root:
        ap.error("give one or more kernel files, or --root DIR")

    suffixes = tuple(s if s.startswith(".") else "." + s for s in args.suffix.split(","))
    kernel_files = [k.resolve() for k in args.kernels]
    if args.root:
        kernel_files += discover_kernels(args.root.resolve(), suffixes)
    kernel_files = sorted(set(kernel_files))
    if not kernel_files:
        raise SystemExit("no kernel translation units found")

    search_roots = [Path(r).resolve() for r in args.search_roots]
    repo_root = _pick_repo_root(args.repo_root, args.root, kernel_files, search_roots)
    if not search_roots:
        search_roots = [repo_root]

    sets = EmuleSets.collect(args.surface_source)
    results = classify_tree(kernel_files, repo_root, search_roots, sets,
                            seed_siblings=args.seed_siblings)

    out_dir = Path(args.out_dir)
    json_path, md_path = _write_outputs(results, out_dir, repo_root, len(sets.tile))
    if not args.quiet:
        _print_summary(results)
        print(f"\nmanifest: {json_path}\nreport:   {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
