#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Aggregate emule ASAN sweep findings into a deduplicated report.

Consumes a directory of per-entry `<slug>.log` + `<slug>.xml` produced by
`sweep.py run --asan` (one dir per shard, merged) and emits:

  --out-summary    summary.md   — compact, for the Actions run summary page
  --out-dev        findings.md  — one section per finding, with evidence
  --out-headline   headline.json — machine-readable counts (trend/alerting/Slack)

Exit status: non-zero when the run found any violation, or when it produced no
data at all. That is what makes the CI lane fail on an ASAN error instead of
reporting it quietly.

Why the LOG and not just the XML: a sanitizer violation ends in abort(), so the
only record of *what* was violated is the report the dying process wrote to
stderr. The XML says a test crashed with signal 6; the log says it was an
Out-of-Bounds Write at reduce_h_neg.cpp:128. Both are needed — the XML makes the
crash countable and separates it from an ordinary functional failure, the log
makes it identifiable.

Deduplication is the point of this script. One root cause fires in every op that
touches the offending kernel, so a raw `grep -c '[ASAN ERROR]'` overstates the
bug count by an order of magnitude. Findings are keyed on
(category, kernel source file:line) — stable across ops and across runs.

CAVEAT the report states explicitly: counts are a FLOOR, not a census. The
sanitizers abort() on the first violation in a process, so a test that trips two
different checks only ever reports the first. `--asan` forks per test to keep
that blast radius to one test, but within a single test the masking remains.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict
from pathlib import Path

# ---------------------------------------------------------------------------
# Log grammar. Formats are taken verbatim from the panic call sites:
#   [emule] include/jit_hw/asan/*.h, api/dataflow/asan/asan_dataflow.h
#   [metal] tt_metal/impl/emulation/{emule_sanitizers.cpp,host_sanitizers.hpp},
#           host_api/tt_metal.cpp
# Every site is "[ASAN ERROR] <Category>: <detail>", so the category is the text
# between the tag and the first colon.
# ---------------------------------------------------------------------------
ASAN_LINE_RE = re.compile(r"\[ASAN ERROR\]\s+([^:]+):\s*(.*)$")

# Context block emitted by __emule_asan_panic -> __emule_asan_print_trace.
KERNEL_CTX_RE = re.compile(r"^\s*kernel:\s+(\S+)")
# Backtrace frames print as "    #<n> <resolved>", where <resolved> is usually
# "<symbol> at <file>:<line>", possibly with " <- (inlined by) ..." appended.
FRAME_RE = re.compile(r"^\s*#\d+\s+(.*)$")
# Frames print as "<sym> at <file>:<line>" and sometimes "<file>:<line>:<col>".
# A line of 0 means the symbolizer had no line info (e.g. "unity_18_cxx.cxx:0:0")
# and must not be mistaken for a real location — see credible_line().
AT_FILE_LINE_RE = re.compile(r"\bat\s+(\S+?):(\d+)")

# pytest -v prints the nodeid before running each test, which is what lets a log
# region be attributed to a test (the whole reason -v is passed in ASAN mode).
NODEID_RE = re.compile(r"^(\S+\.py::\S+)")
# ...but under --forked the report is not adjacent to that progress line at all:
# pytest-forked replays a failed child's captured output inside the FAILURES
# section, where each test is introduced by an "____ test_name ____" banner. That
# banner is therefore the marker that actually precedes an [ASAN ERROR] block.
FAIL_HEADER_RE = re.compile(r"^_{3,}\s+(.+?)\s+_{3,}\s*$")

# pytest-forked encodes a child killed by a signal as an <error>, not a <failure>.
CRASH_MSG_RE = re.compile(r"CRASHED with signal (\d+)")

# Proof that tt-metal selected the emulator. Logged by the parent process at
# import (rtoptions.cpp), so unlike "execute_program_emulated" it survives
# pytest-forked discarding a passing child's output.
EMULE_MODE_RE = re.compile(r"simulator/emule target device")

# Frames inside the sanitizer machinery itself are never the offending line.
MACHINERY_MARKERS = ("__emule_asan_", "emule_asan_panic", "abort", "raise")

# Non-kernel translation units: a frame here is plumbing, not the bug site.
# Includes the C++ runtime / libc frames that sit under the emule runner — a
# post-execution check's backtrace is *entirely* these, and picking one of them
# yields a stable-looking but meaningless site (thread.cc:82).
PLUMBING_FILES = (
    "asan_l1_checks.h", "asan_cb.h", "asan_dataflow.h", "emule_asan.h",
    "jit_kernel_stubs.hpp", "cb_api.h", "dataflow_api.h", "dataflow_buffer.h",
    "circular_buffer.h",
    "emule_sanitizers.cpp", "host_sanitizers.hpp", "emulated_program_runner.cpp",
    "wrapper.cpp", "thread.cc", "thread", "invoke.h", "std_function.h",
)
# Any path under emule's kernel-API tree is plumbing by construction: it is the
# shim layer every kernel calls through, never the kernel at fault. Matching on
# the path (not a basename list) is what stops a new header from silently
# becoming a bogus finding site — e.g. Dirty CB reports its captured call site as
# `.../include/jit_hw/api/dataflow/dataflow_buffer.h:68` for every DFB-routed
# kernel, which would collapse unrelated bugs into one key.
PLUMBING_PATH_MARKERS = ("/jit_hw/",)
# Unity-build translation units of libtt_metal: real file, never a kernel.
UNITY_TU_RE = re.compile(r"^unity_\d+.*\.(cxx|cpp)$")
PLUMBING_SYMBOLS = ("std::thread", "__libc", "start_thread", "clone",
                    "execute_program_emulated", "launch_cores",
                    "_Function_handler", "fiber_trampoline", "emule_fiber")


def basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def is_plumbing(path: str) -> bool:
    if any(m in path for m in PLUMBING_PATH_MARKERS):
        return True
    name = basename(path)
    return name in PLUMBING_FILES or bool(UNITY_TU_RE.match(name))


def credible_line(line: str) -> bool:
    """A `:0` line number means the symbolizer produced no line info; treating it
    as a location invents a site that does not exist."""
    return line.isdigit() and int(line) > 0


def display_site(site: str) -> str:
    """An entry-scoped key (`@<slug>`) means no source location was resolvable —
    render that as unresolved rather than leaking the internal key form."""
    if not site:
        return "(unresolved)"
    return f"(unresolved, in {site[1:]})" if site.startswith("@") else site


class Finding:
    """One deduplicated violation: a (category, site) pair plus its evidence."""

    def __init__(self, category, site, kernel):
        self.category = category
        self.site = site            # "<file>:<line>" or "" when unresolved
        self.kernel = kernel        # kernel source path from the context block
        self.count = 0              # raw [ASAN ERROR] occurrences
        self.entries = Counter()    # entry slug -> occurrences
        self.tests = OrderedDict()  # nodeid -> None (ordered, deduped)
        self.details = []           # first few raw detail strings
        self.signals = Counter()    # XML crash signals seen for the owning tests

    @property
    def key(self):
        # `site` is already fully resolved by the caller — a file:line, a kernel
        # basename (resolve_site), or an entry-scoped `@<slug>` when nothing was
        # resolvable — so this is the single definition of a finding's identity.
        return f"{self.category}|{self.site or '?'}"

    def add(self, *, entry, nodeid, detail, kernel, signal=None):
        self.count += 1
        self.entries[entry] += 1
        if nodeid:
            self.tests.setdefault(nodeid, None)
        if signal is not None:
            self.signals[signal] += 1
        if kernel and not self.kernel:
            self.kernel = kernel
        if detail and len(self.details) < 3 and detail not in self.details:
            self.details.append(detail)


def pick_site(frames):
    """Choose the most informative source location from a backtrace.

    Preference order:
      1. the frame containing `kernel_main` — the offending line in the kernel
         the JIT compiled, which is what a metal owner needs to see;
      2. the first frame in a non-plumbing file — for host-API checks
         (Use-After-Free, Metadata Overflow) there is no kernel frame at all,
         and the useful location is the calling op;
      3. nothing, rather than a misleading sanitizer-internal line.
    """
    for text in frames:
        if "kernel_main" in text:
            m = AT_FILE_LINE_RE.search(text)
            if m and credible_line(m.group(2)) and not is_plumbing(m.group(1)):
                return f"{basename(m.group(1))}:{m.group(2)}"
    for text in frames:
        if any(m in text for m in MACHINERY_MARKERS + PLUMBING_SYMBOLS):
            continue
        m = AT_FILE_LINE_RE.search(text)
        if m and credible_line(m.group(2)) and not is_plumbing(m.group(1)):
            return f"{basename(m.group(1))}:{m.group(2)}"
    return ""


def resolve_site(detail, frames, kernel):
    """Final location for a finding, most authoritative source first.

    The message wins over the backtrace: a check only embeds an `at <file>:<line>`
    in its own text when that location is the one it deliberately captured, and
    for the post-execution checks (Dirty CB, Object Intent) it is the ONLY real
    location — they fire after the kernel returned, so the backtrace holds
    nothing but runner and libc frames.

    Falling back to the kernel's basename (never its full path) keeps a finding
    key stable across workspaces and across runs.
    """
    return site_from_detail(detail) or pick_site(frames) or basename(kernel or "")


def site_from_detail(detail):
    """Some checks embed the location in the message itself rather than relying
    on the backtrace. Dirty CB is the important one: it is a POST-EXECUTION
    check, so by the time it fires the kernel has returned and the backtrace
    holds only runner frames — the reserve/wait call site captured via
    __builtin_FILE()/__builtin_LINE() is the only real location available."""
    m = AT_FILE_LINE_RE.search(detail)
    if m and credible_line(m.group(2)) and not is_plumbing(m.group(1)):
        return f"{basename(m.group(1))}:{m.group(2)}"
    return ""


def parse_log(path: Path):
    """Extract findings from one entry log.

    Returns (list of raw hit dicts, reached_emule, hw_markers). A hit is
    assembled by scanning forward from the [ASAN ERROR] line through the context
    + backtrace block that __emule_asan_panic prints immediately after it.
    """
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as e:
        sys.stderr.write(f"WARN: cannot read {path}: {e}\n")
        return [], False, False

    hits = []
    nodeid = ""
    reached = False
    hw = False
    for i, line in enumerate(lines):
        # Either marker proves the emulator was selected. The banner is the
        # reliable one: "execute_program_emulated" is logged inside the test
        # process, and pytest-forked replays a forked child's output only when the
        # child fails, so on an all-passing entry it never reaches the log.
        if EMULE_MODE_RE.search(line) or "execute_program_emulated" in line:
            reached = True
        if ("Established firmware bundle version" in line
                or "Mapped hugepage" in line or "KMD version" in line):
            hw = True
        # Whichever marker was seen most recently wins: the -v progress line in
        # the run region, the FAILURES banner once pytest starts replaying
        # captured child output (which is where forked reports land).
        hm = FAIL_HEADER_RE.match(line)
        if hm:
            nodeid = hm.group(1).strip()
        else:
            m = NODEID_RE.match(line)
            if m:
                nodeid = m.group(1)
        am = ASAN_LINE_RE.search(line)
        if not am:
            continue
        category, detail = am.group(1).strip(), am.group(2).strip()

        # Walk the report block that follows: context lines, then frames. Stop at
        # the next [ASAN ERROR] or after a bounded window (the panic mutex means
        # only one full report prints, but a truncated log must not run away).
        kernel, frames = "", []
        for j in range(i + 1, min(i + 120, len(lines))):
            nxt = lines[j]
            if "[ASAN ERROR]" in nxt:
                break
            km = KERNEL_CTX_RE.match(nxt)
            if km:
                kernel = km.group(1)
                continue
            fm = FRAME_RE.match(nxt)
            if fm:
                frames.append(fm.group(1).strip())
        site = resolve_site(detail, frames, kernel)
        hits.append({
            "category": category, "detail": detail, "kernel": kernel,
            "site": site, "nodeid": nodeid,
        })
    return hits, reached, hw


def find_xml(log: Path) -> Path:
    """The JUnit file for an entry log.

    sweep.py asks pytest for `<slug>.xml`, but tt-metal's root conftest renames
    the report to `<slug>_<YYYYmmdd>_<HHMMSS>.xml` whenever CI=true (so that
    serial pytest invocations don't clobber one report). Matching the exact name
    therefore finds nothing in CI while working locally, which silently drops
    every crash-signal correlation from the report. Accept both spellings, newest
    last (the timestamp suffix sorts lexically).
    """
    exact = log.with_suffix(".xml")
    if exact.is_file():
        return exact
    stamped = sorted(log.parent.glob(f"{log.stem}_*.xml"))
    return stamped[-1] if stamped else exact


def parse_xml(path: Path):
    """Per-entry XML outcome counts + the set of tests killed by a signal."""
    out = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0,
           "crashed": {}, "usable": False}
    if not path.is_file():
        return out
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return out
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    for s in suites:
        if s.tag != "testsuite":
            continue
        for tc in s.findall("testcase"):
            out["tests"] += 1
            for child in tc:
                if child.tag == "skipped":
                    out["skipped"] += 1
                elif child.tag in ("failure", "error"):
                    out["failures" if child.tag == "failure" else "errors"] += 1
                    msg = child.get("message") or ""
                    cm = CRASH_MSG_RE.search(msg)
                    if cm:
                        nid = f"{tc.get('classname', '')}::{tc.get('name', '')}"
                        out["crashed"][nid] = int(cm.group(1))
    out["usable"] = out["tests"] > 0
    return out


def collect(results_dir: Path):
    findings = {}
    entries = {}
    for log in sorted(results_dir.rglob("*.log")):
        slug = log.stem
        hits, reached, hw = parse_log(log)
        xml = parse_xml(find_xml(log))
        entries[slug] = {
            "reached_emule": reached, "hw_markers": hw,
            "hits": len(hits), **{k: xml[k] for k in
                                  ("tests", "failures", "errors", "skipped", "usable")},
            "crashed": xml["crashed"],
        }
        # Match a log hit to its XML outcome by test name: the log nodeid is
        # `path/to/test.py::test_x[param]` while JUnit uses
        # `classname::test_x[param]`, so only the trailing component is common.
        crashed = {nid.rsplit("::", 1)[-1]: sig for nid, sig in xml["crashed"].items()}
        for h in hits:
            # With no resolvable location, scope the key to the entry rather than
            # letting every unresolved finding of a category share one key: two
            # different bugs in two different ops would otherwise merge into a
            # single line and one of them would vanish from the report. Over-
            # reporting is recoverable in triage; under-reporting hides a bug.
            site = h["site"] or f"@{slug}"
            f = findings.get(f"{h['category']}|{site}")
            if f is None:
                f = Finding(h["category"], site, h["kernel"])
                findings[f.key] = f
            sig = crashed.get(h["nodeid"].rsplit("::", 1)[-1]) if h["nodeid"] else None
            f.add(entry=slug, nodeid=h["nodeid"], detail=h["detail"],
                  kernel=h["kernel"], signal=sig)
    return findings, entries


def render_summary(findings, entries, *, arch, pin):
    total = len(findings)
    raw = sum(f.count for f in findings.values())
    cats = sorted({f.category for f in findings.values()})
    reached = sum(1 for e in entries.values() if e["reached_emule"])
    invalid = [s for s, e in entries.items() if not e["reached_emule"]]
    hw = [s for s, e in entries.items() if e["hw_markers"]]
    # Ran but recorded no test: wallclock SIGTERM, a collection crash, or a
    # partial XML write. Its "no findings" covers nothing and must not be
    # counted as coverage.
    empty = sorted(s for s, e in entries.items()
                   if e["reached_emule"] and e["tests"] - e["skipped"] <= 0)
    executed = sum(max(e["tests"] - e["skipped"], 0) for e in entries.values())

    # No entries at all means no shard produced results — a build failure, or
    # shards that never started (a GitHub Actions outage does exactly this: they
    # die in "Set up job", before ci-asan-sweep.sh and therefore before its
    # validity gate can run). Reporting that as "no findings" would read as a
    # clean sweep, which is the one thing this lane must never do.
    if not entries:
        return "\n".join([
            f"## emule ASAN sweep — {arch}",
            "",
            f"- **pin:** `{pin}`",
            "",
            "> ### NO DATA — this run proves nothing",
            ">",
            "> No shard produced any result file, so no test was executed and no"
            " sanitizer ran. This is **not** a clean sweep; it is an absent one."
            " Check whether the build job failed or the shard jobs never started,"
            " then re-run.",
            "",
        ]) + "\n"

    L = [
        f"## emule ASAN sweep — {arch}",
        "",
        f"- **pin:** `{pin}`",
        f"- **entries:** {len(entries)} ({reached} reached the emulator"
        + (f", {len(empty)} executed nothing" if empty else "") + ")",
        f"- **tests executed:** {executed}",
        f"- **distinct findings:** **{total}**",
        f"- **categories:** {len(cats)}" + (f" — {', '.join(cats)}" if cats else ""),
        f"- **raw `[ASAN ERROR]` lines:** {raw} (deduplicated into the {total} above)",
        "",
    ]
    if hw:
        L += [f"> **INVALID:** real-hardware markers in {len(hw)} entry log(s) — "
              "this did not run on the emulator.", ""]
    if invalid:
        L += [f"> **{len(invalid)} entry/entries never reached the emulator** — their "
              "'no findings' result is not evidence of anything.", ""]
    if empty:
        L += [f"> **{len(empty)} entry/entries executed no tests** (`{'`, `'.join(empty)}`) —"
              " they reached the emulator but recorded nothing, so they contribute no"
              " coverage. Check their logs for a collection crash or a wallclock kill.", ""]
    if total:
        L += ["| Category | Site | Entries | Tests | Raw |",
              "|---|---|---:|---:|---:|"]
        for f in sorted(findings.values(),
                        key=lambda x: (-len(x.entries), x.category, x.site)):
            L.append(f"| {f.category} | `{display_site(f.site)}` | "
                     f"{len(f.entries)} | {len(f.tests)} | {f.count} |")
        L.append("")
    else:
        L += ["No `[ASAN ERROR]` findings in this run.", ""]
    L += ["<sub>Counts are a floor, not a census: the sanitizers abort() on the "
          "first violation in a process, so a test that trips two checks reports "
          "only the first. Per-test forking bounds the masking to one test.</sub>"]
    return "\n".join(L) + "\n"


def render_dev(findings, *, arch, pin):
    L = [f"# emule ASAN sweep findings — {arch} @ `{pin}`", ""]
    if not findings:
        L += ["No findings.", ""]
        return "\n".join(L)
    L += ["> The **category**, **site** and **kernel** of a finding are exact. The"
          " **test** list is approximate: under `--forked` pytest replays a"
          " crashed child's output inside the FAILURES section, so a report sitting"
          " on a section boundary can be attributed to its neighbour. Use it to"
          " find a reproducer, not as a precise owner list.", ""]
    for f in sorted(findings.values(), key=lambda x: (x.category, x.site)):
        L += [f"## {f.category} — `{display_site(f.site)}`", "",
              f"- **key:** `{f.key}`",
              f"- **kernel:** `{f.kernel or '(none reported)'}`",
              f"- **raw occurrences:** {f.count} across {len(f.entries)} entry/entries",
              ]
        if f.signals:
            L.append("- **crash signals:** "
                     + ", ".join(f"signal {s} ×{n}" for s, n in sorted(f.signals.items())))
        L.append(f"- **entries:** {', '.join(sorted(f.entries))}")
        if f.tests:
            shown = list(f.tests)[:8]
            more = f" (+{len(f.tests) - len(shown)} more)" if len(f.tests) > len(shown) else ""
            L += ["- **tests:**"] + [f"  - `{t}`" for t in shown]
            if more:
                L.append(f"  - {more.strip()}")
        if f.details:
            L += ["- **messages:**"] + [f"  - `{d}`" for d in f.details]
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True,
                    help="dir of per-entry <slug>.log / <slug>.xml (shards merged)")
    ap.add_argument("--arch", default="blackhole")
    ap.add_argument("--pin-sha", default="")
    ap.add_argument("--out-summary", default=None)
    ap.add_argument("--out-dev", default=None)
    ap.add_argument("--out-headline", default=None)
    args = ap.parse_args()

    results = Path(args.results_dir)
    if not results.is_dir():
        sys.stderr.write(f"ERROR: --results-dir not a directory: {results}\n")
        return 2

    pin = args.pin_sha or "unknown"
    findings, entries = collect(results)

    if args.out_summary:
        Path(args.out_summary).write_text(
            render_summary(findings, entries, arch=args.arch, pin=pin))
    if args.out_dev:
        Path(args.out_dev).write_text(
            render_dev(findings, arch=args.arch, pin=pin))
    if args.out_headline:
        Path(args.out_headline).write_text(json.dumps({
            "arch": args.arch,
            "pin_sha": pin,
            "entries": {
                "total": len(entries),
                "reached_emule": sum(1 for e in entries.values() if e["reached_emule"]),
                "invalid": sorted(s for s, e in entries.items() if not e["reached_emule"]),
            },
            # True when no shard produced results: downstream must not read
            # "0 findings" as "clean". See render_summary's NO DATA branch.
            "no_data": len(entries) == 0,
            "findings": {
                "total": len(findings),
                "categories": len({f.category for f in findings.values()}),
                "raw_lines": sum(f.count for f in findings.values()),
                "by_category": dict(Counter(f.category for f in findings.values())),
                "keys": sorted(findings),
            },
        }, indent=2) + "\n")

    if not entries:
        print("::error::ASAN sweep produced NO DATA — no shard result files were found, so "
              "no test ran and no sanitizer executed. This is not a clean sweep.")
    print(f"entries={len(entries)} findings={len(findings)} "
          f"raw={sum(f.count for f in findings.values())}")
    for k in sorted(findings):
        print(f"  FINDING  {k}")

    # A sanitizer violation is a real defect, so the run goes RED for it — every
    # finding, no allowlist. NO DATA fails too: a run that tested nothing has not
    # shown the tree is clean.
    if not entries:
        print("::error::FAILING: the sweep produced no data.")
        return 1
    if findings:
        print(f"::error::FAILING: {len(findings)} ASAN finding(s). "
              "See the run summary and findings.md.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
