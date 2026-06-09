---
name: bh-ci-sweep
description: Reproduce the BH post-commit CI sanity sweep against the current tt-metal pin, producing the same exec + dev reports under snapshots/bh_sanity/. Distilled from the original sweep that closed parent issue #59 (followups #123–#132).
---

# /bh-ci-sweep — Reproduce the BH post-commit sanity sweep

End-to-end recipe for re-running the BH post-commit CI test set under
emule and producing two paired reports — exec (≤1 page leadership
summary) and dev (full per-file/per-class detail). This is the
specialization of the generic `/status-snapshot` skill that locked in
the lessons learned during the original sweep (parent #59).

## When to invoke

- After a tt-metal pin bump, to re-derive the headline `% passing`
  against the new pin.
- After landing one of the #123–#132 followups, to confirm the
  expected unblock landed.
- Periodically (monthly?) to track drift over time.

**Do NOT invoke** for generic status snapshots against other
suites — use `/status-snapshot` for that. This skill bakes in
BH-post-commit-specific decisions (predicate, manifest path, target
SKU, wallclock policy).

## Hard constraints

- Run from the tt-emule branch you want to measure (typically `main`
  or `arminale/bh-ci-sweep`).
- tt-metal must be checked out at the SHA from `tt-metal-pin.txt`.
- tt-umd submodule must be clean (no in-flight working-tree mods to
  `device/chip/sw_emule_chip.cpp` etc. — those bias the run).
- The sweep is **point-in-time**. Each invocation overwrites the
  previous reports + per-file XMLs under `snapshots/bh_sanity/`. Back
  up via `tar czf snapshots/bh_sanity/backups/<timestamp>.tar.gz …`
  before re-running if you want to preserve.

## Pre-flight checklist

```bash
# 1. tt-emule branch + tt-metal pin alignment
cd /localdev/<user>/tt-emule
git -C ../tt-metal rev-parse HEAD                      # must equal tt-metal-pin.txt's first non-comment line
grep -vE '^\s*(#|$)' tt-metal-pin.txt | head -1
git -C ../tt-metal/tt_metal/third_party/umd status -s  # must be empty

# 2. Build is current
cmake --build ../tt-metal/build_emule --target ttnn    # expect "ninja: no work to do"

# 3. pytest-timeout present in the toolchain venv (one-time install)
/opt/ttmlir-toolchain/venv/bin/pip show pytest-timeout 2>/dev/null \
  || /opt/ttmlir-toolchain/venv/bin/pip install pytest-timeout==2.4.0

# 4. No stray pytest / runner processes
ps -ef | grep -E 'pytest|run_perfile' | grep -v grep   # empty
```

## The five scripts, in order

Each script is at `snapshots/bh_sanity/<name>`:

### Step 1 — Expand the manifest (`expand_manifest.py`)

```bash
python3 snapshots/bh_sanity/expand_manifest.py
# writes: snapshots/bh_sanity/expanded_manifest.yaml
```

Reads `tt-metal/tests/pipeline_reorg/ttnn-tests.yaml`, applies the
predicate **`merge_gate is falsy AND 'bh_p150b_civ2' in skus`** (the
BH post-commit lane), expands directory references via
`find tests/ttnn/... -name 'test_*.py'`. Output: one flat list of
~264 test files keyed by manifest entry slug. ~1 min.

### Step 2 — Pre-filter for collection errors (`pre_filter_collection.py`)

```bash
python3 snapshots/bh_sanity/pre_filter_collection.py
# writes: snapshots/bh_sanity/file_classification.yaml
```

Runs `pytest --collect-only` on each expanded file under the emule env,
classifies into RUN (collects OK), DROP (`ModuleNotFoundError` or
`ImportError`), EMPTY. ~25–45 min one-time. **The DROP list typically
captures ~21 files** failing on `transformers` / `IPython` /
`safetensors` / `torchvision` (toolchain venv is leaner than tt-metal's
own `python_env`). Tracked as a known env gap in **#132**.

### Step 3 — Generate the per-file runner (`generate_perfile.py`)

```bash
python3 snapshots/bh_sanity/generate_perfile.py
# writes: snapshots/bh_sanity/run_perfile.sh
```

Emits one pytest invocation per RUN file with:

- `timeout 1350` (22.5-min wallclock backstop)
- `pytest --timeout 300 -v -m "not disable_fast_runtime_mode"`
- **No `--forked`** by default (saves ~3h vs forked; we add it on retry
  for SIGABRT / hang cases)
- `--junitxml=$OUT/<file-slug>.xml > $OUT/<file-slug>.log 2>&1`
- Idempotency: each block is wrapped in `elif [ -s "$OUT/<slug>.xml" ]
  then : skip` so resuming an interrupted run skips completed files.
- `ONLY=<entry-slug>` env var filters to just one manifest entry's
  files; `ONLY_FILE=<rel>` filters to one specific file.

### Step 4 — Run the sweep

```bash
RESUME=0 bash -c 'source snapshots/bh_sanity/env_setup.sh \
                && bash snapshots/bh_sanity/run_perfile.sh'
```

`env_setup.sh` exports `PYTHONPATH`, `LD_LIBRARY_PATH`, `TT_METAL_HOME`,
`PATH` (to put the toolchain pytest first), and clears the JIT cache
(unless `RESUME=1` is set — keep the warm cache when resuming a partial
sweep).

Wallclock: 6–18h depending on emule's pace. Most files finish in
seconds; the heavy ones (eltwise, conv, fused norms) dominate.

To resume after an interruption: `RESUME=1` keeps the JIT cache warm
and the idempotency check skips already-done files.

### Step 5 — Triage killed files (the second sweep)

After the main sweep finishes, scan for **files with a `.log` but no
`.xml`** — those are killed mid-run.

```bash
cd snapshots/bh_sanity/bh_emule
for log in *.log; do
    base="${log%.log}"
    [ ! -f "${base}.xml" ] && echo "killed: $base"
done
```

For each killed file, classify via the abort signature:

| Log shows | Diagnosis | Action |
|---|---|---|
| `Fatal Python error: Aborted` | SIGABRT mid-run — emule hit a C-level assertion. Without `--forked` it kills the whole pytest. | Re-run with `--forked` (5400s wallclock plenty). |
| No abort signature, last log line is mid-line (no `PASSED` suffix), mtime ≈ start + 1350s | Wallclock SIGTERM caught a slow but functional run. | Re-run with bumped wallclock (7200s); `--forked` optional. |
| No abort, log frozen mid-run, mtime not at the cap | **Hang** in a C-extension. `pytest-timeout`'s thread-based interrupt can't kill C code. | Re-run with `--forked` — per-test timeout will fire in the child process. |

Templates in the repo (modify and reuse):

- `run_perfile_retry_forked.sh` — for SIGABRT files
- `run_perfile_retry_wallclock.sh` — for bumped-budget retries
- `run_perfile_retry_forked2.sh` — for hangs discovered after the
  no-forked main sweep (combines forked + bumped budget)

### Step 6 (optional) — Placeholder uncovered op families (`synthesize_skip_xml.py`)

If an op family is known 100% uncovered (e.g. `test_conv3d.py` —
emule has no conv3d), don't burn 30–45 min running it:

```bash
python3 snapshots/bh_sanity/synthesize_skip_xml.py \
    tests/ttnn/unit_tests/operations/conv/test_conv3d.py
```

This writes a placeholder XML with N failure testcases (matching the
pre-filter's collected count), each marked
`<failure message="placeholder: would have failed">`. The parser
counts them in the F bucket as "Unclassified" (the placeholder message
doesn't match any of the 5 blocker patterns — intentional, so a future
reader can grep for it).

### Step 7 — Parse + report (`parse_and_report.py`)

```bash
python3 snapshots/bh_sanity/parse_and_report.py \
    --xml-dir snapshots/bh_sanity/bh_emule \
    --log-dir snapshots/bh_sanity/bh_emule \
    --out-exec snapshots/bh_sanity/bh_sanity_status_exec.md \
    --out-dev  snapshots/bh_sanity/bh_sanity_status_dev.md \
    --suite-name bh_sanity --variant-label bh_emule \
    --audit-log snapshots/bh_sanity/audit_bh_sanity.log \
    --runner   snapshots/bh_sanity/run_perfile.sh \
    --manifest /localdev/<user>/tt-metal/tests/pipeline_reorg/ttnn-tests.yaml \
    --baseline silicon-passing
```

Writes both reports. Key sections in dev:
- Per-testcase blocker-class breakdown (use the
  `fail_class_counts` Counter — **do not** revert to per-file sample).
- **JIT compile-error root causes (top 5)** — scans per-file logs for
  C++ `error: ...` lines; correlates the top-5 against the followup
  issues.
- **Top 5 files by Numeric drift (PCC/ATOL)** — for PCC investigation.
- **Excluded test files** — the env-side DROP list.

### Step 8 — Verify (`verify.sh`)

```bash
bash snapshots/bh_sanity/verify.sh
# expect "VERIFY PASS"
```

Jargon scan, section-presence check, totals consistency, audit log
integration, path resolution. Each check is one line on success.

## Gotchas (the lessons learned)

These all bit us in the original sweep. Each cross-references the
followup issue that tracks the underlying bug.

### `--continue-on-collection-errors` is broken in pytest 9.0.3

The flag's help says "Force test execution even if collection errors
occur." Reality: it suppresses the `Interrupted: N errors during
collection` message but still exits in ~0.2s without running any
tests. Verified empirically. **Don't depend on it.** Pre-filter via
`pre_filter_collection.py` instead — drop collection-error files
up-front so pytest never sees them.

### `pytest-timeout` can't interrupt C-extension hangs without `--forked`

`pytest --timeout 300` uses thread-based interrupts that only work on
pure-Python code. If a test hangs inside a ttnn C call (e.g.
`emulated_program_runner` waiting on a sync primitive that the mock
never resolves), the timeout fires but pytest can't actually pre-empt
the C call. The whole pytest process stays stuck until the shell
`timeout` SIGTERMs it.

**Mitigation**: `--forked` runs each test in a subprocess; per-test
timeout SIGTERMs the subprocess. Adds ~0.3s overhead per test
(~3h for a 40k-test sweep), so we drop it for the main sweep and
re-enable for the retry passes.

### Wallclock SIGTERM = no XML

pytest writes its JUnit XML at session end, in one shot, from the
`pytest_sessionfinish` hook. SIGTERM mid-run → no XML produced. The
log file usually exists (pytest writes test names as it goes) but
ends mid-line with no summary.

**Detection**: `[ -f <slug>.log ] && [ ! -f <slug>.xml ]`. Built into
the `verify.sh` check.

### Hangs vs slow-tests look identical without `--forked`

Both produce "log exists, XML doesn't, mtime ≈ start + wallclock".
The difference: a HANG froze on test N out of M and the log mtime is
stale (mid-run); a SLOW run kept producing output up to the cap.
Compare the file's PASSED token count to the pre-filter collected
count to disambiguate.

### Per-testcase classification beats per-file sampling

The naive approach — "take the file's first failure sample, classify
it, count the whole file's F bucket under that class" — loses ~3,500
of 7,725 jit_compile_kernel failures because each file's first
failure happened to be a different class. Use the
`fail_class_counts: Counter` field on each by_file entry; classify
every testcase as it's parsed; aggregate by summing across files.

### The `PASSED`/`FAILED` token doesn't end with newline

pytest `-v` writes `tests/.../test_X[params]` (waiting to append the
verdict), but the test's own logger output (`| DEBUG | ttnn |
manage_config | ...`) gets concatenated in flight. So lines often
look like `…test_X[params]FAILED2026-06-08 02:02:55 | DEBUG | …`.
**The JUnit XML is authoritative.** `grep ' FAILED$'` will under-count;
either use `grep FAILED` (anywhere on line) for log inspection, or
just trust the XML.

### Placeholder XMLs distort "Unclassified"

A placeholder synthesized via `synthesize_skip_xml.py` (e.g. conv3d's
1,552) shows up in the "Unclassified" blocker bucket because its
generic placeholder message doesn't match any of the 5 patterns. **This
is intentional** — but be ready to call it out when reading the dev
report's Top 3 gaps section.

## Worked example: the 2026-06-08 snapshot

```
Pins:
  tt-emule           main @ 6bf1c6b
  tt-metal           detached @ 3d6927b (matches tt-metal-pin.txt)
  tt-umd (submodule) 6c95df2b (v0.9.5-dev.260424-100-g6c95df2b), clean
  pytest             /opt/ttmlir-toolchain/venv/bin/pytest 9.0.3 (+ pytest-timeout 2.4.0)

Pipeline:
  Step 1 — expand_manifest.py            → 14 entries, 264 expanded files (~1 min)
  Step 2 — pre_filter_collection.py      → 243 RUN, 21 DROP, 0 EMPTY (~10 min)
  Step 3 — generate_perfile.py           → run_perfile.sh, 243 invocations
  Step 4 — main sweep (no --forked)      → 230 XMLs, 13 killed (~9h)
  Step 5 — triage + 3 retry passes       → 8 SIGABRT files via --forked, 5 wallclock
                                            (~2.5h total retry time)
  Step 6 — conv3d placeholder            → synthesize_skip_xml.py, 1552 placeholder Fs
  Step 7 — parse_and_report.py
  Step 8 — verify.sh → VERIFY PASS

Headline (after retries + placeholder):
  Total tests collected:  38,926
  Total tests executed:   35,407 (excl. S + X)
  Passing:                24,272  =  68.6% of executed
  Files fully passing:    144 / 243

Blocker breakdown (per-testcase):
  Missing op / unjitted symbol (JIT compile errors): 7,725
  Numeric drift (PCC/ATOL):                          1,710
  Unclassified (1,552 placeholder + 32 real):        1,584
  Mock-API drift:                                       81
                                              total: 11,100 F

Top 5 JIT compile errors driving the failures:
  6,525  redefinition of 'align_power_of_2'                    → #123
  3,316  no matching function for call to 'read_with_state'    → #124
  3,278  use of undeclared identifier 'llk_math_eltwise_binary_init'  → #125
  2,004  no member named 'get_noc_addr'                        → #126
  1,761  'async_read_barrier' ... does not refer to a template → #127

If #123–#127 land, % passing projects to ~93%.
```

## References

- **Parent (closed)**: tenstorrent/tt-emule#59
- **JIT root causes (open followups)**: #123, #124, #125, #126, #127
- **Coverage gaps**: #128 (conv3d), #129 (fused-norm sharded), #130
  (pool family)
- **Numeric drift investigation**: #131
- **Env-side gap**: #132 (toolchain venv missing test deps)
- **Generic skill** this one specializes: `.claude/skills/status-snapshot/SKILL.md`
- **Original coverage audit**: `docs/bh-sanity-coverage.md`
