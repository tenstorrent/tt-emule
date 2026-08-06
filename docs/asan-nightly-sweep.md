# Nightly ASAN Sweep

Runs tt-metal's post-commit ttnn entry set under emule with the sanitizers armed
(`TT_METAL_EMULE_ASAN=1`) to find ASAN violations in tt-metal's kernels.

This is the automated form of the manual ASAN sweeps: the emule sanitizers only
find bugs when something drives real workloads through them, and until this lane
the only thing running with ASAN on in CI was the death-test suite (Tier 3a of
`scripts/run_regression_*.sh`), which proves the *checks* work, not that *metal*
is clean.

## Relationship to the Post-Commit Sweep

Both lanes run the same entry set from the same manifest
(`tt-metal/tests/pipeline_reorg/ttnn_sanity_tests.yaml`) through the same runner
(`scripts/post_commit_sweep/sweep.py`), at the same pinned tt-metal SHA:

| | Nightly Post-Commit Sweep | Nightly ASAN Sweep |
|---|---|---|
| Sanitizers | off | on (all checks) |
| Question | does metal work under emule? | does metal violate memory rules while working? |
| Output | pass rate | deduplicated findings |
| Forking | hybrid (fork only on retry) | always, per test |
| Schedule | 03:30 UTC | 11:00 UTC |

Running at the same pin is deliberate: the post-commit sweep's result is the
ASAN-off baseline for this one. A test that already fails without the sanitizers
is not evidence of an ASAN finding.

## Cadence and cost

Nightly at 11:00 UTC, after the 03:30 post-commit sweep (240 min ceiling) and the
07:00 metal-upstream nightly have cleared, so the three never contend for xlarge
runners. `max-parallel: 2` across 6 shards follows the fleet policy from issue
#189.

ASAN is meaningfully slower than the plain sweep: the L1 chokepoint instruments
every kernel memory access, Object Intent snapshots and `memcmp`s live L1 buffers
around each launch, per-test forking costs ~0.8 s per test, and the kernel JIT
cache starts cold because ASAN is part of the cache key (kernels compile with
`-g`), so it can never reuse the plain sweep's cached `.so` files.

## Why the runner flags are load-bearing

`--asan` implies three things that are correctness requirements, not tuning:

- **`-s`** — pytest's fd-level capture points fd 2 at a temp file it drains only
  during teardown. An `[ASAN ERROR]` report ends in `abort()`, so teardown never
  runs and the whole report is discarded; the log shows only `Fatal Python error:
  Aborted`. Without `-s` the lane reports zero findings while looking healthy.
- **`-v`** — prints each nodeid before its test runs, which is what lets
  `parse_asan_results.py` attribute a log region to a test.
- **`--forked`** — the sanitizers `abort()` on the first violation, so without
  per-test forking one finding ends the entry's process and hides every later one
  in the same file. Forked, only the child dies: the parent runs the remaining
  tests and still writes a complete JUnit XML, so one entry can report many
  findings.

## Findings are report-only; invalid runs are not

A finding never fails the run — the aggregate job publishes it. An **invalid**
run does fail, because the two are not the same claim: if no entry reached the
emulator, "0 findings" means the lane is broken, not that metal is clean.
`ci-asan-sweep.sh` enforces the `emule_postflight` contract per shard, failing if
no log contains `execute_program_emulated`, if no logs were produced at all, or
if real-hardware markers appear.

## Reading the report

Findings are keyed on `(category, site)` and deduplicated. One root cause fires
in every op that touches the offending kernel, so a raw
`grep -c '[ASAN ERROR]'` overstates the bug count by an order of magnitude; the
report shows both the raw line count and the distinct finding count.

`site` is resolved most-authoritative-first: a location embedded in the message
wins over the backtrace, because the post-execution checks (Dirty CB, Object
Intent) fire after the kernel has returned and their backtrace holds only runner
and libc frames. Otherwise the `kernel_main` frame wins, then the first
non-plumbing frame, then the kernel's basename.

**Counts are a floor, not a census.** A test that trips two different checks only
ever reports the first, since the first one aborts. Per-test forking bounds that
masking to a single test rather than a whole file, but it does not remove it.

`.github/known-asan-findings.txt` splits the report into "known" and "NEW". It is
deliberately empty until findings are triaged at the current pin — allowlisting
an untriaged key hides a real metal bug.

## Per-check selection

The `asan-checks` dispatch input takes `all` or a comma-separated subset of
`uaf, host_align, metadata, oob, padding, semaphore, cb_boundary, cb_reservation,
noc_race, noc_align, dirty_cb, object_intent`. It becomes
`TT_METAL_EMULE_ASAN_CHECKS` for the run. Absent, empty, or `all` means every
check, so a run that does not set it behaves exactly as before the option existed.

Selecting a subset is how you isolate one check when another is firing first:
because a violation aborts the process, the loudest check hides the rest, and
narrowing the list is more direct than reading around the noise.

Two deliberate behaviours:

- **An unrecognized name falls back to every check**, not to none. A typo that
  silently disabled the sanitizers would produce a clean report that means
  nothing. The adapter echoes an active subset into the log so what actually ran
  is auditable, and the run summary is only ever a statement about the checks that
  were enabled.
- **CB Reservation Overflow stays on even when deselected.** Gating it would turn
  an over-reserve into a silent deadlock on the space wait instead of a clear
  report, so it is structurally load-bearing rather than optional.

The mask is parsed once on the host and armed per launch — the same path
`cb_boundary_strict` already took — so the kernel-side checks read it out of the
per-fiber state rather than calling `getenv` on the L1 access path. The
name↔bit mapping is defined once, in tt-emule's
`include/jit_hw/internal/emule_thread_ctx.h`, and consumed by both repos.
`TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB` still works as a subtractive override.

## Running it locally

```bash
# One entry, sanitizers on — the fast loop when triaging a single finding.
python3 scripts/post_commit_sweep/sweep.py run \
    --arch blackhole \
    --manifest "$TT_METAL_DIR/tests/pipeline_reorg/ttnn_sanity_tests.yaml" \
    --tt-metal-dir "$TT_METAL_DIR" --out-dir /tmp/asan-out \
    --asan --only ttnn-reduce-group

# Aggregate whatever is in an output dir.
python3 scripts/asan_sweep/parse_asan_results.py \
    --results-dir /tmp/asan-out --arch blackhole \
    --known .github/known-asan-findings.txt \
    --out-summary /tmp/summary.md --out-dev /tmp/findings.md
```

`--dry-run` prints each entry's command without executing it, which is the
cheapest way to confirm the injected flags.

Build the emule tt-metal at the pin, not at main: the sanitizer state is shared
between the host lib and the JIT kernel headers, so a host lib older than the
kernel-side headers segfaults inside the ASAN hooks rather than reporting
anything. `.github/scripts/ci-build.sh` builds the right configuration.
