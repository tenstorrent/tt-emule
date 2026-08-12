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

## What makes the run fail

A sanitizer violation is a defect, so the run goes **red** for it — every
finding, no allowlist. Two layers fail independently:

- each **sweep shard** fails when any of its own entries hits an `[ASAN ERROR]`,
  so the run's job list shows exactly which buckets violated. The verdict lands
  at the **end** of the shard: every entry still runs to completion (a violation
  aborts only the offending test's forked child), and the findings check is the
  script's last step;
- the **report job** fails when any finding exists across the whole set, when
  the sweep produced **no data** (no shard result files at all), or when any
  shard failed to report in — it was INVALID, crashed, or never started.

An **invalid** run fails for a different reason than a finding does: if no entry
reached the emulator, "0 findings" means the lane is broken, not that metal is
clean.

`ci-asan-sweep.sh` enforces the `emule_postflight` contract per shard, failing if
no log contains the emule-mode banner, if no logs were produced at all, or if
real-hardware markers appear.

**Per-shard verdicts, one aggregate.** The sweep matrix runs with `fail-fast`
off, so a red shard (findings or INVALID) never cancels the other five, and the
report job runs on `always()` to judge the whole set. A shard that dies before
its validity gate would leave a hole — it contributes no results, which reads
exactly like "that shard found nothing" — so each shard writes a `shard-<n>.ok`
marker **only** when it passes its validity gate, and the report fails if any
marker is missing. A slice of the test set that never ran cannot be called clean.

That gate runs *inside* a shard, so it cannot speak for shards that never
started — a failed build, or an Actions outage killing jobs during "Set up job".
The report covers that case separately: when no shard produced any result file at
all, the summary, the `no_data` flag in `headline.json`, and the Slack post all
say **NO DATA** rather than "no findings", and the parser emits a CI error
annotation. An absent sweep and a clean sweep must never look alike.

## Following a run while it happens

Each entry's pytest output is **teed**: it streams into the job log as it runs and
is also written to the artifact. A sweep can take two hours, so a job log showing
only the command line makes a running shard impossible to follow and hides a
violation until the artifact is downloaded.

Each entry is wrapped in a collapsible `::group::`, and after it closes an
ungrouped one-liner reports what that entry tripped — a `::warning::` with the
per-category counts if anything fired, otherwise a plain "no ASAN hits". So the
run reads as a scannable timeline even with every group collapsed, and hits
surface as annotations on the run page.

## Buckets and owners

The sweep is split into buckets grouped by **code owner**, not by size, so a red
job names the team to hand it to. The table lives in
`scripts/asan_sweep/buckets.py` and drives the job matrix directly, so the job
names cannot drift from what actually runs.

| Bucket | Owner | Entries |
|---|---|---|
| `eltwise` | eltwise | eltwise-1..4 |
| `matmul-reduce-fused` | mmfusedreduce | matmul, reduce, fused-1, fused-2 |
| `conv-sdpa` | convolutions + sdpa | conv, sdpa, indexer-score |
| `data-movement` | ops-data-movement | data-movement |
| `pool` | convolutions | pool |
| `misc` | ttnn-core | misc-ops, core-ttnn-unit-test |

Ownership comes from tt-metal's CODEOWNERS rules for each entry's test path.
`matmul-reduce-fused` is kept apart from `conv-sdpa` because those teams share no
members at all, whereas convolutions and sdpa do overlap.

Sizes are uneven as a consequence: `eltwise` is ~63% of the test set and is the
run's critical path, while `data-movement` is a few minutes. Splitting it is a
wall-clock decision rather than an ownership one, so it stays whole for now.

The `plan` step fails the build if the manifest gains an entry that no bucket
claims. An unassigned entry would simply never be swept, and every bucket would
still report success — coverage would shrink with nothing to show for it.

The `only` dispatch input collapses the matrix to a single ad-hoc bucket, for
iterating on one entry without paying for a whole sweep.

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

## Per-check selection

The `asan-checks` dispatch input becomes `TT_METAL_EMULE_ASAN_CHECKS` for the
run: `all` (the default), or a comma-separated allowlist of `uaf, host_align,
metadata, oob, padding, semaphore, cb_boundary, cb_reservation, noc_race,
noc_align, dirty_cb, object_intent`. Absent, empty, or `all` means every check,
so a run that does not set it behaves exactly as before the option existed.

Narrowing matters because a violation aborts the process: the loudest check hides
every other one in that test. An unrecognized name falls back to every check
rather than to none — a typo must not be able to silence the sanitizers and report
a clean run — and `cb_reservation` stays on even when the list names something
else, since gating it turns an over-reserve into a silent deadlock instead of a
clear report.

**Skipping just Dirty CB** has its own dispatch flag, `skip-dirty-cb`, because it
is the common case: Dirty CB is usually the check that fires first, so turning
only it off is how you see what it was masking (that is how earlier manual sweeps
surfaced the OOB and CB-undersize bugs underneath it). The flag sets
`TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB=1`, a subtractive override applied after the
check list, and leaves every other check on.

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
    --out-summary /tmp/summary.md --out-dev /tmp/findings.md
```

`--dry-run` prints each entry's command without executing it, which is the
cheapest way to confirm the injected flags.

Build the emule tt-metal at the pin, not at main: the sanitizer state is shared
between the host lib and the JIT kernel headers, so a host lib older than the
kernel-side headers segfaults inside the ASAN hooks rather than reporting
anything. `.github/scripts/ci-build.sh` builds the right configuration.
