# Parallel sweep plan: remaining BH pytest failures

Companion to `docs/bh-bring-up-methodology.md`. The methodology
encodes how to fix **one** failure; this doc is the action plan for
attacking the remaining N failures in parallel without wasting agent
budget on stale baseline entries or fighting each other for the same
file.

## Status entering the sweep

- Baseline: `known-failures-blackhole-pytests.txt` recorded
  78 fail / 73 pass of 151 entries at commit `0dade57`.
- Two methodology cases have landed since (`e99b4cd`, `670d934`).
  Both likely unblocked entire failure families:
  - MEM_ZEROS restoration → every test that uses `zero_tile` /
    `zero_buffer` / the silicon zero-fill idiom (suspected ~30+
    entries — reduction family, softmax family, several SIGABRTs).
  - `fast_tilize_init_skip_remap` shim → tilize callers on the BH
    branch (suspected ~5-10 entries).
- Net: baseline counts are stale. **The sweep MUST start with a
  re-baseline**, not the existing file.

## Phase 0 — re-baseline (single sequential run, ~4 hours)

```bash
TT_EMULE_ARCH=blackhole bash scripts/run_ttnn_pytests_blackhole.sh \
    2>&1 | tee /tmp/bh_sweep_v2.log
TT_EMULE_ARCH=wormhole  bash scripts/run_ttnn_pytests_wormhole.sh  \
    2>&1 | tee /tmp/wh_sweep_v2.log
```

Then regenerate the failures file (using the parser already proven
during the v1 baseline):

```bash
python3 .github/scripts/parse-pytest-log.py /tmp/bh_sweep_v2.log \
    > known-failures-blackhole-pytests.txt
```
(parser is in this branch's history at `0dade57`'s commit message — promote
it to a script if not already done.)

Commit the new baseline as a single commit on `arminale/bh-parity`.
The parallel sweep operates on **this** file, not the v1 baseline.

**Stopping rule for Phase 0**: any entry where WH also FAILS gets
flagged as pre-existing emule bug, NOT included in the parallel
sweep. File a separate issue per such entry.

## Phase 1 — cluster by signature class

Group the v2 failures into clusters. From the v1 data the suspected
post-fix clusters are:

| Cluster | Signature | Estimated count | Methodology branch |
|---|---|---|---|
| **L1**: missing BH-only LLK shim | `error: use of undeclared identifier 'X'` where X is in `ckernels/blackhole/.../experimental/llk_*` | 10-20 | /compute-llk-bringup, Strategy A |
| **L2**: missing non-LLK BH symbol | `undeclared identifier` for non-LLK BH function | 5-10 | implement-mock, Strategy A |
| **N**: wrong output (PCC / ATOL) | pytest assertion at the comparison line, no compile error | 10-20 | /memory-debug discriminator |
| **C**: SIGABRT signal 6 | `:-1: running the test CRASHED with signal 6` after kernel JIT-compiled cleanly | unknown (was 519 subtests pre-fix; re-baseline tells truth) | /memory-debug §Crashes, gdb |
| **T**: hang / timeout | `pytest killed by 900s wall-clock` | <5 | inspect runner state; CB/sem sync most likely |
| **R**: pre-existing emule bug (fails on WH too) | matches anything else **and** WH also fails | filed as issues, NOT in sweep | n/a |

**Clustering script** (deterministic, run after Phase 0):

```bash
python3 .github/scripts/cluster-failures.py \
    known-failures-blackhole-pytests.txt \
    --wh-baseline /tmp/wh_sweep_v2.log \
    --out clusters/
```

Produces one file per cluster + a `pre-existing.txt` for class R.
The L1 cluster should be the **largest** — that's the dominant
failure mode and where parallel agents pay off the most.

## Phase 2 — parallel attack (the bulk of the work)

Use the `Workflow` tool with one sub-agent per failure entry (NOT
per cluster — agents need entry-level context). Concurrency caps at
~16 per `Workflow` invocation; for >16 entries, batch via
`pipeline()`.

### Per-agent contract

Each sub-agent gets:
1. **Entry name + test path** (e.g. `dm_test_repeat_interleave` +
   `tests/ttnn/unit_tests/operations/data_movement/test_repeat_interleave.py`)
2. **Cluster classification** (L1 / L2 / N / C / T) — tells the agent
   which methodology branch to take
3. **`docs/bh-bring-up-methodology.md`** as the workflow reference

Each sub-agent runs:
1. Step A (rerun on BH against HEAD). PASS → report "resolved",
   stop.
2. Step A.5 (rerun on WH). FAIL on WH → report "pre-existing emule
   bug", stop. Don't fix.
3. Step B-G per its cluster's methodology branch.
4. **Stops at**: a single-file minimal diff. Tests pass on BH AND
   no regression on WH AND no regression on the WH sentinel
   (`test_untilize_single_core_interleaved_to_sharded`). Reports
   the diff + verification output.

### Per-agent return shape (StructuredOutput)

```json
{
  "entry": "dm_test_repeat_interleave",
  "outcome": "fixed" | "resolved_by_upstream" | "preexisting_emule_bug" | "deferred",
  "cluster": "L1" | "L2" | "N" | "C" | "T",
  "diff_summary": "added fast_tilize_init_skip_remap shim in api/compute/tilize.h",
  "files_touched": ["include/jit_hw/api/compute/tilize.h"],
  "bh_before": "24 failed / 8 passed",
  "bh_after":  "32 passed",
  "wh_sentinel": "12 passed",
  "deferred_reason": "<only if outcome == deferred>"
}
```

### Concurrency limits

- **Same file**: agents whose `files_touched` overlap **must** be
  serialized. Detect by static analysis: each cluster's likely-target
  file set is known (e.g. all L1 hits land in
  `include/jit_hw/api/compute/`). Serialize within a directory.
- **Same JIT cache**: agents that JIT-compile concurrently will
  thrash the cache. Each agent must `rm -rf /tmp/tt_emule_jit_*`
  in its own pre-flight (already standard in the methodology).

### Token budget

Per the `Workflow` tool's defaults, ~16 concurrent agents × ~100k
tokens per agent ≈ 1.6M tokens per pass. For ~50 estimated
remaining failures, that's 3-4 sequential passes. Budget the whole
sweep at **~5M output tokens**, hard ceiling.

## Phase 3 — review pass

After every Workflow returns:
1. Aggregate `outcome` field across all sub-agents.
2. For `fixed` outcomes: apply the diffs in dependency order
   (cluster L1 first since it unblocks runtime tests; then L2; then
   N; then C; then T).
3. **Re-run the full BH sweep** between cluster transitions — fixes
   in one cluster often unblock entries in others (Case 1 fixed 30+
   entries in one diff). Re-cluster the diminished failure list
   before the next pass.
4. For `preexisting_emule_bug` outcomes: file GitHub issues with
   the agent's WH-also-fails evidence. These move out of the BH
   parity scope.

## Phase 4 — stopping conditions

Stop the parallel sweep (declare BH bring-up done) when **any** of:

- The BH pytest pass rate matches WH's pass rate within ±2 entries.
- Three consecutive sweep passes produce zero new `fixed` outcomes
  (every remaining failure is `deferred` or `preexisting_emule_bug`).
- Token budget exhausted; produce a summary doc of what's left.

## Phase 5 — wire CI for sustained coverage

Once the bulk of failures are fixed, the BH ttnn pytest matrix
(already added in commit `c8a1e7c`) enforces no regressions. At
that point:

1. Move `known-failures-blackhole-pytests.txt` to a CI-classified
   allowlist (only the still-failing entries — `deferred` or known
   `preexisting_emule_bug`).
2. New entries that fail on BH but not in the allowlist fail the PR.
3. Match the structure of `known-failures-wormhole.txt` (already
   wired in the workflow).

## Out of scope

- **Implementing missing tests** — only fixing emule's response to
  the existing curated list.
- **Refactoring the skills** — the methodology doc is the
  authoritative workflow.
- **The 8 entries marked "FAIL without pytest summary" in v1
  baseline** — these are SIGABRTs that crashed the process before
  pytest could write its summary. They go in cluster C and need
  gdb-driven triage; that's its own follow-up after the L1/L2
  clusters land.

## Risks

1. **Sub-agents bikeshedding**: each agent is told to stop at
   "single-file minimal diff". The contract is strict; the methodology
   doc's "no defensive error paths for impossible scenarios" rule
   (from CLAUDE.md) applies recursively to sub-agents.
2. **Same-file collisions**: enumerated in Phase 2's "Concurrency
   limits". The cluster-to-directory mapping is deterministic
   enough that pre-batching by directory is sufficient.
3. **Pre-existing-emule-bug rate**: if Phase 0's WH re-baseline
   shows many entries fail on WH too, the curated WH list itself
   is stale and needs maintenance. That's a finding, not a sweep
   failure — the methodology surfaces it directly.
4. **Re-baseline cost (~4 hours)**: hard to compress. Phase 0 is
   the largest sequential bottleneck. Accept it.
