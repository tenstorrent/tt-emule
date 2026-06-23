# Post-commit pass-rate sweep (nightly CI)

The post-commit sweep runs tt-metal's per-arch **post-commit ttnn test lane**
under the tt-emule backend and reports a headline **pass rate** plus failure
clustering. Unlike the PR regression (`pr-metal-regression.yml`), which is a
zero-tolerance merge gate over a *curated* all-passing subset, the sweep runs
the **whole** post-commit lane — including the parts emule does not yet
support — so the pass rate is a **growing coverage metric, not a gate**.

It runs nightly (`.github/workflows/nightly-post-commit-sweep.yml`) and is
reproducible locally with the same scripts.

- **Scripts:** `scripts/post_commit_sweep/` (shared by CI and local)
- **CI adapter:** `.github/scripts/ci-post-commit-sweep.sh` (one shard)
- **Workflow:** `.github/workflows/nightly-post-commit-sweep.yml`
- **Lineage:** generalizes the one-off pipeline on branch
  `arminale/bh-ci-sweep` (`snapshots/bh_sanity/` + the `/bh-ci-sweep` skill).
  That pipeline was local/manual/single-box (6–18h); this is committed,
  arch-parameterized, sharded, and CI-wired.

## What "the post-commit lane" means

The source of truth is tt-metal's `tests/pipeline_reorg/ttnn-tests.yaml` — the
same manifest tt-metal's own CI consumes. The sweep selects the entries where
**`merge_gate` is falsy AND the arch's SKU is in `skus`**:

| emule arch | manifest SKU (entry filter) | emule device (runtime) | cluster descriptor |
|---|---|---|---|
| `blackhole` | `bh_p150b_civ2` | `P100` | `blackhole_P100.yaml` |
| `wormhole`  | `wh_n300_civ2`  | `N150` | `wormhole_N150.yaml` |

The SKU and the emule device are **independent axes**: the SKU picks *which
manifest entries* run; the device/descriptor is *how emule presents the device*
(single-device, mirroring `scripts/run_ttnn_pytests_<arch>.sh`). Today the
nightly runs **blackhole only** (14 entries); wormhole is a one-line matrix add.

## Granularity: one manifest entry = one unit of work

The unit of work — and the sharding unit, and the JUnit XML unit — is a
**manifest entry** (e.g. `ttnn eltwise group 1`, often a whole directory), not
an individual file. This is the faithful "what metal runs" representation and
keeps the work list small (~14 units), trivially shardable.

### Identical-invocation invariant

Every test path, original env var, and pytest arg from the manifest `cmd` is
preserved verbatim. The only differences are an **additive** layer:

- emule env (`TT_METAL_EMULE_MODE`, `TT_METAL_SLOW_DISPATCH_MODE`),
- variant env (`MESH_DEVICE`, `TT_METAL_MOCK_CLUSTER_DESC_PATH`),
- runtime env (`PYTHONPATH`/`LD_LIBRARY_PATH`/`TT_METAL_HOME`/…),
- the `timeout <wallclock>` wrapper, `--junitxml=`, redirect,

plus two status-snapshot transformations the built-in audit allows:

1. **`-x` stripped** — else the first failure truncates the entry and distorts
   the headline.
2. **import-error pre-filter** — see "The pytest-9.0.3 collection-error
   pre-filter" below.

(`--forked` is *not* a static transformation — it is applied per-pass by the
hybrid runner; see "Run modes".)

## Run modes — the hybrid default

At entry granularity a whole directory runs in one pytest process, so a single
C-level hang (pytest-timeout cannot preempt C code) or `SIGABRT` voids the
entry's **entire** XML. `--forked` (per-test process isolation) prevents that —
but it is expensive.

**Measured `--forked` overhead** (`ttnn data movement group`, 103 executed
tests, warm JIT cache, identical results both ways): forked **144s** vs
no-forked **63s** → **~0.79 s/test** (process spawn + `import ttnn` + emule
device init paid per test instead of once per entry). Extrapolated to a full BH
sweep (~35k executed tests / 6 shards ≈ 5.9k tests/shard) that is **~1.3 h/shard
of forking overhead** — real, even if within the 4-hour shard timeout.

So the default is **hybrid**:

1. **Pass 1 — no-forked** (fast: one process per entry).
2. **Pass 2 — `--forked` retry of only the entries that left no usable XML**
   (missing / malformed / zero-testcase = a crash, hang, or wallclock SIGTERM).
   The retry isolates the offending test so the rest of the entry is recorded,
   paying the fork cost on just the failed tail instead of the whole suite.

Each shard does its own two passes (self-contained — works the same in CI and
locally). Overrides: `--forked` / `SWEEP_FORKED=1` forks every entry (no retry);
`--no-retry` / `SWEEP_NO_RETRY=1` runs pass 1 only (fastest, least robust).

`sweep.py expand` runs the audit; it fails closed if the emitted invocation
diverges from the manifest beyond the allowed layer.

## The pytest-9.0.3 collection-error pre-filter

`--continue-on-collection-errors` is **broken in pytest 9.0.3**: a single
`ImportError` (e.g. a test file that transitively imports `transformers` /
`IPython`, absent from the lean toolchain venv) aborts the **entire** entry —
pytest collects the good items then runs **zero**. The flag suppresses the
message but not the bail-out.

So `sweep.py` runs a fast `--collect-only` pre-pass per entry, identifies the
import-error files, and neutralizes them two ways (both required):

- **drops** any matching explicit positional target (`file.py::node`) — here
  `--ignore` is overridden by the explicit arg (`found no collectors`);
- **`--ignore=<file>`** for directory-target entries where the bad file is
  collected via the directory.

The dropped files are written to `<slug>.excluded.json` so the parser still
reports them as **excluded files** (an env gap, not an emule failure) — they are
*not* counted in the pass-rate denominator.

## Wallclock + fleet policy

- Per-entry wallclock backstop = `3 ×` the manifest SKU timeout (minutes),
  capped at 90 min. Hitting the cap surfaces the entry as **truncated** in the
  report (it contributes 0 to the headline).
- Fleet policy (issue **#189** — "pr-metal-regression is DDoSing our CI
  fleet"): the nightly runs **off-peak (02:00 UTC)**, with **6 shards at
  `max-parallel: 2`**, behind a `concurrency` group that forbids overlapping
  runs, every job `continue-on-error`. It deliberately trades wall-clock for a
  gentle footprint and **never gates a merge**.

## Reporting

`parse_sweep_results.py` aggregates every shard's JUnit XML + logs and emits:

- `exec.md` — leadership headline + top gaps,
- `dev.md` — full per-file / per-class detail (blocker taxonomy, JIT compile-
  error root causes, top numeric-drift files, excluded + truncated entries),
- `headline.json` — machine-readable headline (for trend tracking / alerting),
- `summary.md` — compact block rendered into the GitHub Actions **job summary**.

The nightly also uploads the reports + XMLs as artifacts (30-day) and posts the
headline to Slack (`.github/notify-stub.md`, `NIGHTLY_SLACK_WEBHOOK_URL`; no-ops
if unset).

### Blocker taxonomy

Each failing testcase is classified (per-testcase, not per-file-sample):

| Class | Meaning | Typical fix |
|---|---|---|
| Missing op / unjitted symbol | JIT compile error — emule has no mock for an LLK/NoC symbol | add the shim (`/compute-llk-bringup`) |
| Numeric drift (PCC/ATOL) | wrong values — usually data-movement/tilize/untilize modelling | `/memory-debug` |
| Setup crash | SIGSEGV/SIGABRT during setup | triage root cause |
| Mock-API drift | signature/attribute mismatch vs the current pin | `/uplift` |
| Multi-host / grid gate | needs >1 device / more cores than emule presents | out of scope (single-device) |

## Running it locally

```bash
# Full blackhole sweep (slow — hours; mirrors the nightly):
TT_METAL_DIR=../tt-metal bash scripts/post_commit_sweep/run_post_commit_sweep.sh

# Fast debug loop — one small entry through the whole pipeline:
TT_METAL_DIR=../tt-metal SWEEP_ONLY=ttnn-data-movement-group \
  bash scripts/post_commit_sweep/run_post_commit_sweep.sh

# Just expand + audit the entry list (no execution):
python3 scripts/post_commit_sweep/sweep.py expand \
  --arch blackhole --manifest ../tt-metal/tests/pipeline_reorg/ttnn-tests.yaml
```

Reports land in `$OUT_DIR` (default `../sweep-out/<arch>/`). `pytest-timeout` is
installed into the toolchain venv on first run; `pytest-forked` / `pytest-split`
are already present.

## Adding wormhole (or a new arch)

1. Add the arch to `ARCH_CONFIG` in `sweep.py` (SKU + mesh device + descriptor).
2. Add it to the `matrix.arch` list in `nightly-post-commit-sweep.yml` (both the
   `sweep` and `report` jobs).

No other change — the scripts and parser are arch-agnostic.
