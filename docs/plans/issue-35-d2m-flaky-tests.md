# Plan: Root-cause and fix the 68 D2M flaky tests (Issue #35)

> **Status**: Shelved. Plan written for a future agent to pick up. See https://github.com/tenstorrent/tt-emule/issues/35.

## Context

PR #22's CI surfaced 1 WH + 67 BH D2M entries that flipped from FAIL to PASS between consecutive runs. They're marked `flaky:` in `.github/known-failures-d2m-{wormhole,blackhole}.txt` to unblock the PR, but the underlying nondeterminism is unfixed. The goal is to (a) classify every flaky entry into `stable_pass` / `stable_fail` / truly flaky via repeated sampling, then (b) fix every truly-flaky entry so the suite is deterministic.

Key facts from exploration:

- **All 68 entries cover only 3 test files**: `test_dma.py`, `test_dram_ops.py`, `test_rand.py` under `/localdev/arminale/tt-mlir/test/python/golden/d2m/`. Restricting the sampling experiment to these saves ~10× wall time.
- **Likely single root cause for `test_dram_ops` (52/68 entries)**: `tt-mlir/tools/builder/base/builder.py::_generate_random_tensor` generates inputs via unseeded `torch.randn` / `torch.randint`. Different inputs each run → different output magnitudes → PCC straddles threshold.
- **`test_dma` (14 entries)** compares bit-for-bit roundtrips, so RNG content shouldn't matter — flakiness here points at JIT-cache pollution or emule state, not inputs.
- **`test_rand_reproducibility[ttmetal]` (2 entries: WH + BH)** already uses a fixed device-side seed. Flakiness is in emule's `rand_tile` PRNG init, not Python RNG.
- **pytest conftest is NOT discovered via PYTHONPATH** — must use `PYTEST_PLUGINS=<module>` env-var to inject a plugin without modifying tt-mlir. Env propagates through `pytest --forked`.
- **`pytest-randomly` is not in `/opt/ttmlir-toolchain/venv`** and would not be the right fix anyway (it randomizes per-run; we want determinism).
- **`run_d2m_regression.sh` enumerates test files via `ls test_*.py`** and ignores positional args (they go into `PYTEST_EXTRA_ARGS`). Phase A needs a small `--test-file` override to avoid running the full 33-file suite.

## Approach

### Phase A — Local N-run sampling (~8 h wall time, mostly background)

Goal: collect per-test pass/fail outcomes across N=5 runs per arch against a fixed tt-emule + tt-metal + tt-mlir SHA.

1. **Patch `run_d2m_regression.sh`** to accept a repeatable `--test-file <name>` override that bypasses the `ls test_*.py` enumeration. Same change is useful for general dev work.
2. **Run 5 cold-cache passes per arch**, scoped to the 3 relevant files, sequentially per the project rule:

   ```bash
   for arch in blackhole wormhole; do
     for i in 1 2 3 4 5; do
       rm -rf /tmp/tt_emule_jit_*
       TT_EMULE_ARCH=$arch \
       D2M_XML_DIR=/tmp/d2m-stability/$arch/run$i \
       bash run_d2m_regression.sh --serial \
         --test-file test_dma.py \
         --test-file test_dram_ops.py \
         --test-file test_rand.py
     done
   done
   ```

   Output layout: `/tmp/d2m-stability/<arch>/run<i>/<file>.xml`. 5 × 3 × 2 = 30 XMLs.

### Phase B — Classification

New tool: `/localdev/arminale/tt-emule/scripts/d2m_stability_report.py` (~120 LOC, reuses `_node_id` parsing from `.github/scripts/classify-d2m-results.py`).

- Inputs: `--xml-root /tmp/d2m-stability/<arch>`, `--allowlist .github/known-failures-d2m-<arch>.txt`, `--only-flaky` (filter to entries with the `flaky:` prefix), `--json-out`.
- Per node-id: count PASS / FAIL / SKIP across the N XML files. Classify:
  - `stable_pass` (N PASS, 0 FAIL) → remove from allowlist.
  - `stable_fail` (0 PASS, N FAIL) → demote `flaky:` to plain known-failure; file a real bug.
  - `flaky` (1 ≤ PASS ≤ N-1) → keep `flaky:`, root-cause in Phase C.
  - `absent` (never collected) → likely stale, warn.
- Output: a Markdown table to stdout plus machine-readable JSON for re-use in the issue comment.
- **Do not** auto-edit the allowlist files. Emit a separate sed-script patch for human review.

### Phase C — Root-cause and fix

Working hypothesis: most `test_dram_ops` flakes resolve once `_generate_random_tensor` is seeded; the rest cluster around JIT cache and emule PRNG state.

**Fix 1 — deterministic test inputs (covers most `test_dram_ops`)**

Add `/localdev/arminale/tt-emule/scripts/d2m_seed_plugin.py`: a pytest plugin module that registers an autouse fixture seeding `torch` (and `numpy`, `random`) from a hash of `request.node.nodeid` — reproducible across runs, distinct per parametrization.

Wire it into `run_d2m_regression.sh` by exporting `PYTEST_PLUGINS=d2m_seed_plugin` and prepending `scripts/` to `PYTHONPATH`. `pytest --forked` propagates env to subprocesses, so the plugin loads in every forked worker.

Why this over modifying tt-mlir's builder: smallest blast radius, lives in the repo we own, doesn't require an upstream PR to land before fixing CI here. If the upstream `_generate_random_tensor` change happens later, the plugin can be deleted.

**Fix 2 — JIT cache hygiene (test_dma, test_rand)**

If sampling shows these are still flaky after Fix 1, audit:
- Whether `rm -rf /tmp/tt_emule_jit_*` runs between *parametrizations within a file* (currently it only runs between files).
- The cache-key composition in `tt-metal/tt_metal/impl/emulation/emulated_program_runner.cpp:1109-1145` — confirm it includes all inputs that would change emitted code (bindings, compile args, source mtime).
- For `test_rand_reproducibility`: run it alone with a cold cache N=20 times. If it passes deterministically alone but fails in-suite, the bug is intra-file ordering — fix with a per-test cache nuke fixture.

**Fix 3 — per-entry investigation for any remaining flakes**

After Fix 1 and (if needed) Fix 2, re-sample. For each node-id still classified `flaky`, run an isolation script:

```bash
# scratch tool, not committed
for i in $(seq 1 20); do
  rm -rf /tmp/tt_emule_jit_*
  TT_EMULE_ARCH=blackhole D2M_XML_DIR=/tmp/iso/$i \
    bash run_d2m_regression.sh --serial \
      --test-file <file> -- -k "<test_id_substring>"
done
python scripts/d2m_stability_report.py --xml-root /tmp/iso ...
```

Triage paths from here are case-by-case: PCC threshold too tight, emule modeling gap, etc. Each remaining bug gets its own follow-up issue if not trivially fixable.

### Phase D — Allowlist cleanup and re-verification

1. For each `stable_pass` entry: delete the `flaky:` line from the allowlist.
2. For each `stable_fail` entry: strip the `flaky:` prefix, keep as plain known-failure. File one tracking issue per cluster (e.g., "BH `test_dram_binary_add` consistently fails — investigate").
3. For surviving `flaky` entries after Fix 1+2: keep `flaky:` and annotate the allowlist comment with the suspected cause.
4. **Full re-verification**: one BH D2M `--serial` pass over all 33 files + one WH pass. Classifier output must show 0 new failures and 0 newly-passing.
5. Update issue #35 with the final per-entry classification table, then close.

## Critical files

- `/localdev/arminale/tt-emule/run_d2m_regression.sh` — add `--test-file` override (Phase A) and `PYTEST_PLUGINS=d2m_seed_plugin` + `PYTHONPATH` wiring (Fix 1).
- `/localdev/arminale/tt-emule/scripts/d2m_stability_report.py` — NEW classifier tool (Phase B).
- `/localdev/arminale/tt-emule/scripts/d2m_seed_plugin.py` — NEW pytest plugin (Fix 1).
- `/localdev/arminale/tt-emule/.github/known-failures-d2m-{wormhole,blackhole}.txt` — Phase D cleanup target.
- Reference for plugin design: `/localdev/arminale/tt-mlir/tools/builder/base/builder.py::_generate_random_tensor`.
- Reference for `flaky:` semantics (do NOT modify): `/localdev/arminale/tt-emule/.github/scripts/classify-d2m-results.py`.

## Verification

End-to-end success criteria:

1. **Phase A artifacts**: 30 XML files exist under `/tmp/d2m-stability/`, each with non-zero test counts. Spot-check one to confirm the JUnit format is what the report tool expects.
2. **Phase B report**: every one of the 68 entries falls into `stable_pass` / `stable_fail` / `flaky` / `absent`. Print the count of each.
3. **Fix 1 plugin loads**: smoke-test by running `PYTEST_PLUGINS=d2m_seed_plugin python -m pytest --collect-only test_dram_ops.py 2>&1 | grep -i d2m_seed` and confirm the plugin reports its fixture.
4. **Determinism check**: re-run Phase A sampling (N=3 is fine) after Fix 1. The number of `flaky` entries must drop substantially. Target: ≤ 5 surviving flakes.
5. **Final regression**: one BH `--serial` D2M pass over all 33 test files post-cleanup → classifier reports `0 new failures, 0 newly-passing`.
6. **Issue close**: post the final classification table to #35; close once allowlist is cleaned and any surviving `flaky:` entries have a documented reason.

## Out of scope

- **Modifying tt-mlir source** to seed `_generate_random_tensor` upstream. The plugin in tt-emule achieves the same effect; an upstream PR is a follow-up if/when there's appetite.
- **Modifying `classify-d2m-results.py`'s `flaky:` semantics**. The current behavior (suppress both PASS and FAIL, warn on stale matches) is correct.
- **Adding `pytest-randomly`**. It's the wrong tool here — it picks a random seed per run and prints it, which makes flakes still appear; we want fixed-per-test seeds.
- **Running the sampling in parallel mode**. Saved project rule, and would hide cache-interaction signal.
- **Building CI infrastructure for ongoing flake detection**. Separate effort.

## Risks

- **`_generate_random_tensor` may not be the only input source**. If `test_dram_ops` remains flaky after Fix 1, audit `golden_input_output_tensors` provenance — there may be a flatbuffer path that bypasses the builder.
- **PCC thresholds may be inherently tight** for some ops on BH; deterministic-but-near-threshold inputs could still produce occasional FAILs. In that case the test (or threshold) needs an upstream fix.
- **Forked plugin propagation**: pytest's `--forked` spawns workers via `os.fork`, so env-based plugin loading should Just Work, but verify before kicking off all 30 runs.
- **Wall time**: 8 h is acceptable as background, but if Fix 1 doesn't reduce flake count meaningfully we'd need additional sampling rounds. Budget 2–3 additional days for true (c)-bucket root-causing.
