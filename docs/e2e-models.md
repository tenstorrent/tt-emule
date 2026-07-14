<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# End-to-end model regression

Runs full tt-metal model demos (e.g. `simple_text_demo.py`) end-to-end under
software emulation, without any local edits to the upstream demo. This gives a
reproducible CI signal for whole-model correctness (token accuracy) and
crash-free execution, complementing the op-level ttnn pytest and C++ regressions.

## Why a vendored copy

Running a model demo on emule needs two behavioral changes that upstream does not
expose as run-time knobs, and cannot be merged upstream (the emule team is
codeowner of `tests/emule/` only):

1. **Host sampling** — force `_supports_on_device_sampling = False`
   (`models/tt_transformers/tt/model.py` enables on-device sampling when the
   vocab fits, which hits an unported `manual_seed` path under emule).
2. **Trace off** — emule runs slow-dispatch / teacher-forcing only.

Rather than monkeypatch at run time, the emule-owned test is a **vendored copy**
of the demo with those two lines applied inline, each tagged `# emule:`:

- `tt-metal/tests/emule/models/test_tt_transformers_text_demo.py` — copy of
  `models/tt_transformers/demo/simple_text_demo.py` + the two `# emule:` edits.
  Its header records the source path, the vendored pin SHA, and the re-sync
  recipe.
- `tt-metal/tests/emule/models/conftest.py` — copy of the demo's options-only
  conftest (so `request.config.getoption(...)` resolves). All fixtures come from
  the root tt-metal `conftest.py`.

Everything else — parametrization, model/tokenizer creation, reference-`.refpt`
accuracy — is reused verbatim.

## How it runs

Layered exactly like the ttnn pytest regression:

- `scripts/run_e2e_models.sh` — the runner. Reads `TT_EMULE_ARCH`
  (`wormhole` → `MESH_DEVICE=N150` + `wormhole_N150.yaml`; `blackhole` → `P150` +
  `blackhole_P150.yaml`), sets the emule env, and drives a curated `run_model`
  entry list against the vendored test. Round-robin `SHARD_INDEX/SHARD_COUNT`,
  junit XML, PASS/FAIL summary, exit code. `run_e2e_models_{wormhole,blackhole}.sh`
  are thin arch wrappers.
- `.github/scripts/ci-e2e-models.sh` — CI wrapper (mirrors `ci-ttnn-pytests.sh`):
  env contract, `_ttnn.so` symlink, `HF_HOME` cache, tee log, exit-code
  propagation.
- `e2e-models` job in `.github/workflows/nightly-metal-upstream.yml` — a matrix
  over `arch × shard`, `continue-on-error: true`, fanning out in parallel with
  the `test` (regression) and `ttnn-pytest` jobs. It is fed by a **second**
  `build-pinned` job in that workflow that builds tt-metal at the **pinned** SHA
  (`tt-metal-pin.txt`, same as PR Metal Regression), so — unlike the main-tip
  regression/ttnn-pytest lanes — the model signal is reproducible and moves only
  when a PR bumps the pin. Installs `transformers`/`torchvision` into the
  toolchain venv (no torch bump), caches the ungated
  `unsloth/Llama-3.2-1B-Instruct` weights (no HF token), and uploads the log +
  junit XML.

## Current coverage

Llama-3.2-1B-Instruct (ungated unsloth mirror):

| Entry | Selector | What it checks |
|---|---|---|
| `llama1b_token_matching` | `-k performance-ci-token-matching --max_generated_tokens 48` | Teacher-forced top1/top5 vs `reference_outputs/Llama-3.2-1B-Instruct.refpt`. Primary correctness gate; validates deep-position SDPA-decode. Capped at 48 tokens (the row defaults to 500; at ~20s/token that overruns the per-entry timeout). Prefill is the reference's first half (~512 tokens) so decode starts at pos 512 — the cap keeps the multi-chunk deep-position path, just over a shorter span. |
| `llama1b_batch1` | `-k "performance and batch-1" --max_generated_tokens 32` | Full 16-layer prefill+decode generative smoke (host sampling); crash-free E2E. |

Blackhole is **warning-only** until validated, and runs the batch-1 smoke only:
the token-matching gate is wormhole-only because upstream's centralized accuracy
thresholds have no P150 (blackhole) entry (the emule blackhole compute itself is
correct — identical top1/top5 to wormhole). The whole `e2e-models` job is
`continue-on-error`, so blackhole never blocks regardless.

## Running locally

```bash
export TT_METAL_DIR=/path/to/tt-metal
TT_METAL_DIR=$TT_METAL_DIR bash scripts/run_e2e_models_wormhole.sh
```

Expect `Results: 2 passed, 0 failed`. The token-matching entry clears the
`.refpt` top1/top5 thresholds on the SDPA-decode-fixed tree over its 48-token
window (top1 89.58% / top5 97.92%).

## Adding a model

- **Same demo, new model**: add a `run_model <name> <HF_MODEL> -k "<selector>"`
  line in `scripts/run_e2e_models.sh`. No new test file. The runner forces
  `CI=false` so the demo's `is_ci_env` gate (`if is_ci_env: if not ci_only: skip`)
  never drops a non-`ci_only` entry — your `-k` selection is the sole filter.
- **Different demo / family**: vendor that demo the same way (copy + its
  `# emule:` edits + header) under `tests/emule/models/`, then add a runner
  entry.

## Re-syncing on a tt-metal pin bump

The vendored demo tracks upstream only as of the pin SHA in its header. On a pin
bump, re-run `git show <new-pin>:models/tt_transformers/demo/simple_text_demo.py`
into the vendored file and re-apply the two `# emule:` markers. This is a
`/uplift` Step 5 item (see `.claude/skills/uplift/SKILL.md`) — the demo is not in
the C++ regression, so upstream drift is otherwise uncaught until the nightly e2e
lane runs against the new pin (same lesson as the SDPA-pytest compile break).
