# tt-emule — Claude project context

This file is the entry-point context for any agent working on
tt-emule. It defines the project rules **and** the in-`.claude/`
orientation guide (skills, agents, references) for implementing
faithful mock APIs.

## Project rules (read first, override defaults)

- Always try to reduce the API surface included in `jit_hw/` — prefer
  the real tt-metal headers when possible.
- tt-emule-backed software emulation always runs in **slow dispatch**
  mode (`TT_METAL_SLOW_DISPATCH_MODE=1`).
- The default architecture is **wormhole n150** unless specified
  otherwise.
- Compile with **clang-20** (the toolchain pinned by
  `cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake`).
- Always run the per-arch regression scripts after code changes and
  log the **full** output. Run them **sequentially** (shared JIT
  cache).
- **Prefer making an existing tt-metal test pass over adding a new
  one.** The goal is faithful silicon behavior, which the canonical
  tt-metal suites already assert — a new test is a last resort, only
  when the coverage genuinely does not exist and cannot be reached by
  fixing the mock so an existing test passes. When a new tt-metal test
  is truly unavoidable, add it under **`tests/emule/`** (the dedicated
  emule-only test tree in tt-metal — e.g. `tests/emule/ccl/`), **never**
  under `tests/ttnn/...` or another canonical suite. Its fixtures
  (`mesh_device`, `device_params`, `silicon_arch_name`, …) resolve from
  the root tt-metal `conftest.py`, so no per-dir conftest is needed.
  Wire it into the emule pytest runners (`scripts/run_ttnn_pytests_*.sh`)
  and bump `tt-metal-pin.txt` to the companion commit. Format Python to
  tt-metal's pre-commit config (**black line-length 120**, isort,
  autoflake) before pushing.
- When fixing a failure in a mock API, the goal is **faithful to the
  canonical silicon implementation**. Don't create parallel or
  different code paths to work around bugs — if it works in silicon,
  the mock must be functional and correct against it.
- Be **skeptical of automated PR review comments** (e.g. GitHub
  Copilot's reviewer). Judge each suggestion against these rules
  before acting — don't apply it reflexively. Decline, with a reasoned
  reply that cites the code, when a suggestion would add a defensive
  or divergent code path, mask a caller bug instead of surfacing it,
  or otherwise contradict the faithful-to-silicon / single-code-path
  rules above. Prefer a no-op `ASSERT` that documents a contract over
  a silent clamp.
- The authoritative `src/` + `include/` file/symbol index lives at
  `references/structure.yaml` (replaces `STRUCTURE.md`). Query with
  `scripts/find_symbol.py <name>` (flags: `--kind`, `--path-prefix`,
  `--list-symbols`, `--summary`; `--supports <sym>` / `--shadows
  <include>` for an early-detect "is this kernel symbol/include modelable
  by emule?" probe); plain `grep` works for casual lookups. The `symbols`
  (and `path`/`section`) of every entry are **regenerated** from the
  source by `scripts/gen_structure.py --write` and enforced in sync by a
  pre-commit hook + the Structure Index CI gate (`--check`) — so don't
  hand-edit `symbols`; instead run `--write` after adding/removing/
  renaming a file or top-level symbol. The `summary` field IS
  hand-written prose: a brand-new file needs a one-line summary (the
  generator inserts a `TODO:` sentinel that fails `--check` until you
  replace it).
- `IMPLEMENTATION_REPORT.md` is the documentation entry point and
  index; per-subsystem references live under `docs/`
  (l1/dram/dest/cb/noc-emulation, metal-integration,
  tilize-untilize-pack, cb-dataformat, mem-zeros-handling,
  DFB/QUASAR). Read the relevant doc before changing a subsystem
  and update it in the same change. Verify every claim against
  actual code before writing it down — the docs must never assert
  something the source doesn't do.
- Committed documentation describes the project's current, actual
  scope and behavior — not how it got there. Don't include historical
  information, change narratives, or changelogs (e.g. "previously/now",
  "the bug was", "the fix") unless explicitly asked; that belongs in
  untracked local notes.

## When to use what

| Need | Use |
|---|---|
| What does silicon API X do, so I can mock it? | `/arch-lookup "X"` skill |
| Step-by-step workflow for adding a new mock | `/implement-mock <api>` skill |
| Add a compute-kernel LLK shim (`<op>_tile` in `include/jit_hw/api/compute/`) | `/compute-llk-bringup` skill |
| Diagnose ATOL/PCC failures (wrong bytes, partial zeros, off-by-N) | `/memory-debug` skill |
| A tt-metal pin bump turned the regression red (device-open crash, JIT-compile error, hang) | `/uplift` skill |
| Review an open PR, address its comments, and rebase/prep it to hand off for landing | `/shepherd-emule-pr` skill |
| Parallelize a sweep of ≥4 similar mocks | `/parallel-mock-implementation` skill |
| Map HW concept → existing emule strategy | `references/emule-mapping.md` |
| Where in the pipeline to inject a change | [`docs/api-injection-points.md`](../docs/api-injection-points.md) |
| Verify a mock is complete | `/verify-mock` skill |
| About to add a hack to make something pass (or editing code that carries one) | `/workarounds` skill |
| Deep arch question on Wormhole/Blackhole/Quasar | sage agents (launched by arch-lookup) |

## Skills

- **arch-lookup** — orchestrates parallel `sage-*` agents to answer
  silicon HW questions with an emule-mapping perspective. Adapted
  from `tt-metal/tt_metal/tt-llk/.claude/skills/arch-lookup`.
- **implement-mock** — end-to-end workflow: arch-lookup → pick
  strategy (A/B/C) → implement → verify → document.
- **memory-debug** — diagnostic playbook for data-corruption / ATOL
  failures. The discriminator (exact-zero vs random-wrong), the
  5-step trace pass (compute→CB→writer→L1), and five recurring
  root-cause classes (bank topology, per-shard math, face-3 loss,
  DAZ/FTZ, extern array mismatch). Distilled from rounds 7-12.
- **compute-llk-bringup** — specialization of `implement-mock` for
  compute LLK shims (Strategy A applied to ops under
  `include/jit_hw/api/compute/`). Has the shim-pattern catalog,
  `sfpu_split_includes.h` wiring, polynomial-port recipe, PCC triage.
- **parallel-mock-implementation** — batching primitive: spawn one
  sub-agent per file via the Workflow tool, orchestrator handles
  centralized wiring afterward. Invoked from `/implement-mock` and
  `/compute-llk-bringup` for sweeps.
- **uplift** — methodology for tt-metal/tt-umd pin-bump regressions:
  read-the-artifact triage, the two-stage build-and-run bisection
  (metal-source vs umd bump), proving the mechanism from the upstream
  PR diff, the two emule fix classes (shared-runtime behavioral
  regression vs jit_hw API-surface drift), the cross-repo push chain,
  and verification (oracle before/after + curated-suite membership
  via `--collect-only`). Prove by build+run, never by `git log`.
- **shepherd-emule-pr** — methodology for taking an open PR to
  mergeable against the repo ethos: gather all three review surfaces,
  verify every comment by build (the JIT compile-probe) not by trusting
  "addressed", own ethos pass for the bug classes reviewers miss
  (dangling refactored symbol, divergent-duplicate/ODR, silent silicon
  divergence, superseded-by-main), then fix / register via
  `/workarounds` / trim → rebase locally → hand the developer a
  change+verification summary and drafted per-point comment to review
  and push (the human stays in the loop for the force-push).

## Agents (launched by skills, can be invoked directly via Agent tool)

| Agent | Scope | Primary docs |
|---|---|---|
| `sage-wormhole` | `tt_llk_wormhole_b0/` | DeepWiki tt-isa-documentation + assembly.yaml |
| `sage-blackhole` | `tt_llk_blackhole/` | DeepWiki + assembly.yaml |
| `sage-quasar` | `tt_llk_quasar/` | Confluence + assembly.yaml (NO DeepWiki) |

Each agent has MCP access for the relevant doc source and Read/Glob/Grep
for the LLK code. Each agent's response ends with an **"Emule mapping"**
+ **"Recommendation"** section — that's the emule-specific adaptation
beyond the upstream tt-llk versions.

## References

- `references/emule-mapping.md` — HW concept → emule simulation strategy catalog
- [`../docs/api-injection-points.md`](../docs/api-injection-points.md) — where in the pipeline emule intercepts

## Skills (additional)

- `skills/verify-mock` — pre-implementation through verification (invoke as `/verify-mock`)
- `skills/workarounds` — registry of deliberate, non-ideal workarounds currently
  live in the tree (hacks accepted to make something pass, **not** normal practice).
  Read before adding any workaround or when editing code that carries one; each
  entry records the hack, why it bends the rules, the real root cause, and the
  path to removing it (invoke as `/workarounds`).

## Quick links to project-level context

- [`BUILD_GUIDE.md`](../BUILD_GUIDE.md) — build + test setup
  (`TT_METAL_DIR`, regression scripts)
- [`IMPLEMENTATION_REPORT.md`](../IMPLEMENTATION_REPORT.md) — docs
  index across the per-subsystem references under `docs/`.

## Source skills (upstream — for reference)

- `tt-metal/tt_metal/tt-llk/.claude/skills/{arch-lookup,debug-kernel,port-kernel,run-test}/` — original tt-llk skills
- `tt-metal/tt_metal/tt-llk/.claude/agents/{sage-wormhole,sage-blackhole,sage-quasar,llk-debugger,llk-test-runner}.md` — original tt-llk agents

The tt-emule versions diverge from upstream in three places:
1. Search scope adds emule paths (`tt-emule/include/jit_hw/`, runner cpp)
2. Every response ends with **"Emule mapping"** + **"Recommendation"**
3. References cite the emule-mapping catalog so agents speak the
   same vocabulary about A/B/C strategies
