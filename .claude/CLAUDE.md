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
- When fixing a failure in a mock API, the goal is **faithful to the
  canonical silicon implementation**. Don't create parallel or
  different code paths to work around bugs — if it works in silicon,
  the mock must be functional and correct against it.
- The authoritative `src/` + `include/` file/symbol index lives at
  `references/structure.yaml` (replaces `STRUCTURE.md`). Query with
  `scripts/find_symbol.py <name>` (flags: `--kind`, `--path-prefix`,
  `--list-symbols`, `--summary`); plain `grep` works for casual
  lookups. Keep it in sync when you add/remove/rename a file or
  top-level symbol.
- `IMPLEMENTATION_REPORT.md` is the documentation entry point and
  index; per-subsystem references live under `docs/`
  (l1/dram/dest/cb/noc-emulation, metal-integration,
  tilize-untilize-pack, cb-dataformat, mem-zeros-handling,
  DFB/QUASAR). Read the relevant doc before changing a subsystem
  and update it in the same change. Verify every claim against
  actual code before writing it down — the docs must never assert
  something the source doesn't do.

## When to use what

| Need | Use |
|---|---|
| What does silicon API X do, so I can mock it? | `/arch-lookup "X"` skill |
| Step-by-step workflow for adding a new mock | `/implement-mock <api>` skill |
| Add a compute-kernel LLK shim (`<op>_tile` in `include/jit_hw/api/compute/`) | `/compute-llk-bringup` skill |
| Diagnose ATOL/PCC failures (wrong bytes, partial zeros, off-by-N) | `/memory-debug` skill |
| A tt-metal pin bump turned the regression red (device-open crash, JIT-compile error, hang) | `/uplift` skill |
| Parallelize a sweep of ≥4 similar mocks | `/parallel-mock-implementation` skill |
| Map HW concept → existing emule strategy | `references/emule-mapping.md` |
| Where in the pipeline to inject a change | [`docs/api-injection-points.md`](../docs/api-injection-points.md) |
| Verify a mock is complete | `/verify-mock` skill |
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
