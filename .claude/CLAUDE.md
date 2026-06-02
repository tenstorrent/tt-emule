# tt-emule .claude/ — agent-context for emule mock-API work

This `.claude/` directory provides skills, agents, and references
tailored to **implementing faithful mock APIs in tt-emule**. The
overall project context lives in `tt-emule/CLAUDE.md` (one level up):
build conventions, regression rules, architecture defaults.

This file is the in-`.claude/` orientation guide.

## When to use what

| Need | Use |
|---|---|
| What does silicon API X do, so I can mock it? | `/arch-lookup "X"` skill |
| Step-by-step workflow for adding a new mock | `/implement-mock <api>` skill |
| Add a compute-kernel LLK shim (`<op>_tile` in `include/jit_hw/api/compute/`) | `/compute-llk-bringup` skill |
| Parallelize a sweep of ≥4 similar mocks | `/parallel-mock-implementation` skill |
| Map HW concept → existing emule strategy | `references/emule-mapping.md` |
| Where in the pipeline to inject a change | `references/api-injection-points.md` |
| Verify a mock is complete | `references/stub-checklist.md` |
| Deep arch question on Wormhole/Blackhole/Quasar | sage agents (launched by arch-lookup) |

## Skills

- **arch-lookup** — orchestrates parallel `sage-*` agents to answer
  silicon HW questions with an emule-mapping perspective. Adapted
  from `tt-metal/tt_metal/tt-llk/.claude/skills/arch-lookup`.
- **implement-mock** — end-to-end workflow: arch-lookup → pick
  strategy (A/B/C) → implement → verify → document.
- **compute-llk-bringup** — specialization of `implement-mock` for
  compute LLK shims (Strategy A applied to ops under
  `include/jit_hw/api/compute/`). Has the shim-pattern catalog,
  `sfpu_split_includes.h` wiring, polynomial-port recipe, PCC triage.
- **parallel-mock-implementation** — batching primitive: spawn one
  sub-agent per file via the Workflow tool, orchestrator handles
  centralized wiring afterward. Invoked from `/implement-mock` and
  `/compute-llk-bringup` for sweeps.

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

- `emule-mapping.md` — HW concept → emule simulation strategy catalog
- `api-injection-points.md` — where in the pipeline emule intercepts
- `stub-checklist.md` — pre-implementation through verification

## Quick links to project-level context

- `tt-emule/CLAUDE.md` — project conventions (clang-20, slow dispatch,
  wormhole n150 default, always run regressions)
- `BUILD_GUIDE.md` — build + test setup (`TT_METAL_DIR`, regression scripts)

## Source skills (upstream — for reference)

- `tt-metal/tt_metal/tt-llk/.claude/skills/{arch-lookup,debug-kernel,port-kernel,run-test}/` — original tt-llk skills
- `tt-metal/tt_metal/tt-llk/.claude/agents/{sage-wormhole,sage-blackhole,sage-quasar,llk-debugger,llk-test-runner}.md` — original tt-llk agents

The tt-emule versions diverge from upstream in three places:
1. Search scope adds emule paths (`tt-emule/include/jit_hw/`, runner cpp)
2. Every response ends with **"Emule mapping"** + **"Recommendation"**
3. References cite the emule-mapping catalog so agents speak the
   same vocabulary about A/B/C strategies
