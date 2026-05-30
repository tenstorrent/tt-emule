#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# scripts/dispatch_llk_bringup.sh — parallel LLK bring-up harness.
#
# Dispatches one Claude Code agent per ttnn op in the embedded manifest.
# Each agent works in an isolated git worktree with its own JIT cache dir,
# follows the .claude/skills/op-bring-up/SKILL.md loop end-to-end, and
# returns a structured PASS/FAIL/STUCK line.
#
# See ~/.claude/plans/vivid-foraging-nebula.md for the full plan +
# methodology this script implements.

set -euo pipefail

# ============================================================================
# Defaults
# ============================================================================
PARALLEL=1
TARGETS="all"
BASE_BRANCH="main"
PROMOTION_BRANCH="arminale/mass-llk-bringup"
NO_PROMOTE=0
DRY_RUN=0
RESUME_LOG=""
# Where to place per-agent worktrees + per-agent logs.
WORKTREE_ROOT="${LLK_BRINGUP_WORKTREE_ROOT:-/tmp/llk-bringup}"
RESULTS_DIR="${LLK_BRINGUP_RESULTS_DIR:-$WORKTREE_ROOT/results}"
BRINGUP_LOG_FILE="docs/notes/llk-bringup-log.md"
# Serialize cherry-picks into the promotion branch across parallel agents.
PROMOTION_LOCK="$WORKTREE_ROOT/promotion.lock"

# The Claude CLI binary. Override via $CLAUDE_BIN if installed under a
# different name (e.g., on machines that have both `claude` and `claude-code`).
CLAUDE_BIN="${CLAUDE_BIN:-claude}"

# tt-metal source path — must be a sibling of tt-emule per BUILD_GUIDE Phase 0.
TT_EMULE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TT_METAL_DIR="${TT_METAL_DIR:-$(cd "$TT_EMULE_DIR/../tt-metal" && pwd)}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"

# ============================================================================
# Embedded manifest. Format (tab-separated): ID test_path llk_family tier
# Source of truth: the "Target manifest" section of vivid-foraging-nebula.md.
# Tiers: E (easy), M (medium), H (hard / fresh LLK family).
# ============================================================================
MANIFEST=$(cat <<'EOF'
1	data_movement/test_tilize.py	tilize compute	E
2	data_movement/test_tilize_with_val_padding.py	tilize + zero-fill	M
3	data_movement/test_tilize_with_zero_padding.py	tilize + zero-fill	M
4	data_movement/test_tilizer.py	tilize variant	E
5	data_movement/test_untilize.py	untilize compute	M
6	data_movement/test_untilize_with_unpadding.py	untilize + unpadding	M
7	data_movement/test_pad.py	pad compute	M
8	data_movement/test_pad_subcoregrids.py	sharded pad	H
9	data_movement/test_permute.py	permute (transpose generalization)	M
10	data_movement/test_slice.py	slice / strided read	M
11	data_movement/test_slice_write.py	strided write	M
12	data_movement/test_indexed_fill.py	indexed fill	E
13	data_movement/test_stack.py	concat axis-0 variant	E
14	data_movement/test_dropout.py	random + mask	M
15	data_movement/test_full_like.py	shape-derived fill	E
16	data_movement/test_sort.py	sort	H
17	data_movement/test_tosa_gather.py	tosa gather	E
18	data_movement/test_tosa_scatter.py	tosa scatter	M
19	reduce/test_argmax.py	argmax (reduce with index)	M
20	reduce/test_max.py	reduce max	E
21	reduce/test_sum.py	reduce sum	E
22	reduce/test_cumsum.py	cumsum	H
23	reduce/test_cumprod.py	cumprod	H
24	reduce/test_topk.py	topk	H
25	reduce/test_moe.py	moe reduce	H
26	fused/test_eltwise_softmax_in_place.py	softmax	H
27	fused/test_softmax.py	softmax variants	H
28	fused/test_batch_norm.py	batch norm	H
29	fused/test_group_norm.py	group norm	H
30	matmul/test_linear.py	matmul + bias	E
31	matmul/test_addmm.py	matmul fused with add	E
32	matmul/test_custom_grids.py	matmul with custom core grids	M
33	transformers/test_concatenate_heads.py	reshape + concat	M
EOF
)

# ============================================================================
# Helpers
# ============================================================================

usage() {
    cat <<USAGE
Usage: $(basename "$0") [options]

  --parallel N             Number of agents to run concurrently. Default: 1.
  --targets <selector>     Which manifest rows to dispatch. One of:
                             all                  every row
                             E|M|H                effort tier
                             N,N,N or N-M         explicit IDs / ranges
                             data_movement|reduce|fused|matmul|transformer
                                                  category prefix (matches test_path)
  --base-branch <name>     Branch to base each agent's worktree on. Default: main.
  --promotion-branch <name> Branch to cherry-pick reviewer-APPROVEd commits into.
                           Default: arminale/mass-llk-bringup. Must already exist
                           and be checked out in the main worktree (this script's
                           cwd). Parallel agents serialize via flock.
  --no-promote             Disable auto-promotion. Agent commits stay in worktrees.
  --dry-run                Print the dispatch table and exit. No agents launched.
  --resume <log-path>      Skip rows whose previous run already passed per <log-path>.
  -h, --help               This help.

Env overrides:
  CLAUDE_BIN                       Claude Code CLI binary (default: claude)
  LLK_BRINGUP_WORKTREE_ROOT        Where to place per-agent worktrees + logs
                                   (default: /tmp/llk-bringup)
  TT_METAL_DIR                     tt-metal source root (default: ../tt-metal)
  BUILD_DIR                        tt-metal/build_emule path

Examples:
  $(basename "$0") --targets E --dry-run
  $(basename "$0") --targets E --parallel 4
  $(basename "$0") --targets 1,4,12
  $(basename "$0") --targets H --parallel 1 \\
      --merge-target arminale/llk-bringup-batch-1
USAGE
}

# Parse selector → list of IDs separated by newlines.
select_ids() {
    local selector="$1"
    case "$selector" in
        all)
            echo "$MANIFEST" | awk -F'\t' '{print $1}'
            ;;
        E|M|H)
            echo "$MANIFEST" | awk -F'\t' -v tier="$selector" '$4 == tier {print $1}'
            ;;
        data_movement|reduce|fused|matmul|transformer|transformers)
            local prefix="$selector"
            # Accept "transformer" as alias for "transformers"
            [ "$prefix" = "transformer" ] && prefix="transformers"
            echo "$MANIFEST" | awk -F'\t' -v p="$prefix/" 'index($2, p) == 1 {print $1}'
            ;;
        *)
            # Comma/range list like "1,3,12-15"
            local ids=()
            local IFS=','
            for tok in $selector; do
                if [[ "$tok" == *-* ]]; then
                    local lo="${tok%-*}" hi="${tok#*-}"
                    for ((i=lo; i<=hi; i++)); do ids+=("$i"); done
                else
                    ids+=("$tok")
                fi
            done
            printf '%s\n' "${ids[@]}"
            ;;
    esac
}

# Look up a manifest row by ID. Echoes test_path<TAB>llk_family<TAB>tier.
manifest_row() {
    local id="$1"
    echo "$MANIFEST" | awk -F'\t' -v id="$id" '$1 == id {print $2 "\t" $3 "\t" $4}'
}

# Build the per-agent prompt. Args: id test_path llk_family tier worktree jit_cache.
make_prompt() {
    local id="$1" test_path="$2" llk_family="$3" tier="$4" worktree="$5" jit_cache="$6"
    cat <<PROMPT
You are agent #${id} dispatched by scripts/dispatch_llk_bringup.sh.

Assignment:
  test file:                tests/ttnn/unit_tests/operations/${test_path}
  predicted LLK family:     ${llk_family}
  effort tier:              ${tier}
  tt-emule worktree (cwd):  ${worktree}
  tt-metal source:          ${TT_METAL_DIR}
  build artifact:           ${BUILD_DIR}
  JIT cache dir:            ${jit_cache}

Your job: bring up ONE ttnn op end-to-end by following the
.claude/skills/op-bring-up/SKILL.md loop. Steps:

1. Verify cwd is ${worktree}. The tt-emule worktree is already set up.
2. Set up env:
     export TT_METAL_DIR=${TT_METAL_DIR}
     export BUILD_DIR=${BUILD_DIR}
     export TT_EMULE_JIT_CACHE_DIR=${jit_cache}
     export TT_METAL_HOME=\${TT_METAL_DIR}
     export TT_METAL_RUNTIME_ROOT=\${TT_METAL_DIR}
     export TT_METAL_EMULE_MODE=1
     export TT_METAL_SLOW_DISPATCH_MODE=1
     export TT_METAL_MOCK_CLUSTER_DESC_PATH=\${TT_METAL_DIR}/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml
     export MESH_DEVICE=N150
     export PYTHONPATH=\${TT_METAL_DIR}/ttnn:\${TT_METAL_DIR}/tools:\${BUILD_DIR}/lib:\${TT_METAL_DIR}
     export LD_LIBRARY_PATH=\${BUILD_DIR}/lib
3. Clear the assigned JIT cache: rm -rf \${TT_EMULE_JIT_CACHE_DIR}
4. Run: /opt/ttmlir-toolchain/venv/bin/pytest \${TT_METAL_DIR}/tests/ttnn/unit_tests/operations/${test_path} -s --tb=short
5. If the run is all-PASS, skip to step 9. Otherwise classify the first failure
   per skill Step 4 (missing header → redirect shim; undeclared name → add
   declaration mirroring upstream's signature; PCC fail → walk the runtime-PCC
   table).
6. Add the shim(s) in this worktree under include/jit_hw/, mirroring upstream
   paths from tt_metal/hw/inc/api/... or ttnn/cpp/ttnn/kernel_lib/...
7. Re-run step 4. Repeat 5-7 until the test passes.
8. If you exhaust 5 iterations without convergence, or hit a gap that's clearly
   out of scope (sharded TensorAccessor, host→DRAM, requires_fast_runtime_mode_off),
   stop and emit a STUCK line — don't force it.
9. Promote: edit ${TT_EMULE_DIR}/scripts/run_ttnn_pytests.sh (NOT in your
   worktree — that file is shared; instead, your worktree has its own copy.
   Edit ./scripts/run_ttnn_pytests.sh in your worktree). Add a curated entry
   for the test file (or a -k subset of passing parametrizations).
10. Re-run the full scripts/run_ttnn_pytests.sh sweep in your worktree.
    Confirm: same number of PASS as before + the new entry.
11. Commit your changes in this worktree (DO NOT push):
      git add include/jit_hw/ scripts/run_ttnn_pytests.sh
      git commit -m "llk-bringup: <op> via shim <name>"

Emit on stdout exactly ONE final structured line:
  PASS|${id}|<commit_sha>|<num_parametrizations_passed>
or:
  STUCK|${id}|<one-line description of the gap>
or:
  FAIL|${id}|<one-line reason>

Do not push to remote. Do not modify files outside your worktree. Do not
exceed 5 shim iterations without emitting STUCK.
PROMPT
}

# Build the post-commit review prompt. Args: id, test_path, worktree.
# The reviewer is an independent claude -p invocation; it reads the agent's
# HEAD commit and decides APPROVE / REQUEST_CHANGES.
make_review_prompt() {
    local id="$1" test_path="$2" worktree="$3"
    cat <<PROMPT
You are a code reviewer for the commit produced by agent #${id}, which
was assigned to make this pytest pass under tt-emule by following
.claude/skills/op-bring-up/SKILL.md:
  tests/ttnn/unit_tests/operations/${test_path}

The agent's worktree is ${worktree}; their commit is HEAD.
Upstream tt-metal source for cross-checks: ${TT_METAL_DIR}.

Your job: review the agent's commit. Acceptance criteria:

1. SCOPE — the diff is minimal. Only files needed to (a) close the
   specific JIT-compile or PCC gap that broke this test, and (b)
   promote the test by adding an entry to scripts/run_ttnn_pytests.sh.
   NO unrelated refactors. NO .claude/ edits. NO doc churn. NO
   commented-out blocks. NO speculative changes for hypothetical
   future tests.

2. SHIM CORRECTNESS — any new files under include/jit_hw/ should:
   - Mirror an existing upstream path (e.g.
     include/jit_hw/api/compute/X.h shimming
     ${TT_METAL_DIR}/tt_metal/hw/inc/api/compute/X.h). Verify the
     upstream header exists at the mirrored path.
   - Have function signatures (template params, arg types, return
     type, default args) matching upstream's declarations. Verify by
     reading upstream's header.
   - Be no-op stubs only when emule semantically does not need the
     primitive (sync, remap-configure, packer-init are fine as no-ops).
     Be functional implementations when the LLK actually shapes data.

3. PROMOTION — scripts/run_ttnn_pytests.sh has a new run_pytest line
   for this test. Any -k filter is justified — typically excludes
   sharded / requires_fast_runtime_mode_off / host→DRAM-only variants
   per the methodology. Excluded variants should be principled, not
   arbitrary.

4. NO OVERREACH — agent did not modify files outside its worktree, did
   not push to remote, did not bypass safety checks.

Read the diff: \`git -C ${worktree} show HEAD\`. Read any new shim
files in full. Cross-check signatures against upstream headers under
${TT_METAL_DIR}/tt_metal/hw/inc/ and ${TT_METAL_DIR}/ttnn/cpp/ttnn/
kernel_lib/.

Emit on stdout exactly ONE final structured line:
  REVIEW: APPROVE|<one-line justification, ≤120 chars>
or:
  REVIEW: REQUEST_CHANGES|<actionable feedback, ≤200 chars>

Be lenient on genuinely-trivial diffs (e.g. a one-line type fix to
suppress a signed/unsigned warning is fine — that IS the minimum).
Be strict on shims that don't mirror upstream paths, have wrong
signatures, contain dead code, or touch unrelated files.
PROMPT
}

# Run the reviewer. Args: id, test_path, worktree, [round_suffix].
# Round suffix lets round-1 and round-2 reviews write to separate logs.
# Echoes the verdict line.
review_agent_diff() {
    local id="$1" test_path="$2" worktree="$3" round_suffix="${4:-}"
    local review_log="$RESULTS_DIR/agent-${id}.review${round_suffix}"

    local prompt
    prompt="$(make_review_prompt "$id" "$test_path" "$worktree")"

    (
        cd "$worktree"
        "$CLAUDE_BIN" -p \
            --add-dir "$worktree" \
            --add-dir "$TT_METAL_DIR" \
            --dangerously-skip-permissions \
            "$prompt"
    ) > "$review_log" 2>&1 || true

    local verdict
    verdict="$(grep -E '^REVIEW:' "$review_log" | tail -1 || true)"
    if [ -z "$verdict" ]; then
        verdict="REVIEW: REQUEST_CHANGES|reviewer produced no structured verdict (see $review_log)"
    fi
    echo "$verdict"
}

# Build the round-2 revision prompt. Args: id, test_path, worktree, feedback.
# Agent is instructed to amend their existing HEAD (not stack a second commit).
make_revision_prompt() {
    local id="$1" test_path="$2" worktree="$3" feedback="$4"
    cat <<PROMPT
You are agent #${id} returning for the SINGLE revision round permitted.
Your previous bring-up commit was reviewed; reviewer asked for changes:

  Reviewer feedback: ${feedback}

Address the feedback by AMENDING your existing commit (not stacking a new one).

1. cd ${worktree}
2. Re-read the reviewer feedback and make exactly the changes they asked for.
   Do not add unrelated changes.
3. Re-run the test to confirm it still passes:
     export TT_EMULE_JIT_CACHE_DIR=/tmp/tt_emule_jit_cache_llk_bringup_${id}
     (rest of env per round-1 setup)
     /opt/ttmlir-toolchain/venv/bin/pytest \\
       \${TT_METAL_DIR}/tests/ttnn/unit_tests/operations/${test_path} -s --tb=short
4. If the test still passes:
     git add <modified files>
     git commit --amend --no-edit
5. Emit on stdout exactly ONE final structured line:
     PASS|${id}|<new_commit_sha>|<num_parametrizations_passed>
   or:
     STUCK|${id}|<one-line: why you cannot address the feedback>

This is your ONLY revision attempt. The reviewer will rerun exactly once more.
If your revision is also rejected, the harness will write a handoff doc and
will not cycle again. Do not push. Do not modify files outside your worktree.
PROMPT
}

# Run the agent's round-2 revision. Args: id, test_path, llk_family, tier,
# worktree, jit_cache, reviewer_feedback. Echoes the structured agent result.
dispatch_revision() {
    local id="$1" test_path="$2" llk_family="$3" tier="$4" worktree="$5"
    local jit_cache="$6" feedback="$7"
    local agent_log="$RESULTS_DIR/agent-${id}.log.round2"

    local prompt
    prompt="$(make_revision_prompt "$id" "$test_path" "$worktree" "$feedback")"

    local rc=0
    (
        cd "$worktree"
        "$CLAUDE_BIN" -p \
            --add-dir "$worktree" \
            --add-dir "$TT_METAL_DIR" \
            --dangerously-skip-permissions \
            "$prompt"
    ) > "$agent_log" 2>&1 || rc=$?

    local result
    result="$(grep -E '^(PASS|STUCK|FAIL)\|' "$agent_log" | tail -1 || true)"
    if [ -z "$result" ]; then
        result="STUCK|${id}|round-2 agent produced no structured result (rc=$rc, see $agent_log)"
    fi
    echo "$result"
}

# Serialize cherry-picks into the promotion branch. Args: id, commit_sha.
# Returns 0 on success (cherry-pick landed); 1 on conflict or wrong branch.
cherry_pick_into_promotion_branch() {
    local id="$1" commit_sha="$2"

    # Concurrency: hold an exclusive lock for the duration of the cherry-pick.
    (
        flock -x 9
        local current_branch
        current_branch="$(git -C "$TT_EMULE_DIR" symbolic-ref --short HEAD 2>/dev/null || echo DETACHED)"
        if [ "$current_branch" != "$PROMOTION_BRANCH" ]; then
            echo "[$id] PROMOTION SKIPPED: main worktree on '$current_branch', expected '$PROMOTION_BRANCH'" >&2
            exit 2
        fi
        if git -C "$TT_EMULE_DIR" cherry-pick "$commit_sha" > "$RESULTS_DIR/agent-${id}.cherrypick" 2>&1; then
            echo "[$id] cherry-picked $commit_sha → $PROMOTION_BRANCH ($(git -C "$TT_EMULE_DIR" rev-parse --short HEAD))"
            exit 0
        else
            git -C "$TT_EMULE_DIR" cherry-pick --abort 2>/dev/null || true
            echo "[$id] CONFLICT cherry-picking $commit_sha — see $RESULTS_DIR/agent-${id}.cherrypick"
            exit 1
        fi
    ) 9>"$PROMOTION_LOCK"
}

# Write a handoff doc after two rejected review rounds.
# Args: id, test_path, llk_family, agent_commit, review1, review2, [agent_round2_result]
write_handoff_doc() {
    local id="$1" test_path="$2" llk_family="$3" agent_commit="$4"
    local review1="$5" review2="$6" agent_round2_result="${7:-}"
    local doc_path="$TT_EMULE_DIR/docs/notes/llk-bringup-handoff-${id}.md"
    mkdir -p "$(dirname "$doc_path")"

    {
        echo "# LLK bring-up handoff: agent #${id}"
        echo ""
        echo "Generated by \`scripts/dispatch_llk_bringup.sh\` on $(date -u +%Y-%m-%dT%H:%M:%SZ)."
        echo ""
        echo "## Assignment"
        echo ""
        echo "- Test: \`tests/ttnn/unit_tests/operations/${test_path}\`"
        echo "- LLK family (predicted): ${llk_family}"
        echo "- Promotion branch: \`${PROMOTION_BRANCH}\`"
        echo ""
        echo "## Agent's commit"
        echo ""
        echo "\`${agent_commit}\` — preserved in object db; cherry-pickable if you"
        echo "decide to override the reviewer."
        echo ""
        echo "\`\`\`"
        git -C "$TT_EMULE_DIR" show --stat "$agent_commit" 2>/dev/null | head -20
        echo "\`\`\`"
        echo ""
        echo "## Review round 1"
        echo ""
        echo "\`${review1}\`"
        echo ""
        if [ -n "$agent_round2_result" ]; then
            echo "## Agent's round-2 revision attempt"
            echo ""
            echo "\`${agent_round2_result}\`"
            echo ""
            echo "Round-2 log: \`${RESULTS_DIR}/agent-${id}.log.round2\`"
            echo ""
        fi
        echo "## Review round 2"
        echo ""
        echo "\`${review2}\`"
        echo ""
        echo "## Why this is a handoff and not auto-promoted"
        echo ""
        echo "After two review cycles, the diff still does not meet the harness's"
        echo "acceptance criteria (scope / shim correctness / promotion / no overreach)."
        echo "The harness does not cycle further."
        echo ""
        echo "## Recommended next actions"
        echo ""
        echo "1. Read the round-2 reviewer feedback above. Decide whether it's:"
        echo "   - A real concern → address manually and cherry-pick yourself."
        echo "   - Overly strict → cherry-pick \`${agent_commit}\` into \`${PROMOTION_BRANCH}\`"
        echo "     and document the override here."
        echo "2. If the agent's approach is fundamentally wrong, this op may need a"
        echo "   different bring-up strategy (split shims, defer until upstream lands,"
        echo "   etc.). Update the manifest accordingly."
        echo ""
        echo "## Logs"
        echo ""
        echo "- Agent round 1: \`${RESULTS_DIR}/agent-${id}.log\`"
        echo "- Review round 1: \`${RESULTS_DIR}/agent-${id}.review\`"
        if [ -n "$agent_round2_result" ]; then
            echo "- Agent round 2: \`${RESULTS_DIR}/agent-${id}.log.round2\`"
            echo "- Review round 2: \`${RESULTS_DIR}/agent-${id}.review.round2\`"
        fi
        echo "- Agent worktree: \`${WORKTREE_ROOT}/agent-${id}\` (still on disk for inspection)"
    } > "$doc_path"

    echo "[$id] handoff doc: $doc_path"
}

# Dispatch one agent. Args: id.
dispatch_one() {
    local id="$1"
    local row
    row="$(manifest_row "$id")"
    if [ -z "$row" ]; then
        echo "[$id] ERROR: no manifest row for ID=$id" >&2
        return 1
    fi
    IFS=$'\t' read -r test_path llk_family tier <<<"$row"

    local worktree="$WORKTREE_ROOT/agent-$id"
    local jit_cache="/tmp/tt_emule_jit_cache_llk_bringup_$id"
    local agent_log="$RESULTS_DIR/agent-$id.log"
    local result_line="$RESULTS_DIR/agent-$id.result"

    if [ "$DRY_RUN" = "1" ]; then
        printf '[dry-run] id=%-3s tier=%s test=%s\n  worktree=%s\n  jit_cache=%s\n' \
            "$id" "$tier" "$test_path" "$worktree" "$jit_cache"
        return 0
    fi

    # Skip already-passed IDs if resuming.
    if [ -n "$RESUME_LOG" ] && grep -qE "^\| ${id} \|.*\| PASS \|" "$RESUME_LOG" 2>/dev/null; then
        echo "[$id] SKIP (resume: previously PASS)"
        return 0
    fi

    # Set up worktree (idempotent — remove stale first).
    if [ -e "$worktree" ]; then
        git worktree remove --force "$worktree" 2>/dev/null || rm -rf "$worktree"
    fi
    git worktree add --quiet "$worktree" "$BASE_BRANCH"

    # Build prompt + dispatch.
    local prompt
    prompt="$(make_prompt "$id" "$test_path" "$llk_family" "$tier" "$worktree" "$jit_cache")"

    echo "[$id] dispatching (tier=$tier test=$test_path worktree=$worktree)"
    local rc=0
    (
        cd "$worktree"
        "$CLAUDE_BIN" -p \
            --add-dir "$worktree" \
            --add-dir "$TT_METAL_DIR" \
            --dangerously-skip-permissions \
            "$prompt"
    ) > "$agent_log" 2>&1 || rc=$?

    # Last structured agent line.
    local agent_result
    agent_result="$(grep -E '^(PASS|STUCK|FAIL)\|' "$agent_log" | tail -1 || true)"
    if [ -z "$agent_result" ]; then
        agent_result="FAIL|${id}|no structured result line in agent output (rc=$rc, see $agent_log)"
    fi

    # STUCK / FAIL agents: no review, no cycle, no promotion.
    if [[ "$agent_result" != PASS\|* ]]; then
        echo "$agent_result" > "$result_line"
        echo "[$id] $agent_result"
        return 0
    fi

    # Extract round-1 commit SHA from the PASS line: PASS|id|sha|parametrizations
    local round1_sha
    round1_sha="$(echo "$agent_result" | awk -F'|' '{print $3}')"

    # ====== Review round 1 ======
    echo "[$id] agent PASS (round 1) — dispatching reviewer"
    local verdict1
    verdict1="$(review_agent_diff "$id" "$test_path" "$worktree" "")"
    echo "[$id] reviewer (round 1): $verdict1"

    local final
    if [[ "$verdict1" == REVIEW:\ APPROVE\|* ]]; then
        # Auto-promote unless --no-promote.
        if [ "$NO_PROMOTE" = "1" ]; then
            final="$agent_result"
        elif cherry_pick_into_promotion_branch "$id" "$round1_sha"; then
            final="$agent_result"
        else
            final="STUCK|${id}|round-1 APPROVE but promotion cherry-pick failed (see agent-${id}.cherrypick)"
        fi
        echo "$final" > "$result_line"
        echo "[$id] $final"
        return 0
    fi

    # ====== Round 2: re-dispatch agent with feedback, then re-review ======
    local feedback="${verdict1#REVIEW: REQUEST_CHANGES|}"
    echo "[$id] reviewer REQUEST_CHANGES — dispatching round-2 revision"
    local round2_result
    round2_result="$(dispatch_revision "$id" "$test_path" "$llk_family" "$tier" "$worktree" "$jit_cache" "$feedback")"
    echo "[$id] agent (round 2): $round2_result"

    if [[ "$round2_result" != PASS\|* ]]; then
        # Agent couldn't address the feedback; handoff doc + done.
        write_handoff_doc "$id" "$test_path" "$llk_family" "$round1_sha" "$verdict1" "(no round-2 review — agent failed to revise)" "$round2_result"
        final="STUCK|${id}|round-2 agent failed to address review feedback (handoff doc written)"
        echo "$final" > "$result_line"
        echo "[$id] $final"
        return 0
    fi

    local round2_sha
    round2_sha="$(echo "$round2_result" | awk -F'|' '{print $3}')"

    echo "[$id] round-2 agent PASS — dispatching reviewer (round 2)"
    local verdict2
    verdict2="$(review_agent_diff "$id" "$test_path" "$worktree" ".round2")"
    echo "[$id] reviewer (round 2): $verdict2"

    if [[ "$verdict2" == REVIEW:\ APPROVE\|* ]]; then
        if [ "$NO_PROMOTE" = "1" ]; then
            final="$round2_result"
        elif cherry_pick_into_promotion_branch "$id" "$round2_sha"; then
            final="$round2_result"
        else
            final="STUCK|${id}|round-2 APPROVE but promotion cherry-pick failed (see agent-${id}.cherrypick)"
        fi
    else
        # Two rejections; handoff doc + STUCK.
        write_handoff_doc "$id" "$test_path" "$llk_family" "$round2_sha" "$verdict1" "$verdict2" "$round2_result"
        final="STUCK|${id}|reviewer rejected both rounds (handoff doc written)"
    fi

    echo "$final" > "$result_line"
    echo "[$id] $final"
}

# Post-batch summary + log append.
summarize_batch() {
    echo ""
    echo "============================================================"
    echo " Batch summary"
    echo "============================================================"
    local pass=0 stuck=0 fail=0
    for r in "$RESULTS_DIR"/*.result; do
        [ -f "$r" ] || continue
        local line
        line="$(cat "$r")"
        case "${line%%|*}" in
            PASS)  pass=$((pass+1)) ;;
            STUCK) stuck=$((stuck+1)) ;;
            FAIL)  fail=$((fail+1)) ;;
        esac
        printf '  %s\n' "$line"
    done
    echo ""
    echo "  totals: ${pass} PASS / ${stuck} STUCK / ${fail} FAIL"
    echo ""

    # Append to bring-up log.
    local log_path="$TT_EMULE_DIR/$BRINGUP_LOG_FILE"
    mkdir -p "$(dirname "$log_path")"
    if [ ! -f "$log_path" ]; then
        cat > "$log_path" <<'HEADER'
# LLK bring-up log

Append-only audit trail of agent runs dispatched by
`scripts/dispatch_llk_bringup.sh`. One row per dispatched agent.

| ID | test | LLK family | tier | status | commit | parametrizations | timestamp |
|---:|------|-----------|:----:|:------:|--------|:----------------:|-----------|
HEADER
    fi
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for r in "$RESULTS_DIR"/*.result; do
        [ -f "$r" ] || continue
        local line
        line="$(cat "$r")"
        IFS='|' read -r status id rest <<<"$line"
        local row
        row="$(manifest_row "$id")"
        IFS=$'\t' read -r test_path llk_family tier <<<"$row"
        local commit="-" params="-" reason="-"
        case "$status" in
            PASS)  IFS='|' read -r _ _ commit params <<<"$line" ;;
            STUCK) IFS='|' read -r _ _ reason <<<"$line" ;;
            FAIL)  IFS='|' read -r _ _ reason <<<"$line" ;;
        esac
        printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' \
            "$id" "$test_path" "$llk_family" "$tier" "$status" \
            "${commit:--}" "${params:--}" "$ts" >> "$log_path"
    done
    echo "  appended ${pass} + ${stuck} + ${fail} rows to $log_path"
}

# Cherry-pick promotion is now per-agent (see cherry_pick_into_promotion_branch +
# the dispatch_one APPROVE path). This post-batch hook prints the final state
# of the promotion branch so the summary tells the user where to find the work.
summarize_promotion_branch_state() {
    [ "$DRY_RUN" = "1" ] && return 0
    [ "$NO_PROMOTE" = "1" ] && return 0
    echo ""
    echo "============================================================"
    echo " Promotion branch: $PROMOTION_BRANCH"
    echo "============================================================"
    local current_branch
    current_branch="$(git -C "$TT_EMULE_DIR" symbolic-ref --short HEAD 2>/dev/null || echo DETACHED)"
    if [ "$current_branch" != "$PROMOTION_BRANCH" ]; then
        echo "  WARNING: main worktree is on '$current_branch', not '$PROMOTION_BRANCH'."
        echo "  No commits were auto-promoted. Each PASS agent's commit is preserved"
        echo "  in object db; cherry-pick manually if you want them on the promotion branch."
        return 0
    fi
    echo "  HEAD: $(git -C "$TT_EMULE_DIR" rev-parse --short HEAD)"
    local commits_ahead
    commits_ahead="$(git -C "$TT_EMULE_DIR" rev-list --count "$BASE_BRANCH"..HEAD)"
    echo "  Commits ahead of $BASE_BRANCH: $commits_ahead"
    git -C "$TT_EMULE_DIR" log --oneline "$BASE_BRANCH"..HEAD | sed 's/^/  /'
}

# ============================================================================
# Argparse
# ============================================================================
while [ $# -gt 0 ]; do
    case "$1" in
        --parallel)         PARALLEL="$2"; shift 2 ;;
        --targets)          TARGETS="$2"; shift 2 ;;
        --base-branch)      BASE_BRANCH="$2"; shift 2 ;;
        --promotion-branch) PROMOTION_BRANCH="$2"; shift 2 ;;
        --merge-target)     PROMOTION_BRANCH="$2"; shift 2 ;;  # legacy alias
        --no-promote)       NO_PROMOTE=1; shift ;;
        --dry-run)          DRY_RUN=1; shift ;;
        --resume)           RESUME_LOG="$2"; shift 2 ;;
        -h|--help)          usage; exit 0 ;;
        *)                  echo "Unknown flag: $1" >&2; usage; exit 1 ;;
    esac
done

# ============================================================================
# Main
# ============================================================================
mkdir -p "$WORKTREE_ROOT" "$RESULTS_DIR"
# Wipe stale per-run results (logs survive for inspection).
rm -f "$RESULTS_DIR"/*.result

echo "tt-emule:   $TT_EMULE_DIR"
echo "tt-metal:   $TT_METAL_DIR"
echo "build:      $BUILD_DIR"
echo "base:       $BASE_BRANCH"
echo "parallel:   $PARALLEL"
echo "targets:    $TARGETS"
if [ "$NO_PROMOTE" = "1" ]; then
    echo "promotion: disabled (--no-promote)"
else
    echo "promote into: $PROMOTION_BRANCH"
fi
[ -n "$RESUME_LOG" ] && echo "resume log: $RESUME_LOG"
[ "$DRY_RUN" = "1" ] && echo "(dry-run)"
echo ""

# Build the list of IDs to dispatch.
IDS="$(select_ids "$TARGETS")"
if [ -z "$IDS" ]; then
    echo "No manifest rows matched selector '$TARGETS'." >&2
    exit 1
fi
echo "Selected IDs:"
echo "$IDS" | xargs
echo ""

# Validate every selected ID is in the manifest before dispatching.
INVALID=()
while IFS= read -r id; do
    [ -z "$id" ] && continue
    if [ -z "$(manifest_row "$id")" ]; then
        INVALID+=("$id")
    fi
done <<<"$IDS"
if [ "${#INVALID[@]}" -gt 0 ]; then
    echo "Selector '$TARGETS' contains IDs not in the manifest: ${INVALID[*]}" >&2
    echo "Valid IDs are 1..$(echo "$MANIFEST" | wc -l)." >&2
    exit 1
fi

# Export everything dispatch_one needs into a subshell for xargs -P.
export -f dispatch_one make_prompt make_review_prompt review_agent_diff manifest_row
export -f make_revision_prompt dispatch_revision cherry_pick_into_promotion_branch write_handoff_doc
export MANIFEST WORKTREE_ROOT RESULTS_DIR DRY_RUN BASE_BRANCH PROMOTION_BRANCH NO_PROMOTE PROMOTION_LOCK
export TT_EMULE_DIR TT_METAL_DIR BUILD_DIR CLAUDE_BIN RESUME_LOG

# Dispatch in parallel. xargs -I{} reads one input per invocation; -P sets
# the concurrency. No -n flag — -I implies it.
echo "$IDS" | xargs -P "$PARALLEL" -I{} bash -c 'dispatch_one "$@"' _ {}

# Aggregate + log + optional merge.
if [ "$DRY_RUN" = "0" ]; then
    summarize_batch
    summarize_promotion_branch_state
fi
