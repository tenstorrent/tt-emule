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
MERGE_TARGET=""
DRY_RUN=0
RESUME_LOG=""
# Where to place per-agent worktrees + per-agent logs.
WORKTREE_ROOT="${LLK_BRINGUP_WORKTREE_ROOT:-/tmp/llk-bringup}"
RESULTS_DIR="${LLK_BRINGUP_RESULTS_DIR:-$WORKTREE_ROOT/results}"
BRINGUP_LOG_FILE="docs/notes/llk-bringup-log.md"

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
  --merge-target <name>    Branch to cherry-pick successful agent commits into post-batch.
                           If unset, leaves agent worktrees in place for review.
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

# Run the reviewer. Args: id, test_path, worktree. Echoes the verdict line.
review_agent_diff() {
    local id="$1" test_path="$2" worktree="$3"
    local review_log="$RESULTS_DIR/agent-${id}.review"

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

    # If the agent PASSed, dispatch an independent reviewer against the agent's
    # HEAD commit. The PASS only stands if the reviewer also APPROVEs.
    local final="$agent_result"
    if [[ "$agent_result" == PASS\|* ]]; then
        echo "[$id] agent PASS — dispatching reviewer"
        local verdict
        verdict="$(review_agent_diff "$id" "$test_path" "$worktree")"
        echo "[$id] reviewer: $verdict"
        if [[ "$verdict" == REVIEW:\ APPROVE\|* ]]; then
            : # PASS stands as-is.
        else
            # Demote to STUCK so the harness summary + log capture the rejection.
            local reason="${verdict#REVIEW: }"
            final="STUCK|${id}|review demoted: ${reason}"
        fi
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

# Optional cherry-pick of PASS commits into --merge-target.
maybe_merge_pass_commits() {
    [ -z "$MERGE_TARGET" ] && return 0
    [ "$DRY_RUN" = "1" ] && return 0

    echo ""
    echo "============================================================"
    echo " Merging PASS agent commits into $MERGE_TARGET"
    echo "============================================================"

    # Create or check out merge target.
    if git show-ref --quiet "refs/heads/$MERGE_TARGET"; then
        git switch "$MERGE_TARGET"
    else
        git switch -c "$MERGE_TARGET" "$BASE_BRANCH"
    fi

    local picked=0 skipped=0
    for r in "$RESULTS_DIR"/*.result; do
        [ -f "$r" ] || continue
        local line; line="$(cat "$r")"
        [[ "$line" == PASS\|* ]] || continue
        IFS='|' read -r _ id commit _ <<<"$line"
        if [ -z "$commit" ] || [ "$commit" = "-" ]; then
            echo "  [agent $id] no commit SHA in result line; skipping"
            skipped=$((skipped+1))
            continue
        fi
        if git cherry-pick "$commit" 2>/dev/null; then
            picked=$((picked+1))
            echo "  [agent $id] cherry-picked $commit"
        else
            git cherry-pick --abort 2>/dev/null || true
            echo "  [agent $id] CONFLICT cherry-picking $commit — left for manual review"
            skipped=$((skipped+1))
        fi
    done
    echo ""
    echo "  merged: $picked  conflicts/skipped: $skipped"
    echo "  branch: $MERGE_TARGET (HEAD: $(git rev-parse --short HEAD))"
}

# ============================================================================
# Argparse
# ============================================================================
while [ $# -gt 0 ]; do
    case "$1" in
        --parallel)      PARALLEL="$2"; shift 2 ;;
        --targets)       TARGETS="$2"; shift 2 ;;
        --base-branch)   BASE_BRANCH="$2"; shift 2 ;;
        --merge-target)  MERGE_TARGET="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --resume)        RESUME_LOG="$2"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "Unknown flag: $1" >&2; usage; exit 1 ;;
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
[ -n "$MERGE_TARGET" ] && echo "merge into: $MERGE_TARGET"
[ -n "$RESUME_LOG" ]   && echo "resume log: $RESUME_LOG"
[ "$DRY_RUN" = "1" ]   && echo "(dry-run)"
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
export MANIFEST WORKTREE_ROOT RESULTS_DIR DRY_RUN BASE_BRANCH
export TT_EMULE_DIR TT_METAL_DIR BUILD_DIR CLAUDE_BIN RESUME_LOG

# Dispatch in parallel. xargs -I{} reads one input per invocation; -P sets
# the concurrency. No -n flag — -I implies it.
echo "$IDS" | xargs -P "$PARALLEL" -I{} bash -c 'dispatch_one "$@"' _ {}

# Aggregate + log + optional merge.
if [ "$DRY_RUN" = "0" ]; then
    summarize_batch
    maybe_merge_pass_commits
fi
