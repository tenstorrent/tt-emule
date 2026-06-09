#!/usr/bin/env bash
# Master driver for the BH post-commit full sweep.
# End-to-end orchestration of the /status-snapshot skill's Phases 4–8.
#
# Single-command entry: bash snapshots/bh_sanity/run_full_sweep.sh
#
# Expected wallclock: 6–18h, dominated by the per-entry 90-min wallclock
# backstops (most entries finish well before their cap; heavy ones may
# hit it).
#
# Side effects:
#   - clears the JIT cache once at start
#   - writes 14 entries' XML + log under snapshots/bh_sanity/bh_emule/
#   - writes snapshots/bh_sanity/bh_sanity_status_{exec,dev}.md
#   - prints VERIFY PASS / VERIFY FAIL at the end

set -uo pipefail

BH_SANITY_DIR="/localdev/arminale/tt-emule/snapshots/bh_sanity"
TT_METAL_DIR="${TT_METAL_DIR:-/localdev/arminale/tt-metal}"

echo "=================================================================="
echo "=== BH post-commit full sweep — start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=================================================================="

# Phase 4a: env setup (sets PYTHONPATH/LD_LIBRARY_PATH/TT_METAL_HOME,
# clears JIT cache, cd's into $TT_METAL_DIR).
echo
echo "--- env_setup.sh ---"
# shellcheck disable=SC1091
source "$BH_SANITY_DIR/env_setup.sh"

# Phase 4b: run the audited runner. Each of the 14 entries produces
# its own per-entry XML + log under bh_emule/.
echo
echo "--- run_bh_sanity.sh (this is the long step) ---"
t_runner_start=$(date +%s)
bash "$BH_SANITY_DIR/run_bh_sanity.sh"
runner_rc=$?
t_runner_end=$(date +%s)
echo "runner_rc=$runner_rc  runner_elapsed=$((t_runner_end - t_runner_start))s"

# Phase 5–7: parse + classify + write reports. The
# --expected-from-audit flag cross-references the audit log so any
# entry that hit its wallclock SIGTERM appears as TRUNCATED rather
# than silently disappearing.
echo
echo "--- parse_and_report.py ---"
python3 "$BH_SANITY_DIR/parse_and_report.py" \
    --xml-dir "$BH_SANITY_DIR/bh_emule" \
    --log-dir "$BH_SANITY_DIR/bh_emule" \
    --out-exec "$BH_SANITY_DIR/bh_sanity_status_exec.md" \
    --out-dev  "$BH_SANITY_DIR/bh_sanity_status_dev.md" \
    --suite-name bh_sanity \
    --variant-label bh_emule \
    --audit-log "$BH_SANITY_DIR/audit_bh_sanity.log" \
    --runner   "$BH_SANITY_DIR/run_bh_sanity.sh" \
    --manifest "$TT_METAL_DIR/tests/pipeline_reorg/ttnn-tests.yaml" \
    --baseline silicon-passing \
    --expected-from-audit "$BH_SANITY_DIR/audit_bh_sanity.log"
parse_rc=$?

# Phase 8: verify.
echo
echo "--- verify.sh ---"
bash "$BH_SANITY_DIR/verify.sh"
verify_rc=$?

echo
echo "=================================================================="
echo "=== BH post-commit full sweep — end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== runner_rc=$runner_rc  parse_rc=$parse_rc  verify_rc=$verify_rc"
echo "=== Reports:"
echo "===   $BH_SANITY_DIR/bh_sanity_status_exec.md"
echo "===   $BH_SANITY_DIR/bh_sanity_status_dev.md"
echo "=================================================================="

# Non-zero overall if either parse or verify failed. Runner exit code
# is informational only — a non-zero from pytest (e.g. tests failed)
# is expected and not a master-driver failure.
[ "$parse_rc" = "0" ] && [ "$verify_rc" = "0" ]
