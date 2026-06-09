#!/usr/bin/env python3
# Manifest-mode runner generator + audit (Phase 2 + Phase 3 of /status-snapshot).
#
# Reads tt-metal's tests/pipeline_reorg/ttnn-tests.yaml, filters to the BH
# post-commit lane (merge_gate=false AND bh_p150b_civ2 in skus), and emits:
#   - run_bh_sanity.sh   one bash line per selected entry, with emule env
#                        layered in + --junitxml and a per-entry timeout.
#   - audit_bh_sanity.log  per-entry diff between source cmd and emitted line.
#
# Identical-invocation invariant: every test path, every original env var,
# every pytest arg present in the source cmd must appear in the emitted line
# byte-for-byte (modulo argument ordering). The diff between source and
# emitted MUST consist only of: emule env vars, variant env overlays, the
# `timeout` wrapper, --junitxml=…, and the redirection.

import os
import re
import shlex
import sys
from pathlib import Path

import yaml

# -- Inputs ------------------------------------------------------------------

MANIFEST = Path("/localdev/arminale/tt-metal/tests/pipeline_reorg/ttnn-tests.yaml")
OUT_DIR = Path("/localdev/arminale/tt-emule/snapshots/bh_sanity")
TARGET_SKU = "bh_p150b_civ2"

# Single variant for BH emule. Matches scripts/run_ttnn_pytests_blackhole.sh:
# slow dispatch + P100 cluster descriptor + MESH_DEVICE=P100. EMULE_MODE +
# SLOW_DISPATCH are added by the layering step; the rest is the variant
# overlay.
EMULE_ENV = {
    "TT_METAL_EMULE_MODE": "1",
    "TT_METAL_SLOW_DISPATCH_MODE": "1",
}
VARIANT_LABEL = "bh_emule"
VARIANT_ENV = {
    "MESH_DEVICE": "P100",
    # Cluster descriptor path resolved at runtime; the runner script consumes
    # $CLUSTER_EXAMPLES set by the harness (kept consistent with the existing
    # scripts/run_ttnn_pytests_blackhole.sh contract).
    "TT_METAL_MOCK_CLUSTER_DESC_PATH": "$CLUSTER_EXAMPLES/blackhole_P100.yaml",
}

# -- Helpers -----------------------------------------------------------------

ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def split_env_prefix(tokens):
    """Walk leading KEY=value tokens; return (env_dict, remaining_tokens)."""
    env = {}
    i = 0
    while i < len(tokens) and ENV_ASSIGN_RE.match(tokens[i]):
        k, _, v = tokens[i].partition("=")
        env[k] = v
        i += 1
    return env, tokens[i:]


def normalise_args(args):
    """Return a canonical bag of args for byte-equal comparison.

    Pytest args have a mix of `--flag value`, `--flag=value`, and positional
    test paths. For the audit we treat the multiset of tokens after expanding
    `--flag=value` into `--flag value` as the canonical form.
    """
    out = []
    for tok in args:
        if tok.startswith("--") and "=" in tok:
            k, _, v = tok.partition("=")
            out.append(k)
            out.append(v)
        else:
            out.append(tok)
    return sorted(out)


def strip_x_flag(args):
    """Status-snapshot mode: remove pytest's `-x` (stop-on-first-failure).

    Handles three shapes:
      * `-x` alone           → drop the token
      * `-xv` / `-xvv`       → drop the leading `x`, keep the verbose flags
      * `-vx` / `-vvx`       → drop the trailing `x`, keep the verbose flags

    Other combined short flags (`-s`, `-q`, …) are left intact. Returns the
    (new_args_list, was_stripped_bool) pair so the runner header / audit can
    report exactly which tokens were transformed.
    """
    out = []
    stripped = False
    for tok in args:
        if tok == "-x":
            stripped = True
            continue
        if len(tok) > 1 and tok.startswith("-") and not tok.startswith("--") and "x" in tok[1:]:
            new = "-" + tok[1:].replace("x", "")
            if new == "-":
                stripped = True
                continue
            out.append(new)
            stripped = True
            continue
        out.append(tok)
    return out, stripped


def inject_forked(args):
    """Status-snapshot mode: ensure pytest runs each test in its own forked
    process. Without this a single segfault truncates the rest of the entry
    (same blast-radius shape as `-x`), making the `% passing` headline depend
    on which test happens to crash first.

    Returns (new_args_list, was_injected_bool). No-op if `--forked` is already
    present.
    """
    if "--forked" in args:
        return args, False
    return ["--forked"] + args, True


def inject_continue_on_collection_errors(args):
    """Status-snapshot mode: don't let one file's ImportError block the rest.

    pytest's default behavior on collection errors is to abort the entire
    test session (the `!!! Interrupted: N errors during collection !!!`
    bailout). For status snapshots we want the tests that DID collect to
    run; the unimportable files get surfaced separately in the report as
    excluded-files.

    Returns (new_args_list, was_injected_bool). No-op if the flag is already
    present.
    """
    if "--continue-on-collection-errors" in args:
        return args, False
    return ["--continue-on-collection-errors"] + args, True


# Tokens the audit treats as the additive layer between source cmd and emitted
# invocation. Anything that appears in the emitted line and not in the source
# must be one of these.
ADDITIVE_LAYER_TOKENS = {"--forked", "--continue-on-collection-errors", "timeout"}


# -- Phase 1 -----------------------------------------------------------------

assert MANIFEST.is_file(), f"manifest missing: {MANIFEST}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / VARIANT_LABEL).mkdir(parents=True, exist_ok=True)

with MANIFEST.open() as f:
    entries = yaml.safe_load(f)

selected = []
for e in entries:
    if e.get("merge_gate", False):
        continue
    skus = e.get("skus") or {}
    if TARGET_SKU not in skus:
        continue
    selected.append(e)

if not selected:
    print("ERROR: no entries match the predicate", file=sys.stderr)
    sys.exit(1)

# -- Phase 2 — emit runner script -------------------------------------------

runner_lines = [
    "#!/usr/bin/env bash",
    "# AUTOGENERATED by snapshots/bh_sanity/generate.py — do not edit.",
    "#",
    "# Identical-invocation invariant: every test path, every original env var,",
    "# and every pytest arg from tt-metal/tests/pipeline_reorg/ttnn-tests.yaml",
    "# is preserved verbatim below. The diff vs the source cmd consists ONLY of",
    f"# the following additive layer: emule env ({', '.join(EMULE_ENV)}),",
    f"# variant env ({VARIANT_LABEL}: {', '.join(VARIANT_ENV)}), the `timeout`",
    "# wrapper, the --junitxml= flag, the stdout/stderr redirection.",
    "#",
    "# Plus three status-snapshot-mode transformations that the audit allows:",
    "#",
    "#   1. `-x` (stop-on-first-failure) is STRIPPED. Without this, the first",
    "#      failure per entry truncates the rest of the entry — distorting the",
    "#      `% passing` headline metric the snapshot is built to report.",
    "#",
    "#   2. `--forked` is INJECTED (per-test process isolation). Without this,",
    "#      a single segfault crashes the whole pytest process and the rest of",
    "#      the entry doesn't run — same shape of distortion as `-x`.",
    "#",
    "#   3. `--continue-on-collection-errors` is INJECTED. Without this, any",
    "#      one file with an ImportError (e.g. transitively importing a",
    "#      package not in the toolchain venv) aborts the entire test session",
    "#      via `!!! Interrupted: N errors during collection !!!`. With it,",
    "#      the offending file becomes a single `<error>` in the JUnit XML",
    "#      and the rest of the directory's tests run normally. The parser",
    "#      surfaces these files explicitly as excluded-files.",
    "#",
    "# All three transformations widen the test coverage of a single entry's run;",
    "# none alters the test set itself or how individual tests execute.",
    "#",
    "# Re-derive with: snapshots/bh_sanity/generate.py",
    "",
    "set -uo pipefail",
    "",
    ': "${CLUSTER_EXAMPLES:?CLUSTER_EXAMPLES must point at cluster_descriptor_examples}"',
    ': "${TT_METAL_HOME:?TT_METAL_HOME must be set}"',
    f'OUT="{OUT_DIR}/{VARIANT_LABEL}"',
    'mkdir -p "$OUT"',
    "",
]

audit_lines = []
emit_records = []
audit_ok = True


def emit_entry(entry):
    """Parse one manifest entry and return (runner_block, audit_record)."""
    global audit_ok
    name = entry["name"]
    src_cmd = entry["cmd"]
    sku_timeout_min = (entry.get("skus") or {}).get(TARGET_SKU, {}).get("timeout", 30)
    # Wallclock backstop = 3 × silicon-CI SKU budget, capped at 90 min.
    # Bumped from 2×/60-min after the conv group dry-run came within a few
    # minutes of the old cap — emule's per-test cost is heavier than
    # silicon's so the 2× ratio left no margin. 3× restores headroom and the
    # 90-min cap is high enough that no current entry pegs it without genuine
    # runaway. Hitting the cap is still surfaced loudly in the report.
    wallclock = min(3 * int(sku_timeout_min) * 60, 5400)

    src_tokens = shlex.split(src_cmd)
    src_env, src_rest = split_env_prefix(src_tokens)
    if not src_rest:
        return None, {"name": name, "status": "FAIL", "reason": "empty cmd after env"}
    src_exe = src_rest[0]
    src_args = src_rest[1:]
    is_pytest = "pytest" in src_exe

    xml_path = f'$OUT/{slug(name)}.xml'
    log_path = f'$OUT/{slug(name)}.log'

    # Status-snapshot-mode arg transformations (pytest entries only).
    # Non-pytest entries (e.g. ./tests/scripts/run_ttnn_examples.sh) get
    # neither transformation — they wrap their own pytest invocations and we
    # treat them opaquely.
    emit_args = list(src_args)
    x_stripped = False
    forked_injected = False
    coce_injected = False
    if is_pytest:
        emit_args, x_stripped = strip_x_flag(emit_args)
        emit_args, forked_injected = inject_forked(emit_args)
        emit_args, coce_injected = inject_continue_on_collection_errors(emit_args)

    # Re-assemble the invocation. Order:
    #   <emule env> <variant env> <entry env prefix> timeout N <exe> <args> [--junitxml=…]
    emitted_env = {**EMULE_ENV, **VARIANT_ENV, **src_env}
    env_segment = " ".join(f'{k}={shlex.quote(v) if not v.startswith("$") else v}'
                            for k, v in emitted_env.items())
    args_segment = " ".join(shlex.quote(a) for a in emit_args)
    junit_flag = f'--junitxml={xml_path}' if is_pytest else ''

    if is_pytest:
        cmd_line = (
            f'{env_segment} timeout {wallclock} {src_exe} {args_segment} '
            f'{junit_flag} > {log_path} 2>&1'
        )
    else:
        # Non-pytest entry (e.g. ./tests/scripts/run_ttnn_examples.sh): pass
        # the executable+args through verbatim, no --junitxml injection.
        cmd_line = (
            f'{env_segment} timeout {wallclock} {src_exe} {args_segment} '
            f'> {log_path} 2>&1'
        )

    # Per-entry ONLY filter — `ONLY=<slug>` env var skips every entry whose
    # slug doesn't match. `ONLY` unset means "run all" (backward-compatible
    # behavior). Used to step through entries one at a time with checkpoints.
    block = [
        f'if [ -z "${{ONLY:-}}" ] || [ "${{ONLY}}" = "{slug(name)}" ]; then',
        f'  echo "::group::{name}"',
        f'  echo "# source: {MANIFEST}"',
        f'  echo "# entry:  {name}"',
        f'  {cmd_line}',
        f'  rc=$?; echo "rc=$rc (entry={name})"',
        f'  echo "::endgroup::"',
        f'fi',
        "",
    ]

    # ---- Audit ----
    audit_reasons = []
    re_tokens = shlex.split(cmd_line)
    emit_env, emit_rest = split_env_prefix(re_tokens)

    # Drop the `timeout <N>` prefix the layer added.
    if emit_rest[:1] == ["timeout"]:
        emit_rest = emit_rest[2:]
    else:
        audit_reasons.append("emitted line missing `timeout` wrapper")

    if not emit_rest:
        audit_reasons.append("emitted line has no executable")
    else:
        emit_exe = emit_rest[0]
        emit_args = emit_rest[1:]
        if emit_exe != src_exe:
            audit_reasons.append(
                f"executable changed: source={src_exe!r} emitted={emit_exe!r}")

        # Strip the appended --junitxml=… and the trailing redirection from
        # the audit's emitted-args view, since both are part of the additive
        # layer.
        emit_audit_args = emit_rest[1:]
        cleaned = []
        skip_next = False
        for tok in emit_audit_args:
            if skip_next:
                skip_next = False
                continue
            if tok.startswith("--junitxml="):
                continue
            if tok in (">", ">>"):
                skip_next = True
                continue
            if tok in ("2>&1",):
                continue
            cleaned.append(tok)

        # Reverse the status-snapshot-mode transformations before comparing
        # to the source args. The audit then proves the emitted args are
        # byte-equal to source MODULO those two allowed transformations.
        canonical = list(cleaned)
        if forked_injected:
            try:
                canonical.remove("--forked")
            except ValueError:
                audit_reasons.append("forked injection was claimed but `--forked` not present in emitted line")
        if coce_injected:
            try:
                canonical.remove("--continue-on-collection-errors")
            except ValueError:
                audit_reasons.append("coce injection was claimed but `--continue-on-collection-errors` not present in emitted line")
        if x_stripped:
            # Re-introduce the bare `-x` so the comparison passes — except if
            # source actually had `-xv` etc., in which case we need to map our
            # stripped form back. Easier: drop `-x` shape from source side too.
            normalised_src = []
            for tok in src_args:
                if tok == "-x":
                    continue
                if len(tok) > 1 and tok.startswith("-") and not tok.startswith("--") and "x" in tok[1:]:
                    repl = "-" + tok[1:].replace("x", "")
                    if repl != "-":
                        normalised_src.append(repl)
                    continue
                normalised_src.append(tok)
            cmp_src = normalised_src
        else:
            cmp_src = src_args

        if normalise_args(canonical) != normalise_args(cmp_src):
            audit_reasons.append("pytest args differ (modulo ordering and the two snapshot-mode transformations)")
            audit_reasons.append(f"  source args (normalised): {sorted(cmp_src)}")
            audit_reasons.append(f"  emitted args (less --forked / less --junitxml): {sorted(canonical)}")

    # Env-prefix check: every source env var must be present in emitted env,
    # with the same value. Emitted may have extra (emule + variant) — that's
    # the additive layer.
    for k, v in src_env.items():
        if emit_env.get(k) != v:
            audit_reasons.append(
                f"env var lost or changed: {k}={v!r} (emitted={emit_env.get(k)!r})")

    # Additive-only check: every extra emitted env var must come from EMULE_ENV
    # or VARIANT_ENV.
    allowed_extras = set(EMULE_ENV) | set(VARIANT_ENV)
    for k in emit_env:
        if k not in src_env and k not in allowed_extras:
            audit_reasons.append(f"emitted line introduces unexpected env var: {k}")

    status = "PASS" if not audit_reasons else "FAIL"
    if status == "FAIL":
        audit_ok = False
    return block, {
        "name": name,
        "status": status,
        "reasons": audit_reasons,
        "src_cmd": src_cmd.strip(),
        "is_pytest": is_pytest,
        "wallclock": wallclock,
    }


for entry in selected:
    block, record = emit_entry(entry)
    if block:
        runner_lines.extend(block)
    emit_records.append(record)

# Coverage audit: every selected entry must produce exactly one emitted block.
seen_names = set()
for r in emit_records:
    if r["name"] in seen_names:
        audit_ok = False
        r["reasons"].append("entry appeared in more than one emitted block")
    seen_names.add(r["name"])

# -- Phase 3 — write the audit log -------------------------------------------

audit_lines.append(f"AUDIT: BH post-commit / {TARGET_SKU}")
audit_lines.append(f"manifest:   {MANIFEST}")
audit_lines.append(f"predicate:  merge_gate is falsy AND {TARGET_SKU!r} in skus")
audit_lines.append(f"selected:   {len(emit_records)} entries")
audit_lines.append("")

passed = sum(1 for r in emit_records if r["status"] == "PASS")
failed = sum(1 for r in emit_records if r["status"] == "FAIL")
audit_lines.append(f"  PASS: {passed}")
audit_lines.append(f"  FAIL: {failed}")
audit_lines.append("")

for r in emit_records:
    audit_lines.append(f"--- {r['name']} [{r['status']}] ---")
    audit_lines.append(f"  pytest entry: {r.get('is_pytest', '?')}")
    audit_lines.append(f"  wallclock:    {r.get('wallclock', '?')}s")
    audit_lines.append(f"  source cmd:")
    for ln in r["src_cmd"].splitlines():
        audit_lines.append(f"    {ln}")
    if r.get("reasons"):
        audit_lines.append(f"  reasons:")
        for reason in r["reasons"]:
            audit_lines.append(f"    - {reason}")
    audit_lines.append("")

audit_lines.append("AUDIT PASS" if audit_ok else "AUDIT FAIL")

# -- Write outputs -----------------------------------------------------------

runner_path = OUT_DIR / "run_bh_sanity.sh"
audit_path = OUT_DIR / "audit_bh_sanity.log"
runner_path.write_text("\n".join(runner_lines) + "\n")
audit_path.write_text("\n".join(audit_lines) + "\n")
os.chmod(runner_path, 0o755)

print(f"wrote {runner_path}")
print(f"wrote {audit_path}")
print(f"selected: {len(emit_records)} entries — PASS {passed}, FAIL {failed}")
sys.exit(0 if audit_ok else 2)
