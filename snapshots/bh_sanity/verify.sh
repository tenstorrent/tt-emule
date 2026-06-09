#!/usr/bin/env bash
# Production Phase 8 verify. Exits 0 if all checks pass, 1 otherwise.

set -uo pipefail

OUT_DIR="/localdev/arminale/tt-emule/snapshots/bh_sanity"
EXEC="$OUT_DIR/bh_sanity_status_exec.md"
DEV="$OUT_DIR/bh_sanity_status_dev.md"
XML_DIR="$OUT_DIR/bh_emule"
AUDIT="$OUT_DIR/audit_bh_sanity.log"

JARGON_TOKENS=(wave)

fail=0
ok()  { printf '  \033[32mPASS\033[0m  %s\n' "$*"; }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=$((fail+1)); }
section() { echo; echo "== $* =="; }

section "Artefacts present"
for f in "$EXEC" "$DEV"; do
    if [ -s "$f" ]; then ok "$f"; else bad "$f missing or empty"; fi
done
xml_count=$(ls "$XML_DIR"/*.xml 2>/dev/null | wc -l)
log_count=$(ls "$XML_DIR"/*.log 2>/dev/null | wc -l)
if [ "$xml_count" -gt 0 ]; then ok "$xml_count XML files in $XML_DIR"; else bad "no XML files in $XML_DIR"; fi
if [ "$log_count" -gt 0 ]; then ok "$log_count log files in $XML_DIR"; else bad "no log files in $XML_DIR"; fi

section "XML well-formed (sampling all)"
malformed=0
for x in "$XML_DIR"/*.xml; do
    [ -f "$x" ] || continue
    python3 -c "import xml.etree.ElementTree as ET; ET.parse('$x')" 2>/dev/null \
        || { bad "$x fails to parse"; malformed=$((malformed+1)); }
done
[ "$malformed" = "0" ] && ok "all $xml_count XML files parse"

section "Banner present"
grep -q "BH " "$EXEC" 2>/dev/null && ok "exec has a header" || bad "exec missing recognizable header"
grep -q "## Headline" "$EXEC" 2>/dev/null && ok "exec has Headline section" || bad "exec missing Headline"

section "Jargon scan on exec"
for tok in "${JARGON_TOKENS[@]}"; do
    c=$(grep -ic "\\b${tok}\\b" "$EXEC" 2>/dev/null)
    c=${c:-0}
    if [ "$c" = "0" ]; then ok "no '$tok' (count=$c)"; else bad "found '$tok' x$c in exec"; fi
done

section "Section coverage"
for s in "## Headline" "## Status" "## What \"skipped by design\"" "## Top 3 gaps" "## Top 3 next steps" "## Variant readiness" "## Recent landmarks"; do
    if grep -qF "$s" "$EXEC"; then ok "exec has '$s'"; else bad "exec missing '$s'"; fi
done
for s in "## Headline" "## Fully-passing files" "## Partial-pass files" "## All-failing files" "## Blocker taxonomy" "## Coverage analysis" "## Excluded test files (collection errors)" "## Variant readiness" "## Top suggested next steps" "## References" "## Truncated entries"; do
    if grep -qF "$s" "$DEV"; then ok "dev has '$s'"; else bad "dev missing '$s'"; fi
done

section "Audit log integration"
if [ -s "$AUDIT" ]; then
    ok "audit log exists"
    tail -1 "$AUDIT" | grep -q "AUDIT PASS" && ok "audit ends with AUDIT PASS" || bad "audit not AUDIT PASS"
    grep -qF "$AUDIT" "$DEV" && ok "dev references audit log path" || bad "dev does not reference audit log path"
else
    bad "audit log missing"
fi

section "Path resolution (dev report)"
missing=0
while IFS= read -r p; do
    case "$p" in
        *"::"*|*"["*) continue ;;
    esac
    if [ -e "$p" ] || [ -e "/localdev/arminale/tt-emule/$p" ] || [ -e "/localdev/arminale/tt-metal/$p" ]; then
        continue
    fi
    bad "unresolved path in dev: $p"
    missing=$((missing+1))
done < <(grep -oE '`[^`]+`' "$DEV" | tr -d '`' | grep -E '^(/|snapshots/|tests/|docs/|scripts/)' | sort -u)
[ "$missing" = "0" ] && ok "all cited paths resolve" || true

section "Totals consistency (headline P vs aggregated XML)"
xml_p=$(python3 -c "
import xml.etree.ElementTree as ET, glob
p = 0
for path in sorted(glob.glob('$XML_DIR/*.xml')):
    root = ET.parse(path).getroot()
    suites = root.findall('testsuite') if root.tag != 'testsuite' else [root]
    for s in suites:
        for tc in s.findall('testcase'):
            if tc.find('failure') is None and tc.find('error') is None and tc.find('skipped') is None:
                p += 1
print(p)")
hdr_p=$(grep -oE '\| Tests passing \| [0-9]+ \|' "$EXEC" | grep -oE '[0-9]+' | head -1)
if [ "$xml_p" = "$hdr_p" ]; then
    ok "headline P=$hdr_p matches aggregated XML P=$xml_p"
else
    bad "headline P=$hdr_p but aggregated XML P=$xml_p"
fi

section "Truncation surfacing"
trunc_line=$(grep -oE '\| \*\*Truncated entries\*\* \| \*\*[0-9]+\*\* \|' "$EXEC" | head -1)
if [ -n "$trunc_line" ]; then
    n=$(echo "$trunc_line" | grep -oE '[0-9]+')
    ok "exec reports truncated entries count = $n"
else
    bad "exec missing Truncated entries row in Status table"
fi

echo
if [ "$fail" = "0" ]; then
    echo "VERIFY PASS"
    exit 0
else
    echo "VERIFY FAIL ($fail check(s) failed)"
    exit 1
fi
