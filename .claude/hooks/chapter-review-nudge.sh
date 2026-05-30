#!/usr/bin/env bash
#
# PostToolUse nudge — ssm-foundations
#
# After an Edit/Write/MultiEdit, if the touched file is book content, emit a
# reminder (via hookSpecificOutput.additionalContext) pointing the main agent at
# the matching read-only review subagent. The hook does NOT invoke the agent
# itself — it nudges; the main agent decides.
#
#   src/content/chapters/*.mdx  -> chapter-auditor + prose-pedagogy-reviewer
#   companions/**               -> companion-verifier
#   bibliography.bib            -> citation-link-auditor
#
# Contract: read-only, fail-open (never block an edit), fast (<100ms). Requires
# jq (already used by the repo's other hooks).
#
set -uo pipefail

# Fail-open: any read error => no output, exit 0.
input=$(cat 2>/dev/null) || exit 0

# Touched file path. Support both current (.tool_input) and legacy (.parameters)
# payload shapes so the nudge survives a hook-schema change.
fp=$(printf '%s' "$input" | jq -r '.tool_input.file_path // .parameters.file_path // empty' 2>/dev/null) || exit 0
[ -z "$fp" ] && exit 0

msg=""
case "$fp" in
  *src/content/chapters/*.mdx)
    chap=$(basename "$fp")
    msg="Edited chapter ${chap}. Before advancing its frontmatter status:, consider delegating to the chapter-auditor subagent (STYLE.md + mechanical gates) and the prose-pedagogy-reviewer subagent (teaching quality). Both are read-only."
    ;;
  *companions/*)
    msg="Edited companion code (${fp#*companions/}). Consider delegating to the companion-verifier subagent to run the jax/julia/torch suites, check JAX<->Julia parity, and confirm figure/caption claims."
    ;;
  *bibliography.bib)
    msg="Edited bibliography.bib. Consider delegating to the citation-link-auditor subagent for bibkey hygiene, <Cite> resolution, and cross-repo link freshness."
    ;;
  *)
    exit 0
    ;;
esac

# Emit additionalContext (JSON-escaped) and exit 0.
esc=$(printf '%s' "$msg" | jq -Rs .)
cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "PostToolUse",
    "additionalContext": $esc
  }
}
EOF
exit 0
