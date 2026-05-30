# `.claude/` — project subagents for ssm-foundations

Four **read-only review/verify** subagents that isolate high-volume,
well-specified, verifiable checks from the main authoring thread. The principle:
delegate when *intermediate work ≫ conclusion* and the task is specified by a
pointer (a chapter slug), not by the conversation. Authoring stays in the main
thread by design — these agents review and verify; they never write.

## The agents (`.claude/agents/`)

| Agent | Fires when | What it does | Tools | Model |
|---|---|---|---|---|
| **chapter-auditor** | a chapter `.mdx` is drafted/edited, before advancing `status:` | STYLE.md compliance + `status:` truthfulness + notation/ID hygiene; runs `make lint` / `make validate` / status-check. Emits a severity-ranked findings table (Track A/B/C). | Read, Grep, Glob, Bash | inherit |
| **companion-verifier** | companion code under `companions/chXX/**` changes | runs jax/julia/torch suites, checks JAX↔Julia numeric parity, verifies figures exist + caption numbers match. Returns pass/fail + failures only. | Read, Grep, Bash | sonnet |
| **prose-pedagogy-reviewer** | a prose draft reads complete (before the audit) | qualitative teaching-quality review vs STYLE.md §§9–10 and the Ch 1–6 exemplars (voice, narrative, pilot hook, references). | Read, Grep, Glob | inherit |
| **citation-link-auditor** | `bibliography.bib` changes / periodic pre-release | bibkey hygiene + `<Cite>` resolution + unused-entry detection + cross-repo URL freshness (WebFetch). Repo-wide, deliberate cadence. | Read, Grep, Bash, WebFetch | sonnet |

All four are **findings-only**: they report; the main thread decides what to fix.
None has `Write`/`Edit`.

## How they fire (auto-delegation, no explicit ask)

Two layers, because a `description:` enables but does not *guarantee* timely firing:

1. **Precise `description:` fields** — each says exactly when to fire; this drives
   the main agent's auto-delegation.
2. **`PostToolUse` reminder hook** (`hooks/chapter-review-nudge.sh`, wired in
   `settings.json`) — at the moment an `Edit`/`Write`/`MultiEdit` touches book
   content, it injects a reminder naming the matching agent. The hook is
   read-only, fail-open, and does NOT invoke the agent — it nudges.

To invoke explicitly instead, just ask: e.g. *"audit ch08 with chapter-auditor"*
or *"run companion-verifier on ch07"*.

## `settings.json`

- **Scoped Bash allowlist** — `make`, `node scripts/`, `npm run`, `julia`,
  `pytest`, `.venv/bin/pytest`. Note: with the global `Bash(*:*)` + permissive
  mode, this is primarily **documentation of intent**; the hard constraint on
  each agent is its restricted `tools:` list + system-prompt discipline.
- **`PostToolUse` hook** — the nudge described above.

## Design notes

- **Why no authoring agents?** Drafting prose/exercises produces output that *is*
  the deliverable — nothing to isolate, and it's collaborative. It stays in the
  main thread. If Ch 7–17 ever needs orchestrated fan-out, reach for a multi-agent
  **Workflow** (stateful, multi-stage), not a stateless subagent.
- **Division of labor** — `chapter-auditor` = mechanical/standards;
  `prose-pedagogy-reviewer` = qualitative teaching; `companion-verifier` = code +
  figures; `citation-link-auditor` = repo-wide citation/link. They deliberately
  do not overlap (auditor checks per-chapter `<Cite>` resolution; the
  citation-link-auditor owns the repo-wide + URL-freshness sweep).
- All checks **reuse existing repo tooling** (`Makefile`, `scripts/*.mjs`) rather
  than reimplementing — see `../STYLE.md`, `../Makefile`, `../audits/`.
