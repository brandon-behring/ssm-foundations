# Audit Reports

Independent reviews of the `ssm-foundations` repository. The top level
contains one current canonical audit plus this navigation file. Older audits
and draft plans live in `archive/` and are historical evidence only.

## Current Canonical Audit

Read this first for the current standards, infrastructure, and authoring-rigor
state of the book repo:

- [`2026-05-25_standards_vs_post_transformers.md`](2026-05-25_standards_vs_post_transformers.md)

This is the inaugural canonical audit. It establishes the F-numbered findings
format, Track A/B/C remediation taxonomy, bracketed-text status vocabulary, and
the pilot-imminent promotion gate. The format mirrors
[`post_transformers/audits/archive/2026-04-11_repo_maintainability_audit.md`](https://github.com/brandon-behring/post_transformers/blob/main/audits/archive/2026-04-11_repo_maintainability_audit.md)
with the deliberate deviation of text-label status markers
(`[open]` / `[tracked]` / `[fixed]` / `[pilot-blocked]`) in place of emoji.

## Archive

Prior audits live in `archive/` (currently empty — 2026-05-25 is the first
audit). When a future audit supersedes the current canonical, move the prior
file with `git mv` and update `archive/ARCHIVE_INDEX.md`.

Archived files may still contain useful methodology, evidence trails, and
historical rationale, but they are not current truth unless a newer canonical
audit revalidates them.

## Adding A Future Audit

1. Name the file `YYYY-MM-DD_short_scope.md`.
2. Make the new file the only current canonical audit referenced in this README.
3. Move superseded top-level audit files into `archive/` with `git mv`.
4. Create or update `archive/ARCHIVE_INDEX.md` with what remains durable and
   what is stale.
5. Keep command evidence, contradiction tables, assumptions, and unresolved
   questions in the audit itself.
6. Apply the pilot-imminent promotion gate: any finding whose unblock event
   falls within ≤14 days of audit date is Track A regardless of effort.
