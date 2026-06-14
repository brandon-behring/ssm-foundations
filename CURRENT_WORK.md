# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** **Stage 2 (tooling/CI) — PR A shipping.** Stage 1 (the post-M6 beta gate) is COMPLETE
(Ch 1–10 hardened to the Ch 11–17 bar across PRs #32–#36; `claim-skeptic` now reflected on all 17
chapters). **Stage 2 PR A** (`chore/stage2-tooling-ci`) closes the tooling residuals in one PR, no
chapter-content change: **#26** — `generate-status.mjs --check` now regenerates-and-diffs against
`docs/STATUS.md` (not just the Verified date), with a `node --test` suite wired into `make check` via a
new `test-scripts` target; **it caught a real Stage-1 drift on its first run** (ch03 Cites 7→9, ch06 5→4,
ch10 Lines 449→450 → STATUS.md regenerated). **F7/#4** — ch04 Julia folded into the default
`companion-julia-tests` loop (verified 10/10 + full loop green locally; one-time `Pkg.instantiate`
documented). **F26/F27** — confirmed ch01–10 + backfilled ch01–03 literal torch↔jax parity
(`test_*_parity.py`, 9 tests; companion-verifier clean). Doc-sync (audit + DASHBOARD + this file) rides in.

**Why:** the six-chapter campaign (approved 2026-06-10, order **12 → 14 → 16 → 13 → 15 → 17**) is
**COMPLETE — 6/6.** The 17-chapter book is content-complete; cadence proven **11×** (Ch 7–17).

**Next step:** **Stage 2 PR B — the toolkit re-bump.** Bump `@brandon_m_behring/book-scaffold-astro`
4.16.0 → 4.23.0 (closed-upstream **#126** auto-numbered theorem headings @ v4.18 + **#135** sidebar
subtitle @ v4.21; open **#140/#141** base-link bugs don't affect this root-deployed book) and land the
**#14** sidebar subtitle in the same preview-gated PR (`npm run build` + `npm run preview` → verify
theorem heading-numbers match XRef + subtitle shows, before merge=deploy). Then **Stage 3** (hygiene-only
— #14 pulled forward) and the readiness-gated **M7** (fold C1/B empirical results into Ch 15/16/17, flip
`alpha→beta`). **Hold `alpha` until M7.**

Deferred-with-notes in `audits/2026-06-13_post-m6_recheck.md` (accepted, not bugs): R23 (ch08
resolvent — a self-flagged non-implemented sketch), R24 (ch09 residual magnitudes — prose already says
"measured … pinned below 1e-12"), R38 (ch10 Mamba-3 attribution — beta-gate fact-check vs local
`2603.15569.pdf`, low risk). Pilots execute in `post_transformers` (M6 = issue #54).

**Context when I return:**
- Per-chapter cadence (for reference / future books): brief → `/exploring-options` (4 questions) →
  companions-first → prose → wire-up → all four review subagents → one PR (doc-sync rides IN it) →
  merge=deploy → memory. Gates: `make check-local-torch` + `npm run build` (the only MDX compiler).
- **MDX/gotcha catalog (the full campaign's lessons):** every inline `$...$` span on ONE physical
  line (a `-`/`+`/`*`-leading wrapped continuation breaks acorn — ch13); an unquoted frontmatter
  `description:` containing `: ` breaks the build (YAML map — ch15); **the validator greps for
  `<Theorem` even inside `{/* */}` macros comments → never write the literal `<Theorem>` in a
  comment, use "Theorem" without brackets (ch17)**; matplotlib mathtext ≠ KaTeX (`\*` invalid, use
  `^*`); never hard-code Theorem numbers (XRef self-refs; no XRef inside Figure captions).
- **Synthesis-chapter pattern (ch17):** integration companions COMPOSE existing instruments into a
  NEW measured signature (reductions-to-components are consistency checks, not the headline);
  status:implemented needs prose+exercises+companions, NOT theorems (0 is fine); STYLE §13
  positional accommodation for <6 content sections (Ch 5 precedent); verify numerical behavior
  empirically before stating it (the secular-vs-endpoint metric distinction).
- Post-ship checklist (drift guard): a chapter PR updates CLAUDE.md status lines, README (banner +
  row), `docs/DASHBOARD.md` (row + verified + trust notes), regen `docs/STATUS.md`, refresh this file.
