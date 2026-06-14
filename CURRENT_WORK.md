# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** **Post-M7-staging cleanup PR** (`chore/repo-cleanup-banner-bib-astro`) — (1) closed the
stale 2026-05-25 audit-umbrella **#1** (its punch list was long since resolved); (2) corrected **README:5**'s
false "pre-release banner is live site-wide" claim (the toolkit's `PreReleaseBanner` is unwired here — this
book uses the toolkit's auto-injected layouts, so a real banner needs a toolkit `releaseStatus` feature,
requested separately); (3) **de-anonymized the F21 bib entry** — `anonymous2025lyapunov` was not only
Anonymous but carried a *wrong title*; now Halloran, Gulati & Roysdon, "Mamba State-Space Models Are
Lyapunov-Stable Learners", TMLR 2025 (arXiv 2406.00209), bibkey kept for cite stability; (4) bumped **astro
`^6.1.7`→`^6.4.6`** (katex 0.16→0.17 deferred — shared with the toolkit's rehype-katex). `npm run build`
+ `make check` + check-bibkeys green; the astro rebuild is the only deploy-affecting change (content-identical).

Before this, **M7 stage 1 (staging the pilot-results fold-in) shipped** (PR #40 + post_transformers #55):
`docs/m7-pilot-integration-plan.md` = the fold-in catalog (ch17 §17.2→§17.2b / §17.3→§17.3b / §17.5
rewrite) + the C1/B data contracts (idealized→trained-model inputs) + the flip mechanism. M7 itself (fold
C1/B trained-model results into Ch 14–17, flip alpha→beta) stays readiness-gated on the pilots in
post_transformers. Stages 1 (#32–#36), 2 (#37–#38), 3/R38 (#39) before it are all done + deployed.

**Why:** the six-chapter campaign (approved 2026-06-10, order **12 → 14 → 16 → 13 → 15 → 17**) is
**COMPLETE — 6/6.** The 17-chapter book is content-complete; cadence proven **11×** (Ch 7–17).

**Next step:** **M7 completion** — readiness-gated on the C1/B pilots producing trained-model results in
`post_transformers` (per the data contract in `docs/m7-pilot-integration-plan.md`). When results land,
the fold-in is turnkey (that doc's execution checklist): fill §17.2b/§17.3b, rewrite §17.5, update
ch14/15/16 cross-refs, flip alpha→beta + the doc-status sweep. Otherwise only low-priority Track-C
housekeeping remains (F10/F37/F21/F22/F25 — do-when-triggered). **Hold `alpha` until M7.**

Deferred-with-notes in `audits/2026-06-13_post-m6_recheck.md` (accepted, not bugs): R23 (ch08
resolvent — a self-flagged non-implemented sketch), R24 (ch09 residual magnitudes — prose already says
"measured … pinned below 1e-12"). **R38 resolved 2026-06-14** (ch10 Mamba-3 fact-check vs the local
`2603.15569.pdf`: λ=σ(u_t) + RoPE-on-B,C verified correct; MIMO reframed as also a quality lever).
Pilots execute in `post_transformers` (M6 = issue #54).

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
