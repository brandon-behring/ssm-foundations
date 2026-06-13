# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** **Stage 1 (the post-M6 beta gate) is COMPLETE** — Ch 1–10 brought up to the Ch 11–17
quality bar (they predated the `claim-skeptic` discipline). The 17-chapter book stays content-complete
(M6) + deployed; this was hardening, not authoring. **Phase A:** A1 re-verified the 2026-05-27 audit
(30/37 already fixed, *every* CRITICAL incl. the F29 Julia exp-trap math bug); A2 ran the first-ever
`claim-skeptic` over Ch 1–10 (42 findings); A3 → `audits/2026-06-13_post-m6_recheck.md`. **Phase B —
five PRs merged:** **#32** (Ch 1–3: the *false* Jordan-block-size formula R2, Lyapunov reason R1,
Longhorn cite R3), **#33** (Ch 4–6: ZOH-aliasing R4, trilemma-vs-Thm-5.3 R5, HLW cite R6,
theorem-hypotheses R7, + 6 companion eigenvalue comments), **#34** (Ch 7 HiPPO conditioning R8 —
park2024→yu2023, PDF-grounded), **#35** (Ch 8/10: §10.2≢§4.5 R9, S4 Õ(N+L) R22, exp-trap C²), and the
**close-out** (re-sweep follow-ups + hygiene F9/F15/F17/F36 + STYLE §8/§13 + the DASHBOARD done-signal).
**Verification:** the claim-skeptic re-sweep found + fixed 4 residual issues (ch04/ch10); the
citation-link-auditor returned GREEN.

**Why:** the six-chapter campaign (approved 2026-06-10, order **12 → 14 → 16 → 13 → 15 → 17**) is
**COMPLETE — 6/6.** The 17-chapter book is content-complete; cadence proven **11×** (Ch 7–17).

**Next step:** **Stage 2 (tooling/CI).** #26 (`generate-status.mjs --check` validates table *content*,
not just the Verified date); **F7/#4** (add `companions/ch04/julia` to the default `make
companion-julia-tests` loop — deferred from Stage 1 because it needs a local julia `Pkg.instantiate`
+ run to verify; the `Manifest.toml` exists); the **v4.16.0+ toolkit re-bump** for closed-upstream
#126/#135 on an isolated branch (guard open #140/#141). Then **Stage 3** (docs + #14) and the
readiness-gated **M7** (fold C1/B empirical results into Ch 15/16/17, flip `alpha→beta`). **Hold
`alpha` until M7.**

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
