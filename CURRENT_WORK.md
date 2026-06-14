# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** **R38 shipping — the last beta-gate content fact-check; the book-side roadmap is
otherwise in steady state.** Stages 1 (Ch 1–10 hardening, PRs #32–#36) and 2 (tooling/CI + the toolkit
re-bump to 4.23.0, PRs #37–#38) are COMPLETE + deployed. **R38** (`fix/r38-mamba3-mimo-factcheck`)
verified ch10's three Mamba-3 attributions against the paper (`2603.15569.pdf`): **λ = σ(u_t)** (App. A.3
+ Table 8) and **RoPE on B,C** (§3.2 Prop 2/3, §3.4 "RoPE trick") are exactly correct — no change;
**MIMO** (§10.6) was reframed — the paper presents it as a model-quality/expressivity lever (a +1.2-pt
avg downstream gain over SISO, §4.1.1; abstract "for better model performance"), not "engineering rather
than dynamics," while keeping the correct "orthogonal to the discretization/stability story" point. One
prose edit + a STATUS regen (ch10 Lines 450→453); `npm run build` + `make check` green.

**Why:** the six-chapter campaign (approved 2026-06-10, order **12 → 14 → 16 → 13 → 15 → 17**) is
**COMPLETE — 6/6.** The 17-chapter book is content-complete; cadence proven **11×** (Ch 7–17).

**Next step:** **Steady state.** The book is content-complete (M6), beta-gate-hardened (Stages 1–2),
and now R38-fact-checked. Only low-priority Track-C housekeeping remains (F10 dep bumps, F37 dep pins,
F21/F22/F25 bib hygiene — do-when-triggered, not scheduled). The next real milestone is the
readiness-gated **M7** (fold C1/B pilot results into Ch 15/16/17, flip `alpha→beta`), gated on pilot
execution in `post_transformers`. **Hold `alpha` until M7.**

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
