# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** **Stage 2 (tooling/CI) — shipping PR B (the toolkit re-bump); PR A merged + deployed.**
Stage 1 (the post-M6 beta gate) is COMPLETE (Ch 1–10 hardened to the Ch 11–17 bar across PRs #32–#36;
`claim-skeptic` now reflected on all 17 chapters). **PR A** (#37, merged) closed the tooling residuals:
**#26** content-validating `status-check` (regenerate-and-diff + a `node --test` guard in `make check`;
**it caught a real Stage-1 STATUS.md drift on first run** — ch03 Cites 7→9, ch06 5→4, ch10 Lines 449→450),
**F7/#4** ch04-Julia in the default loop (+ a `companion-julia-instantiate` target), **F26/F27** ch01–03
literal torch↔jax parity (9 tests; companion-verifier clean). **PR B** (`chore/stage2-toolkit-rebump`)
re-bumps the toolkit **4.16.0 → 4.23.0** — landing **#126** (theorem headings auto-number to match
`<XRef>`; preview-verified: ch09 `ssd-duality` heading = XRef = "Theorem 9.5") + **#135** (sidebar
subtitle, closing **#14** — the "A scaffold-astro book" placeholder is gone) — #140/#141 confirmed
harmless to this root-deployed book. Preview gate passed (`npm run build` green; headings numbered;
subtitle on all 22 pages). `npm ci`/lockfile pins 4.23.0 (the approved target; `^4.23.0` resolved to a
same-day 4.24.0, re-pinned to the vetted version).

**Why:** the six-chapter campaign (approved 2026-06-10, order **12 → 14 → 16 → 13 → 15 → 17**) is
**COMPLETE — 6/6.** The 17-chapter book is content-complete; cadence proven **11×** (Ch 7–17).

**Next step:** **Stage 2 is complete** once PR B merges + deploys (then close it out: memory + the
roadmap flip). After that, **Stage 3** is minimal — #14 was pulled forward into Stage 2 and F9/F36 shipped
in #36, so it's just any residual hygiene (F8/F11/F0) in one small PR — followed by the readiness-gated
**M7** (fold C1/B empirical results into Ch 15/16/17, flip `alpha→beta`). **Hold `alpha` until M7.**

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
