# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** **M7 — stage 1 (staging the pilot-results fold-in).** Everything before M7 is done +
deployed: Stages 1 (Ch 1–10 hardening, #32–#36), 2 (tooling/CI + toolkit 4.23.0, #37–#38), 3/R38 (the
ch10 Mamba-3 fact-check, #39). M7 (fold C1/B *trained-model* results into Ch 14–17, flip alpha→beta) is
readiness-gated on the pilots executing in `post_transformers` — so stage 1 is **the prep doable now**:
a book-side spec doc (`docs/m7-pilot-integration-plan.md`) with the **fold-in catalog** (the ~23 hedged
passages M7 updates, concentrated in ch17 §17.2→§17.2b / §17.3→§17.3b / §17.5 rewrite) + the **C1/B data
contracts** (idealized→trained-model inputs the pilots must produce, so the fold-in is mechanical) + the
**flip mechanism**, PLUS a cross-repo `tracked` issue in post_transformers surfacing that contract to the
pilots. **Recon finding:** the toolkit's `PreReleaseBanner` is unwired and README:5's "banner is live" is
stale — so the alpha→beta "flip" is currently a doc-status change, not a live banner toggle (recorded in
the spec for M7 completion). No `src/` change → no deploy impact.

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
