# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** **Stage 1 (the post-M6 beta gate) is underway** — bringing Ch 1–10 up to the
Ch 11–17 quality bar (they were authored before the `claim-skeptic` discipline existed). The
17-chapter book itself is content-complete (M6) + deployed; this is hardening, not authoring.
**Phase A (gather) COMPLETE:** A1 re-verified the 2026-05-27 deeper audit against current state
(30/37 already fixed — *every* CRITICAL, incl. the F29 Julia exp-trap **math bug**, F2/F3, F4, F6);
A2 ran the **first-ever `claim-skeptic` Workflow** over Ch 1–10 (run `wf_647b37ff-ec5`; **42 findings**
— genuine math errors + misattributions, not just nits); A3 merged + triaged →
**`audits/2026-06-13_post-m6_recheck.md`** (**12 must-fix + 23 should-fix**). **Phase B (remediate)
underway:** **B1a** ships the Ch 1–3 fixes — R2 (the Jordan-block-size formula was *false*, refuted
by the chapter's own Ex 3.1), R1 (Lyapunov kernel-trivial reason), R3 (Longhorn `<Cite>` key +
DeltaNet/Longhorn explicit-vs-implicit attribution), R10/R11 (ch01 energy caption + degenerate-node
table). Gate green (validate + build).

**Why:** the six-chapter campaign (approved 2026-06-10, order **12 → 14 → 16 → 13 → 15 → 17**) is
**COMPLETE — 6/6.** The 17-chapter book is content-complete; cadence proven **11×** (Ch 7–17).

**Next step:** continue Phase B PRs (each: `make check-local-torch` + `npm run build` + review
subagents → push → CI → merge → deploy → live check; doc-sync rides IN). Backlog = the recheck audit:
- **B1b** — Ch 4–6 prose + companions: R4 (ZOH-modulus claim, self-contradictory + load-bearing for
  the Ch 6 symplectic motivation), R5 (the trilemma/NoteBox contradict the chapter's own Theorem 5.3),
  R6 (HLW `<Cite>` key → `hairer2006geometric`), R7 (modified-Hamiltonian theorem missing analyticity
  + Δ-threshold), R12/R14/R16/R17/R18/R19, F16; companion eigenvalue comments R13/R15 (√15/4 →
  √15.75/2, 6 files); F7/#4 (ch04 julia → default gate).
- **B2** — Ch 7–10 + the deferred **F14/R8** (HiPPO conditioning: **verify vs the yu2023/park2024
  PDFs first** — R8 says park2024 proves no conditioning result; sub-quadratic is yu2023's, for a
  *different* matrix) + R9 (ch10 §10.2≢§4.5) + R20–R27 + R38 (Mamba-3 fact-check vs local
  `2603.15569.pdf`) + hygiene F9/F36/F15/F17.
- **B3** — `citation-link-auditor` repo sweep (cross-checks the R3/R6 cite-key fixes). **B4** — STYLE
  §8 refresh + remove the DASHBOARD `claim-skeptic`-not-run note (`docs/DASHBOARD.md:53/61/63` —
  Stage-1 done-signal, after B1+B2 land).

Then **Stage 2** (tooling #26 + the v4.16.0+ toolkit re-bump for closed-upstream #126/#135 on an
isolated branch, guarding open #140/#141), **Stage 3** (docs + #14), and the readiness-gated **M7**
(fold C1/B empirical results into Ch 15/16/17, flip `alpha→beta`). **Hold `alpha` until M7.** The
pilots execute in `post_transformers` (the book owes them nothing; M6 surfaced as issue #54).

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
